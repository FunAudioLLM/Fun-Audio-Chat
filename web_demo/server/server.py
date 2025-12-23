# Copyright (c) 2025, Alibaba Cloud and its affiliates;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
server script for FunAudioChat s2s model
"""

import argparse
import asyncio
import os
import aiohttp
from aiohttp import web
import numpy as np
import sphn
import soundfile as sf
from datetime import datetime
import torch
import torchaudio
import json
import math
import uuid
import queue
import sys
import time
import threading
from threading import Thread
import multiprocessing as mp
from multiprocessing import Process, Queue as MPQueue

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  
from transformers import AutoConfig, AutoProcessor, AutoModelForSeq2SeqLM

from web_demo.server.protocal import encode_handshake, decode_message
from web_demo.server.funaudiochat_infer import FunaudioChatStreamer, remove_generate_text_special_token

from utils.constant import *
from utils.cosyvoice_detokenizer import get_audio_detokenizer, tts_infer_streaming


def log(level, message):
    print(f"[{level.upper()}] {message}")

def tts_worker_process(input_queue: MPQueue, output_queue: MPQueue, control_queue: MPQueue, tts_gpu: int = 1):
    """
    TTS independent process1, avoiding Python GIL
    
    Args:
        input_queue: (uuid, tokens_list, offset, finalize)
        output_queue: (uuid, audio_array)
        control_queue: ('init_cache', uuid) / ('clear_cache', uuid) / ('stop', None)
        tts_gpu: GPU device id for TTS model (default: 1)
    """

    log("info", f"[TTS Process] Starting TTS worker process on cuda:{tts_gpu}...")
    torch.cuda.set_device(tts_gpu)
    tts_device = torch.device(f"cuda:{tts_gpu}")
    
    log("info", "[TTS Process] Loading TTS model...")
    tts_model = get_audio_detokenizer()
    
    tts_spk_emb_path = tts_model_config['spk_emb_path']
    tts_spk_embedding = torch.load(tts_spk_emb_path)["中文女"]["embedding"]
    tts_spk_embedding = tts_spk_embedding.to(tts_device)
    
    log("info", "[TTS Process] TTS model loaded successfully")
    
    running = True
    while running:
        try:
            # check control queue
            try:
                while not control_queue.empty():
                    cmd, data = control_queue.get_nowait()
                    if cmd == 'init_cache':
                        uuid_str = data
                        tts_model.model.hift_cache_dict[uuid_str] = None
                        log("info", f"[TTS Process] Initialized cache for {uuid_str}")
                    elif cmd == 'clear_cache':
                        uuid_str = data
                        if uuid_str in tts_model.model.hift_cache_dict:
                            del tts_model.model.hift_cache_dict[uuid_str]
                            log("info", f"[TTS Process] Cleared cache for {uuid_str}")
                    elif cmd == 'stop':
                        running = False
                        log("info", "[TTS Process] Received stop command")
                        break
            except:
                pass
            
            if not running:
                break
            
            try:
                task = input_queue.get(timeout=0.1)
            except:
                continue
            
            uuid_str, tokens_list, offset, finalize = task
            
            queue_size = input_queue.qsize()
            if queue_size > 10:
                log("warning", f"[TTS Process] Input queue backlog: {queue_size} tasks pending")
            
            if uuid_str not in tts_model.model.hift_cache_dict:
                log("info", f"[TTS Process] Skipping task for cleared session {uuid_str[:8]}...")
                continue
            
            this_tokens = torch.tensor(tokens_list, dtype=torch.long, device=tts_device).view(1, -1)
            
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            log("info", f"[{timestamp}] [TTS Process] TTS input: {len(tokens_list)} tokens, offset: {offset}")
            
            speech = tts_infer_streaming(
                tts_model,
                tts_spk_embedding,
                this_tokens,
                offset,
                uuid_str,
                finalize=finalize,
                token_hop_len=TOKEN_HOP_LEN,
                pre_lookahead_len=PRE_LOOKAHEAD_LEN,
                device=f"cuda:{tts_gpu}"
            )
            
            if speech is not None and speech.shape[-1] > 0:
                speech_array = speech[0].cpu().numpy()
                output_queue.put((uuid_str, speech_array))
                
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                log("info", f"[{timestamp}] [TTS Process] TTS generated: {speech_array.shape[-1]} samples")
            
        except Exception as e:
            log("error", f"[TTS Process] Error: {e}")
            import traceback
            traceback.print_exc()
    
    log("info", "[TTS Process] TTS worker process stopped")


class GlobalModelManager:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def initialize(self, model_path: str, target_sample_rate: int = 16000):
        if self._initialized:
            log("info", "initialized, skipping ...")
            return
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_sample_rate = target_sample_rate
        
        log("info", f"loading s2s model to {self.device}...")
        
        config = AutoConfig.from_pretrained(model_path)
        text_config = getattr(config, "text_config", None)
        if text_config and getattr(text_config, "model_type", None) in ["qwen3_moe", ]:
            setattr(text_config, "output_router_logits", False)
        
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, 
            config=config, 
            torch_dtype=torch.bfloat16
        ).to(self.device)
        
        # set gen args
        self.gen_kwargs = DEFAULT_S2M_GEN_KWARGS.copy()
        if 'bad_words_ids' not in self.gen_kwargs or self.gen_kwargs['bad_words_ids'] is None:
            self.gen_kwargs['bad_words_ids'] = [[self.processor.tokenizer.convert_tokens_to_ids('<|audio_bos|>'),
                                                self.processor.tokenizer.convert_tokens_to_ids('<|sil|>')]]

        self.model.sp_gen_kwargs = DEFAULT_SP_GEN_KWARGS.copy()
        
        log("info", f"s2s model loaded (: {self.device})")
        
        self._initialized = True
        log("info", f"waiting for tts model loading ... ")


class ServerState:
    def __init__(self, model_manager: GlobalModelManager, sample_rate: int = 24000, output_dir: str = "./output", tts_gpu: int = 1):
        self.model_manager = model_manager
        self.sample_rate = sample_rate
        self.output_dir = output_dir
        self.tts_gpu = tts_gpu
        self.lock = asyncio.Lock()
        
        self.template = AUDIO_TEMPLATE
        self.APAD_TOKEN = AUDIO_PAD_TOKEN
        self.token_fps = TOKEN_FPS
        self.system_prompt = SPOKEN_S2M_PROMPT

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "input"), exist_ok=True)
        log("info", f"Output directory: {self.output_dir}")
        
        # global TTS queue
        self.tts_input_queue = MPQueue()   
        self.tts_output_queue = MPQueue()  
        self.tts_control_queue = MPQueue() 
        
        self.tts_process = Process(
            target=tts_worker_process, 
            args=(self.tts_input_queue, self.tts_output_queue, self.tts_control_queue, self.tts_gpu),
            daemon=True
        )
        self.tts_process.start()
        log("info", f"Global TTS process started (pid: {self.tts_process.pid})")
    
    def stop_tts_process(self):
        if self.tts_process and self.tts_process.is_alive():
            self.tts_control_queue.put(('stop', None))
            self.tts_process.join(timeout=5.0)
            if self.tts_process.is_alive():
                log("warning", "TTS process did not stop in time, terminating...")
                self.tts_process.terminate()
            log("info", "Global TTS process stopped")

    async def handle_chat(self, request):
        # Set heartbeat interval to 30 seconds and timeout to 60 seconds to prevent connections from being disconnected by intermediate proxies/firewalls
        ws = web.WebSocketResponse(heartbeat=30.0, receive_timeout=None)
        await ws.prepare(request)

        client_id = f"Client-{id(ws)}"
        turn_counter = 0
        is_recording = True 
        
        # system_prompt for current session
        session_system_prompt = self.system_prompt
        
        # history
        messages = []
        audio_list = []

        async def recv_loop():
            nonlocal close, is_recording, turn_counter, opus_reader
            try:
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.ERROR:
                        log("error", f"{ws.exception()}")
                        break
                    elif message.type == aiohttp.WSMsgType.CLOSED:
                        break
                    elif message.type != aiohttp.WSMsgType.BINARY:
                        log("error", f"unexpected message type {message.type}")
                        continue
                    message = message.data
                    if not isinstance(message, bytes):
                        log("error", f"unsupported message type {type(message)}")
                        continue
                    if len(message) == 0:
                        log("warning", "empty message")
                        continue
                    
                    try:
                        decoded = decode_message(message)
                        msg_type = decoded['type']
                        
                        if msg_type == 'audio':
                            if is_recording:
                                payload = decoded['data']
                                log("info", f"Received audio data: {len(payload)} bytes")
                                pcm = opus_reader.append_bytes(payload)
                                if pcm is not None and len(pcm) > 0:
                                    await pcm_queue.put(pcm)
                        
                        elif msg_type == 'control':
                            action = decoded['action']
                            if action == 'pause':
                                log("info", f"Received PAUSE signal")
                                is_recording = False
                                await save_audio_queue.put(('pause', None))
                            elif action == 'start':
                                log("info", f"Received START signal")
                                is_recording = True
                                turn_counter += 1
                                await save_audio_queue.put(('start', None))
                            elif action == 'endTurn':
                                log("info", f"Received END_TURN signal")
                                is_recording = False
                                await save_audio_queue.put(('pause', None))
                            else:
                                log("info", f"Received control: {action}")
                        
                        elif msg_type == 'ping':
                            log("info", "Received PING")
                        
                        elif msg_type == 'text':
                            log("info", f"Received text: {decoded['data']}")
                        
                        elif msg_type == 'metadata':
                            metadata = decoded['data']
                            if isinstance(metadata, dict) and 'system_prompt' in metadata:
                                nonlocal session_system_prompt
                                session_system_prompt = metadata['system_prompt']
                                log("info", f"Received custom system prompt: {session_system_prompt[:100]}...")
                            else:
                                log("info", f"Received metadata: {metadata}")
                        
                        else:
                            log("warning", f"Unknown message type: {msg_type}")
                            
                    except Exception as e:
                        log("error", f"Failed to decode message: {e}")
                        kind = message[0]
                        log("info", f"Trying old protocol, kind={kind}")
                        if kind == 1:  # audio
                            if is_recording:
                                payload = message[1:]
                                log("info", f"Received audio data (old): {len(payload)} bytes")
                                pcm = opus_reader.append_bytes(payload)
                                if pcm is not None and len(pcm) > 0:
                                    await pcm_queue.put(pcm)
                        elif kind == 2:  # pause signal
                            log("info", f"Received PAUSE signal (old)")
                            is_recording = False
                            await save_audio_queue.put(('pause', None))
                        elif kind == 3:  # start signal
                            log("info", f"Received START signal (old)")
                            is_recording = True
                            turn_counter += 1
                            await save_audio_queue.put(('start', None))
                        
            finally:
                close = True
                log("info", "connection closed")

        async def save_audio_loop():
            """Async tasks for handling audio saving and inference"""
            nonlocal all_recorded_pcm, turn_counter, messages, audio_list, audio_buffer_list, audio_buffer_lock, all_generated_audio, reset_first_frame, reset_send_state, this_uuid, tts_offset, cur_audio_tokens, opus_reader, accumulate_tts_tokens
            
            current_generation = {
                'streamer': None,
                'accumulated_text': '',
                'generation_thread': None,
                'is_generating': False,
                'interrupt': False  
            }
            
            while True:
                if close:
                    return
                try:
                    signal_type, _ = await asyncio.wait_for(save_audio_queue.get(), timeout=0.1)
                    
                    if signal_type == 'pause':
                        # save audio and model inference
                        if all_recorded_pcm is not None and len(all_recorded_pcm) > 0:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"{client_id}_turn{turn_counter}_input.wav"
                            filepath = os.path.join(self.output_dir, "input", filename)
                            
                            audio_duration = len(all_recorded_pcm) / self.sample_rate
                            
                            # resampling
                            if self.sample_rate != self.model_manager.target_sample_rate:
                                audio_tensor = torch.from_numpy(all_recorded_pcm).unsqueeze(0)
                                resampler = torchaudio.transforms.Resample(
                                    self.sample_rate, 
                                    self.model_manager.target_sample_rate
                                )
                                audio_tensor = resampler(audio_tensor)
                                audio_for_model = audio_tensor.squeeze(0).numpy()
                            else:
                                audio_for_model = all_recorded_pcm
                            
                            audio_tensor = torch.from_numpy(audio_for_model).unsqueeze(0)
                            torchaudio.save(filepath, audio_tensor, self.model_manager.target_sample_rate)
                            log("info", f"Saved audio to {filepath}, length: {audio_duration:.2f}s")
                            
                            # Prepare conversation messages
                            if len(messages) == 0:
                                messages = [{"role": "system", "content": session_system_prompt}]
                        
                            max_messages = 1 + MAX_HISTORY_TURNS * 2  # 1 for system prompt
                            if len(messages) >= max_messages:
                                messages_to_remove = len(messages) - max_messages + 2  
                                if messages_to_remove > 0:
                                    user_messages_removed = sum(1 for m in messages[1:1+messages_to_remove] if m['role'] == 'user')
                                    messages = [messages[0]] + messages[1+messages_to_remove:]
                                    audio_list = audio_list[user_messages_removed:]
                                    log("info", f"Trimmed history: removed {messages_to_remove} messages, {user_messages_removed} audio items")
                            
                            message_item = {"role": "user", "content": self.template}
                            audio_tokens = self.APAD_TOKEN * int(math.ceil(audio_duration * self.token_fps))
                            audio_item = {'path': filepath, 'token': audio_tokens, 'text': ''}
                            
                            messages.append(message_item)
                            audio_list.append(json.dumps(audio_item))
                            
                            log("info", f"[Turn {turn_counter}] Preparing model input: {len(messages)} messages, {len(audio_list)} audio items")
                            log("info", f"[Turn {turn_counter}] Message history: {[m['role'] for m in messages]}")
                            
                            log("info", f"[Turn {turn_counter}] Queue status: audio_tokens={audio_tokens_queue.qsize()}, opus_bytes={opus_bytes_queue.qsize()}")
                            
                            text = self.model_manager.processor.apply_chat_template(
                                messages,
                                add_generation_prompt=True,
                                tokenize=False
                            )
                            inputs = self.model_manager.processor(
                                text=text,
                                audio=audio_list,
                                return_tensors="pt",
                                return_token_type_ids=False
                            ).to(self.model_manager.device)
                            
                            log("info", f"start inference of ({audio_duration:.2f}s audio)...")
                            
                            msg = b"\x02" + bytes("[Processing...]", encoding="utf8")
                            await ws.send_bytes(msg)
                            
                            group_size = getattr(self.model_manager.model.config.audio_config, 'group_size', 5)
                            streamer = FunaudioChatStreamer(
                                self.model_manager.processor,
                                skip_prompt=True,
                                group_size=group_size
                            )
                            
                            gen_kwargs_with_streamer = self.model_manager.gen_kwargs.copy()
                            gen_kwargs_with_streamer['streamer'] = streamer
                            
                            generation_error = {'error': None}
                            generation_start_time = time.time()
                            
                            def run_generation():
                                try:
                                    log("info", f"[Turn {turn_counter}] Generation thread started, input_ids shape: {inputs['input_ids'].shape}")
                                    self.model_manager.model.generate(**inputs, **gen_kwargs_with_streamer)
                                    elapsed = time.time() - generation_start_time
                                    log("info", f"[Turn {turn_counter}] Generation thread completed in {elapsed:.2f}s")
                                except Exception as e:
                                    generation_error['error'] = e
                                    elapsed = time.time() - generation_start_time
                                    log("error", f"[Turn {turn_counter}] Generation failed after {elapsed:.2f}s: {e}")
                                    import traceback
                                    traceback.print_exc()
                            
                            generation_thread = Thread(target=run_generation)
                            generation_thread.start()
                            
                            current_generation['streamer'] = streamer
                            current_generation['accumulated_text'] = ""
                            current_generation['generation_thread'] = generation_thread
                            current_generation['is_generating'] = True
                            
                            last_step = 0
                            accumulated_text = ""
                            accumulated_audio_ids = []  
                            first_audio_batch = True  
                            loop_count = 0
                            last_step_change_time = time.time()
                            stuck_warning_shown = False
                            
                            log("info", f"Start monitoring the generation results loop (turn {turn_counter})")
                            while generation_thread.is_alive() or last_step < len(streamer.get_step_results()):
                                loop_count += 1
                                if current_generation['interrupt']:
                                    log("info", "Generation interrupted by new turn")
                                    break
                                
                                if generation_error['error'] is not None and not generation_thread.is_alive():
                                    raise generation_error['error']
                                step_results = streamer.get_step_results()
                                
                                current_steps = len(step_results)
                                if current_steps != last_step:
                                    last_step_change_time = time.time()
                                    stuck_warning_shown = False
                                else:
                                    if not stuck_warning_shown and (time.time() - last_step_change_time) > 30:
                                        log("warning", f"[Turn {turn_counter}] Generation appears stuck: no new steps for 10s (steps={current_steps}, loop={loop_count})")
                                        stuck_warning_shown = True
                                
                                if loop_count % 100 == 0:
                                    elapsed_since_last_step = time.time() - last_step_change_time
                                    log("info", f"Monitor loop #{loop_count}: thread_alive={generation_thread.is_alive()}, steps={len(step_results)}, last_step={last_step}, stuck_time={elapsed_since_last_step:.1f}s")
                                
                                for i in range(last_step, len(step_results)):
                                    step = step_results[i]
                                    
                                    # Get the newly added text
                                    if step['new_text_str']:
                                        new_text = step['new_text_str']
                                        if new_text and not new_text.startswith('<|') and new_text.strip():
                                            accumulated_text += new_text
                                            current_generation['accumulated_text'] = accumulated_text  
                                            try:
                                                text_buffer_queue.put_nowait(new_text)
                                                log("info", f"Buffered text: {new_text}")
                                            except Exception as send_err:
                                                log("error", f"Failed to buffer text '{new_text}': {send_err}")
                                                current_generation['interrupt'] = True
                                                break
                                    
                                    # Get audio tokens and put them into the queue (for TTS)
                                    if 'new_audio_ids' in step:
                                        audio_ids = step['new_audio_ids']
                                        if audio_ids is not None and len(audio_ids) > 0:
                                            if first_audio_batch:
                                                log("info", f"Skipping first audio batch (prompt): shape={audio_ids.shape}, tokens={audio_ids.shape[1]}")
                                                first_audio_batch = False
                                            else:
                                                try:
                                                    valid_codes = []
                                                    for code in audio_ids[0]:
                                                        code_val = code.item() if torch.is_tensor(code) else code
                                                        if 0 <= code_val < 6561:
                                                            valid_codes.append(code_val)
                                                    accumulated_audio_ids.extend(valid_codes)
                                                    
                                                    for code in valid_codes:
                                                        audio_tokens_queue.put_nowait(code)
                                                    
                                                    if len(valid_codes) > 0:
                                                        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                                                        log("info", f"[{timestamp}] Added {len(valid_codes)} valid audio codes to queue (skipped {len(audio_ids[0]) - len(valid_codes)} invalid)")
                                                    else:
                                                        log("warning", f"No valid audio codes in this batch (all {len(audio_ids[0])} tokens were invalid)")
                                                except Exception as e:
                                                    log("error", f"Failed to process audio tokens: {e}")
                                
                                last_step = len(step_results)
                                await asyncio.sleep(0.05)
                            
                            log("info", f"Monitor loop ended: total_loops={loop_count}, final_steps={len(streamer.get_step_results())}, interrupted={current_generation['interrupt']}")
                            generation_thread.join()
                            
                            # If interrupted, skip subsequent processing
                            if current_generation['interrupt']:
                                log("info", "Skipping post-generation processing due to interrupt")
                                current_generation['is_generating'] = False
                                current_generation['streamer'] = None
                                current_generation['accumulated_text'] = ''
                                current_generation['generation_thread'] = None
                                continue 
                            
                            # Mark TTS generation as complete
                            tts_generation_complete['flag'] = True
                            log("info", "TTS generation marked as complete")
                            
                            # Wait for the encoding thread to process all remaining audio
                            max_wait_time = 15  # Wait up to 15 seconds
                            wait_interval = 0.1
                            waited = 0
                            while waited < max_wait_time:
                                with audio_buffer_lock:
                                    buffer_empty = len(audio_buffer_list) == 0
                                
                                if buffer_empty and opus_bytes_queue.qsize() == 0:
                                    log("info", f"All audio encoded and sent after {waited:.1f}s")
                                    break
                                
                                await asyncio.sleep(wait_interval)
                                waited += wait_interval
                            
                            if waited >= max_wait_time:
                                log("warning", f"Encoding timeout after {max_wait_time}s")
                            
                            # Get final results
                            final_results = streamer.get_accumulated_results()
                            generate_text = final_results['text_str']
                            
                            log("info", f"Generation completed: {generate_text}")
                            
                            # Save the complete generated audio
                            assistant_audio_path = None
                            assistant_audio_duration = 0.0
                            if len(all_generated_audio) > 0:
                                try:
                                    # Concatenate all generated audio segments
                                    full_generated_audio = np.concatenate(all_generated_audio)
                                    assistant_audio_duration = len(full_generated_audio) / self.sample_rate
                                    log("info", f"Concatenated generated audio: {len(all_generated_audio)} segments, total length {assistant_audio_duration:.2f}s ({len(full_generated_audio)} samples)")
                                    
                                    # Save as WAV file
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    output_filename = f"{client_id}_turn{turn_counter}_output_{timestamp}.wav"
                                    assistant_audio_path = os.path.join(self.output_dir, output_filename)
                                    
                                    # Ensure audio is in float32 format, range [-1, 1]
                                    audio_to_save = full_generated_audio.astype(np.float32)
                                    if np.abs(audio_to_save).max() > 1.0:
                                        audio_to_save = audio_to_save / np.abs(audio_to_save).max()
                                    
                                    sf.write(assistant_audio_path, audio_to_save, self.sample_rate)
                                    log("info", f"Saved generated audio: {assistant_audio_path}")
                                    
                                    # Clear accumulated audio for the next turn
                                    all_generated_audio.clear()
                                except Exception as e:
                                    log("error", f"Failed to save generated audio: {e}")
                                    import traceback
                                    traceback.print_exc()
                                    assistant_audio_path = None
                                    assistant_audio_path = None
                            
                            # Clean up and send the final text
                            clean_text = remove_generate_text_special_token(generate_text)
                            if clean_text != accumulated_text:
                                try:
                                    text_buffer_queue.put_nowait(f"\n[Final: {clean_text}]")
                                except Exception as e:
                                    log("error", f"Failed to send final text: {e}")
                            
                            messages.append({"role": "assistant", "content": clean_text})
                            
                            # reset
                            current_generation['is_generating'] = False
                            current_generation['streamer'] = None
                            current_generation['accumulated_text'] = ''
                            current_generation['generation_thread'] = None
                            
                            log("info", f"Processing completed")
                        else:
                            log("warning", "No audio data to save")
                    
                    elif signal_type == 'start':
                        # If there is an ongoing inference, set the interrupt flag and wait
                        if current_generation['is_generating']:
                            log("info", "Interrupting current generation...")
                            current_generation['interrupt'] = True
                            if current_generation['generation_thread'] is not None:
                                current_generation['generation_thread'].join(timeout=2.0)
                                if current_generation['generation_thread'].is_alive():
                                    log("warning", "Generation thread did not stop in time")
                            current_generation['is_generating'] = False
                            current_generation['interrupt'] = False
                            log("info", "Previous generation stopped")
                        
                        all_recorded_pcm = None
                        with audio_buffer_lock:
                            all_generated_audio.clear() 
                            audio_buffer_list.clear()  
                        
                        cleared_count = 0
                        while not opus_bytes_queue.empty():
                            try:
                                opus_bytes_queue.get_nowait()
                                cleared_count += 1
                            except queue.Empty:
                                break
                        if cleared_count > 0:
                            log("info", f"Cleared {cleared_count} opus frames from queue")
                        
                        opus_reader = sphn.OpusStreamReader(self.sample_rate)
                        log("info", "Reset opus_reader for new turn")
                        
                        cleared_pcm = 0
                        try:
                            while True:
                                pcm_queue.get_nowait()
                                cleared_pcm += 1
                        except asyncio.QueueEmpty:
                            pass
                        if cleared_pcm > 0:
                            log("info", f"Cleared {cleared_pcm} PCM chunks from queue")
                        
                        reset_first_frame['flag'] = True 
                        reset_send_state['flag'] = True  
                        tts_generation_complete['flag'] = False
                        frame_generation_complete['flag'] = False
                        audio_send_started['flag'] = False  
                        
                        # Reset TTS-related variables (protected with a lock)
                        with tts_state_lock:
                            cur_audio_tokens.clear()
                            tts_offset = 0
                            accumulate_tts_tokens = 0 
                            old_uuid = this_uuid
                            this_uuid = str(uuid.uuid4())
                            
                            tts_control_queue.put(('clear_cache', old_uuid))
                            tts_control_queue.put(('init_cache', this_uuid))
                        
                        cleared_tokens = 0
                        try:
                            while True:
                                audio_tokens_queue.get_nowait()
                                cleared_tokens += 1
                        except queue.Empty:
                            pass
                        if cleared_tokens > 0:
                            log("info", f"Cleared {cleared_tokens} audio tokens from queue")
                        
                        cleared_texts = 0
                        try:
                            while True:
                                text_buffer_queue.get_nowait()
                                cleared_texts += 1
                        except queue.Empty:
                            pass
                        if cleared_texts > 0:
                            log("info", f"Cleared {cleared_texts} buffered texts from queue")
                        
                        log("info", "Cleared audio buffer and TTS state for new recording")
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    log("error", f"Inference failed: {e}")
                    import traceback
                    traceback.print_exc()
                    try:
                        error_msg = b"\x05" + bytes(f"Processing failed: {str(e)}", encoding="utf8")
                        await ws.send_bytes(error_msg)
                    except:
                        pass

        async def accumulate_pcm_loop():
            nonlocal all_recorded_pcm
            total_samples = 0
            last_all_recorded_pcm_id = id(all_recorded_pcm) if all_recorded_pcm is not None else None
            
            while True:
                if close:
                    return
                
                current_id = id(all_recorded_pcm) if all_recorded_pcm is not None else None
                if all_recorded_pcm is None and last_all_recorded_pcm_id is not None:
                    total_samples = 0
                    log("info", "Reset PCM accumulator for new turn")
                last_all_recorded_pcm_id = current_id
                
                # Read PCM data from the queue
                try:
                    pcm = await asyncio.wait_for(pcm_queue.get(), timeout=0.01)
                except asyncio.TimeoutError:
                    continue
                
                if pcm is None or len(pcm) == 0:
                    continue
                
                if is_recording:
                    pcm_samples = len(pcm)
                    total_samples += pcm_samples
                    log("info", f"Accumulating PCM: {pcm_samples} samples (total: {total_samples}, {total_samples/self.sample_rate:.2f}s)")
                    
                    if all_recorded_pcm is None:
                        all_recorded_pcm = pcm
                    else:
                        all_recorded_pcm = np.concatenate((all_recorded_pcm, pcm))

        def tts_sender_thread_func():
            """[Multiprocessing] Send audio tokens to the TTS process"""
            nonlocal tts_offset, cur_audio_tokens, this_uuid, max_tts_tokens, tts_generation_complete, accumulate_tts_tokens
            
            token_hop_len = 15
            pre_lookahead_len = 3
            
            log("info", "TTS sender thread started")
            
            while not close:
                try:
                    finalize = False
                    try:
                        audio_token = audio_tokens_queue.get(timeout=0.1)
                        
                        with tts_state_lock:
                            cur_audio_tokens.append(audio_token)
                            if len(cur_audio_tokens) < tts_offset + token_hop_len + pre_lookahead_len:
                                continue
                    except queue.Empty:
                        if tts_generation_complete['flag']:
                            finalize = True
                            frame_generation_complete['flag'] = True
                            with tts_state_lock:
                                if len(cur_audio_tokens) <= tts_offset + 1 + pre_lookahead_len:
                                    continue
                        else:
                            continue
                    
                    with tts_state_lock:
                        tokens_to_send = cur_audio_tokens.copy()
                        local_tts_offset = tts_offset
                        local_this_uuid = this_uuid
                    
                    tts_input_queue.put((local_this_uuid, tokens_to_send, local_tts_offset, finalize))
                    
                    # update tts offset for streaming tts
                    with tts_state_lock:
                        tts_offset += token_hop_len
                        if tts_offset >= max_tts_tokens:
                            tts_offset -= max_tts_tokens
                            cur_audio_tokens = cur_audio_tokens[max_tts_tokens:]
                            accumulate_tts_tokens += max_tts_tokens
                            if accumulate_tts_tokens >= MAX_TTS_HISTORY:
                                accumulate_tts_tokens = 0
                                tts_control_queue.put(('clear_cache', this_uuid))
                                tts_control_queue.put(('init_cache', this_uuid))
                    
                except Exception as e:
                    log("error", f"TTS sender thread error: {e}")
                    import traceback
                    traceback.print_exc()
            
            log("info", "TTS sender thread stopped")

        def tts_receiver_thread_func():
            """[Multiprocessing] Receiving output from the TTS process (only messages belonging to the current session)"""
            log("info", f"TTS receiver thread started for session {this_uuid}")
            
            while not close:
                try:
                    try:
                        uuid_str, speech_array = tts_output_queue.get(timeout=0.1)
                    except:
                        continue
                    
                    with tts_state_lock:
                        current_uuid = this_uuid
                    
                    if uuid_str != current_uuid:
                        log("info", f"Discarding TTS output from old session {uuid_str[:8]}... (current: {current_uuid[:8]}...)")
                        continue
                    
                    with audio_buffer_lock:
                        all_generated_audio.append(speech_array.copy())
                        audio_buffer_list.append(speech_array.copy())
                    
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    log("info", f"[{timestamp}] Received TTS audio: {speech_array.shape[-1]} samples")
                    
                except Exception as e:
                    log("error", f"TTS receiver thread error: {e}")
                    import traceback
                    traceback.print_exc()
            
            log("info", "TTS receiver thread stopped")

        def encode_thread_func():
            """Encoding thread: Take complete frames from audio_buffer, encode them into Opus, and place them into opus_bytes_queue"""
            nonlocal reset_first_frame, tts_generation_complete, frame_generation_complete
            
            local_opus_writer = sphn.OpusStreamWriter(self.sample_rate)
            frame_size = int(self.sample_rate * 0.04)  # 80ms = 1920 samples @ 24kHz
            local_buffer = np.array([], dtype=np.float32)
            is_first_frame = True  
            first_frame_start_time = None 
            
            log("info", "Encode thread started")
            
            while not close:
                try:
                    if reset_first_frame['flag']:
                        is_first_frame = True
                        first_frame_start_time = None
                        reset_first_frame['flag'] = False
                        local_buffer = np.array([], dtype=np.float32)
                        log("info", "Reset first frame flag for new conversation turn")
                    
                    with audio_buffer_lock:
                        if len(audio_buffer_list) > 0:
                            chunk = audio_buffer_list.pop(0)
                        else:
                            chunk = None
                    
                    if chunk is None:
                        if frame_generation_complete['flag'] and len(local_buffer) > 0:
                            log("info", f"TTS completed, flushing remaining {len(local_buffer)} samples")
                            
                            if is_first_frame:
                                is_first_frame = False
                                log("info", "Skipping first frame delay due to TTS completion")
                            
                            while len(local_buffer) >= frame_size:
                                frame = local_buffer[:frame_size]
                                local_buffer = local_buffer[frame_size:]
                                
                                opus_bytes = local_opus_writer.append_pcm(frame)
                                if opus_bytes is not None and len(opus_bytes) > 0:
                                    opus_bytes_queue.put(opus_bytes)
                            
                            # Process the last incomplete frame (pad with zeros)
                            if len(local_buffer) > 0:
                                padding = np.zeros(frame_size - len(local_buffer), dtype=np.float32)
                                frame = np.concatenate([local_buffer, padding])
                                local_buffer = np.array([], dtype=np.float32)
                                
                                opus_bytes = local_opus_writer.append_pcm(frame)
                                if opus_bytes is not None and len(opus_bytes) > 0:
                                    opus_bytes_queue.put(opus_bytes)
                                    log("info", "Encoded final partial frame")
                            
                            log("info", "All audio flushed to queue")
                        
                        time.sleep(0.01)
                        continue
                    
                    # Add chunk to local buffer
                    local_buffer = np.concatenate([local_buffer, chunk])
                    
                    # If this is the first frame, record the start time
                    if is_first_frame and first_frame_start_time is None and len(local_buffer) > 0:
                        first_frame_start_time = time.time()
                        log("info", f"First audio data received, waiting before encoding...")
                    
                    # Check immediately if there are complete frames
                    while len(local_buffer) >= frame_size:
                        # If this is the first frame, wait some time to accumulate data
                        if is_first_frame:
                            if first_frame_start_time is not None:
                                elapsed = time.time() - first_frame_start_time
                            
                            is_first_frame = False
                            log("info", f"Starting to encode (buffer: {len(local_buffer)} samples)")
                        
                        frame = local_buffer[:frame_size]
                        local_buffer = local_buffer[frame_size:]
                        
                        # Encode to Opus
                        opus_bytes = local_opus_writer.append_pcm(frame)
                        if opus_bytes is not None and len(opus_bytes) > 0:
                            opus_bytes_queue.put(opus_bytes)
                        
                except Exception as e:
                    log("error", f"Encode thread error: {e}")
                    import traceback
                    traceback.print_exc()
            
            log("info", "Encode thread stopped")

        async def send_audio_loop():
            """Async coroutine: Read encoded Opus data from the queue and send at fixed time intervals"""
            nonlocal reset_send_state, audio_send_started
            frame_interval = 0.010 
            next_send_time = None  
            frames_sent = 0  
            
            while not close:
                try:
                    if reset_send_state['flag']:
                        next_send_time = None
                        frames_sent = 0
                        reset_send_state['flag'] = False
                        log("info", "Reset send state for new turn")
                    
                    try:
                        opus_bytes = opus_bytes_queue.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.005)
                        continue
                    
                    current_time = time.time()
                    
                    if next_send_time is None:
                        next_send_time = current_time
                    
                    wait_time = next_send_time - current_time
                    
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                    
                    await ws.send_bytes(b"\x01" + opus_bytes)
                    frames_sent += 1
                    
                    # When the first frame is sent, notify the text sending thread to start
                    if frames_sent == 1:
                        audio_send_started['flag'] = True
                        log("info", "Audio sending started, text buffer can start sending")
                    
                    next_send_time += frame_interval
                    
                    lag = time.time() - next_send_time
                    if lag > 0.5:
                        log("warning", f"Send lag detected: {lag*1000:.0f}ms, resetting time base")
                        next_send_time = time.time() + frame_interval
                    
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    log("info", f"[{timestamp}] Sent frame #{frames_sent} (queue: {opus_bytes_queue.qsize()}, lag: {lag*1000:.1f}ms)")
                    
                except Exception as e:
                    log("error", f"Send coroutine error: {e}")
                    import traceback
                    traceback.print_exc()
            
            log("info", f"Send coroutine stopped, total frames sent: {frames_sent}")

        async def send_text_loop():
            """Asynchronous coroutine: Read text from the text buffer queue and send it at 200ms intervals after audio transmission begins."""
            nonlocal audio_send_started
            text_interval = 0.2  # 200ms per text
            texts_sent = 0
            current_turn_started = False  
            
            log("info", "Text send coroutine started")
            
            while not close:
                try:
                    try:
                        text = text_buffer_queue.get_nowait()
                    except queue.Empty:
                        if current_turn_started and not audio_send_started['flag']:
                            current_turn_started = False
                            log("info", "New turn detected, resetting text send state")
                        await asyncio.sleep(0.05)
                        continue
                    
                    if not current_turn_started:
                        log("info", f"Waiting for audio to start before sending text: '{text[:20]}...'")
                        while not close and not audio_send_started['flag']:
                            await asyncio.sleep(0.05)
                        
                        if close:
                            log("info", "Text send coroutine stopped while waiting for audio")
                            break
                        
                        current_turn_started = True
                        log("info", "Audio started, beginning text transmission at 200ms intervals")
                    
                    try:
                        msg = b"\x02" + bytes(text, encoding="utf8")
                        await ws.send_bytes(msg)
                        texts_sent += 1
                        log("info", f"Sent buffered text #{texts_sent}: {text}")
                    except Exception as send_err:
                        log("error", f"Failed to send text '{text}': {send_err}")
                        break
                    
                    await asyncio.sleep(text_interval)
                    
                except Exception as e:
                    log("error", f"Text send coroutine error: {e}")
                    import traceback
                    traceback.print_exc()
            
            log("info", f"Text send coroutine stopped, total texts sent: {texts_sent}")

        log("info", "accepted connection")
        close = False
        all_recorded_pcm = None  # Store all recorded PCM data
        all_generated_audio = []  # Store all generated audio segments
        audio_buffer_list = []  # Audio buffer between TTS process and main process
        audio_buffer_lock = threading.Lock()  # Protect audio_buffer_list and all_generated_audio
        reset_first_frame = {'flag': True}  # Control encoding thread to reset first frame delay (dict for thread sharing)
        reset_send_state = {'flag': False}  # Control send coroutine to reset frame count and time base
        tts_generation_complete = {'flag': False}  # Mark whether audio tokens generation is complete
        frame_generation_complete = {'flag': False}  # Mark whether TTS generation is finished
        audio_send_started = {'flag': False}  # Mark whether audio sending has started
        save_audio_queue = asyncio.Queue()  # Signal queue for saving audio
        audio_tokens_queue = queue.Queue()  # Thread-safe queue to avoid blocking event loop
        pcm_queue = asyncio.Queue()  # Queue for decoded PCM data
        opus_bytes_queue = queue.Queue()  # Thread-safe queue for encoded Opus byte data
        text_buffer_queue = queue.Queue()  # Thread-safe queue for buffering text to be sent
        cur_audio_tokens = []
        tts_offset = 0
        accumulate_tts_tokens = 0
        max_tts_tokens = MAX_TTS_TOKENS
        this_uuid = str(uuid.uuid4())
        tts_state_lock = threading.Lock()  # Protect cur_audio_tokens, tts_offset, this_uuid
        
        # TTS 
        tts_input_queue = self.tts_input_queue
        tts_output_queue = self.tts_output_queue
        tts_control_queue = self.tts_control_queue
        
        tts_control_queue.put(('init_cache', this_uuid))
        log("info", f"Initialized TTS cache for session {this_uuid}")

        opus_reader = sphn.OpusStreamReader(self.sample_rate)
        
        loop = asyncio.get_event_loop()
        
        tts_sender_thread = Thread(target=tts_sender_thread_func, daemon=True)
        tts_receiver_thread = Thread(target=tts_receiver_thread_func, daemon=True)
        encode_thread = Thread(target=encode_thread_func, daemon=True)
        
        tts_sender_thread.start()
        tts_receiver_thread.start()
        encode_thread.start()
        
        log("info", "All workers started (TTS in separate process, encode thread active)")
        # Send the handshake.
        handshake = encode_handshake(version=0, model=0)
        await ws.send_bytes(handshake)
        log("info", "Sent handshake")
        
        try:
            await asyncio.gather(
                recv_loop(), 
                save_audio_loop(),
                accumulate_pcm_loop(),
                send_audio_loop(),
                send_text_loop(),
            )
        finally:
            close = True
            log("info", "Waiting for workers to stop...")
            
            tts_control_queue.put(('clear_cache', this_uuid))
            log("info", f"Cleared TTS cache for session {this_uuid}")
            
            tts_sender_thread.join(timeout=2.0)
            tts_receiver_thread.join(timeout=2.0)
            encode_thread.join(timeout=2.0)
            
            log("info", "All worker threads stopped")
        
        log("info", "done with connection")
        return ws

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=11235, type=int)
    parser.add_argument("--sample-rate", default=24000, type=int, help="Audio sample rate (Opus)")
    parser.add_argument("--model-sample-rate", default=16000, type=int, help="Model sample rate")
    parser.add_argument("--output-dir", default="./output", type=str, help="Directory to save input audio files")
    parser.add_argument("--model-path", type=str, default="model/s2s", help="Path to S2S model")
    parser.add_argument("--tts-gpu", default=1, type=int, help="GPU device id for TTS model (default: 1)")

    args = parser.parse_args()

    log("info", f"Initializing server with Opus sample rate: {args.sample_rate}, Model sample rate: {args.model_sample_rate}")
    
    model_manager = GlobalModelManager()
    try:
        model_manager.initialize(
            model_path=args.model_path,
            target_sample_rate=args.model_sample_rate
        )
    except Exception as e:
        log("error", f"Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    state = ServerState(
        model_manager=model_manager,
        sample_rate=args.sample_rate, 
        output_dir=args.output_dir,
        tts_gpu=args.tts_gpu
    )
    
    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)
    
    protocol = "http"
    log("info", f"Access the Web UI directly at {protocol}://{args.host}:{args.port}")
    web.run_app(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
