import os
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1+PTX")
import sys
import uuid
import json
import re
import time
import torch
torch._C._jit_set_nvfuser_enabled(False)
torch._C._jit_set_texpr_fuser_enabled(False)
try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass
import torchaudio
import soundfile as sf
import librosa
from flask import Flask, request, jsonify, send_file, Response, stream_with_context
from flask_cors import CORS
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoProcessor

sys.path.append(os.getcwd())
from funaudiochat.register import register_funaudiochat
register_funaudiochat()

from utils.cosyvoice_detokenizer import get_audio_detokenizer, token2wav
from utils.constant import (
    DEFAULT_S2M_GEN_KWARGS,
    DEFAULT_SP_GEN_KWARGS,
    DEFAULT_S2M_PROMPT,
    FUNCTION_CALLING_PROMPT,
    AUDIO_TEMPLATE,
)

app = Flask(__name__)
CORS(app)

MODEL_DEFAULT_ID = "Fun-Audio-Chat-8B"
PERSONAPLEX_MODEL_ID = "nvidia/personaplex-7b-v1"
ASR_MODEL_SIZE = os.environ.get("ASR_MODEL_SIZE", "base")
ASR_LANGUAGE = os.environ.get("ASR_LANGUAGE", "en")
SPK_EMB_PATHS = [
    'pretrained_models/Fun-CosyVoice3-0.5B-2512/spk_emb.pt',
    'utils/new_spk2info.pt',
]

# Global model variables
model = None
processor = None
tts_model = None
current_model_id = None
device = None
output_dir = './output_audio'
asr_model = None


def resolve_model_path(model_id: str) -> str:
    if os.path.isdir(model_id):
        return model_id

    local_candidate = os.path.join('pretrained_models', model_id)
    if os.path.isdir(local_candidate):
        return local_candidate

    return model_id


def is_personaplex(model_id: str) -> bool:
    return PERSONAPLEX_MODEL_ID in model_id.lower() or 'personaplex' in model_id.lower()


def load_model_if_needed(model_id: str):
    global model, processor, tts_model, current_model_id, device

    if model is not None and model_id == current_model_id:
        return

    print(f'Loading model: {model_id}')
    force_cpu = os.environ.get('FORCE_CPU', '').strip().lower() in ('1', 'true', 'yes')
    device = torch.device('cpu' if force_cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model_path = resolve_model_path(model_id)

    if is_personaplex(model_id):
        if not os.path.isdir(model_path) and model_id == PERSONAPLEX_MODEL_ID:
            raise RuntimeError(
                'PersonaPlex model not found. Download it and set PERSONAPLEX_MODEL_PATH or place it '
                'under pretrained_models/personaplex-7b-v1.'
            )

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map='auto'
    ).eval()

    if hasattr(model, 'config') and hasattr(model.config, 'attn_implementation'):
        model.config.attn_implementation = 'eager'
    if hasattr(model, 'config'):
        setattr(model.config, '_attn_implementation', 'eager')
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'config') and hasattr(model.language_model.config, 'attn_implementation'):
        model.language_model.config.attn_implementation = 'eager'
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'config'):
        setattr(model.language_model.config, '_attn_implementation', 'eager')

    if hasattr(model, 'sp_gen_kwargs'):
        # Match web_demo defaults to avoid CRQ dimension mismatches.
        model.sp_gen_kwargs.update(DEFAULT_SP_GEN_KWARGS)
        model.sp_gen_kwargs.update({
            'text_greedy': False,
            'disable_speech': False,
        })

    tts_model = get_audio_detokenizer()
    current_model_id = model_id
    os.makedirs(output_dir, exist_ok=True)
    print('Models loaded successfully!')


def parse_tools(tools_raw: str | None):
    if not tools_raw:
        return None
    try:
        tools = json.loads(tools_raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid tools JSON: {exc}") from exc
    if not isinstance(tools, list):
        raise ValueError('Tools must be a JSON array')
    return tools


def build_system_prompt(system_prompt: str, tools: list | None):
    if not tools:
        return system_prompt

    tools_definition = "\n".join([json.dumps(tool, ensure_ascii=False) for tool in tools])
    tools_prompt = FUNCTION_CALLING_PROMPT.replace("{tools_definition}", tools_definition)
    if system_prompt:
        return f"{system_prompt}\n\n{tools_prompt}"
    return tools_prompt


def ensure_speech_prompt(system_prompt: str) -> str:
    if DEFAULT_S2M_PROMPT and DEFAULT_S2M_PROMPT not in system_prompt:
        if system_prompt.strip():
            return f"{system_prompt}\n\n{DEFAULT_S2M_PROMPT}"
        return DEFAULT_S2M_PROMPT
    return system_prompt


def strip_tool_calls(text: str) -> str:
    text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
    text = re.sub(r"\[[a-zA-Z0-9_]+\s+[^\]]+\]", "", text)
    text = re.sub(r"\{\s*\"(action|tool|name|function)\".*?\}", "", text, flags=re.DOTALL)
    return text


def extract_tool_calls(text: str, tools: list | None = None):
    tool_calls = []

    matches = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    for match in matches:
        try:
            data = json.loads(match)
        except json.JSONDecodeError:
            continue
        name = data.get('name') or data.get('function') or 'tool'
        arguments = data.get('arguments') or {}
        tool_calls.append({
            'name': name,
            'arguments': json.dumps(arguments, ensure_ascii=False) if not isinstance(arguments, str) else arguments,
        })

    bracket_matches = re.findall(r"\[([a-zA-Z0-9_]+)\s+([^\]]+)\]", text)
    for name, args in bracket_matches:
        parsed_args = {}
        for key, value in re.findall(r'([a-zA-Z0-9_]+)\s*=\s*\"([^\"]*)\"', args):
            parsed_args[key] = value
        if not parsed_args:
            parsed_args = {"input": args.strip()}
        tool_calls.append({
            'name': name,
            'arguments': json.dumps(parsed_args, ensure_ascii=False),
        })

    call_matches = re.findall(r"([a-zA-Z0-9_]+)\s*\(([^)]*)\)", text)
    for name, args in call_matches:
        args = args.strip()
        if not args:
            parsed_args = {}
        else:
            string_match = re.match(r'\"([^\"]+)\"', args) or re.match(r"\'([^\']+)\'", args)
            if string_match:
                parsed_args = {"input": string_match.group(1)}
            else:
                parsed_args = {"input": args}
        tool_calls.append({
            'name': name,
            'arguments': json.dumps(parsed_args, ensure_ascii=False),
        })

    json_matches = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    for match in json_matches:
        try:
            data = json.loads(match)
        except json.JSONDecodeError:
            continue
        name = data.get('name') or data.get('function') or data.get('tool') or data.get('action')
        arguments = {k: v for k, v in data.items() if k not in {'name', 'function', 'tool', 'action'}}
        if not name:
            if tools and len(tools) == 1:
                name = tools[0].get('name') or 'tool'
            else:
                continue
        tool_calls.append({
            'name': name,
            'arguments': json.dumps(arguments, ensure_ascii=False),
        })

    return tool_calls


def fallback_tool_calls(text: str, tools: list | None):
    if not tools or len(tools) != 1:
        return []
    tool = tools[0] or {}
    name = tool.get('name') or 'tool'
    tool_blob = json.dumps(tool, ensure_ascii=False).lower()
    if 'weather' not in name.lower() and 'weather' not in tool_blob:
        return []

    city = extract_city_from_text(text)

    if not city:
        return []

    return [{
        'name': name,
        'arguments': json.dumps({'city': city}, ensure_ascii=False),
    }]


def extract_city_from_text(text: str):
    match = re.search(r"weather\s*(?:in|for)\s+([A-Za-z][A-Za-z\- ]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()

    match = re.search(r"\bin\s+([A-Z][A-Za-z\-]+)\b", text)
    if match:
        return match.group(1).strip()

    return None


def extract_location_from_text(text: str):
    match = re.search(r"\b(?:in|for|at)\s+([A-Za-z][A-Za-z\- ]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def extract_math_expression(text: str):
    match = re.search(r"([-+/*().\d\s]{3,})", text)
    if not match:
        return None
    expr = match.group(1).strip()
    if any(char.isdigit() for char in expr):
        return expr
    return None


def extract_translation(text: str):
    match = re.search(r"translate\s+(.+?)\s+to\s+([A-Za-z \-]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    match = re.search(r"translate\s+(.+?)\s+into\s+([A-Za-z \-]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, None


def extract_search_query(text: str):
    match = re.search(r"(?:search for|look up|find)\s+(.+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def find_tool_by_keywords(tools: list | None, keywords: list[str]):
    if not tools:
        return None
    for tool in tools:
        blob = f"{tool.get('name', '')} {tool.get('description', '')}".lower()
        if any(keyword in blob for keyword in keywords):
            return tool
    return None


def infer_tool_calls_from_text(text: str, tools: list | None):
    if not text or not tools:
        return []

    tool = find_tool_by_keywords(tools, ['weather'])
    if tool:
        city = extract_city_from_text(text)
        if city:
            return [{
                'name': tool.get('name') or 'tool',
                'arguments': json.dumps({'city': city}, ensure_ascii=False),
            }]

    tool = find_tool_by_keywords(tools, ['time', 'clock'])
    if tool and re.search(r"\btime\b", text, flags=re.IGNORECASE):
        location = extract_location_from_text(text) or 'UTC'
        return [{
            'name': tool.get('name') or 'tool',
            'arguments': json.dumps({'location': location}, ensure_ascii=False),
        }]

    tool = find_tool_by_keywords(tools, ['calculate', 'calculator', 'math'])
    if tool:
        expr = extract_math_expression(text)
        if expr:
            return [{
                'name': tool.get('name') or 'tool',
                'arguments': json.dumps({'expression': expr}, ensure_ascii=False),
            }]

    tool = find_tool_by_keywords(tools, ['translate', 'translation'])
    if tool:
        text_value, target_language = extract_translation(text)
        if text_value and target_language:
            return [{
                'name': tool.get('name') or 'tool',
                'arguments': json.dumps({'text': text_value, 'target_language': target_language}, ensure_ascii=False),
            }]

    tool = find_tool_by_keywords(tools, ['summarize', 'summary'])
    if tool and re.search(r"\bsummar", text, flags=re.IGNORECASE):
        return [{
            'name': tool.get('name') or 'tool',
            'arguments': json.dumps({'text': text.strip()}, ensure_ascii=False),
        }]

    tool = find_tool_by_keywords(tools, ['search', 'web'])
    if tool:
        query = extract_search_query(text)
        if query:
            return [{
                'name': tool.get('name') or 'tool',
                'arguments': json.dumps({'query': query}, ensure_ascii=False),
            }]

    return []


def load_asr_model():
    global asr_model
    if asr_model is not None:
        return asr_model

    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        print(f'ASR unavailable: {exc}')
        return None

    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    compute_type = 'float16' if torch.cuda.is_available() else 'int8'
    try:
        asr_model = WhisperModel(ASR_MODEL_SIZE, device=device_name, compute_type=compute_type)
        return asr_model
    except Exception as exc:
        print(f'ASR load failed on {device_name}: {exc}')
        if device_name != 'cpu':
            try:
                asr_model = WhisperModel(ASR_MODEL_SIZE, device='cpu', compute_type='int8')
                return asr_model
            except Exception as exc_cpu:
                print(f'ASR load failed on cpu: {exc_cpu}')
        asr_model = None
        return None


def transcribe_audio_array(audio_array, sr: int):
    model = load_asr_model()
    if model is None:
        return None

    if sr != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
        sr = 16000

    audio_array = audio_array.astype("float32")
    try:
        segments, _ = model.transcribe(
            audio_array,
            beam_size=5,
            language=ASR_LANGUAGE,
        )
        text = ''.join([segment.text for segment in segments]).strip()
        return text or None
    except Exception as exc:
        print(f'ASR transcription failed: {exc}')
        return None


def override_weather_tool_call(tool_calls: list, transcript: str, tools: list | None):
    if not transcript or not tools or len(tools) != 1:
        return tool_calls
    tool = tools[0] or {}
    name = tool.get('name') or 'tool'
    tool_blob = json.dumps(tool, ensure_ascii=False).lower()
    if 'weather' not in name.lower() and 'weather' not in tool_blob:
        return tool_calls

    city = extract_city_from_text(transcript)
    if not city:
        return tool_calls

    return [{
        'name': name,
        'arguments': json.dumps({'city': city}, ensure_ascii=False),
    }]


def load_audio(file_storage, target_sr: int):
    temp_path = os.path.join(output_dir, f'input_{uuid.uuid4()}.webm')
    file_storage.save(temp_path)
    audio_array, sr = librosa.load(temp_path, sr=target_sr, mono=True)
    return audio_array, sr


def prepare_inputs(audio_array, system_prompt: str):
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": AUDIO_TEMPLATE},
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=text,
        audio=[audio_array],
        return_tensors="pt",
        return_token_type_ids=False,
    ).to(device)
    return inputs


def get_generate_ids(outputs):
    if isinstance(outputs, tuple):
        return outputs[0]
    if hasattr(outputs, 'sequences'):
        return outputs.sequences
    return outputs


def decode_generation(outputs, inputs):
    generate_ids = get_generate_ids(outputs)
    speech_ids = None
    if isinstance(outputs, tuple) and len(outputs) > 1:
        speech_ids = outputs[1]

    if hasattr(inputs, 'input_ids') and inputs.input_ids is not None:
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
    text = processor.decode(generate_ids[0], skip_special_tokens=True)

    if speech_ids is not None and hasattr(speech_ids, 'numel') and speech_ids.numel() > 0:
        audio_tokens = speech_ids[0].to(dtype=torch.long)
    else:
        audio_tokens = generate_ids[0][generate_ids[0] >= 32000]
    return text, audio_tokens


def load_speaker_embedding():
    for path in SPK_EMB_PATHS:
        if not os.path.exists(path):
            continue
        data = torch.load(path)
        if isinstance(data, dict):
            if '中文女' in data and 'embedding' in data['中文女']:
                return data['中文女']['embedding']
            if 'embedding' in data:
                return data['embedding']
    return None


def synthesize_audio(audio_tokens):
    if audio_tokens is None or audio_tokens.numel() == 0:
        return None

    # Move to CPU and drop special/pad tokens that CosyVoice doesn't model.
    audio_tokens = audio_tokens.detach().to('cpu')
    audio_tokens = audio_tokens[(audio_tokens >= 0) & (audio_tokens < 6561)]
    if audio_tokens.numel() == 0:
        return None

    spk_embedding = load_speaker_embedding()
    if spk_embedding is None:
        print('No speaker embedding found; skipping audio synthesis.')
        return None

    try:
        audio_output = token2wav(
            tts_model,
            audio_tokens,
            spk_embedding,
        )
    except Exception as exc:
        print(f'TTS failed: {exc}')
        return None

    output_path = os.path.join(output_dir, f'output_{uuid.uuid4()}.wav')
    audio_np = audio_output.detach().cpu().numpy()
    if audio_np.ndim > 1:
        audio_np = audio_np[0]
    sf.write(output_path, audio_np, 24000)
    return output_path


def chunk_text(text: str, chunk_size: int = 48):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def get_generation_kwargs():
    kwargs = dict(DEFAULT_S2M_GEN_KWARGS)
    kwargs['use_cache'] = True

    if not kwargs.get('bad_words_ids'):
        try:
            kwargs['bad_words_ids'] = [[
                processor.tokenizer.convert_tokens_to_ids('<|audio_bos|>'),
                processor.tokenizer.convert_tokens_to_ids('<|sil|>')
            ]]
        except Exception:
            pass

    eos_token_id = getattr(processor.tokenizer, 'eos_token_id', None)
    if eos_token_id is None and hasattr(model.config, 'text_config'):
        eos_token_id = getattr(model.config.text_config, 'eos_token_id', None)
    if eos_token_id is not None:
        kwargs['eos_token_id'] = eos_token_id

    pad_token_id = getattr(processor.tokenizer, 'pad_token_id', None)
    if pad_token_id is None and hasattr(model.config, 'text_config'):
        pad_token_id = getattr(model.config.text_config, 'pad_token_id', None)
    if pad_token_id is not None:
        kwargs['pad_token_id'] = pad_token_id

    return kwargs


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None, 'model_id': current_model_id})


@app.route('/process-audio', methods=['POST'])
def process_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        system_prompt = request.form.get('system_prompt', DEFAULT_S2M_PROMPT)
        system_prompt = ensure_speech_prompt(system_prompt)
        tools = parse_tools(request.form.get('tools'))
        model_id = request.form.get('model', MODEL_DEFAULT_ID).strip() or MODEL_DEFAULT_ID

        load_model_if_needed(model_id)

        target_sr = 24000 if is_personaplex(model_id) else 16000
        audio_array, _ = load_audio(audio_file, target_sr)
        system_prompt = build_system_prompt(system_prompt, tools)

        inputs = prepare_inputs(audio_array, system_prompt)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **get_generation_kwargs(),
            )

        response_text, audio_tokens = decode_generation(outputs, inputs)
        tool_calls = extract_tool_calls(response_text, tools)
        if not tool_calls:
            tool_calls = fallback_tool_calls(response_text, tools)
        if tools:
            transcript = transcribe_audio_array(audio_array, target_sr)
            if transcript:
                tool_calls = override_weather_tool_call(tool_calls, transcript, tools)
                if not tool_calls:
                    tool_calls = infer_tool_calls_from_text(transcript, tools)
                if not tool_calls:
                    tool_calls = fallback_tool_calls(transcript, tools)
        if tools and not tool_calls:
            tool_calls = infer_tool_calls_from_text(response_text, tools)
        response_text = strip_tool_calls(response_text).strip()
        if tool_calls and not response_text:
            response_text = "Calling tool..."

        audio_url = None
        output_path = synthesize_audio(audio_tokens)
        if output_path:
            audio_url = f'/audio/{os.path.basename(output_path)}'

        return jsonify({
            'text': response_text,
            'audio_url': audio_url,
            'tool_calls': tool_calls,
            'status': 'success'
        })

    except Exception as exc:
        print(f'Error: {str(exc)}')
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


@app.route('/process-audio-stream', methods=['POST'])
def process_audio_stream():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        system_prompt = request.form.get('system_prompt', DEFAULT_S2M_PROMPT)
        system_prompt = ensure_speech_prompt(system_prompt)
        tools = parse_tools(request.form.get('tools'))
        model_id = request.form.get('model', MODEL_DEFAULT_ID).strip() or MODEL_DEFAULT_ID

        load_model_if_needed(model_id)

        target_sr = 24000 if is_personaplex(model_id) else 16000
        audio_array, _ = load_audio(audio_file, target_sr)
        system_prompt = build_system_prompt(system_prompt, tools)

        inputs = prepare_inputs(audio_array, system_prompt)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **get_generation_kwargs(),
            )

        response_text, audio_tokens = decode_generation(outputs, inputs)
        tool_calls = extract_tool_calls(response_text, tools)
        if not tool_calls:
            tool_calls = fallback_tool_calls(response_text, tools)
        if tools:
            transcript = transcribe_audio_array(audio_array, target_sr)
            if transcript:
                tool_calls = override_weather_tool_call(tool_calls, transcript, tools)
                if not tool_calls:
                    tool_calls = infer_tool_calls_from_text(transcript, tools)
                if not tool_calls:
                    tool_calls = fallback_tool_calls(transcript, tools)
        if tools and not tool_calls:
            tool_calls = infer_tool_calls_from_text(response_text, tools)
        response_text = strip_tool_calls(response_text).strip()
        if tool_calls and not response_text:
            response_text = "Calling tool..."
        output_path = synthesize_audio(audio_tokens)
        audio_url = f'/audio/{os.path.basename(output_path)}' if output_path else None

        def generate_events():
            for chunk in chunk_text(response_text):
                yield f"data: {json.dumps({'delta': chunk})}\n\n"
                time.sleep(0.02)

            for tool in tool_calls:
                yield f"data: {json.dumps({'tool_call': tool})}\n\n"

            if audio_url:
                yield f"data: {json.dumps({'audio_url': audio_url})}\n\n"

            yield "data: [DONE]\n\n"

        return Response(
            stream_with_context(generate_events()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
            }
        )

    except Exception as exc:
        print(f'Error: {str(exc)}')
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


@app.route('/audio/<filename>', methods=['GET'])
def get_audio(filename):
    return send_file(os.path.join(output_dir, filename), mimetype='audio/wav')


if __name__ == '__main__':
    load_model_if_needed(MODEL_DEFAULT_ID)
    app.run(host='0.0.0.0', port=11236, debug=False, threaded=True)
