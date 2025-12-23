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
常量配置文件
存储推理过程中使用的提示词、模板和其他配置常量
"""

# ============= Prompts =============
# S2T (Speech to Text) 模式提示词 - 仅生成文本
DEFAULT_S2T_PROMPT = "You are asked to generate text tokens."

# S2M (Speech to Joint speech-text response) 模式提示词 - 同时生成文本和语音
DEFAULT_S2M_PROMPT = "You are asked to generate both text and speech tokens at the same time."

# S2M 口语
SPOKEN_S2M_PROMPT = DEFAULT_S2M_PROMPT + " " + "你的名字是小云。你是一位来自杭州的温柔友善的女孩，声音甜美，举止亲切。你的回复语气自然友好，力求沟通简洁明了。你的回复简短，通常只有一到三句话，避免使用正式的称谓和重复的短语。你能用恰当的声音回复，遵循用户的指示，并能共情他们的情绪。你能用恰当的方言回复，会说四川话和粤语。"

# Function Calling
FUNCTION_CALLING_PROMPT = DEFAULT_S2T_PROMPT + """
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_definition}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

# ============= Templates =============
# 音频输入模板
AUDIO_TEMPLATE = '<|audio_bos|><|AUDIO|><|audio_eos|>'

# 音频填充token
AUDIO_PAD_TOKEN = "<|audio_pad|>"
AUDIO_BOS_TOKEN = "<|audio_bos|>"


# ============= Model Config =============
# token帧率 (fps)
TOKEN_FPS = 25

DEFAULT_SP_GEN_KWARGS = {
    'text_greedy': False,
    'only_crq_sampling': True,
    'disable_speech': False,
    'force_text_abos': True,
}

DEFAULT_S2M_GEN_KWARGS = {
    'do_sample': True,
    'temperature': 0.8,
    'top_p': 0.9,
    'top_k': 0,
    'num_beams': 1,
    'max_new_tokens': 512,
    'repetition_penalty': 1.2,
    'length_penalty': 1.0,
    'eos_token_id': 151645,
    'pad_token_id': 151643,
}

MAX_HISTORY_TURNS = 8  # Keep only the latest 8 rounds of conversation (16 messages: 8 user + 8 assistant)

# ============= TTS Config =============
TOKEN_HOP_LEN = 15
PRE_LOOKAHEAD_LEN = 3

MAX_TTS_TOKENS = TOKEN_FPS * 8
MAX_TTS_HISTORY = TOKEN_FPS * 20

tts_model_config = {
    "spk_emb_path": "utils/new_spk2info.pt",
}