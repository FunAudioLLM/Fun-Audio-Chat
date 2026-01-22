# Fun-Audio-Chat

<p align="right">
  <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
</p>

<div align="center">

<img src="assets/通义百聆.png" alt="通义百聆" height="80">

**Fun-Audio-Chat** 是一个专为自然、低延迟语音交互打造的大型音频语言模型。

[![arXiv](https://img.shields.io/badge/arXiv-2512.20156-red)](https://arxiv.org/pdf/2512.20156)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-模型-yellow)](https://huggingface.co/FunAudioLLM/Fun-Audio-Chat-8B)
[![ModelScope](https://img.shields.io/badge/ModelScope-模型-orange)](https://modelscope.cn/FunAudioLLM/Fun-Audio-Chat-8B)
[![演示](https://img.shields.io/badge/演示-页面-green)](https://funaudiollm.github.io/funaudiochat)

</div>

---

## 📋 目录

- [概述](#overview)
- [最新动态](#news)
- [安装](#installation)
- [快速开始](#quick-start)
- [评测](#evaluation)
- [训练](#training)
- [致谢](#acknowledgments)
- [许可证](#license)
- [实习生招聘](#-实习生招聘)
- [联系我们](#contact)

---

## <a id="overview"></a>📖 概述

**Fun-Audio-Chat** 是一个专为自然、低延迟语音交互打造的大型音频语言模型。它引入了**双分辨率语音表征**（高效的5Hz共享骨干网络 + 25Hz精细化头部），在保持高语音质量的同时大幅降低计算开销，并采用**Core-Cocktail训练策略**来保持强大的文本LLM能力。该模型在语音问答、音频理解、语音函数调用、语音指令遵循和语音情感共鸣等基准测试中均取得了顶尖成绩。

<div align="center">
<img src="assets/Results.png" alt="Fun-Audio-Chat 评测结果" width="95%">
</div>

### 核心特性

- **双分辨率语音表征**：高效的5Hz帧率（相比其他模型的12.5Hz或25Hz），将GPU训练时间减少近50%，同时保持高语音质量
- **业界领先性能**：在同等规模模型（约8B参数）中，在OpenAudioBench、VoiceBench、UltraEval-Audio、MMAU、MMAU-Pro、MMSU、Speech-ACEBench、Speech-BFCL、Speech-SmartInteract、VStyle等评测集上排名领先
- **全面的能力覆盖**：支持语音问答、音频理解、语音函数调用、语音指令遵循、语音情感共鸣

<div align="center">
<img src="assets/Architecture.png" alt="Fun-Audio-Chat 架构图" width="95%">
</div>

---

## <a id="news"></a>📰 最新动态

- **[2025.12.23]** Fun-Audio-Chat-8B（模型、训练和推理代码）发布，在语音问答、音频理解、语音函数调用、语音指令遵循和语音情感共鸣等多个基准测试中取得业界领先性能

---

## <a id="installation"></a>🔧 安装

### 1. 环境要求

- Python == 3.12
- PyTorch == 2.8.0
- ffmpeg
- 显存要求：推理需要 ~24GB，训练需要 4×80GB

### 2. 克隆仓库

```bash
git clone --recurse-submodules https://github.com/FunAudioLLM/Fun-Audio-Chat
cd Fun-Audio-Chat
```

### 3. 安装依赖

```bash
apt install ffmpeg
# 建议创建新的conda环境
conda create -n FunAudioChat python=3.12 -y
conda activate FunAudioChat
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### 4. 下载预训练模型

预训练模型需要放置在 `pretrained_models/` 目录下：

**使用 HuggingFace 下载：**
```bash
pip install huggingface-hub
hf download FunAudioLLM/Fun-Audio-Chat-8B --local-dir ./pretrained_models/Fun-Audio-Chat-8B
hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --local-dir ./pretrained_models/Fun-CosyVoice3-0.5B-2512
```

**或使用 ModelScope 下载：**
```bash
modelscope download --model FunAudioLLM/Fun-Audio-Chat-8B --local_dir pretrained_models/Fun-Audio-Chat-8B
modelscope download --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --local_dir pretrained_models/Fun-CosyVoice3-0.5B-2512
```

**目录结构：**
```
pretrained_models/
├── Fun-Audio-Chat-8B/     # 8B参数主模型
└── Fun-CosyVoice3-0.5B-2512/  # 语音合成模型
```

---

## <a id="quick-start"></a>🚀 快速开始

### 运行示例脚本

```bash
export PYTHONPATH=`pwd`
python examples/infer_s2t.py
python examples/infer_s2s.py
```

### Web 演示

**服务端：**
```bash
# 启动服务器
pip install sphn aiohttp

# 使用另一张 GPU 以获得更好的性能
python -m web_demo.server.server --model-path pretrained_models/Fun-Audio-Chat-8B --port 11236 --tts-gpu 1
```

**客户端：**
```bash
cd web_demo/client
# 1. 使用 NVM 管理 Node 版本（如未安装请先安装 NVM）
# 安装 NVM（如需要）：
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash

# 使用项目推荐的 Node 版本
nvm use

# 2. 生成 SSL 证书（cert.pem 和 key.pem）
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# 3. 创建 .env.local 文件并添加配置
cat > .env.local << 'EOF'
VITE_QUEUE_API_PATH=/api
EOF

# 4. 安装依赖
npm install

# 5. 运行开发服务器
npm run dev
```

更多详情请参阅 [`web_demo/server/README.md`](web_demo/server/README.md) 和 [`web_demo/client/README.md`](web_demo/client/README.md)。

---

## <a id="evaluation"></a>📊 评测

### 1. S2T（语音转文字）

推理时使用 [`utils/constant.py`](utils/constant.py) 中的 `DEFAULT_S2T_PROMPT`。推理脚本请参考 [`examples/infer_s2t.py`](examples/infer_s2t.py)。

- **VoiceBench**：数据和评测脚本可在 [Kimi-Audio-Evalkit](https://github.com/MoonshotAI/Kimi-Audio-Evalkit) 获取
- **OpenAudioBench**：数据和评测脚本可在 [OpenAudioBench](https://huggingface.co/datasets/baichuan-inc/OpenAudioBench) 获取

### 2. S2S（语音转语音）

推理时使用 [`utils/constant.py`](utils/constant.py) 中的 `DEFAULT_S2M_PROMPT`。推理脚本请参考 [`examples/infer_s2s.py`](examples/infer_s2s.py)。

- **UltraEval-Audio**：数据和评测脚本可在 [UltraEval-Audio](https://github.com/OpenBMB/UltraEval-Audio) 获取

### 3. 音频理解与语音识别

#### 音频理解

推理时使用 [`utils/constant.py`](utils/constant.py) 中的 `DEFAULT_S2T_PROMPT`。推理脚本请参考 [`examples/infer_s2t.py`](examples/infer_s2t.py)。

- **MMAU**：数据和评测脚本可在 [Kimi-Audio-Evalkit](https://github.com/MoonshotAI/Kimi-Audio-Evalkit)（MMAU评测部分）获取
- **MMSU**：数据和评测脚本可在 [MMSU_Bench](https://github.com/dingdongwang/MMSU_Bench) 获取
- **MMAU-Prompt**：数据和评测脚本可在 [MMAUPro](https://github.com/sonalkum/MMAUPro) 获取

**音频理解任务的指令格式：**
- 对于选择题：`f"{question} Choose the correct option from the following options:\n(A){choice_a}\n(B){choice_b}\n(C){choice_c}\n(D){choice_d}"`（如有更多选项请相应扩展）
- 对于非选择题：`f"{question}"`

关于 `question` 和 `choices` 请参考各数据集中的相应文本。

#### 语音识别（ASR）

**评测工具**：使用 [whisper_normalizer](https://github.com/kurianbenoy/whisper_normalizer) 和 [compute-wer](https://github.com/pengzhendong/compute-wer) 计算 WER/CER。

**ASR 指令**：`Please help me transcribe the audio.`

### 4. 语音函数调用

推理时使用 [`utils/constant.py`](utils/constant.py) 中的 `FUNCTION_CALLING_PROMPT`。注意：需要将 `{tools_definition}` 占位符替换为适当的工具定义。推理脚本和工具定义格式请参考 [`examples/infer_s2t.py`](examples/infer_s2t.py)。


- **SpeechFCEval**：数据和评测脚本可在 [SpeechFCEval](https://huggingface.co/datasets/FunAudioLLM/SpeechFCEval) 获取
- 部分数据和评测脚本来自 [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard/bfcl_eval) 和 [ACEBench](https://github.com/chenchen0103/ACEBench/tree/main/model_eval)。感谢他们的贡献。

### 5. 语音指令遵循

推理时使用 [`utils/constant.py`](utils/constant.py) 中的 `SPOKEN_S2M_PROMPT`。推理脚本请参考 [`examples/infer_s2s.py`](examples/infer_s2s.py)。

- **VStyle**：数据和评测脚本可在 [VStyle](https://github.com/alibaba/vstyle) 获取

---

## <a id="training"></a>🎓 训练

### 0. 环境配置

**安装第三方库：**
```bash
pip install flash-attn --no-build-isolation
cd third_party/LLaMA-Factory
pip install -e ".[metrics]" --no-build-isolation
```

### 1. 准备数据

**参考数据：**

将 [GSQA/spoken-alpaca-gpt4](https://huggingface.co/datasets/GSQA/spoken-alpaca-gpt4) 数据下载到 `training/datasets/spoken-alpaca-gpt4` 目录。

**执行格式转换：**
```bash
cd ../../training
python process/data_process.py --debug
```

在 [`training/data/dataset_info.json`](training/data/dataset_info.json) 中配置您的数据集。

### 2. 配置训练参数

编辑 [`training/configs/sft.yaml`](training/configs/sft.yaml)：

```yaml
model_name_or_path: ../pretrained_models/Fun-Audio-Chat-8B
dataset: your_dataset
template: funaudiochat
output_dir: saves/your_experiment
```

### 3. 开始训练

```bash
bash run_shell/run.sh
```

### 4. 监控训练

训练日志保存在 `training/logs/` 目录，模型检查点保存在配置的 `output_dir` 中。

---

## <a id="acknowledgments"></a>🙏 致谢

本项目基于以下优秀的开源项目构建：

- [**Transformers**](https://github.com/huggingface/transformers)
- [**LlamaFactory**](https://github.com/hiyouga/LLaMA-Factory)
- [**Moshi**](https://github.com/kyutai-labs/moshi)
- [**CosyVoice**](https://github.com/FunAudioLLM/CosyVoice)

---

## Citation

如果您觉得本模型对您有帮助，请引用我们的论文：

```bibtex
@article{funaudiochat2025,
  title={Fun-Audio-Chat Technical Report},
  author={Qian Chen and Luyao Cheng and Chong Deng and Xiangang Li and Jiaqing Liu and Chao-Hong Tan and Wen Wang and Junhao Xu and Jieping Ye and Qinglin Zhang and Qiquan Zhang and Jingren Zhou},
  year={2025},
  eprint={2512.20156},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2512.20156},
}

@misc{tan2025drvoiceparallelspeechtextvoice,
  title={DrVoice: Parallel Speech-Text Voice Conversation Model via Dual-Resolution Speech Representations}, 
  author={Chao-Hong Tan and Qian Chen and Wen Wang and Chong Deng and Qinglin Zhang and Luyao Cheng and Hai Yu and Xin Zhang and Xiang Lv and Tianyu Zhao and Chong Zhang and Yukun Ma and Yafeng Chen and Hui Wang and Jiaqing Liu and Xiangang Li and Jieping Ye},
  year={2025},
  eprint={2506.09349},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2506.09349}, 
}
```


## <a id="license"></a>📄 许可证

Fun-Audio-Chat 是由阿里云开发的用于自然语音交互的大型音频语言模型，采用 Apache License (Version 2.0) 许可证。
本产品包含多个采用其他开源许可证的第三方组件。
详情请参阅 [NOTICE](NOTICE) 文件。

许可证详情请参阅 [LICENSE](LICENSE) 文件。

---

## 💼 实习生招聘

研究实习生岗位开放（工作地点：杭州、北京、上海）。

**研究方向：** 多模态交互、多模态理解与生成等

**联系方式：** dengchong.d@alibaba-inc.com 和 tanqing.cq@alibaba-inc.com

---

## <a id="contact"></a>📮 联系我们

如有任何问题或建议，请通过以下方式联系我们：

- 🐛 提交 [Issue](https://github.com/FunAudioLLM/Fun-Audio-Chat/issues)
- 💡 提交 Pull Request
- 📧 发送邮件
- 💬 加入官方钉钉交流群

<img src="assets/dingding.png" width="250px">

---

<div align="center">

**如果本项目对您有帮助，请给我们一个 ⭐ Star！**

</div>

