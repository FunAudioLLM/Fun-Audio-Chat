# Server

Backend server for the FunAudioChat S2S (Speech-to-Speech) demo.

## Requirements

- Python 3.8+
- CUDA enabled GPU (default: cuda:0 for S2S model, cuda:1 for TTS model, TTS GPU is configurable)
- Dependencies installed (see project root requirements)

## Run the Server

From the project root directory, run:

```bash
python -m web_demo.server.server
```

### GPU Compatibility Launcher (sm_121 and newer)

If you see NVRTC errors like `invalid value for --gpu-architecture` on newer GPUs,
use the compatibility launcher to set conservative CUDA env defaults before importing
Torch:

```bash
FUN_AUDIOCHAT_CUDA_COMPAT=1 python -m web_demo.server.server_gpu_compat --host 0.0.0.0 --port 11235
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--host` | str | `localhost` | Server host address |
| `--port` | int | `11235` | Server port |
| `--sample-rate` | int | `24000` | Audio sample rate (Opus) |
| `--model-sample-rate` | int | `16000` | Model sample rate |
| `--output-dir` | str | `./output` | Directory to save input audio files |
| `--model-path` | str | `model/s2s` | Path to S2S model |
| `--tts-gpu` | int | `1` | GPU device id for TTS model |

### Example

```bash
# Run server on custom host and port
python -m web_demo.server.server --host 0.0.0.0 --port 8088

# Specify custom model path
python -m web_demo.server.server --model-path /path/to/your/model

# Use a different GPU for TTS model
python -m web_demo.server.server --tts-gpu 0
```

## Configuration

Additional model parameters can be configured in `utils/constants.py`:

- `SPOKEN_S2M_PROMPT`: Default system prompt for the model
- `MAX_HISTORY_TURNS`: Maximum conversation history turns (default: 8)
- `DEFAULT_S2M_GEN_KWARGS`: Generation parameters (temperature, top_p, top_k, etc.)
- `DEFAULT_SP_GEN_KWARGS`: Speech generation parameters

## License

The present code is provided under the MIT license.
