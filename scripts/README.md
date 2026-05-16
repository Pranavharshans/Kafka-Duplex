# Scripts

Standalone command-line scripts live here.

Planned scope:

- data preparation scripts
- one-off conversion utilities
- training launch scripts
- local validation runners

Current Stage 1 additions:

- `build_stage1_dataset.py`: builds `ASR` and `TTS` JSONL manifests from a LibriSpeech-style directory
- `launch_stage1_training.py`: validates Stage 1 training config and manifest presence
