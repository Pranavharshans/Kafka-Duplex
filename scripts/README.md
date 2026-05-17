# Scripts

Standalone command-line scripts live here.

Planned scope:

- data preparation scripts
- one-off conversion utilities
- training launch scripts
- local validation runners

Current Stage 1 additions:

- `build_stage1_dataset.py`: builds `ASR` and `TTS` JSONL manifests from a LibriSpeech-style directory
- `fetch_stage1_hf_dataset.py`: downloads a prebuilt Stage 1 manifest snapshot from Hugging Face into the `/workspace` paths expected by training
- `eval_stage1_checkpoint.py`: loads a Stage 1 checkpoint, runs task-aware greedy generation on a few manifest examples, and reports generated-vs-target token overlap
- `launch_stage1_training.py`: validates Stage 1 training config and manifest presence, and now fails fast on broken CUDA unless CPU fallback is explicitly allowed
- `run_stage1_from_hf.sh`: end-to-end remote launcher for a fresh GPU box that pulls the repo, verifies CUDA, fetches the HF manifest snapshot, installs Stage 1 deps including `transformers`, and starts Stage 1 training under `/workspace`
