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
- `final_eval_stage1_checkpoint.py`: runs a fuller final Stage 1 checkpoint eval with decoded ASR text, `WER`, token-match summaries, and optional TTS WAV export when the codec supports decode
- `launch_stage1_training.py`: validates Stage 1 training config and manifest presence, and now fails fast on broken CUDA unless CPU fallback is explicitly allowed
- `remap_stage1_manifest_tokens.py`: rebuilds Stage 1 manifests from the bootstrap token layout into a tokenizer-backed interface with reserved special tokens and speech-token block
- `run_stage1_from_hf.sh`: end-to-end remote launcher for a fresh GPU box that pulls the repo, verifies CUDA, fetches the HF manifest snapshot, installs Stage 1 deps including `transformers`, and starts Stage 1 training under `/workspace`
- `upload_stage1_baseline_to_hf.py`: packages a Stage 1 checkpoint, config, and optional eval/log artifacts and uploads them to a Hugging Face model repo
