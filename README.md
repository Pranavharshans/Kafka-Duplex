# Kafka-Duplex

Research prototype for an action-conditioned full-duplex speech model.

- Architecture spec: [ARCHITECTURE.md](ARCHITECTURE.md)
- Current focus: validate explicit `LISTEN` / `SPEAK` / `BACKCHANNEL` control in a 200ms chunked duplex loop
- Measured codec assumption: the tested `CosyVoice-300M` tokenizer path emits `10` speech tokens per `200ms`
- Runtime example env: [configs/runtime.example.env](configs/runtime.example.env)
- Real token probe: `python3 scripts/probe_cosyvoice_tokens.py --input-wav <file.wav>`
- Stage 1 training now defaults to `/workspace/kafka_duplex_runs/...` so Vast.ai TensorBoard can read event files without a second log root
- Stage 1 dataset build can combine multiple LibriSpeech subsets by repeating `--input-root`
- Prebuilt Stage 1 manifests can be fetched from Hugging Face with `python3 scripts/fetch_stage1_hf_dataset.py`
- Fresh large-GPU boxes can bootstrap directly from Hugging Face with `bash scripts/run_stage1_from_hf.sh`

This repository is intentionally architecture-first and prototype-oriented.
