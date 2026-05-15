# Kafka-Duplex

Research prototype for an action-conditioned full-duplex speech model.

- Architecture spec: [ARCHITECTURE.md](ARCHITECTURE.md)
- Current focus: validate explicit `LISTEN` / `SPEAK` / `BACKCHANNEL` control in a 200ms chunked duplex loop
- Measured codec assumption: the tested `CosyVoice-300M` tokenizer path emits `10` speech tokens per `200ms`
- Runtime example env: [configs/runtime.example.env](configs/runtime.example.env)
- Real token probe: `python3 scripts/probe_cosyvoice_tokens.py --input-wav <file.wav>`

This repository is intentionally architecture-first and prototype-oriented.
