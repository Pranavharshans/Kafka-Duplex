# Kafka-Duplex Phase Tracker

This file tracks implementation phases, their intended outcomes, and completion state.

Status key:

- `[x]` completed
- `[ ]` not started

## Phase 0: Foundation

- `[x]` Create repository skeleton
- `[x]` Define canonical token, action, and chunk schema

Outcome:

- Top-level folders exist for `data`, `models`, `training`, `inference`, `scripts`, `configs`, and `eval`
- A shared schema module exists for chunk-level data structures and special tokens

## Phase 1: Fake End-to-End Loop

- `[x]` Build mocked 200ms duplex loop with fake speech tokens
- `[x]` Implement chunk assembler/parser utilities

Outcome:

- The control loop works without real audio dependencies
- `LISTEN`, `SPEAK`, and `BACKCHANNEL` chunk paths can be serialized and parsed consistently

## Phase 2: Real Speech I/O Plumbing

- `[ ]` Integrate speech tokenizer and decoder
- `[ ]` Build realtime local duplex runner around chunked audio I/O

Outcome:

- `200ms audio -> speech tokens -> decoded audio` works in practice
- Codec rate, chunking assumptions, and latency are validated

Current status:

- Phase 2 scaffolding is implemented with `audio.py`, `codec.py`, offline validation, and prerecorded-audio runners
- Real codec validation found that the tested `CosyVoice-300M` tokenizer path emits `10 speech tokens / 200ms`
- Adapter wiring and docs must follow the measured token rate rather than the earlier `5 tokens / 200ms` assumption

## Phase 3: Minimal Model

- `[ ]` Implement LM extension with expanded vocabulary and action head
- `[ ]` Train Stage 1 modality alignment

Outcome:

- Base model can process the schema correctly
- ASR/TTS token mapping works well enough to continue

## Phase 4: Half-Duplex Dialogue

- `[ ]` Train Stage 2 half-duplex dialogue model
- `[ ]` Add baseline evaluation for spoken response quality and stability

Outcome:

- Spoken input -> spoken response works in a turn-based setting
- The project has a meaningful non-duplex baseline

## Phase 5: Duplex Policy

- `[ ]` Create Stage 3 chunked duplex dataset with deterministic action labels
- `[ ]` Train `LISTEN` vs `SPEAK` duplex policy first
- `[ ]` Add `BACKCHANNEL` as the third action after 2-class stability

Outcome:

- Action-conditioned duplex behavior is trainable and measurable
- Yield and interruption behavior can be derived and evaluated

## Phase 6: Runtime Validation

- `[ ]` Build streaming inference runner with chunk-level logging
- `[ ]` Run live interaction tests and record failure modes

Outcome:

- The architecture is validated end to end under realtime conditions
- Bottlenecks are visible in timing, action selection, generation, or audio quality
