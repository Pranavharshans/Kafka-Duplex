# Kafka-Duplex: Architecture & PRD

## 1. SYSTEM OVERVIEW

**Kafka-Duplex** is a research-first full-duplex spoken interaction model built to test explicit turn-taking control, not human-level conversational polish. It listens in 200ms chunks, predicts an action for the current chunk, and only generates agent audio when that action says it should.

| Metric | Target |
|--------|--------|
| Trainable Parameters | ~360M |
| Turn-taking latency | 200-400ms |
| Chunk size | 200ms |
| Simultaneous listen+speak | Yes |
| Core novelty | Explicit action head + action-conditioned generation |
| Hardware (inference) | 8GB+ GPU |
| Hardware (comfortable) | 12GB+ GPU |
| Training cost | ~$60-80 cloud |
| Training time | ~45-55 GPU-hours |

## 2. CORE DESIGN

The system is built around one central idea:

- **Turn-taking is a first-class prediction target**
- The model should **explicitly decide** whether to `LISTEN`, `SPEAK`, or `BACKCHANNEL`
- Content generation is then conditioned on that action instead of being forced to emerge purely from silence tokens

This is intentionally a prototype architecture. The goal is to validate whether explicit action-conditioned duplex generation works at small scale with a single LM backbone.

## 3. ARCHITECTURE DIAGRAM

```text
USER AUDIO IN (mic)                                                     AGENT AUDIO OUT (speaker)
     |                                                                                 ^
     v                                                                                 |
+----------------+    10 tokens / 200ms     +---------------------------+    10 tokens / 200ms     +----------------+
| CosyVoice      |------------------------->| SmolLM2-360M              |-------------------------->| CosyVoice      |
| Encoder        |                          | + 3-way Action Head       |                           | Decoder        |
| (frozen)       |                          | + Action Token Injection  |                           | (frozen)       |
+----------------+                          +---------------------------+                           +----------------+
                                                    |
                                                    v
                                action ∈ { LISTEN, SPEAK, BACKCHANNEL }
```

Per 200ms chunk:

1. Encode user audio into 10 speech tokens
2. Run the LM over the new chunk context
3. Predict a chunk-level action after the user tokens are consumed
4. Inject the predicted action token into the sequence
5. Either emit deterministic silence or generate agent tokens conditioned on that action

## 4. LLM BACKBONE

**Base**: SmolLM2-360M class backbone

| Parameter | Value |
|-----------|-------|
| Architecture | Causal decoder-only LM |
| Scale target | ~360M parameters |
| Context window | 8192 tokens |
| Position encoding | RoPE |
| Training mode | Full fine-tuning |
| Added module | Small 3-way action classification head |
| Final model role | Speech-text-speech policy + generation core |

Rationale for `~360M` instead of `135M`:

- `135M` is likely too small to jointly learn speech grounding, dialogue policy, turn-taking, and speech token generation
- `360M` is still cheap enough for fast iteration, but gives more headroom for the duplex policy

## 5. VOCABULARY (53,260 tokens)

```text
Tokens 0-49151:     BPE text tokens
Tokens 49152-53247: CosyVoice speech codes (4096 codebook)
Token 53248:        [CHUNK]         200ms chunk boundary
Token 53249:        [ASR]           ASR task prefix
Token 53250:        [TTS]           TTS task prefix
Token 53251:        [SOS]           Start of speech
Token 53252:        [EOS]           End of speech
Token 53253:        [SOT]           Start of text
Token 53254:        [EOT]           End of text
Token 53255:        <sil_sp>        Silent speech token
Token 53256:        <sil_txt>       No text this chunk
Token 53257:        [LISTEN]        Action token: listen
Token 53258:        [SPEAK]         Action token: speak
Token 53259:        [BACKCHANNEL]   Action token: backchannel
```

Notes:

- The action is predicted by a parallel head
- The predicted action is also materialized as a real token in the sequence for conditioning and debugging
- `YIELD` and `INTERRUPT` are **derived events**, not primary labels

## 6. SEQUENCE FORMATS

### Stage 1 — ASR

```text
[ASR] [SOS] sp1 sp2 sp3 ... spN [EOS] [SOT] txt1 txt2 ... txtM [EOT]
```

### Stage 1 — TTS

```text
[TTS] [SOT] txt1 txt2 ... txtM [EOT] [SOS] sp1 sp2 sp3 ... spN [EOS]
```

### Stage 2 — Half-Duplex Dialogue

```text
[SOS] user_sp1..N [EOS] [SOT] agent_txt1..M [EOT] [SOS] agent_sp1..K [EOS] ...
```

### Stage 3 — Full-Duplex (200ms chunks)

Measured codec note:

- On the tested `CosyVoice-300M` speech tokenizer path, the real token rate is **10 speech tokens per 200ms**
- Earlier `5 tokens / 200ms` assumptions were incorrect for this runtime

Canonical chunk layout:

```text
[CHUNK] usr_sp1 usr_sp2 usr_sp3 usr_sp4 usr_sp5 usr_sp6 usr_sp7 usr_sp8 usr_sp9 usr_sp10 [ACTION] agt_txt1 agt_txt2 agt_sp1 agt_sp2 agt_sp3 agt_sp4 agt_sp5 agt_sp6 agt_sp7 agt_sp8 agt_sp9 agt_sp10
```

Total: `24 tokens / chunk`

Action-specific behavior:

#### LISTEN

```text
[CHUNK] usr1 usr2 usr3 usr4 usr5 usr6 usr7 usr8 usr9 usr10 [LISTEN] <sil_txt> <sil_txt> <sil_sp> <sil_sp> <sil_sp> <sil_sp> <sil_sp> <sil_sp> <sil_sp> <sil_sp> <sil_sp> <sil_sp>
```

- No generation after the action token
- Agent output is deterministic silence

#### SPEAK

```text
[CHUNK] usr1 usr2 usr3 usr4 usr5 usr6 usr7 usr8 usr9 usr10 [SPEAK] txt1 txt2 sp1 sp2 sp3 sp4 sp5 sp6 sp7 sp8 sp9 sp10
```

- Full generation path
- Agent may emit internal text scaffold plus speech tokens

#### BACKCHANNEL

```text
[CHUNK] usr1 usr2 usr3 usr4 usr5 usr6 usr7 usr8 usr9 usr10 [BACKCHANNEL] <sil_txt> <sil_txt> bch1 bch2 bch3 bch4 bch5 bch6 bch7 bch8 bch9 bch10
```

- No text tokens
- Generate only a short speech acknowledgment

### Derived transition events

These are computed from action transitions, not directly labeled:

```python
yield_event = prev_action == SPEAK and cur_action == LISTEN and user_speaking
interrupt_event = prev_action == LISTEN and cur_action == SPEAK and user_speaking
```

### Context window math

- `8192 / 24 = 341` chunks
- `341 x 200ms = 68.2 seconds` maximum rolling context

## 7. ACTION HEAD

The model predicts the current chunk action **after consuming the user speech tokens for that chunk**.

Minimal design:

```python
action_head = MLP(hidden_size -> 128 -> 3)
classes = [LISTEN, SPEAK, BACKCHANNEL]
```

Prediction point:

- Input sequence up to: `[CHUNK] usr_sp1..10`
- Read hidden state at the last user speech token position
- Produce 3-way action logits

Conditioning mechanism:

1. Predict action with the parallel head
2. Append the corresponding action token to the sequence
3. Generate the remaining agent tokens conditioned on that token

This keeps the action:

- explicitly supervised
- inspectable in logs
- available to the generator as a causal input

## 8. TRAINING STAGES

### Stage 1: Modality Alignment

| | |
|--|--|
| **Goal** | Teach speech <-> text mapping |
| **Tasks** | ASR + TTS |
| **Data** | LibriSpeech 960h |
| **Init** | Pretrained SmolLM2-scale checkpoint with expanded vocab |
| **Batch** | 32 effective |
| **Seq len** | 1024 |
| **Epochs** | 3 |
| **Loss** | Standard CE on target modality only |
| **Success** | Model can map speech to text and text to speech tokens |

### Stage 2: Half-Duplex Dialogue

| | |
|--|--|
| **Goal** | Learn spoken request -> agent response behavior |
| **Tasks** | User speech -> agent text -> agent speech |
| **Data** | 10K synthetic spoken dialogues |
| **Init** | Stage 1 checkpoint |
| **Batch** | 64 effective |
| **Seq len** | 4096 |
| **Epochs** | 4-5 |
| **Loss** | Agent-token CE only |
| **Success** | Coherent spoken responses in turn-based mode |

### Stage 3: Full-Duplex Action-Conditioned Training

| | |
|--|--|
| **Goal** | Learn chunk-level timing policy and chunk-conditioned generation |
| **Tasks** | LISTEN vs SPEAK vs BACKCHANNEL |
| **Data** | 15K chunked conversations (synthetic + augmented) |
| **Format** | `[CHUNK] usr_sp10 [ACTION] agt_txt2 agt_sp10` |
| **Init** | Stage 2 checkpoint |
| **Batch** | 64 effective |
| **Seq len** | 8192 |
| **Epochs** | 8-10 |
| **Loss** | `CE(tokens) + lambda * CE(action)` |
| **Success** | Stable chunk actions, low-latency duplex loop, interruption/yield behavior emerges from transitions |

Recommended loss:

```python
total_loss = token_loss + 0.5 * action_loss
```

Important training note:

- Early Stage 3 can use teacher-forced action tokens
- Later Stage 3 should introduce predicted-action rollouts to reduce train/infer mismatch

## 9. DATA GENERATION

### Stage 1

```python
from datasets import load_dataset
ds = load_dataset("openslr/librispeech_asr", "all")
```

Then tokenize audio with the CosyVoice encoder.

### Stage 2

```python
# 1. Generate text dialogues with an LLM
# 2. Synthesize each turn with CosyVoice TTS
# 3. Re-encode audio into speech tokens
# 4. Serialize into half-duplex training sequences
```

### Stage 3

```python
# 1. Take Stage 2 dialogues
# 2. Chunk both sides into 200ms windows
# 3. Encode each chunk into 10 speech tokens
# 4. Label each chunk action:
#       BACKCHANNEL if short overlapping acknowledgment
#       SPEAK if agent speech is present
#       LISTEN otherwise
# 5. Derive yield/interrupt only for evaluation
# 6. Serialize as [CHUNK] usr_sp10 [ACTION] agt_txt2 agt_sp10
```

Chunk labeling skeleton:

```python
def get_action_label(chunk):
    user_speaking = not all_silent(chunk.user_speech_tokens)
    agent_speaking = not all_silent(chunk.agent_speech_tokens)
    is_bch = agent_speaking and chunk.is_short_ack

    if is_bch:
        return BACKCHANNEL
    if agent_speaking:
        return SPEAK
    return LISTEN
```

Prototype assumption:

- Label quality only needs to be good enough to validate the architecture
- Perfectly human-like backchannel timing is not required in v1

## 10. INFERENCE

### Per-chunk runtime

```text
1. Capture 200ms of user audio
2. Encode to 10 speech tokens
3. Append [CHUNK] + user tokens
4. Run LM forward and predict action
5. Append action token
6. If LISTEN:
       append deterministic silence tokens
   If SPEAK:
       generate 2 text + 10 speech tokens
   If BACKCHANNEL:
       generate 10 speech tokens only
7. Decode speech tokens and play audio
```

### Realtime loop

```python
while True:
    user_audio = capture_200ms()
    user_tokens = cosyvoice_encode(user_audio)              # -> 10 tokens

    context.extend([CHUNK] + user_tokens)
    h = model.forward_until_current_position(context)
    action = action_head(h.last_user_token_state)

    if action == LISTEN:
        context.extend([LISTEN, SIL_TXT, SIL_TXT] + [SIL_SP] * 5)
        continue

    if action == SPEAK:
        context.append(SPEAK)
        agent_tokens = model.generate(max_new_tokens=12)    # 2 txt + 10 speech
    else:
        context.append(BACKCHANNEL)
        agent_tokens = [SIL_TXT, SIL_TXT] + model.generate(max_new_tokens=5)

    context.extend(agent_tokens)
    play(cosyvoice_decode(agent_tokens[-5:]))
```

### Latency estimate

```text
LISTEN chunk:
  encode + forward + action = ~25-35ms

SPEAK chunk:
  encode + forward + action + generation + decode = ~55-75ms

Average expected chunk cost:
  ~40-50ms depending on speaking ratio
```

The main win is not just speed. It is:

- deterministic silence during listen chunks
- direct observability of the timing policy
- a cleaner training target for duplex control

### Memory estimate

```text
Model (bf16, ~360M):      ~0.8-1.0 GB
KV cache (8192 ctx):      ~3-4 GB
CosyVoice enc + dec:      ~0.3-0.6 GB
-----------------------------------------
Total VRAM:               ~5-6 GB practical
```

## 11. EVALUATION METRICS

| Metric | Target | Stage |
|--------|--------|-------|
| ASR WER | Report | 1 |
| TTS intelligibility | Report | 1 |
| Response relevance | Report | 2 |
| Action accuracy | >85% | 3 |
| LISTEN/SPEAK/BACKCHANNEL F1 | Report | 3 |
| Turn-taking latency | <400ms | 3 |
| Yield rate within 400ms | >80% | 3 |
| Interrupt precision | Report | 3 |
| FD-Bench score | Report | 3 |

Prototype interpretation:

- For v1, action quality matters more than human-likeness
- If the action head is robust and the loop stays stable, the architecture passes the main test

## 12. KNOWN RISKS

1. **Action errors are highly visible**
   A wrong timing decision hurts more than a mediocre lexical choice.

2. **Backchannel labels may be noisy**
   The class is useful, but its heuristics may be messy in synthetic data.

3. **Fixed per-chunk token budget may be too rigid**
   `2 text + 10 speech` is a prototype assumption, not a proven optimum.

4. **Frozen speech stack may cap quality**
   If the codec or decoder behaves poorly in chunked streaming mode, the LM cannot fully rescue it.

5. **Synthetic data may teach brittle timing**
   Acceptable for architecture validation, but likely not enough for natural production behavior.

## 13. WHY THIS VERSION

Compared with the earlier pure next-token duplex design, this version:

- gives explicit supervision for turn-taking
- reduces the amount of implicit behavior the LM must discover
- makes logging and debugging much easier
- allows deterministic listen behavior
- keeps the system as a single trainable LM instead of splitting policy and generation into separate large models

This is the right v1 for fast experimentation.

## 14. REFERENCES

- [OmniFlatten (2410.17799)](https://arxiv.org/abs/2410.17799) — flattened speech-text sequence design
- [SyncLLM (2409.15594)](https://arxiv.org/abs/2409.15594) — time-synchronous spoken interaction
- [Chronological Thinking (2510.05150)](https://arxiv.org/abs/2510.05150) — think-while-listen style reasoning
- [Full-Duplex-Bench (2503.04721)](https://arxiv.org/abs/2503.04721) — duplex evaluation setup
- [Sommelier (2603.25750)](https://arxiv.org/abs/2603.25750) — speech data pipeline ideas
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) — speech tokenizer / decoder stack
- [SmolLM2](https://huggingface.co/collections/HuggingFaceTB/smollm2-673c4e5f04d95e387e51d7e0) — lightweight LM family
