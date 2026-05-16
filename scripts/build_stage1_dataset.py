"""Build Stage 1 ASR/TTS alignment manifests from a LibriSpeech-style directory."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kafka_duplex import AudioBuffer, chunk_audio, concatenate, create_codec, load_wav_mono_pcm16
from kafka_duplex.stage1 import (
    LibriSpeechUtterance,
    Stage1AlignmentExample,
    deterministic_split,
    speech_to_vocab_ids,
    text_to_mock_ids,
    write_jsonl,
)


SUPPORTED_AUDIO_EXTENSIONS = {".flac", ".wav"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage 1 alignment manifests.")
    parser.add_argument("--input-root", required=True, help="LibriSpeech root or subset directory.")
    parser.add_argument("--output-root", required=True, help="Output directory for JSONL manifests.")
    parser.add_argument("--codec", default="mock", help="Codec adapter: mock or cosyvoice.")
    parser.add_argument("--val-ratio", type=float, default=0.02, help="Validation ratio.")
    parser.add_argument("--seed", type=int, default=13, help="Deterministic split seed.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max utterances to process.")
    parser.add_argument("--num-workers", type=int, default=4, help="Parallel workers for encoding.")
    return parser.parse_args()


def read_audio(path: str) -> AudioBuffer:
    input_path = Path(path)
    if input_path.suffix.lower() == ".wav":
        return load_wav_mono_pcm16(str(input_path))

    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("Reading FLAC requires soundfile to be installed.") from exc

    samples, sample_rate = sf.read(str(input_path), dtype="int16", always_2d=False)
    if getattr(samples, "ndim", 1) != 1:
        raise ValueError(f"Only mono audio is supported for Stage 1 build: {path}")
    return AudioBuffer(samples=[int(sample) for sample in samples.tolist()], sample_rate=int(sample_rate))


def discover_utterances(root: Path) -> list[LibriSpeechUtterance]:
    utterances: list[LibriSpeechUtterance] = []
    for transcript_path in sorted(root.rglob("*.trans.txt")):
        speaker_id = transcript_path.parent.parent.name
        chapter_id = transcript_path.parent.name
        with transcript_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                utterance_id, transcript = line.split(" ", 1)
                audio_path = None
                for extension in SUPPORTED_AUDIO_EXTENSIONS:
                    candidate = transcript_path.parent / f"{utterance_id}{extension}"
                    if candidate.exists():
                        audio_path = candidate
                        break
                if audio_path is None:
                    continue
                utterances.append(
                    LibriSpeechUtterance(
                        utterance_id=utterance_id,
                        speaker_id=speaker_id,
                        chapter_id=chapter_id,
                        transcript=transcript.strip(),
                        audio_path=str(audio_path),
                    )
                )
    return utterances


def encode_utterance(utterance: LibriSpeechUtterance, codec_name: str) -> tuple[Stage1AlignmentExample, Stage1AlignmentExample]:
    codec = create_codec(codec_name)
    audio = read_audio(utterance.audio_path)
    chunks = chunk_audio(audio, chunk_ms=200)
    raw_speech_token_ids: list[int] = []
    for chunk in chunks:
        raw_speech_token_ids.extend(codec.encode_chunk(chunk))
    speech_token_ids = speech_to_vocab_ids(raw_speech_token_ids)

    text_token_ids = text_to_mock_ids(utterance.transcript)
    common = {
        "transcript": utterance.transcript,
        "text_token_ids": text_token_ids,
        "speech_token_ids": speech_token_ids,
        "source_audio_path": utterance.audio_path,
        "speaker_id": utterance.speaker_id,
        "chapter_id": utterance.chapter_id,
        "utterance_id": utterance.utterance_id,
        "metadata": {
            "num_chunks": len(chunks),
            "num_speech_tokens": len(speech_token_ids),
            "num_text_tokens": len(text_token_ids),
        },
    }
    return (
        Stage1AlignmentExample(task="ASR", example_id=f"{utterance.utterance_id}-asr", **common),
        Stage1AlignmentExample(task="TTS", example_id=f"{utterance.utterance_id}-tts", **common),
    )


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    utterances = discover_utterances(input_root)
    if args.limit > 0:
        utterances = utterances[: args.limit]
    if not utterances:
        raise RuntimeError(f"No LibriSpeech-style utterances found under {input_root}.")

    train_utterances, val_utterances = deterministic_split(utterances, val_ratio=args.val_ratio, seed=args.seed)

    def build_partition(items: list[LibriSpeechUtterance]) -> list[Stage1AlignmentExample]:
        examples: list[Stage1AlignmentExample] = []
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            for asr_example, tts_example in executor.map(lambda item: encode_utterance(item, args.codec), items):
                examples.append(asr_example)
                examples.append(tts_example)
        return examples

    train_examples = build_partition(train_utterances)
    val_examples = build_partition(val_utterances)

    train_path = output_root / "train.stage1.jsonl"
    val_path = output_root / "val.stage1.jsonl"
    train_count = write_jsonl(train_path, train_examples)
    val_count = write_jsonl(val_path, val_examples)

    print(
        " ".join(
            [
                "stage1_dataset",
                f"codec={args.codec}",
                f"input_root={input_root}",
                f"train_examples={train_count}",
                f"val_examples={val_count}",
                f"train_manifest={train_path}",
                f"val_manifest={val_path}",
            ]
        )
    )


if __name__ == "__main__":
    main()
