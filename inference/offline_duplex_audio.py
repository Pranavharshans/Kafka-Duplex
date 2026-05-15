"""Offline prerecorded-audio runner for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass, field

from kafka_duplex.audio import AudioBuffer, AudioChunk, chunk_audio, concatenate
from kafka_duplex.chunks import build_backchannel_chunk, build_chunk, build_listen_chunk, build_speak_chunk, Token
from kafka_duplex.codec import SpeechCodec, timed_roundtrip
from kafka_duplex.schema import DuplexAction


@dataclass(slots=True)
class OfflineEvent:
    """One offline duplex event derived from prerecorded audio."""

    chunk_index: int
    user_rms: float
    action: DuplexAction
    user_tokens: list[int]
    agent_tokens: list[int]
    serialized_chunk: list[Token]


@dataclass(slots=True)
class OfflineRunResult:
    """Artifacts produced by the offline prerecorded runner."""

    events: list[OfflineEvent] = field(default_factory=list)
    decoded_agent_audio: AudioBuffer | None = None
    rolling_context: list[Token] = field(default_factory=list)


class OfflineDuplexRunner:
    """Run the Phase 1 control flow over prerecorded audio chunks."""

    def __init__(self, codec: SpeechCodec, *, chunk_ms: int = 200) -> None:
        self.codec = codec
        self.chunk_ms = chunk_ms

    def run(self, audio: AudioBuffer) -> OfflineRunResult:
        chunks = chunk_audio(audio, chunk_ms=self.chunk_ms)
        events: list[OfflineEvent] = []
        decoded_buffers: list[AudioBuffer] = []
        rolling_context: list[Token] = []

        for chunk in chunks:
            user_tokens, _, _ = timed_roundtrip(self.codec, chunk)
            action = self._choose_action(chunk)
            agent_text, agent_speech = self._agent_payload(chunk.index, action)

            if action == DuplexAction.LISTEN:
                record = build_listen_chunk(
                    user_tokens,
                    user_is_speaking=not chunk.is_silent,
                    metadata={"source": "offline", "chunk_index": chunk.index},
                )
                agent_tokens = []
            elif action == DuplexAction.BACKCHANNEL:
                record = build_backchannel_chunk(
                    user_tokens,
                    speech_tokens=agent_speech,
                    user_is_speaking=not chunk.is_silent,
                    metadata={"source": "offline", "chunk_index": chunk.index},
                )
                agent_tokens = agent_speech
            else:
                record = build_speak_chunk(
                    user_tokens,
                    text_tokens=agent_text,
                    speech_tokens=agent_speech,
                    user_is_speaking=False,
                    metadata={"source": "offline", "chunk_index": chunk.index},
                )
                agent_tokens = agent_speech

            serialized = build_chunk(record)
            rolling_context.extend(serialized)
            if agent_tokens:
                decoded_buffers.append(self.codec.decode_chunk(agent_tokens, sample_rate=audio.sample_rate))

            events.append(
                OfflineEvent(
                    chunk_index=chunk.index,
                    user_rms=chunk.rms,
                    action=action,
                    user_tokens=user_tokens,
                    agent_tokens=agent_tokens,
                    serialized_chunk=serialized,
                )
            )

        return OfflineRunResult(
            events=events,
            decoded_agent_audio=concatenate(decoded_buffers) if decoded_buffers else AudioBuffer(samples=[], sample_rate=audio.sample_rate),
            rolling_context=rolling_context,
        )

    @staticmethod
    def _choose_action(chunk: AudioChunk) -> DuplexAction:
        if chunk.is_silent:
            return DuplexAction.SPEAK
        if (chunk.index + 1) % 4 == 0:
            return DuplexAction.BACKCHANNEL
        return DuplexAction.LISTEN

    @staticmethod
    def _agent_payload(chunk_index: int, action: DuplexAction) -> tuple[list[int], list[int]]:
        if action == DuplexAction.BACKCHANNEL:
            base = 7000 + (chunk_index * 10)
            return [], [base + offset for offset in range(5)]

        if action == DuplexAction.SPEAK:
            text_base = 1000 + (chunk_index * 2)
            speech_base = 9000 + (chunk_index * 10)
            return [text_base, text_base + 1], [speech_base + offset for offset in range(5)]

        return [], []
