"""Mocked duplex loop for validating Phase 1 control flow."""

from __future__ import annotations

from dataclasses import dataclass, field

from kafka_duplex.chunks import Token, build_backchannel_chunk, build_chunk, build_listen_chunk, build_speak_chunk, parse_chunk
from kafka_duplex.schema import DuplexAction, DuplexChunkRecord


@dataclass(slots=True)
class MockDuplexConfig:
    """Configuration for the mocked duplex loop."""

    total_chunks: int = 12
    backchannel_every: int = 4
    user_pause_every: int = 3


@dataclass(slots=True)
class MockChunkEvent:
    """Single logged chunk event from the mocked runner."""

    step: int
    user_is_speaking: bool
    action: DuplexAction
    chunk_tokens: list[Token]
    parsed_chunk: DuplexChunkRecord


@dataclass(slots=True)
class MockDuplexResult:
    """Output of a mocked duplex session."""

    events: list[MockChunkEvent] = field(default_factory=list)
    rolling_context: list[Token] = field(default_factory=list)


class MockDuplexRunner:
    """Small deterministic duplex runner for token-only architecture validation."""

    def __init__(self, config: MockDuplexConfig | None = None) -> None:
        self.config = config or MockDuplexConfig()

    def run(self) -> MockDuplexResult:
        result = MockDuplexResult()
        for step in range(self.config.total_chunks):
            user_is_speaking = not self._is_user_pause(step)
            user_tokens = self._fake_user_tokens(step, user_is_speaking)
            action = self._choose_action(step, user_is_speaking)
            record = self._build_record(step, user_tokens, action, user_is_speaking)
            chunk_tokens = build_chunk(record)
            parsed_chunk = parse_chunk(chunk_tokens)

            result.rolling_context.extend(chunk_tokens)
            result.events.append(
                MockChunkEvent(
                    step=step,
                    user_is_speaking=user_is_speaking,
                    action=action,
                    chunk_tokens=chunk_tokens,
                    parsed_chunk=parsed_chunk,
                )
            )
        return result

    def _is_user_pause(self, step: int) -> bool:
        return (step + 1) % self.config.user_pause_every == 0

    def _choose_action(self, step: int, user_is_speaking: bool) -> DuplexAction:
        if user_is_speaking:
            if (step + 1) % self.config.backchannel_every == 0:
                return DuplexAction.BACKCHANNEL
            return DuplexAction.LISTEN
        return DuplexAction.SPEAK

    def _build_record(
        self,
        step: int,
        user_tokens: list[int],
        action: DuplexAction,
        user_is_speaking: bool,
    ) -> DuplexChunkRecord:
        metadata = {
            "step": step,
            "mode": "mock",
            "user_is_speaking": user_is_speaking,
        }

        if action == DuplexAction.LISTEN:
            return build_listen_chunk(
                user_tokens,
                user_is_speaking=user_is_speaking,
                metadata=metadata,
            )

        if action == DuplexAction.BACKCHANNEL:
            return build_backchannel_chunk(
                user_tokens,
                speech_tokens=self._fake_agent_speech_tokens(step, base=7000),
                user_is_speaking=user_is_speaking,
                metadata=metadata,
            )

        return build_speak_chunk(
            user_tokens,
            text_tokens=self._fake_agent_text_tokens(step),
            speech_tokens=self._fake_agent_speech_tokens(step, base=9000),
            user_is_speaking=user_is_speaking,
            metadata=metadata,
        )

    @staticmethod
    def _fake_user_tokens(step: int, user_is_speaking: bool) -> list[int]:
        base = 100 if user_is_speaking else 10
        start = base + (step * 10)
        return [start + idx for idx in range(10)]

    @staticmethod
    def _fake_agent_text_tokens(step: int) -> list[int]:
        start = 1000 + (step * 2)
        return [start, start + 1]

    @staticmethod
    def _fake_agent_speech_tokens(step: int, *, base: int) -> list[int]:
        start = base + (step * 10)
        return [start + idx for idx in range(10)]


def format_event(event: MockChunkEvent) -> str:
    """Format a mock chunk event for human-readable logging."""

    speaking_state = "USER_SPEAKING" if event.user_is_speaking else "USER_PAUSE"
    chunk_text = " ".join(str(token) for token in event.chunk_tokens)
    return f"step={event.step:02d} {speaking_state} action={event.action.value:<11} | {chunk_text}"
