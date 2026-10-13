"""Judge-only JSON stopping; importing this module never imports torch/HF.

The state consumes one genuine generation callback per generated token.  It does
not turn an already generated string into a stop receipt.  ``finish`` removes
only padding appended *after* a recorded callback stop, retaining every raw ID.
Only explicit v7 judge calls enable this stopping contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


JUDGE_MAX_NEW_TOKENS = 160
JSON_STOP_PROTOCOL = "judge_json_object_v7"


class StopContractError(ValueError):
    """A caller violated the judge-only, greedy, per-token stop contract."""


@dataclass(frozen=True)
class JsonClose:
    eligible: bool
    closed: bool
    close_character_index: int | None = None
    invalid_nesting: bool = False


def first_json_object_close(text: str) -> JsonClose:
    """Find the first root-object close, without validating or slicing the JSON.

    Only JSON whitespace may precede the root ``{``.  A fence, prose prefix or
    other root value is not eligible.  Strings, escaped characters, nested
    objects and arrays are tracked.  Everything after the close is untouched;
    a strict downstream parser must reject junk, extra objects and bad syntax.
    """
    if not isinstance(text, str):
        raise StopContractError("decoder must return str")
    index = 0
    while index < len(text) and text[index] in " \t\r\n":
        index += 1
    if index == len(text) or text[index] != "{":
        return JsonClose(eligible=False, closed=False)
    stack: list[str] = []
    in_string = False
    escaped = False
    for offset in range(index, len(text)):
        character = text[offset]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "{[":
            stack.append(character)
        elif character in "}]":
            expected = "{" if character == "}" else "["
            if not stack or stack[-1] != expected:
                return JsonClose(True, False, invalid_nesting=True)
            stack.pop()
            if not stack:
                return JsonClose(True, True, close_character_index=offset)
    return JsonClose(True, False)


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise StopContractError(f"{name} must be int >= {minimum}")
    return value


@dataclass
class _Row:
    observed_ids: list[int] = field(default_factory=list)
    previous_text: str = ""
    decoded_prefix_rewrites: int = 0
    reason: str | None = None
    stop_token_count: int | None = None
    stop_callback_step: int | None = None
    text_at_stop: str | None = None
    close_character_index: int | None = None
    triggers: list[str] = field(default_factory=list)


class JsonObjectStopState:
    """Per-batch-row receipts for greedy, non-speculative judge generation.

    ``decode`` receives generated IDs only and must be deterministic.  Use
    tokenizer.decode(ids, skip_special_tokens=True,
    clean_up_tokenization_spaces=False).  Re-decoding the complete prefix on
    every step handles tokenizers that rewrite a decoded Unicode prefix.

    Priority when a callback sees simultaneous conditions: EOS, JSON close,
    then length. All conditions remain in ``stop_triggers``. A JSON close at
    token 160 is ``json_closed`` with ``cap_reached=True``; an EOS on that same
    callback takes precedence. The external deadline remains an exception.
    """

    def __init__(
        self,
        decode: Callable[[Sequence[int]], str],
        prompt_width: int,
        batch_size: int,
        eos_token_ids: Sequence[int],
        pad_token_id: int,
        max_new_tokens: int = JUDGE_MAX_NEW_TOKENS,
        stage: str = "judge",
    ) -> None:
        if stage != "judge":
            raise StopContractError("JSON stopping is restricted to stage='judge'")
        if type(max_new_tokens) is not int or max_new_tokens != JUDGE_MAX_NEW_TOKENS:
            raise StopContractError("judge max_new_tokens must be exactly 160")
        if not callable(decode):
            raise StopContractError("decode must be callable")
        self.stage = stage
        self.decode = decode
        self.prompt_width = _integer(prompt_width, "prompt_width")
        self.batch_size = _integer(batch_size, "batch_size", 1)
        self.pad_token_id = _integer(pad_token_id, "pad_token_id")
        self.eos_token_ids = frozenset(
            _integer(value, "eos_token_id") for value in eos_token_ids
        )
        if not self.eos_token_ids:
            raise StopContractError("nonempty actual EOS IDs are required for HF row padding")
        self.max_new_tokens = max_new_tokens
        self.callback_steps = 0
        self._rows = [_Row() for _ in range(self.batch_size)]

    def _validate_rows(self, generated_rows: Sequence[Sequence[int]]) -> list[list[int]]:
        if len(generated_rows) != self.batch_size:
            raise StopContractError("batch size changed")
        rows = [list(row) for row in generated_rows]
        for row in rows:
            for token in row:
                _integer(token, "generated token ID")
        return rows

    def observe(self, generated_rows: Sequence[Sequence[int]]) -> list[bool]:
        """Record an actual post-token callback and return shape-(batch,) mask.

        Greedy generation must call once after each single emitted token. A
        first call on a full output, skipped step, replay or prefix mutation is
        refused. A finished row may receive only the configured HF pad ID.
        """
        if self.stage != "judge" or self.max_new_tokens != JUDGE_MAX_NEW_TOKENS:
            raise StopContractError("judge stage/cap changed")
        rows = self._validate_rows(generated_rows)
        expected_length = self.callback_steps + 1
        if expected_length > self.max_new_tokens:
            raise StopContractError("callback beyond judge token cap")
        # Validate the entire batch before changing any receipt.
        for state, tokens in zip(self._rows, rows):
            if len(tokens) != expected_length:
                raise StopContractError("one callback per single generated token is required")
            if tokens[:-1] != state.observed_ids:
                raise StopContractError("generated token prefix changed")
            if state.reason is not None and tokens[-1] != self.pad_token_id:
                raise StopContractError("finished row received non-pad token")
        masks: list[bool] = []
        for state, tokens in zip(self._rows, rows):
            state.observed_ids = tokens
            if state.reason is not None:
                masks.append(True)
                continue
            text = self.decode(tokens)
            if not isinstance(text, str):
                raise StopContractError("decoder must return str")
            if not text.startswith(state.previous_text):
                state.decoded_prefix_rewrites += 1
            state.previous_text = text
            close = first_json_object_close(text)
            eos_now = tokens[-1] in self.eos_token_ids
            cap_now = len(tokens) == self.max_new_tokens
            triggers = []
            if eos_now:
                triggers.append("eos")
            if close.closed:
                triggers.append("json_closed")
            if cap_now:
                triggers.append("length")
            if triggers:
                state.reason = triggers[0]
                state.stop_token_count = len(tokens)
                state.stop_callback_step = expected_length
                state.text_at_stop = text
                state.close_character_index = close.close_character_index
                state.triggers = triggers
            masks.append(bool(triggers))
        self.callback_steps = expected_length
        return masks

    def finish(self, full_generated_rows: Sequence[Sequence[int]]) -> list[dict[str, Any]]:
        """Project receipts without discovering a new stop or trimming text.

        Returned sequences must exactly equal the last observed callback. Only
        known HF padding after the recorded stop is excluded from ``text`` and
        ``output_token_ids``; ``full_generated_token_ids`` keeps it verbatim.
        ``valid_output_tokens`` means non-post-stop-padding count, not quality.
        A stop from an unknown external criterion is refused, never called EOS.
        """
        rows = self._validate_rows(full_generated_rows)
        results = []
        for state, full_ids in zip(self._rows, rows):
            if state.reason is None or state.stop_token_count is None:
                raise StopContractError("no actual terminal callback receipt")
            if full_ids != state.observed_ids:
                raise StopContractError("returned sequence differs from final callback")
            output_ids = full_ids[: state.stop_token_count]
            tail = full_ids[state.stop_token_count :]
            if any(token != self.pad_token_id for token in tail):
                raise StopContractError("post-stop tail is not HF padding")
            text = self.decode(output_ids)
            if text != state.text_at_stop:
                raise StopContractError("decoder changed since stop callback")
            results.append({
                "text": text,
                "finish_reason": state.reason,
                "output_tokens": len(output_ids),
                "valid_output_tokens": len(output_ids),
                "output_token_ids": list(output_ids),
                "full_generated_token_ids": list(full_ids),
                "padding_tokens_after_stop": len(tail),
                "stop_callback_step": state.stop_callback_step,
                "stop_triggers": list(state.triggers),
                "json_closed_fired": "json_closed" in state.triggers,
                "json_close_character_index": state.close_character_index,
                "eos_observed": "eos" in state.triggers,
                "cap_reached": "length" in state.triggers,
                "decoded_prefix_rewrites": state.decoded_prefix_rewrites,
            })
        return results


def _build_hf_adapter(state: JsonObjectStopState, stopping_base: type, tensor_factory: Callable):
    """Dependency seam for source-class/fake-tensor tests, not an HF substitute."""
    if state.stage != "judge" or state.max_new_tokens != JUDGE_MAX_NEW_TOKENS:
        raise StopContractError("HF JSON adapter is judge-only with cap 160")

    class JudgeJsonObjectStoppingCriteria(stopping_base):
        def __call__(self, input_ids, scores, **kwargs):
            if len(input_ids.shape) != 2 or input_ids.shape[0] != state.batch_size:
                raise StopContractError("HF input_ids must have shape (batch, sequence)")
            if kwargs.get("new_token_length", 1) != 1:
                raise StopContractError("speculative/multi-token callbacks are unsupported")
            if input_ids.shape[1] <= state.prompt_width:
                raise StopContractError("HF callback has no generated token")
            generated = input_ids[:, state.prompt_width :].tolist()
            return tensor_factory(state.observe(generated), input_ids.device)

    return JudgeJsonObjectStoppingCriteria()


def make_hf_stopping_criteria(state: JsonObjectStopState):
    """Construct a real Transformers subclass; this is the only heavy-import seam.

    The caller must use do_sample=False, num_beams=1, max_new_tokens=160 and the
    same nonempty EOS/pad IDs recorded by state. Do not replace model EOS IDs or
    inject synthetic EOS. The local source check is not a GPU runtime test.
    """
    if state.stage != "judge" or state.max_new_tokens != JUDGE_MAX_NEW_TOKENS:
        raise StopContractError("HF JSON adapter is judge-only with cap 160")
    import torch
    from transformers import StoppingCriteria

    return _build_hf_adapter(
        state,
        StoppingCriteria,
        lambda values, device: torch.tensor(values, dtype=torch.bool, device=device),
    )


