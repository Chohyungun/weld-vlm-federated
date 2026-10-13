"""JSON stop state and adapter checks; no model package is imported."""

from __future__ import annotations

import json
import ast
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from corpus.production.judge_stopping import (
    JsonObjectStopState, StopContractError, _build_hf_adapter,
    first_json_object_close, make_hf_stopping_criteria,
)


PAD = 0
EOS = 99


def state_for(mapping=None, batch_size=1, prompt_width=0, **kwargs):
    mapping = mapping or {1: "{", 2: "}", 3: '"a":1', 4: "x", 5: "{}junk"}
    def decode(ids):
        return "".join(mapping.get(token, "") for token in ids)
    return JsonObjectStopState(decode, prompt_width, batch_size, [EOS], PAD, **kwargs)


def emit(state, token_rows):
    result = None
    for width in range(1, len(token_rows[0]) + 1):
        result = state.observe([row[:width] for row in token_rows])
    return result


class ScannerTests(unittest.TestCase):
    def test_empty_whitespace(self):
        self.assertFalse(first_json_object_close(" \t\n").eligible)

    def test_empty_object(self):
        self.assertEqual(first_json_object_close("{} ").close_character_index, 1)

    def test_nested_object_array(self):
        text = '{"a":[{"b":{}},[1,2]],"c":3}'
        self.assertEqual(first_json_object_close(text).close_character_index, len(text)-1)

    def test_quoted_braces(self):
        self.assertTrue(first_json_object_close('{"a":"} [ { ]"}').closed)

    def test_escaped_quote(self):
        self.assertTrue(first_json_object_close(json.dumps({"x": 'a"}b'})).closed)

    def test_escaped_backslash(self):
        self.assertTrue(first_json_object_close(json.dumps({"x": "\\"})).closed)

    def test_incomplete_escaped_string(self):
        self.assertFalse(first_json_object_close('{"x":"abc\\').closed)

    def test_mismatched_nesting(self):
        result = first_json_object_close('{"a":[1}}')
        self.assertFalse(result.closed)
        self.assertTrue(result.invalid_nesting)

    def test_non_object_roots(self):
        for text in ('[]', '"{}"', 'true', 'null', '1', 'note {}', '```json\n{}\n```'):
            with self.subTest(text=text):
                self.assertFalse(first_json_object_close(text).eligible)

    def test_unicode_prefix_not_json_whitespace(self):
        self.assertFalse(first_json_object_close('\ufeff{}').eligible)

    def test_trailing_junk_is_not_hidden(self):
        self.assertEqual(first_json_object_close('{}junk').close_character_index, 1)

    def test_scanner_is_not_json_validator(self):
        self.assertTrue(first_json_object_close('{bad}').closed)
        with self.assertRaises(json.JSONDecodeError):
            json.loads('{bad}')


class StateTests(unittest.TestCase):
    def test_actual_close_receipt(self):
        state = state_for()
        self.assertEqual(emit(state, [[1,3,2]]), [True])
        receipt = state.finish([[1,3,2]])[0]
        self.assertEqual(receipt['finish_reason'], 'json_closed')
        self.assertEqual(receipt['text'], '{"a":1}')
        self.assertTrue(receipt['json_closed_fired'])
        self.assertEqual(receipt['output_tokens'], 3)

    def test_same_token_junk_preserved(self):
        state = state_for()
        state.observe([[5]])
        receipt = state.finish([[5]])[0]
        self.assertEqual(receipt['text'], '{}junk')
        with self.assertRaises(json.JSONDecodeError):
            json.loads(receipt['text'])

    def test_same_token_extra_object_preserved(self):
        state = state_for({1: '{}{}'})
        state.observe([[1]])
        with self.assertRaises(json.JSONDecodeError):
            json.loads(state.finish([[1]])[0]['text'])

    def test_duplicate_keys_remain_for_strict_parser(self):
        state = state_for({1: '{"x":1,"x":2}'})
        state.observe([[1]])
        def reject_duplicates(pairs):
            if len(dict(pairs)) != len(pairs):
                raise ValueError('duplicate')
            return dict(pairs)
        with self.assertRaises(ValueError):
            json.loads(state.finish([[1]])[0]['text'], object_pairs_hook=reject_duplicates)

    def test_unicode_decoder_prefix_rewrite(self):
        outputs = {(): '', (1,): '{"x":"\ufffd', (1,2): '{"x":"가', (1,2,3): '{"x":"가"}'}
        state = JsonObjectStopState(lambda ids: outputs[tuple(ids)], 0, 1, [EOS], PAD)
        self.assertEqual(emit(state, [[1,2,3]]), [True])
        receipt = state.finish([[1,2,3]])[0]
        self.assertEqual(receipt['text'], '{"x":"가"}')
        self.assertEqual(receipt['decoded_prefix_rewrites'], 1)

    def test_brace_inside_split_string_does_not_stop(self):
        state = state_for({1: '{"x":"', 2: '}', 3: '"}'})
        self.assertEqual(state.observe([[1]]), [False])
        self.assertEqual(state.observe([[1,2]]), [False])
        self.assertEqual(state.observe([[1,2,3]]), [True])

    def test_eos_before_close_truthful(self):
        state = state_for()
        emit(state, [[1,EOS]])
        receipt = state.finish([[1,EOS]])[0]
        self.assertEqual(receipt['finish_reason'], 'eos')
        self.assertEqual(receipt['text'], '{')
        self.assertFalse(receipt['json_closed_fired'])
        self.assertEqual(receipt['output_token_ids'], [1,EOS])

    def test_eos_without_json(self):
        state = state_for()
        state.observe([[EOS]])
        self.assertEqual(state.finish([[EOS]])[0]['finish_reason'], 'eos')

    def test_length_without_synthetic_eos(self):
        state = state_for()
        tokens = [4] * 160
        emit(state, [tokens])
        receipt = state.finish([tokens])[0]
        self.assertEqual(receipt['finish_reason'], 'length')
        self.assertFalse(receipt['eos_observed'])
        self.assertNotIn(EOS, receipt['full_generated_token_ids'])

    def test_close_at_cap_preserves_both_triggers(self):
        state = state_for({1: ' ', 2: '{}'})
        tokens = [1] * 159 + [2]
        emit(state, [tokens])
        receipt = state.finish([tokens])[0]
        self.assertEqual(receipt['finish_reason'], 'json_closed')
        self.assertEqual(receipt['stop_triggers'], ['json_closed', 'length'])

    def test_eos_at_cap_has_precedence(self):
        state = state_for()
        tokens = [4] * 159 + [EOS]
        emit(state, [tokens])
        self.assertEqual(state.finish([tokens])[0]['stop_triggers'], ['eos', 'length'])

    def test_same_callback_eos_close_records_both(self):
        # Deliberately fake EOS decoder tests tie semantics, not tokenizer behavior.
        state = state_for({EOS: '{}'})
        state.observe([[EOS]])
        receipt = state.finish([[EOS]])[0]
        self.assertEqual(receipt['finish_reason'], 'eos')
        self.assertEqual(receipt['stop_triggers'], ['eos', 'json_closed'])
        self.assertTrue(receipt['json_closed_fired'])

    def test_mixed_rows_preserve_full_padding(self):
        state = state_for({1: '{}', 2: '{', 3: '}'}, batch_size=3)
        self.assertEqual(state.observe([[1], [2], [2]]), [True, False, False])
        self.assertEqual(state.observe([[1,PAD], [2,EOS], [2,3]]), [True, True, True])
        receipts = state.finish([[1,PAD], [2,EOS], [2,3]])
        self.assertEqual([row['finish_reason'] for row in receipts], ['json_closed','eos','json_closed'])
        self.assertEqual(receipts[0]['full_generated_token_ids'], [1,PAD])
        self.assertEqual(receipts[0]['output_token_ids'], [1])
        self.assertEqual(receipts[0]['valid_output_tokens'], 1)
        self.assertEqual(receipts[0]['padding_tokens_after_stop'], 1)

    def test_pad_equal_eos_not_misreported(self):
        state = JsonObjectStopState(lambda ids: ''.join({1:'{}',2:'{',3:'}'}.get(x,'') for x in ids), 0, 2, [EOS], EOS)
        state.observe([[1],[2]])
        state.observe([[1,EOS],[2,3]])
        receipt = state.finish([[1,EOS],[2,3]])[0]
        self.assertEqual(receipt['finish_reason'], 'json_closed')
        self.assertFalse(receipt['eos_observed'])

    def test_nonpad_after_stop_refused(self):
        state = state_for({1:'{}'}, batch_size=2)
        state.observe([[1],[4]])
        with self.assertRaisesRegex(StopContractError, 'non-pad'):
            state.observe([[1,4],[4,4]])

    def test_first_callback_cannot_be_posthoc_full_output(self):
        with self.assertRaisesRegex(StopContractError, 'one callback'):
            state_for().observe([[1,2]])

    def test_replayed_callback_refused(self):
        state = state_for()
        state.observe([[1]])
        with self.assertRaises(StopContractError):
            state.observe([[1]])

    def test_token_prefix_mutation_refused(self):
        state = state_for()
        state.observe([[1]])
        with self.assertRaisesRegex(StopContractError, 'prefix changed'):
            state.observe([[4,2]])

    def test_batch_shape_change_refused(self):
        with self.assertRaisesRegex(StopContractError, 'batch size'):
            state_for().observe([[1],[1]])

    def test_finish_without_terminal_callback_refused(self):
        state = state_for()
        state.observe([[1]])
        with self.assertRaisesRegex(StopContractError, 'no actual'):
            state.finish([[1]])

    def test_finish_never_discovers_close(self):
        with self.assertRaisesRegex(StopContractError, 'no actual'):
            state_for().finish([[1,2]])

    def test_returned_sequence_mismatch_refused(self):
        state = state_for({1:'{}'})
        state.observe([[1]])
        with self.assertRaisesRegex(StopContractError, 'differs'):
            state.finish([[1,PAD]])

    def test_decoder_drift_refused(self):
        state = state_for({1:'{}'})
        state.observe([[1]])
        state.decode = lambda ids: '{}changed'
        with self.assertRaisesRegex(StopContractError, 'decoder changed'):
            state.finish([[1]])

    def test_fence_does_not_create_json_stop(self):
        state = state_for({1:'```json\n{}\n```'})
        self.assertEqual(state.observe([[1]]), [False])
        state.observe([[1,EOS]])
        self.assertEqual(state.finish([[1,EOS]])[0]['finish_reason'], 'eos')

    def test_judge_stage_required(self):
        with self.assertRaisesRegex(StopContractError, 'judge'):
            state_for(stage='generation')

    def test_exact_cap_required(self):
        for cap in (159, 384, True):
            with self.subTest(cap=cap), self.assertRaises(StopContractError):
                state_for(max_new_tokens=cap)

    def test_nonempty_eos_required(self):
        with self.assertRaisesRegex(StopContractError, 'nonempty'):
            JsonObjectStopState(str, 0, 1, [], PAD)

    def test_bool_id_refused(self):
        with self.assertRaises(StopContractError):
            state_for().observe([[True]])


class FakeInput:
    def __init__(self, rows, device='fake-cpu'):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))
        self.device = device

    def __getitem__(self, key):
        first, second = key
        return FakeInput([row[second] for row in self.rows[first]], self.device)

    def tolist(self):
        return self.rows


class FakeStoppingBase:
    """Minimal superclass seam; this test does not claim a real HF execution."""


def stopping_base():
    return FakeStoppingBase


class AdapterTests(unittest.TestCase):
    def test_adapter_base_and_batch_mask(self):
        base = stopping_base()
        state = state_for({1:'{}',2:'{'}, batch_size=2, prompt_width=3)
        adapter = _build_hf_adapter(state, base, lambda values, device: {'mask':values,'device':device,'shape':(len(values),),'dtype':'bool'})
        self.assertIsInstance(adapter, base)
        result = adapter(FakeInput([[77,78,79,1],[80,81,82,2]]), None)
        self.assertEqual(result, {'mask':[True,False], 'device':'fake-cpu','shape':(2,), 'dtype':'bool'})

    def test_adapter_uses_padded_prompt_width(self):
        state = state_for({1:'{}'}, prompt_width=4)
        adapter = _build_hf_adapter(state, stopping_base(), lambda values, device: values)
        self.assertEqual(adapter(FakeInput([[0,0,11,12,1]]), None), [True])
        self.assertEqual(state.finish([[1]])[0]['full_generated_token_ids'], [1])

    def test_adapter_refuses_speculative_multitoken(self):
        adapter = _build_hf_adapter(state_for(), stopping_base(), lambda values, device: values)
        with self.assertRaisesRegex(StopContractError, 'multi-token'):
            adapter(FakeInput([[1]]), None, new_token_length=2)

    def test_factory_stage_guard_before_heavy_import(self):
        state = state_for()
        state.stage = 'generation'
        with self.assertRaises(StopContractError):
            make_hf_stopping_criteria(state)

    def test_module_has_no_eager_model_imports(self):
        from corpus.production import judge_stopping

        source = ast.parse(Path(judge_stopping.__file__).read_text(encoding='utf-8'))
        for node in source.body:
            if isinstance(node, ast.Import):
                self.assertFalse(any(name.name in {'torch', 'transformers'} for name in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertNotIn(node.module, {'torch', 'transformers'})


class _Values:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values

    def sum(self, dim):
        return _Values([sum(row) for row in self.values])


class _Encoded(dict):
    def to(self, device):
        return self


class _Tokenizer:
    pad_token_id = PAD

    def __init__(self):
        self.mapping = {1: '{}junk', 2: '{', 3: '}', 4: 'x', 5: '{}'}
        self.decode_options = []

    def apply_chat_template(self, messages, **kwargs):
        return messages[-1]['content']

    def __call__(self, texts, **kwargs):
        return _Encoded(input_ids=FakeInput([[77, 78] for _ in texts]),
                        attention_mask=_Values([[1, 1] for _ in texts]))

    def decode(self, tokens, **kwargs):
        self.decode_options.append(kwargs)
        return ''.join(self.mapping.get(token, '') for token in tokens)


class _Model:
    """Greedy per-token test driver, not a real model or GPU runtime."""
    device = 'fake-cpu'

    def __init__(self, streams, *, invoke_callbacks=True):
        self.streams = streams
        self.invoke_callbacks = invoke_callbacks
        self.generation_config = SimpleNamespace(eos_token_id=EOS)
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        rows = [list(row) for row in kwargs['input_ids'].tolist()]
        done = [False] * len(rows)
        for step in range(kwargs['max_new_tokens']):
            for i, row in enumerate(rows):
                token = PAD if done[i] else self.streams[i][step]
                row.append(token)
            if self.invoke_callbacks:
                for criterion in kwargs['stopping_criteria']:
                    result = criterion(FakeInput(rows), None)
                    mask = result if isinstance(result, list) else [result] * len(rows)
                    done = [old or new for old, new in zip(done, mask)]
            done = [old or row[-1] == EOS for old, row in zip(done, rows)]
            if all(done) or step + 1 == max(map(len, self.streams)):
                break
        return FakeInput(rows)


def _backend(streams, *, enabled=True, stage='judge', cap=160, invoke_callbacks=True):
    from corpus.production import infer as I

    model = I.TransformersBackend.__new__(I.TransformersBackend)
    model.spec = {'max_input_tokens': 8192, 'max_new_tokens': cap}
    model.stage = stage
    model.json_object_stopping = False
    model.identity = {'backend': 'transformers'}
    model.model = _Model(streams, invoke_callbacks=invoke_callbacks)
    model.tokenizer = _Tokenizer()
    model.torch = SimpleNamespace(
        cuda=SimpleNamespace(reset_peak_memory_stats=lambda _: None,
                             max_memory_allocated=lambda _: 0,
                             max_memory_reserved=lambda _: 0),
        inference_mode=nullcontext, bool='bool',
        tensor=lambda values, dtype, device: list(values))
    if enabled:
        model.enable_json_object_stopping()
    return model


class BackendTests(unittest.TestCase):
    def _run(self, model):
        fake_transformers = SimpleNamespace(StoppingCriteria=FakeStoppingBase,
                                            StoppingCriteriaList=list)
        with patch.dict(sys.modules, torch=model.torch, transformers=fake_transformers):
            return model.generate_batch(['candidate'] * len(model.model.streams),
                                        'v7 system supplied explicitly', float('inf'))

    def test_explicit_enable_and_actual_callback_receipts(self):
        model = _backend([[5], [2, 3]])
        rows = self._run(model)
        self.assertEqual([r['finish_reason'] for r in rows], ['json_closed', 'json_closed'])
        self.assertEqual(rows[0]['full_generated_token_ids'], [5, PAD])
        self.assertEqual(rows[0]['output_token_ids'], [5])
        self.assertEqual(rows[0]['padding_tokens_after_stop'], 1)
        self.assertEqual(rows[1]['stop_callback_step'], 2)
        self.assertEqual(model.identity['json_stop_protocol'], 'judge_json_object_v7')
        self.assertEqual(model.model.kwargs['eos_token_id'], EOS)
        self.assertEqual(model.model.kwargs['max_new_tokens'], 160)
        self.assertFalse(model.model.kwargs['do_sample'])
        self.assertEqual(model.model.kwargs['num_beams'], 1)
        self.assertTrue(all(x.get('clean_up_tokenization_spaces') is False
                            for x in model.tokenizer.decode_options))

    def test_same_token_junk_is_preserved_by_backend(self):
        rows = self._run(_backend([[1]]))
        self.assertEqual(rows[0]['text'], '{}junk')
        self.assertEqual(rows[0]['finish_reason'], 'json_closed')
        with self.assertRaises(json.JSONDecodeError):
            json.loads(rows[0]['text'])

    def test_backend_never_infers_posthoc_close(self):
        with self.assertRaisesRegex(StopContractError, 'no actual terminal'):
            self._run(_backend([[5]], invoke_callbacks=False))

    def test_legacy_backend_does_not_enable_json_stopper(self):
        model = _backend([[2, EOS]], enabled=False, stage='generate', cap=384)
        rows = self._run(model)
        self.assertEqual(rows, [{'text': '{', 'finish_reason': 'eos',
                                 'input_tokens': 2, 'output_tokens': 2}])
        self.assertEqual(len(model.model.kwargs['stopping_criteria']), 1)
        self.assertNotIn('json_stop_protocol', model.identity)
        self.assertNotIn('eos_token_id', model.model.kwargs)

    def test_enable_refuses_wrong_stage_or_cap(self):
        from corpus.production import infer as I
        for stage, cap in [('generate', 160), ('judge', 384), ('judge', True)]:
            with self.subTest(stage=stage, cap=cap), self.assertRaises(I.StageRefused):
                _backend([[5]], stage=stage, cap=cap)

    def test_enabled_file_fingerprint_change_is_refused(self):
        from corpus.production import infer as I
        model = _backend([[5]])
        model.identity['json_stop_implementation_sha256'] = '0' * 64
        with self.assertRaisesRegex(I.StageRefused, 'contract or implementation changed'):
            self._run(model)

    def test_json_closed_needs_explicit_row_permission(self):
        from corpus.production import infer as I
        row = {'id': 'case', 'text': '{}', 'finish_reason': 'json_closed',
               'input_tokens': 2, 'output_tokens': 1}
        spec = {'max_input_tokens': 8192, 'max_new_tokens': 160}
        with self.assertRaises(I.StageRefused):
            I._check_rows([row], ['case'], spec)
        I._check_rows([row], ['case'], spec, allow_json_closed=True)
        with self.assertRaises(I.StageRefused):
            I._check_rows([row], ['case'], spec, allow_json_closed=1)


if __name__ == '__main__':
    unittest.main(verbosity=2)


