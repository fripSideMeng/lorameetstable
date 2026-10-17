from __future__ import annotations
import argparse
import csv
import json
import math
import re
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator
import torch
import torch.nn as nn
from peft import PeftModel
from peft.tuners.lora.layer import LoraLayer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
PAD_SEG = -1
SPECIAL_SEG = -2
CANONICAL_TASKS = ('cta', 'cpa', 'cea', 'hvm', 'em')
SEGMENT_ORDER = ('instruction', 'table', 'focus', 'query_value', 'candidates', 'answer')

def canonical_task(task: str) -> str:
    slug = str(task).lower().strip()
    for base in CANONICAL_TASKS:
        if slug == base or slug.startswith(base + '_'):
            return base
    return slug

class SegmentVocab:

    def __init__(self) -> None:
        self._to_id: dict[str, int] = {}
        self._to_name: dict[int, str] = {}
        for name in SEGMENT_ORDER:
            self.to_id(name)

    def to_id(self, name: str) -> int:
        if name not in self._to_id:
            new_id = len(self._to_id)
            self._to_id[name] = new_id
            self._to_name[new_id] = name
        return self._to_id[name]

    def id_to_name(self) -> dict[int, str]:
        return dict(self._to_name)

@dataclass
class Example:
    ex_id: str
    task: str
    input_ids: list[int]
    attention_mask: list[int]
    segment_ids: list[int]
    labels: list[int]
    segment_names: dict[int, str]
    segment_texts: dict[int, str]
PAYLOAD_START_PATTERNS = ('(?im)^[ \\t]*Record A\\s*:', '(?im)^[ \\t]*Record 1\\s*:', '(?im)^[ \\t]*Left record\\s*:', '(?im)^[ \\t]*Entity A\\s*:', '(?im)^[ \\t]*Entity 1\\s*:', '(?im)^[ \\t]*Input record\\s*:', '(?im)^[ \\t]*Table\\s*:')
FOCUS_START_PATTERNS = ('(?im)^[ \\t]*Head column\\s*:', '(?im)^[ \\t]*Target entity column\\s*:', '(?im)^[ \\t]*Target column\\s*:', '(?im)^[ \\t]*Tail column\\s*:', '(?im)^[ \\t]*Target cell(?: located[^:]*)?\\s*:', '(?im)^[ \\t]*Target row\\s*:', '(?im)^[ \\t]*Target pair\\s*:', '(?im)^[ \\t]*Question\\s*:', '(?im)^[ \\t]*Query\\s*:', '(?im)^[ \\t]*Output\\s*:', '(?im)^[ \\t]*Answer\\s*:')
CANDIDATE_PATTERNS = ('(?im)^[ \\t]*Candidate column headers\\s*:', '(?im)^[ \\t]*Candidate headers\\s*:', '(?im)^[ \\t]*Candidate labels\\s*:', '(?im)^[ \\t]*Candidates\\s*:')
CEA_LOCATOR_PATTERN = re.compile('(?im)^[ \\t]*Target cell(?: located[^:]*)?\\s*:[^\\r\\n]*(?:\\r?\\n|$)')

def find_markdown_table_span(text: str) -> tuple[int, int] | None:
    lines = text.splitlines(keepends=True)
    line_spans: list[tuple[int, int, str]] = []
    cursor = 0
    for line in lines:
        start = cursor
        cursor += len(line)
        line_spans.append((start, cursor, line))
    first: int | None = None
    for (index, (_, _, line)) in enumerate(line_spans):
        if line.lstrip().startswith('|'):
            first = index
            break
    if first is None:
        return None
    last = first
    for index in range(first + 1, len(line_spans)):
        if line_spans[index][2].lstrip().startswith('|'):
            last = index
        else:
            break
    return (line_spans[first][0], line_spans[last][1])

def find_earliest_regex(text: str, patterns: Iterable[str], start: int=0) -> re.Match[str] | None:
    best: re.Match[str] | None = None
    for pattern in patterns:
        absolute = re.compile(pattern).search(text, start)
        if absolute is None:
            continue
        if best is None or absolute.start() < best.start():
            best = absolute
    return best

def append_nonempty(segments: list[dict[str, str]], name: str, text: str) -> None:
    if text == '':
        return
    if text.strip() == '' and segments:
        segments[-1]['text'] += text
        return
    segments.append({'name': name, 'text': text})

def split_cea_suffix(suffix: str) -> list[dict[str, str]]:
    match = CEA_LOCATOR_PATTERN.search(suffix)
    if match is None:
        return [{'name': 'focus', 'text': suffix}] if suffix else []
    focus_end = match.end()
    segments: list[dict[str, str]] = []
    append_nonempty(segments, 'focus', suffix[:focus_end])
    append_nonempty(segments, 'query_value', suffix[focus_end:])
    return segments

def split_hvm_suffix(suffix: str) -> list[dict[str, str]]:
    match = find_earliest_regex(suffix, CANDIDATE_PATTERNS)
    if match is None:
        return [{'name': 'focus', 'text': suffix}] if suffix else []
    before = suffix[:match.start()]
    candidates = suffix[match.start():]
    segments: list[dict[str, str]] = []
    if before.strip():
        append_nonempty(segments, 'focus', before)
        append_nonempty(segments, 'candidates', candidates)
    else:
        append_nonempty(segments, 'candidates', before + candidates)
    return segments

def split_task_suffix(task: str, suffix: str) -> list[dict[str, str]]:
    task = canonical_task(task)
    if not suffix:
        return []
    if task == 'cea':
        return split_cea_suffix(suffix)
    if task == 'hvm':
        return split_hvm_suffix(suffix)
    return [{'name': 'focus', 'text': suffix}]

def auto_segment_prompt(prompt: str, task: str) -> list[dict[str, str]]:
    table_span = find_markdown_table_span(prompt)
    if table_span is not None:
        (table_start, table_end) = table_span
        segments: list[dict[str, str]] = []
        append_nonempty(segments, 'instruction', prompt[:table_start])
        append_nonempty(segments, 'table', prompt[table_start:table_end])
        for segment in split_task_suffix(task, prompt[table_end:]):
            append_nonempty(segments, segment['name'], segment['text'])
        reconstructed = ''.join((segment['text'] for segment in segments))
        if reconstructed != prompt:
            raise RuntimeError(f'Segmentation changed the original {task!r} prompt.')
        return segments
    payload_match = find_earliest_regex(prompt, PAYLOAD_START_PATTERNS)
    if payload_match is None:
        return [{'name': 'instruction', 'text': prompt}]
    payload_start = payload_match.start()
    focus_match = find_earliest_regex(prompt, FOCUS_START_PATTERNS, start=payload_match.end())
    payload_end = focus_match.start() if focus_match is not None else len(prompt)
    segments = []
    append_nonempty(segments, 'instruction', prompt[:payload_start])
    append_nonempty(segments, 'table', prompt[payload_start:payload_end])
    append_nonempty(segments, 'focus', prompt[payload_end:])
    reconstructed = ''.join((segment['text'] for segment in segments))
    if reconstructed != prompt:
        raise RuntimeError(f'Fallback segmentation changed the original {task!r} prompt.')
    return segments

def get_record_segments(record: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    task = str(record.get('task', 'unknown')).lower()
    if 'segments' in record:
        raw_segments = record['segments']
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ValueError(f"Invalid segments for example {record.get('id', '?')}")
        segments: list[dict[str, str]] = []
        for segment in raw_segments:
            if 'name' not in segment or 'text' not in segment:
                raise ValueError("Every explicit segment requires 'name' and 'text'.")
            append_nonempty(segments, str(segment['name']), str(segment['text']))
    else:
        if 'input' not in record:
            raise ValueError(f"Example {record.get('id', '?')} has neither 'segments' nor 'input'.")
        segments = auto_segment_prompt(str(record['input']), task)
    if 'target' in record:
        answer = str(record['target'])
    elif 'answer' in record:
        answer = str(record['answer'])
    else:
        raise ValueError(f"Example {record.get('id', '?')} has neither 'target' nor 'answer'.")
    return (segments, answer)

def overlap_segment(start: int, end: int, spans: list[tuple[int, int, int]]) -> int:
    best_segment = SPECIAL_SEG
    best_overlap = 0
    for (left, right, segment_id) in spans:
        overlap = max(0, min(end, right) - max(start, left))
        if overlap > best_overlap:
            best_segment = segment_id
            best_overlap = overlap
    return best_segment

def _needed_length(segments: list[dict[str, str]], answer: str, tok: Any) -> int:
    full = ''.join((segment['text'] for segment in segments)) + answer
    length = len(tok(full, add_special_tokens=True)['input_ids'])
    if tok.eos_token_id is not None:
        length += 1
    return length

def fit_within_max_length(segments: list[dict[str, str]], answer: str, tok: Any, max_length: int) -> tuple[list[dict[str, str]], bool]:
    if _needed_length(segments, answer, tok) <= max_length:
        return (segments, True)
    table_index = next((i for (i, seg) in enumerate(segments) if seg['name'] == 'table'), None)
    if table_index is None:
        return (segments, False)
    segments = [dict(seg) for seg in segments]
    table_lines = segments[table_index]['text'].splitlines(keepends=True)
    n_data = len(table_lines) - 2
    if n_data < 1:
        return (segments, _needed_length(segments, answer, tok) <= max_length)

    def keep_rows(keep: int) -> bool:
        segments[table_index]['text'] = ''.join(table_lines[:2 + keep])
        return _needed_length(segments, answer, tok) <= max_length
    if not keep_rows(1):
        return (segments, False)
    (low, high, best) = (1, n_data, 1)
    while low <= high:
        mid = (low + high) // 2
        if keep_rows(mid):
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    keep_rows(best)
    return (segments, True)

def encode_record(record: dict[str, Any], tok: Any, max_length: int, vocab: SegmentVocab) -> Example | None:
    (segments, answer) = get_record_segments(record)
    (segments, fits) = fit_within_max_length(segments, answer, tok, max_length)
    if not fits:
        return None
    prompt = ''.join((segment['text'] for segment in segments))
    full_text = prompt + answer
    spans: list[tuple[int, int, int]] = []
    segment_names: dict[int, str] = {}
    segment_texts: dict[int, str] = {}
    cursor = 0
    for segment in segments:
        text = segment['text']
        segment_id = vocab.to_id(segment['name'])
        start = cursor
        end = start + len(text)
        spans.append((start, end, segment_id))
        segment_names[segment_id] = segment['name']
        segment_texts[segment_id] = text
        cursor = end
    answer_id = vocab.to_id('answer')
    answer_start = cursor
    answer_end = answer_start + len(answer)
    spans.append((answer_start, answer_end, answer_id))
    segment_names[answer_id] = 'answer'
    segment_texts[answer_id] = answer
    encoding = tok(full_text, add_special_tokens=True, truncation=True, max_length=max_length, return_offsets_mapping=True)
    input_ids = list(encoding['input_ids'])
    attention_mask = list(encoding['attention_mask'])
    offsets = list(encoding['offset_mapping'])
    segment_ids: list[int] = []
    labels: list[int] = []
    for (token_id, (token_start, token_end)) in zip(input_ids, offsets):
        if token_start == token_end:
            segment_id = SPECIAL_SEG
        else:
            segment_id = overlap_segment(token_start, token_end, spans)
        segment_ids.append(segment_id)
        labels.append(token_id if segment_id == answer_id else -100)
    appended_eos = False
    if tok.eos_token_id is not None and len(input_ids) < max_length and (input_ids[-1] != tok.eos_token_id):
        input_ids.append(tok.eos_token_id)
        attention_mask.append(1)
        segment_ids.append(answer_id)
        labels.append(tok.eos_token_id)
        appended_eos = True
    supervised_tokens = sum((label != -100 for label in labels))
    if supervised_tokens == 0:
        return None
    complete = tok(full_text, add_special_tokens=True, truncation=False, return_offsets_mapping=True)
    complete_answer_tokens = 0
    for (start, end) in complete['offset_mapping']:
        if start == end:
            continue
        overlap = max(0, min(end, answer_end) - max(start, answer_start))
        complete_answer_tokens += int(overlap > 0)
    observed_answer_tokens = sum((seg_id == answer_id for seg_id in segment_ids))
    if appended_eos:
        observed_answer_tokens -= 1
    if observed_answer_tokens < complete_answer_tokens:
        return None
    return Example(ex_id=str(record.get('id', '?')), task=str(record.get('task', 'unknown')).lower(), input_ids=input_ids, attention_mask=attention_mask, segment_ids=segment_ids, labels=labels, segment_names=segment_names, segment_texts=segment_texts)

def load_examples(path: str, tok: Any, max_length: int, vocab: SegmentVocab) -> list[Example]:
    examples: list[Example] = []
    skipped: list[str] = []
    with open(path, encoding='utf-8') as file:
        for (line_number, line) in enumerate(tqdm(file, desc='encoding examples', unit='ex'), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                example = encode_record(record, tok, max_length, vocab)
            except Exception as exc:
                raise RuntimeError(f'Failed on JSONL line {line_number}') from exc
            if example is None:
                skipped.append(str(record.get('id', f'line{line_number}')))
                continue
            examples.append(example)
    total = len(examples) + len(skipped)
    if skipped:
        preview = ', '.join(skipped[:5])
        suffix = ' ...' if len(skipped) > 5 else ''
        print(f'Skipped {len(skipped)}/{total} examples that exceed max_length={max_length} even after trimming the table to a single row (invalid NLL): {preview}{suffix}')
    if not examples:
        raise ValueError(f'No examples fit within max_length={max_length}. Increase --max_length or check the data.')
    print(f'Loaded {len(examples)}/{total} examples within max_length={max_length}')
    return examples

def write_segmentation_preview(path: Path, examples: list[Example], tokenizer: Any, limit: int) -> None:
    with path.open('w', encoding='utf-8') as file:
        for example in examples[:limit]:
            token_counts = defaultdict(int)
            for segment_id in example.segment_ids:
                if segment_id >= 0:
                    token_counts[segment_id] += 1
            output = {'id': example.ex_id, 'task': example.task, 'segments': [{'segment_id': segment_id, 'name': example.segment_names[segment_id], 'characters': len(example.segment_texts[segment_id]), 'tokens': token_counts.get(segment_id, 0), 'text': example.segment_texts[segment_id]} for segment_id in sorted(example.segment_names)], 'decoded_matches': tokenizer.decode(example.input_ids, skip_special_tokens=True)}
            file.write(json.dumps(output, ensure_ascii=False) + '\n')

def make_batch(items: list[Example], pad_id: int) -> dict[str, Any]:
    length = max((len(item.input_ids) for item in items))

    def pad(values: list[int], fill: int) -> list[int]:
        return values + [fill] * (length - len(values))
    return {'input_ids': torch.tensor([pad(item.input_ids, pad_id) for item in items], dtype=torch.long), 'attention_mask': torch.tensor([pad(item.attention_mask, 0) for item in items], dtype=torch.long), 'segment_ids': torch.tensor([pad(item.segment_ids, PAD_SEG) for item in items], dtype=torch.long), 'labels': torch.tensor([pad(item.labels, -100) for item in items], dtype=torch.long), 'items': items}

def homogeneous_batches(examples: list[Example], batch_size: int, pad_id: int) -> Iterator[dict[str, Any]]:
    by_task: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        by_task[example.task].append(example)
    for items in by_task.values():
        for start in range(0, len(items), batch_size):
            yield make_batch(items[start:start + batch_size], pad_id)

def layer_no(name: str) -> int:
    match = re.search('(?:layers|h)\\.(\\d+)', name)
    return int(match.group(1)) if match else -1

class LoraProbe:

    def __init__(self, model: nn.Module):
        self.segment_ids: torch.Tensor | None = None
        self.ablate: set[int] = set()
        self.collect = False
        self.pending: dict[int, torch.Tensor] = {}
        self.names: dict[int, str] = {}
        self.stats: dict[tuple[str, int], list[float | int]] = defaultdict(lambda : [0.0, 0.0, 0])
        self.handles: list[Any] = []
        for (name, module) in model.named_modules():
            if not isinstance(module, LoraLayer):
                continue
            if getattr(module, 'merged_adapters', []):
                raise RuntimeError(f'Adapter is merged in {name}; load an unmerged checkpoint.')
            self.names[id(module)] = name
            self.handles.append(module.register_forward_pre_hook(self._before))
            self.handles.append(module.register_forward_hook(self._after))
        if not self.handles:
            raise RuntimeError('No PEFT LoRA layers found')
        print(f'Hooked {len(self.handles) // 2} LoRA layers')

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def reset_stats(self) -> None:
        self.stats.clear()

    def _before(self, module: LoraLayer, inputs: tuple[Any, ...]) -> None:
        if not inputs or not torch.is_tensor(inputs[0]) or inputs[0].ndim < 3:
            return
        if getattr(module, 'disable_adapters', False):
            return
        x = inputs[0]
        total: torch.Tensor | None = None
        for adapter in list(getattr(module, 'active_adapters', [])):
            if adapter not in module.lora_A:
                continue
            if bool(getattr(module, 'use_dora', {}).get(adapter, False)):
                raise NotImplementedError('DoRA is not supported by this probe')
            lora_a = module.lora_A[adapter]
            lora_b = module.lora_B[adapter]
            dropped = module.lora_dropout[adapter](x.to(lora_a.weight.dtype))
            delta = (lora_b(lora_a(dropped)) * module.scaling[adapter]).to(x.dtype)
            total = delta if total is None else total + delta
        if total is not None:
            self.pending[id(module)] = total

    def _after(self, module: LoraLayer, inputs: tuple[Any, ...], output: Any) -> Any:
        delta = self.pending.pop(id(module), None)
        if delta is None or self.segment_ids is None or (not torch.is_tensor(output)):
            return output
        segment_ids = self.segment_ids.to(delta.device)
        if segment_ids.shape[:2] != delta.shape[:2]:
            raise RuntimeError(f'Mask/output mismatch in {self.names[id(module)]}: mask={tuple(segment_ids.shape)}, delta={tuple(delta.shape)}')
        if self.collect:
            base = output.detach() - delta.detach()
            delta_rms = delta.detach().float().pow(2).mean(-1).sqrt()
            base_rms = base.float().pow(2).mean(-1).sqrt()
            relative = delta_rms / (base_rms + 1e-08)
            name = self.names[id(module)]
            for segment_id in torch.unique(segment_ids).tolist():
                if segment_id < 0:
                    continue
                selected = segment_ids.eq(segment_id)
                self.stats[name, int(segment_id)][0] += float(delta_rms[selected].sum())
                self.stats[name, int(segment_id)][1] += float(relative[selected].sum())
                self.stats[name, int(segment_id)][2] += int(selected.sum())
        if self.ablate:
            remove_mask = torch.zeros_like(segment_ids, dtype=torch.bool)
            for segment_id in self.ablate:
                remove_mask |= segment_ids.eq(segment_id)
            output = output - delta * remove_mask.unsqueeze(-1).to(delta.dtype)
        return output

@contextmanager
def probe_mode(probe: LoraProbe, segment_ids: torch.Tensor, ablate: Iterable[int]=(), collect: bool=False):
    probe.segment_ids = segment_ids
    probe.ablate = set(ablate)
    probe.collect = collect
    try:
        yield
    finally:
        probe.segment_ids = None
        probe.ablate = set()
        probe.collect = False
        probe.pending.clear()

def target_nll(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, int]:
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    total = nn.functional.cross_entropy(shifted_logits.reshape(-1, shifted_logits.size(-1)), shifted_labels.reshape(-1), ignore_index=-100, reduction='sum')
    count = int(shifted_labels.ne(-100).sum())
    return (float(total), count)

def contribution_analysis(model: nn.Module, batches: list[dict[str, Any]], probe: LoraProbe, input_device: torch.device, id_to_name: dict[int, str]) -> list[dict[str, Any]]:
    aggregate: dict[tuple[str, str, str], list[float | int]] = defaultdict(lambda : [0.0, 0.0, 0])
    with torch.inference_mode():
        for batch in tqdm(batches, desc='contribution', unit='batch'):
            task = batch['items'][0].task
            input_ids = batch['input_ids'].to(input_device)
            attention_mask = batch['attention_mask'].to(input_device)
            segment_ids = batch['segment_ids'].to(input_device)
            probe.reset_stats()
            with probe_mode(probe, segment_ids, collect=True):
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            for ((module, segment_id), values) in probe.stats.items():
                segment_name = id_to_name.get(segment_id, f'segment_{segment_id}')
                key = (task, module, segment_name)
                for index in range(3):
                    aggregate[key][index] += values[index]
    rows: list[dict[str, Any]] = []
    for ((task, module, segment_name), values) in sorted(aggregate.items()):
        (delta_sum, ratio_sum, count) = values
        count_int = int(count)
        rows.append({'task': task, 'module': module, 'layer': layer_no(module), 'module_type': module.rsplit('.', 1)[-1], 'segment': segment_name, 'mean_delta_rms': float(delta_sum) / max(count_int, 1), 'mean_delta_to_base_rms': float(ratio_sum) / max(count_int, 1), 'token_count': count_int})
    return rows

def ablation_analysis(model: nn.Module, batches: list[dict[str, Any]], probe: LoraProbe, input_device: torch.device, requested: list[str], id_to_name: dict[int, str]) -> list[dict[str, Any]]:
    name_to_id = {name: segment_id for (segment_id, name) in id_to_name.items()}
    totals: dict[tuple[str, str], list[float | int]] = defaultdict(lambda : [0.0, 0, 0])
    with torch.inference_mode():
        for batch in tqdm(batches, desc='ablation', unit='batch'):
            task = batch['items'][0].task
            input_ids = batch['input_ids'].to(input_device)
            attention_mask = batch['attention_mask'].to(input_device)
            segment_ids = batch['segment_ids'].to(input_device)
            labels = batch['labels'].to(input_device)
            present_ids = {int(value) for value in batch['segment_ids'].unique().tolist() if value >= 0}
            present_names = {id_to_name.get(i) for i in present_ids}
            available = [name for name in requested if name in name_to_id and name in present_names]
            for name in ['none', *available]:
                ablate = [] if name == 'none' else [name_to_id[name]]
                with probe_mode(probe, segment_ids, ablate=ablate):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
                (nll, token_count) = target_nll(logits, labels)
                totals[task, name][0] += nll
                totals[task, name][1] += token_count
                totals[task, name][2] += len(batch['items'])
    baseline: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for ((task, name), values) in totals.items():
        (nll, token_count, example_count) = values
        mean_nll = float(nll) / max(int(token_count), 1)
        if name == 'none':
            baseline[task] = mean_nll
        rows.append({'task': task, 'ablation': name, 'mean_target_token_nll': mean_nll, 'perplexity': math.exp(min(mean_nll, 20)), 'target_tokens': int(token_count), 'examples': int(example_count)})
    for row in rows:
        row['nll_change_vs_none'] = row['mean_target_token_nll'] - baseline[row['task']]
    return sorted(rows, key=lambda row: (row['task'], row['ablation']))

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def find_input_device(model: nn.Module) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        for parameter in model.parameters():
            if parameter.device.type != 'meta':
                return parameter.device
    raise RuntimeError('Could not determine model input device')

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--preview_examples', type=int, default=12)
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='bfloat16')
    parser.add_argument('--ablations', nargs='*', default=['instruction', 'table', 'focus', 'query_value', 'candidates', 'answer'])
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError('A fast tokenizer is required for offset mapping')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=getattr(torch, args.dtype), device_map='auto')
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    model.eval()
    input_device = find_input_device(model)
    vocab = SegmentVocab()
    examples = load_examples(args.data, tokenizer, args.max_length, vocab)
    id_to_name = vocab.id_to_name()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_segmentation_preview(output_dir / 'segmentation_preview.jsonl', examples, tokenizer, args.preview_examples)
    batches = list(homogeneous_batches(examples, args.batch_size, tokenizer.pad_token_id))
    probe = LoraProbe(model)
    try:
        contribution_rows = contribution_analysis(model, batches, probe, input_device, id_to_name)
        write_csv(output_dir / 'lora_token_contributions.csv', contribution_rows)
        ablation_rows = ablation_analysis(model, batches, probe, input_device, args.ablations, id_to_name)
        write_csv(output_dir / 'segment_ablation_nll.csv', ablation_rows)
    finally:
        probe.close()
    print(f'Wrote results to {output_dir}')
if __name__ == '__main__':
    main()
