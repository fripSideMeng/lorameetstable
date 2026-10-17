import collections
import torch
from lora_internal_per_token import find_markdown_table_span
(R_PAD, R_PRE, R_TABLE, R_POST, R_GEN) = (-1, 0, 1, 2, 3)
REGION_NAME = {R_PAD: 'pad', R_PRE: 'pre_table', R_TABLE: 'table', R_POST: 'query', R_GEN: 'generated'}
ABLATE_CHOICES = ('none', 'table', 'pre_table', 'post_table', 'query', 'generated', 'all')
_ABLATED_REGIONS = {'none': frozenset(), 'table': frozenset({R_TABLE}), 'pre_table': frozenset({R_PRE}), 'post_table': frozenset({R_POST, R_GEN}), 'query': frozenset({R_POST}), 'generated': frozenset({R_GEN}), 'all': frozenset({R_PRE, R_TABLE, R_POST, R_GEN})}

def ablated_regions(mode):
    if mode not in _ABLATED_REGIONS:
        raise ValueError(f'unknown --lora_ablate mode {mode!r} (choices: {ABLATE_CHOICES})')
    return _ABLATED_REGIONS[mode]

def generated_token_kept(mode):
    return R_GEN not in ablated_regions(mode)

def table_char_span(content, templated):
    for cand in (content, content.strip()):
        base = templated.rfind(cand)
        if base < 0:
            continue
        span = find_markdown_table_span(cand)
        if span is None:
            return None
        return (base + span[0], base + span[1])
    return None

def region_ids_for_example(offsets, attn, span):
    n = len(offsets)
    ids = [R_PAD] * n
    if span is None:
        return ids
    (t0, t1) = span
    for i in range(n):
        if not int(attn[i]):
            continue
        (a, b) = (int(offsets[i][0]), int(offsets[i][1]))
        if b <= a:
            b = a + 1
        if b <= t0:
            ids[i] = R_PRE
        elif a >= t1:
            ids[i] = R_POST
        else:
            ids[i] = R_TABLE
    return ids

def keep_mask_from_regions(region_ids, mode, device=None, dtype=torch.float32):
    ab = ablated_regions(mode)
    rows = [[0.0 if r in ab else 1.0 for r in row] for row in region_ids]
    return torch.tensor(rows, device=device, dtype=dtype)

class LoraRegionMasker:

    def __init__(self, model, mode='none', scale=1.0):
        self.mode = mode
        self.scale = float(scale)
        self.keep = None
        self.gen_keep = 1.0 if generated_token_kept(mode) else 0.0
        self.force_off = False
        self.enabled = True
        self.handles = []
        self.n_hooked = 0
        self.n_shape_fallback = 0
        self.stats = {'examples': 0, 'no_table_span': 0, 'region_tokens': collections.Counter()}
        for (name, mod) in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and '.lora_B.' in '.' + name + '.':
                self.handles.append(mod.register_forward_hook(self._hook))
                self.n_hooked += 1

    def _hook(self, module, args, output):
        if not self.enabled:
            return output
        if self.force_off:
            return output * 0.0
        g = self.scale
        if self.mode == 'none':
            return output if g == 1.0 else output * g
        if not torch.is_tensor(output) or output.dim() != 3:
            return output
        if self.mode == 'all':
            return output * 0.0
        keep = self.keep
        if keep is not None and output.shape[0] == keep.shape[0] and (output.shape[1] == keep.shape[1]):
            m = keep.to(device=output.device, dtype=output.dtype).unsqueeze(-1)
            return output * m if g == 1.0 else output * (m * g)
        if output.shape[1] != 1:
            self.n_shape_fallback += 1
        if self.gen_keep == 1.0:
            return output if g == 1.0 else output * g
        return output * 0.0

    def note_regions(self, regions):
        for row in regions:
            self.stats['examples'] += 1
            if all((r == R_PAD for r in row)):
                self.stats['no_table_span'] += 1
            for r in row:
                self.stats['region_tokens'][REGION_NAME[r]] += 1

    def stats_dict(self):
        st = dict(self.stats)
        st['region_tokens'] = dict(self.stats['region_tokens'])
        st['n_hooked_lora_B'] = self.n_hooked
        st['scale'] = self.scale
        st['unexpected_multi_token_forwards'] = self.n_shape_fallback
        return st

    def set_prompt_mask(self, keep):
        self.keep = keep

    def clear(self):
        self.keep = None

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

@torch.no_grad()
def check_masker_identities(model, masker, enc, tol=0.0):
    kw = {k: v for (k, v) in enc.items() if k in ('input_ids', 'attention_mask')}
    kw['logits_to_keep'] = 1
    (prev_enabled, prev_force, prev_mode, prev_keep, prev_scale) = (masker.enabled, masker.force_off, masker.mode, masker.keep, masker.scale)
    try:
        masker.scale = 1.0
        masker.enabled = False
        ref_on = model(**kw).logits.float()
        (masker.enabled, masker.force_off, masker.mode, masker.keep) = (True, False, 'none', None)
        got_on = model(**kw).logits.float()
        d_on = (got_on - ref_on).abs().max().item()
        masker.force_off = True
        got_off = model(**kw).logits.float()
        masker.enabled = False
        dis = getattr(model, 'disable_adapter', None)
        if dis is None:
            d_off = float('nan')
        else:
            with dis():
                ref_off = model(**kw).logits.float()
            d_off = (got_off - ref_off).abs().max().item()
        (masker.enabled, masker.force_off) = (True, False)
        (masker.mode, masker.keep) = (prev_mode, prev_keep)
        masker.scale = 1.0
        ref_g = model(**kw).logits.float()
        masker.scale = prev_scale
        d_scale = (model(**kw).logits.float() - ref_g).abs().max().item()
    finally:
        (masker.enabled, masker.force_off, masker.mode, masker.keep, masker.scale) = (prev_enabled, prev_force, prev_mode, prev_keep, prev_scale)
    return {'passthrough_max_abs_diff': d_on, 'all_off_vs_disable_adapter_max_abs_diff': d_off, 'scale': prev_scale, 'scale_vs_unscaled_max_abs_diff': d_scale}

def _self_test():
    ok = fail = 0

    def chk(cond, msg):
        nonlocal ok, fail
        if cond:
            ok += 1
        else:
            fail += 1
            print('FAIL:', msg)
    content = 'Task:\nverify\n\nTable:\n| a | b |\n|:--|:--|\n| 1 | 2 |\n\nStatement: x\nAnswer:'
    templated = '<|hdr|>\n\n' + content + '<|eot|>'
    span = table_char_span(content, templated)
    chk(span is not None, 'table span found')
    chk(templated[span[0]:span[1]].strip().startswith('|'), 'span starts at a table line')
    chk(templated[span[0]:span[1]].strip().endswith('|'), 'span ends at a table line')
    chk('Statement' not in templated[span[0]:span[1]], 'span excludes the query')
    chk('Task:' not in templated[span[0]:span[1]], 'span excludes the instruction')
    chk(table_char_span('no table here', 'no table here') is None, 'no-table -> None')
    trimmed = '<|hdr|>\n\n' + content.strip() + '<|eot|>'
    sp2 = table_char_span('\n\n' + content + '\n\n', trimmed)
    chk(sp2 is not None and trimmed[sp2[0]:sp2[1]].strip().startswith('|') and trimmed[sp2[0]:sp2[1]].strip().endswith('|'), f'trimmed-template span {sp2}')
    chk(table_char_span(content, '<|hdr|>totally different') is None, 'content not found in template -> None')
    (t0, t1) = span
    offs = [(0, 5), (5, t0), (t0 - 1, t0), (t0, t0 + 3), (t1 - 2, t1), (t1, t1 + 4), (t1 + 4, t1 + 9), (0, 0)]
    attn = [1, 1, 1, 1, 1, 1, 1, 0]
    ids = region_ids_for_example(offs, attn, span)
    chk(ids[:3] == [R_PRE, R_PRE, R_PRE], f'pre tokens {ids[:3]}')
    chk(ids[3] == R_TABLE and ids[4] == R_TABLE, f'table tokens {ids[3:5]}')
    chk(ids[5] == R_POST and ids[6] == R_POST, f'post tokens {ids[5:7]}')
    chk(ids[7] == R_PAD, 'pad token')
    chk(region_ids_for_example(offs, attn, None) == [R_PAD] * 8, 'no span -> all pad')
    for (mode, expect) in (('none', [1] * 8), ('table', [1, 1, 1, 0, 0, 1, 1, 1]), ('pre_table', [0, 0, 0, 1, 1, 1, 1, 1]), ('post_table', [1, 1, 1, 1, 1, 0, 0, 1]), ('query', [1, 1, 1, 1, 1, 0, 0, 1]), ('generated', [1] * 8), ('all', [0, 0, 0, 0, 0, 0, 0, 1])):
        m = keep_mask_from_regions([ids], mode)[0].tolist()
        chk(m == [float(v) for v in expect], f'keep mask {mode}: {m}')
    ans = [R_POST, R_POST, R_GEN, R_GEN]
    chk(keep_mask_from_regions([ans], 'query')[0].tolist() == [0.0, 0.0, 1.0, 1.0], 'query: prompt query off, answer tokens on')
    chk(keep_mask_from_regions([ans], 'generated')[0].tolist() == [1.0, 1.0, 0.0, 0.0], 'generated: prompt query on, answer tokens off')
    chk(keep_mask_from_regions([ans], 'post_table')[0].tolist() == [0.0] * 4, 'post_table == query + generated')
    chk(generated_token_kept('none') and generated_token_kept('table') and generated_token_kept('pre_table') and generated_token_kept('query'), 'generated tokens kept for pre/table/none/query')
    chk(not generated_token_kept('post_table') and (not generated_token_kept('all')) and (not generated_token_kept('generated')), 'generated tokens dropped for post_table/generated/all')
    chk(ablated_regions('post_table') == ablated_regions('query') | ablated_regions('generated'), 'post_table is exactly the union of its two halves')

    class Toy(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.lora_B = torch.nn.ModuleDict({'default': torch.nn.Linear(4, 4, bias=False)})

        def forward(self, x):
            return self.lora_B['default'](x)
    toy = Toy()
    torch.nn.init.ones_(toy.lora_B['default'].weight)
    mk = LoraRegionMasker(toy, mode='table')
    chk(mk.n_hooked == 1, f'hooked {mk.n_hooked} lora_B modules')
    x = torch.ones(1, 8, 4)
    mk.set_prompt_mask(keep_mask_from_regions([ids], 'table'))
    y = toy(x)
    chk(y.shape == (1, 8, 4), 'hook preserves shape')
    chk(float(y[0, 3].abs().sum()) == 0.0 and float(y[0, 4].abs().sum()) == 0.0, 'table rows zeroed')
    chk(float(y[0, 0].abs().sum()) > 0.0 and float(y[0, 6].abs().sum()) > 0.0, 'non-table rows untouched')
    y1 = toy(torch.ones(1, 1, 4))
    chk(float(y1.abs().sum()) > 0.0, 'generated token kept under mode=table')
    for m_ in ('post_table', 'generated'):
        mk.mode = m_
        mk.gen_keep = 0.0 if not generated_token_kept(m_) else 1.0
        chk(float(toy(torch.ones(1, 1, 4)).abs().sum()) == 0.0, f'generated token zeroed under mode={m_}')
    mk.mode = 'query'
    mk.gen_keep = 1.0
    chk(float(toy(torch.ones(1, 1, 4)).abs().sum()) > 0.0, 'generated token kept under mode=query')
    mk.mode = 'generated'
    mk.set_prompt_mask(keep_mask_from_regions([ids], 'generated'))
    chk(float(toy(x).abs().min()) > 0.0, 'mode=generated leaves the prompt untouched')
    mk.mode = 'all'
    chk(float(toy(x).abs().sum()) == 0.0, 'mode=all zeroes everything')
    mk.mode = 'none'
    chk(float(toy(x).abs().sum()) > 0.0, 'mode=none is a pass-through')
    ref = toy(x).clone()
    mk.scale = 1.0
    chk(torch.equal(toy(x), ref), 'scale=1.0 is bit-identical to no scaling')
    mk.scale = 0.4
    chk(torch.allclose(toy(x), ref * 0.4, atol=0, rtol=1e-06), 'mode=none applies gamma to every prompt token')
    y1 = toy(torch.ones(1, 1, 4))
    mk.scale = 1.0
    chk(torch.allclose(y1, toy(torch.ones(1, 1, 4)) * 0.4, atol=0, rtol=1e-06), 'gamma applies to generated (decode) tokens too')
    (mk.mode, mk.scale) = ('table', 0.4)
    mk.set_prompt_mask(keep_mask_from_regions([ids], 'table'))
    y = toy(x)
    chk(float(y[0, 3].abs().sum()) == 0.0, 'ablated token stays exactly zero under gamma')
    chk(torch.allclose(y[0, 0], ref[0, 0] * 0.4, atol=0, rtol=1e-06), 'kept token is scaled by gamma under an ablation mode')
    (mk.mode, mk.scale) = ('none', 1.0)
    mk.set_prompt_mask(None)
    mk.remove()
    chk(float(toy(x).abs().sum()) > 0.0, 'hooks removed cleanly')
    print(f'{ok}/{ok + fail} checks passed' if fail == 0 else f'{fail} CHECK(S) FAILED')
    return fail == 0
if __name__ == '__main__':
    raise SystemExit(0 if _self_test() else 1)
