import collections
import json
import math
import os
import torch
FAMILIES = ('q', 'k', 'v', 'o', 'gate', 'up', 'down')

def family_of(name):
    tail = name.rsplit('.', 1)[-1]
    for f in FAMILIES:
        if tail == f'{f}_proj':
            return f
    return None

def _cos(a, b):
    (na, nb) = (a.norm(), b.norm())
    if na == 0 or nb == 0:
        return float('nan')
    return float(a.flatten() @ b.flatten() / (na * nb))

class TargetGradDiagnostic:

    def __init__(self, model, target_role_id, out, verify_steps=(1, 2, 3), verify_n_layers=4, flush_every=50, meta=None):
        self.target_role_id = int(target_role_id)
        self.out = out
        self.flush_every = int(flush_every)
        self._meta = dict(meta or {})
        self.verify_steps = set((int(s) for s in verify_steps))
        self.verify_n_layers = int(verify_n_layers)
        self.mask = None
        self.handles = []
        self.layers = {}
        self.clip = {'pre_norm': None, 'max_grad_norm': None}
        self.steps = []
        self.per_step = collections.defaultdict(lambda : collections.defaultdict(list))
        self.n_target_tokens = 0
        self.n_tokens = 0
        self._attach(model)

    def _attach(self, model):
        for (name, mod) in model.named_modules():
            A = getattr(mod, 'lora_A', None)
            B = getattr(mod, 'lora_B', None)
            if A is None or B is None or (not hasattr(A, 'keys')):
                continue
            keys = list(A.keys())
            if len(keys) != 1:
                raise RuntimeError(f'{name}: expected exactly one LoRA adapter, got {keys}')
            k = keys[0]
            fam = family_of(name)
            if fam is None:
                raise RuntimeError(f'{name}: adapted module is not one of {FAMILIES}')
            (la, lb) = (A[k], B[k])
            ent = {'name': name, 'family': fam, 'A': la, 'B': lb, 'scaling': float(mod.scaling[k]), 'gA': torch.zeros_like(la.weight, dtype=torch.float32), 'gB': torch.zeros_like(lb.weight, dtype=torch.float32), 'cA': torch.zeros_like(la.weight, dtype=torch.float32), 'cB': torch.zeros_like(lb.weight, dtype=torch.float32), 'fA': torch.zeros_like(la.weight, dtype=torch.float32), 'fB': torch.zeros_like(lb.weight, dtype=torch.float32), 'vA': None, 'vB': None, 'X': None, 'Z': None}
            self.layers[name] = ent
            self.handles.append(la.register_forward_hook(self._fwd_A(ent)))
            self.handles.append(lb.register_forward_hook(self._fwd_B(ent)))
        if not self.layers:
            raise RuntimeError('no PEFT LoRA modules found to instrument')
        self.verify_names = sorted(self.layers)[::max(1, len(self.layers) // self.verify_n_layers)]
        self.verify_names = set(self.verify_names[:self.verify_n_layers])

    def install_clip_probe(self, accelerator, max_grad_norm):
        self.clip['max_grad_norm'] = float(max_grad_norm) if max_grad_norm else None
        orig = accelerator.clip_grad_norm_
        probe = self.clip

        def wrapped(*a, **kw):
            n = orig(*a, **kw)
            try:
                probe['pre_norm'] = float(n)
            except (TypeError, ValueError):
                probe['pre_norm'] = None
            return n
        accelerator.clip_grad_norm_ = wrapped
        return wrapped

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def set_batch_mask(self, role_ids):
        self.mask = role_ids.reshape(-1) == self.target_role_id
        self.n_target_tokens += int(self.mask.sum())
        self.n_tokens += int(self.mask.numel())

    def clear_batch_mask(self):
        self.mask = None
        for ent in self.layers.values():
            ent['X'] = ent['Z'] = None

    def _fwd_A(self, ent):

        def hook(_m, inp, out):
            if self.mask is None:
                return
            X = inp[0].reshape(-1, inp[0].shape[-1])
            Z = out.reshape(-1, out.shape[-1])
            m = self.mask.to(X.device)
            ent['X'] = X[m].detach().float()
            ent['Z'] = Z[m].detach().float()
            if self._verifying and ent['name'] in self.verify_names:
                ent['Xall'] = X.detach().float()
                ent['Zall'] = Z.detach().float()
        return hook

    def _fwd_B(self, ent):

        def hook(_m, _inp, out):
            if self.mask is None:
                return
            m = self.mask.to(out.device)
            verify = self._verifying and ent['name'] in self.verify_names

            def on_grad(g):
                D = g.reshape(-1, g.shape[-1])
                Bw = ent['B'].weight.detach().float()
                Dt = D[m].detach().float()
                (Xt, Zt) = (ent['X'], ent['Z'])
                if Xt is not None and Xt.shape[0] > 0:
                    ent['gB'] += Dt.T @ Zt
                    ent['gA'] += (Dt @ Bw).T @ Xt
                if verify and ent.get('Xall') is not None:
                    Da = D.detach().float()
                    vB = Da.T @ ent['Zall']
                    vA = (Da @ Bw).T @ ent['Xall']
                    ent['vB'] = vB if ent['vB'] is None else ent['vB'] + vB
                    ent['vA'] = vA if ent['vA'] is None else ent['vA'] + vA
                    ent['Xall'] = ent['Zall'] = None
                return None
            out.register_hook(on_grad)
        return hook
    _verifying = False

    def begin_microbatch(self, step_idx):
        self._verifying = int(step_idx) in self.verify_steps

    def _clip_coef(self):
        (mg, pre) = (self.clip['max_grad_norm'], self.clip['pre_norm'])
        if not mg or pre is None or pre <= mg:
            return 1.0
        return float(mg / pre)

    def on_optimizer_step(self, step):
        c = self._clip_coef()
        rec = {'step': int(step), 'clip_coef': c, 'pre_clip_grad_norm': self.clip['pre_norm']}
        verify = []
        for name in sorted(self.layers):
            ent = self.layers[name]
            row = {}
            for (tag, W, gt) in (('A', ent['A'].weight, ent['gA']), ('B', ent['B'].weight, ent['gB'])):
                full = W.grad
                if full is None:
                    continue
                full = full.detach().float()
                tgt = gt * c
                oth = full - tgt
                row[f'cos_{tag}'] = _cos(tgt, oth)
                row[f'nt_{tag}'] = float(tgt.norm())
                row[f'no_{tag}'] = float(oth.norm())
                row[f'nf_{tag}'] = float(full.norm())
                nf = row[f'nf_{tag}']
                row[f'ratio_{tag}'] = row[f'nt_{tag}'] / nf if nf > 0 else float('nan')
                row[f'energy_{tag}'] = row[f'nt_{tag}'] ** 2 / nf ** 2 if nf > 0 else float('nan')
                ent['c' + tag] += gt
                ent['f' + tag] += full / c if c > 0 else full
            for (kk, vv) in row.items():
                self.per_step[name][kk].append(vv)
            if int(step) in self.verify_steps and name in self.verify_names:
                for tag in ('A', 'B'):
                    v = ent['v' + tag]
                    g = ent[tag].weight.grad
                    if v is None or g is None:
                        continue
                    g = g.detach().float() / c
                    den = float(g.norm())
                    err = float((v - g).norm()) / den if den > 0 else float('nan')
                    verify.append({'step': int(step), 'layer': name, 'which': tag, 'rel_err': err, 'grad_norm': den})
                    ent['v' + tag] = None
        rec['verify'] = verify
        self.steps.append(rec)
        if self.flush_every and len(self.steps) % self.flush_every == 0:
            try:
                self.finish(self._meta, partial=True)
            except OSError:
                pass
        for ent in self.layers.values():
            ent['gA'].zero_()
            ent['gB'].zero_()
        self.clip['pre_norm'] = None

    def _summary(self, keys):
        out = {}
        for tag in ('A', 'B'):
            vals = [v for name in keys for v in self.per_step[name].get(f'cos_{tag}', []) if not math.isnan(v)]
            rat = [v for name in keys for v in self.per_step[name].get(f'ratio_{tag}', []) if not math.isnan(v)]
            if not vals:
                continue
            t = torch.tensor(vals)
            q = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
            out[tag] = {'n': len(vals), 'cos_mean': float(t.mean()), 'cos_median': float(t.median()), 'cos_std': float(t.std(unbiased=False)), 'cos_quantiles': {f'q{round(p * 100):02d}': float(v) for (p, v) in zip(q.tolist(), torch.quantile(t, q).tolist())}, 'frac_cos_negative': float((t < 0).float().mean()), 'target_norm_ratio_mean': float(torch.tensor(rat).mean()) if rat else None}
            (cts, cos_cum) = ([], [])
            for name in keys:
                ent = self.layers[name]
                (ct, cf) = (ent['c' + tag], ent['f' + tag])
                co = cf - ct
                cos_cum.append(_cos(ct, co))
                cts.append(float(ct.norm()) / float(cf.norm()) if float(cf.norm()) > 0 else float('nan'))
            cos_cum = [v for v in cos_cum if not math.isnan(v)]
            out[tag]['cumulative_cos_mean'] = float(torch.tensor(cos_cum).mean()) if cos_cum else None
            out[tag]['cumulative_cos_per_layer_min'] = min(cos_cum) if cos_cum else None
            out[tag]['cumulative_cos_per_layer_max'] = max(cos_cum) if cos_cum else None
        return out

    def finish(self, meta=None, partial=False):
        if meta is not None:
            self._meta = dict(meta)
        meta = self._meta
        by_fam = collections.defaultdict(list)
        for (name, ent) in self.layers.items():
            by_fam[ent['family']].append(name)
        report = {'meta': dict(meta or {}), 'partial': bool(partial), 'n_layers': len(self.layers), 'n_optimizer_steps': len(self.steps), 'target_token_fraction': self.n_target_tokens / self.n_tokens if self.n_tokens else None, 'n_target_tokens': self.n_target_tokens, 'n_tokens': self.n_tokens, 'clip': {'max_grad_norm': self.clip['max_grad_norm'], 'steps_clipped': sum((1 for s in self.steps if s['clip_coef'] < 1.0))}, 'overall': self._summary(sorted(self.layers)), 'by_family': {f: self._summary(sorted(v)) for (f, v) in sorted(by_fam.items())}, 'reconstruction_check': [v for s in self.steps for v in s['verify']], 'per_step': {n: dict(d) for (n, d) in self.per_step.items()}, 'clip_coef_per_step': [s['clip_coef'] for s in self.steps]}
        os.makedirs(os.path.dirname(os.path.abspath(self.out)) or '.', exist_ok=True)
        with open(self.out, 'w') as f:
            json.dump(report, f, indent=2)
        return report

def print_report(rep):
    print(f"[grad_tgt_b] layers={rep['n_layers']} steps={rep['n_optimizer_steps']} target tokens={rep['n_target_tokens']}/{rep['n_tokens']} ({100 * (rep['target_token_fraction'] or 0):.2f}%)")
    rc = rep['reconstruction_check']
    if rc:
        good = [r['rel_err'] for r in rc if not math.isnan(r['rel_err'])]
        degen = len(rc) - len(good)
        if good:
            print(f'[grad_tgt_b] reconstruction (all-token formula vs autograd .grad): {len(good)} valid checks, worst rel_err={max(good):.3e}, median={sorted(good)[len(good) // 2]:.3e}' + (f'; {degen} skipped (.grad exactly 0)' if degen else ''))
        else:
            print(f'[grad_tgt_b] reconstruction: all {len(rc)} checks had a zero .grad (B has not left its zero init); nothing verified -- widen verify_steps')
    print(f"[grad_tgt_b] clipping active on {rep['clip']['steps_clipped']}/{rep['n_optimizer_steps']} steps (max_grad_norm={rep['clip']['max_grad_norm']})")
    hdr = f"{'scope':10s} {'w':2s} {'cos mean':>9s} {'median':>8s} {'std':>7s} {'frac<0':>7s} {'cum cos':>8s} {'|t|/|f|':>8s}"
    print(hdr)
    print('-' * len(hdr))
    rows = [('overall', rep['overall'])] + [(f, d) for (f, d) in rep['by_family'].items()]
    for (scope, d) in rows:
        for tag in ('A', 'B'):
            s = d.get(tag)
            if not s:
                continue
            print(f"{scope:10s} {tag:2s} {s['cos_mean']:9.4f} {s['cos_median']:8.4f} {s['cos_std']:7.4f} {s['frac_cos_negative']:7.3f} {(s['cumulative_cos_mean'] if s['cumulative_cos_mean'] is not None else float('nan')):8.4f} {(s['target_norm_ratio_mean'] if s['target_norm_ratio_mean'] is not None else float('nan')):8.4f}")

def _self_test():
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model
    ok = True

    def chk(name, cond, extra=''):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{('OK ' if cond else 'FAIL')}] {name}{(' -- ' + extra if extra else '')}")
    torch.manual_seed(0)

    class Tiny(nn.Module):

        def __init__(self, d=16, h=24):
            super().__init__()
            self.q_proj = nn.Linear(d, d, bias=False)
            self.gate_proj = nn.Linear(d, h, bias=False)
            self.down_proj = nn.Linear(h, d, bias=False)

        def forward(self, x):
            return self.down_proj(torch.relu(self.gate_proj(self.q_proj(x))))
    base = Tiny()
    cfg = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=['q_proj', 'gate_proj', 'down_proj'], bias='none')
    model = get_peft_model(base, cfg)
    for (_n, _m) in model.named_modules():
        _B = getattr(_m, 'lora_B', None)
        if _B is not None and hasattr(_B, 'keys'):
            for _k in _B.keys():
                torch.nn.init.normal_(_B[_k].weight, std=0.05)
    diag = TargetGradDiagnostic(model, target_role_id=2, out='/dev/null', verify_steps=(1,), verify_n_layers=99)
    chk('found all three adapted modules', len(diag.layers) == 3, str(sorted(diag.layers)))
    fams = sorted((e['family'] for e in diag.layers.values()))
    chk('families parsed', fams == ['down', 'gate', 'q'], str(fams))
    (B, S, d) = (3, 7, 16)
    x = torch.randn(B, S, d)
    roles = torch.zeros(B, S, dtype=torch.long)
    roles[0, 1:3] = 2
    roles[2, 4] = 2
    diag.begin_microbatch(1)
    diag.set_batch_mask(roles)
    model.zero_grad(set_to_none=True)
    model(x).square().sum().backward()
    worst = 0.0
    for (name, ent) in diag.layers.items():
        for tag in ('A', 'B'):
            g = ent[tag].weight.grad.detach().float()
            v = ent['v' + tag]
            rel = float((v - g).norm()) / float(g.norm())
            worst = max(worst, rel)
    chk('all-token D^T Z / (D B)^T X reproduces autograd .grad', worst < 1e-05, f'worst rel_err={worst:.2e}')
    tgt_ref = {}
    model.zero_grad(set_to_none=True)
    diag.mask = None
    out = model(x)
    m = (roles == 2).unsqueeze(-1).float()
    g_full = 2 * out
    out.backward(g_full * m)
    for (name, ent) in diag.layers.items():
        tgt_ref[name, 'A'] = ent['A'].weight.grad.detach().float().clone()
        tgt_ref[name, 'B'] = ent['B'].weight.grad.detach().float().clone()
    worst = 0.0
    for (name, ent) in diag.layers.items():
        for tag in ('A', 'B'):
            ref = tgt_ref[name, tag]
            got = ent['g' + tag]
            rel = float((got - ref).norm()) / max(float(ref.norm()), 1e-12)
            worst = max(worst, rel)
    chk('gathered-row target gradient matches a target-masked backward', worst < 0.0001, f'worst rel_err={worst:.2e}')
    diag.clip['max_grad_norm'] = 1.0
    diag.clip['pre_norm'] = 4.0
    chk('clip coefficient recovered', abs(diag._clip_coef() - 0.25) < 1e-12)
    diag.clip['pre_norm'] = 0.5
    chk('no clipping when norm is under the cap', diag._clip_coef() == 1.0)
    for (allroles, expect) in ((torch.full((B, S), 2, dtype=torch.long), 'target'), (torch.zeros(B, S, dtype=torch.long), 'other')):
        d2 = TargetGradDiagnostic(model, target_role_id=2, out='/dev/null', verify_steps=())
        d2.begin_microbatch(0)
        d2.set_batch_mask(allroles)
        model.zero_grad(set_to_none=True)
        model(x).square().sum().backward()
        ent = d2.layers[sorted(d2.layers)[0]]
        g = ent['A'].weight.grad.detach().float()
        frac = float(ent['gA'].norm()) / float(g.norm())
        if expect == 'target':
            chk('all-target batch: target carries the whole gradient', abs(frac - 1) < 1e-05, f'{frac:.6f}')
        else:
            chk('all-other batch: target contributes nothing', frac < 1e-06, f'{frac:.2e}')
        d2.remove()
    diag.remove()
    print(f"\n  {('ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED')}")
    return 0 if ok else 1
if __name__ == '__main__':
    raise SystemExit(_self_test())
