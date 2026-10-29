import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import role_low_rank as RLR
MECHANISM = 'soft_role_prompt'
ROLE_TASKS = ('cta', 'cpa')
SLOT_NAMES = ('cta_context', 'cta_target', 'cpa_context', 'cpa_target')
N_SLOTS = len(SLOT_NAMES)
SLOT_INDEX = {n: i for (i, n) in enumerate(SLOT_NAMES)}
SLOT_NONE = -1
ROLE_TO_SLOT = {RLR.ROLE_CTA_TARGET: SLOT_INDEX['cta_target'], RLR.ROLE_CTA_CONTEXT: SLOT_INDEX['cta_context'], RLR.ROLE_CPA_HEAD: SLOT_INDEX['cpa_target'], RLR.ROLE_CPA_TAIL: SLOT_INDEX['cpa_target'], RLR.ROLE_CPA_CONTEXT: SLOT_INDEX['cpa_context']}
SLOT_SOURCES = {n: tuple(sorted((RLR.ROLE_NAMES[rid] for (rid, si) in ROLE_TO_SLOT.items() if si == i))) for (i, n) in enumerate(SLOT_NAMES)}
PARAM_DTYPE = torch.float32
ROLE_DIM = 64
(PARAM_ENCODER, PARAM_DIRECT) = ('encoder', 'direct')
PARAMETERIZATIONS = (PARAM_ENCODER, PARAM_DIRECT)
DEFAULT_PARAM = PARAM_ENCODER
PARAM_TAG = {PARAM_ENCODER: 'p_tuning_special_token_encoder', PARAM_DIRECT: 'prompt_tuning_direct_vectors'}
STATE_FILE = 'soft_role_prompt.pt'
CONFIG_FILE = 'soft_role_prompt_config.json'
FORMULA = 'v_slot = W e_slot + b  (e_slot: learned d_role role embedding, W/b: one SHARED d_role -> d_model prompt encoder); inputs_embeds[t] = E[input_ids[t]] for a text token, = v_slot for an inserted role position; slot in (cta_context | cta_target | cpa_context | cpa_target)'

def task_has_roles(task):
    base = str(task).lower() if task else ''
    return any((base == t or base.startswith(t + '_') for t in ROLE_TASKS))

def task_key(task):
    base = str(task).lower() if task else ''
    for t in ROLE_TASKS:
        if base == t or base.startswith(t + '_'):
            return t
    return None

def slots_for_tasks(tasks):
    keys = {task_key(t) for t in tasks}
    keys.discard(None)
    if not keys:
        raise ValueError(f'none of the training tasks has a target/context vocabulary, so there is no soft-role slot to allocate (tasks={sorted((str(t) for t in tasks))}, role tasks={list(ROLE_TASKS)})')
    want = {f'{k}_context' for k in keys} | {f'{k}_target' for k in keys}
    return tuple((n for n in SLOT_NAMES if n in want))

def slots_from_adapter(adapter_dir):
    cfg = read_config(adapter_dir)
    if not cfg or not cfg.get('slots'):
        raise RuntimeError(f"{adapter_dir}/{CONFIG_FILE} does not name the slots this adapter was trained with; the bank's width and row order cannot be guessed.")
    bad = [n for n in cfg['slots'] if n not in SLOT_INDEX]
    if bad:
        raise RuntimeError(f'{adapter_dir}: unknown soft-role slot(s) {bad}')
    return tuple(cfg['slots'])

def char_cell_ids(text):
    (col, content, _, _) = RLR.scan_table_columns(text)
    line_of = [0] * len(text)
    ln = 0
    for (g, ch) in enumerate(text):
        line_of[g] = ln
        if ch == '\n':
            ln += 1
    return [(line_of[g], col[g]) if content[g] else None for g in range(len(text))]

def insertion_points(tok, raw_prompt, rendered, task, expect_ids=None):
    if not task_has_roles(task):
        ids = tok(rendered, add_special_tokens=False)['input_ids']
        (points, absorbed) = ([], 0)
    else:
        spec = RLR.role_spec(raw_prompt, RLR.task_key(task))
        enc = tok(rendered, return_offsets_mapping=True, add_special_tokens=False)
        (ids, offs) = (enc['input_ids'], enc['offset_mapping'])
        roles = RLR.char_role_ids(rendered, spec)
        cells = char_cell_ids(rendered)
        (points, absorbed, seen) = ([], 0, set())
        for (i, (a, b)) in enumerate(offs):
            marked = False
            last = None
            for g in range(a, b):
                c = cells[g]
                if c is None or c == last:
                    continue
                last = c
                if c in seen:
                    continue
                seen.add(c)
                if marked:
                    absorbed += 1
                    continue
                slot = ROLE_TO_SLOT.get(roles[g])
                if slot is None:
                    continue
                points.append((i, slot))
                marked = True
    if expect_ids is not None and list(ids) != list(expect_ids):
        raise RuntimeError(f"soft-role tokenization ({len(ids)} tokens) does not reproduce the model's own prompt tokenization ({len(expect_ids)} tokens). The insertion points are positions in that sequence; refusing to insert role embeddings at unverified positions.")
    return (ids, points, absorbed)

def insert(ids, labels, points, placeholder_id):
    at = {}
    for (i, s) in points:
        if i in at:
            raise ValueError(f'two markers for token index {i}; insertion_points emits at most one')
        if not 0 <= s < N_SLOTS:
            raise ValueError(f'slot {s} outside 0..{N_SLOTS - 1}')
        at[i] = s
    if at and max(at) >= len(ids):
        raise ValueError(f'marker at token index {max(at)} beyond the {len(ids)}-token sequence')
    (ids2, lab2, slots2) = ([], [] if labels is not None else None, [])
    for (i, t) in enumerate(ids):
        s = at.get(i)
        if s is not None:
            ids2.append(placeholder_id)
            slots2.append(s)
            if lab2 is not None:
                lab2.append(-100)
        ids2.append(t)
        slots2.append(SLOT_NONE)
        if lab2 is not None:
            lab2.append(labels[i])
    if labels is not None and len(labels) != len(ids):
        raise ValueError(f'labels ({len(labels)}) do not match ids ({len(ids)})')
    return (ids2, lab2, slots2)

def pad_slots(slot_lists, width, side='right'):
    out = []
    for s in slot_lists:
        n = width - len(s)
        if n < 0:
            raise ValueError('slot sequence longer than the padded width')
        out.append([SLOT_NONE] * n + list(s) if side == 'left' else list(s) + [SLOT_NONE] * n)
    return out

class SoftRoleBank(nn.Module):

    def __init__(self, d_model, init_std, slots=SLOT_NAMES, seed=0, role_dim=ROLE_DIM, param=DEFAULT_PARAM):
        super().__init__()
        if param not in PARAMETERIZATIONS:
            raise ValueError(f'unknown parameterization {param!r}; known: {list(PARAMETERIZATIONS)}')
        self.param = param
        names = tuple(slots)
        if not names:
            raise ValueError('a soft-role bank needs at least one slot')
        bad = [n for n in names if n not in SLOT_INDEX]
        if bad:
            raise ValueError(f'unknown soft-role slot(s) {bad}; known: {list(SLOT_NAMES)}')
        if len(set(names)) != len(names):
            raise ValueError(f'duplicate soft-role slot in {list(names)}')
        role_dim = int(role_dim)
        if param == PARAM_DIRECT:
            role_dim = 0
        elif not 0 < role_dim < int(d_model):
            raise ValueError(f'role_dim must be a LOWER dimension than d_model ({role_dim} vs {d_model}): the point of the encoder is that the role is learned in a small space and mapped up, not that it is a d_model vector by another name.')
        self.slot_names = tuple((n for n in SLOT_NAMES if n in set(names)))
        g = torch.Generator().manual_seed(int(seed))
        if param == PARAM_DIRECT:
            self.direct = nn.Parameter(torch.randn(len(self.slot_names), int(d_model), generator=g, dtype=PARAM_DTYPE) * float(init_std))
            self.role_embed = None
            self.proj = None
            self._d_model = int(d_model)
            self.role_dim = 0
            self.init_std = float(init_std)
            self.seed = int(seed)
            row = torch.full((N_SLOTS,), -1, dtype=torch.long)
            for (r, n) in enumerate(self.slot_names):
                row[SLOT_INDEX[n]] = r
            self.register_buffer('row_of_slot', row, persistent=False)
            return
        self.direct = None
        self.role_embed = nn.Parameter(torch.randn(len(self.slot_names), role_dim, generator=g, dtype=PARAM_DTYPE) * (float(init_std) * (int(d_model) / role_dim) ** 0.5))
        self.proj = nn.Linear(role_dim, int(d_model), bias=True)
        with torch.no_grad():
            a = torch.randn(int(d_model), role_dim, generator=g, dtype=PARAM_DTYPE)
            (q, r) = torch.linalg.qr(a)
            q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
            self.proj.weight.copy_(q)
            self.proj.bias.zero_()
        self.proj.to(PARAM_DTYPE)
        self.role_dim = role_dim
        self.init_std = float(init_std)
        self.seed = int(seed)
        row = torch.full((N_SLOTS,), -1, dtype=torch.long)
        for (r, n) in enumerate(self.slot_names):
            row[SLOT_INDEX[n]] = r
        self.register_buffer('row_of_slot', row, persistent=False)

    @property
    def vectors(self):
        return self.direct if self.param == PARAM_DIRECT else self.proj(self.role_embed)

    @property
    def d_model(self):
        return self._d_model if self.param == PARAM_DIRECT else self.proj.out_features

    @property
    def n_slots(self):
        return len(self.slot_names)

    def n_params(self):
        if self.param == PARAM_DIRECT:
            n = int(self.direct.numel())
            return {'role_embed': 0, 'encoder': 0, 'direct': n, 'total': n}
        lat = int(self.role_embed.numel())
        enc = int(self.proj.weight.numel() + self.proj.bias.numel())
        return {'role_embed': lat, 'encoder': enc, 'direct': 0, 'total': lat + enc}

    def holds(self, slot):
        i = SLOT_INDEX[slot] if isinstance(slot, str) else int(slot)
        return 0 <= i < N_SLOTS and int(self.row_of_slot[i]) >= 0

    def table(self, dtype, device):
        v = self.vectors.to(device=device, dtype=dtype)
        return torch.cat([v.new_zeros(1, v.shape[1]), v], 0)

    def rows(self, slots):
        dev = slots.device
        held = self.row_of_slot.to(dev)
        lut = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), torch.where(held >= 0, held + 1, held)])
        out = lut[(slots + 1).clamp(min=0, max=N_SLOTS)]
        if bool((out < 0).any()):
            missing = sorted({SLOT_NAMES[int(i)] for i in slots[out < 0].flatten().tolist()})
            raise RuntimeError(f'the batch is marked with slot(s) {missing}, which this bank does not hold (it was built for {list(self.slot_names)}). The adapter and the data belong to different regimes.')
        return out

    def apply_to(self, base_embeds, slots):
        if tuple(slots.shape) != tuple(base_embeds.shape[:2]):
            raise RuntimeError(f'slot ids {tuple(slots.shape)} do not align with the embeddings {tuple(base_embeds.shape[:2])}')
        slots = slots.to(base_embeds.device)
        tab = self.table(base_embeds.dtype, base_embeds.device)
        oh = F.one_hot(self.rows(slots), self.n_slots + 1).to(tab.dtype)
        soft = oh @ tab
        keep = (slots < 0).unsqueeze(-1)
        return torch.where(keep, base_embeds, soft)

    def extra_repr(self):
        return f"slots={self.n_slots} ({', '.join(self.slot_names)}), d_role={self.role_dim} -> d_model={self.d_model}"

def attach(model, slots=SLOT_NAMES, seed=0, role_dim=ROLE_DIM, param=DEFAULT_PARAM):
    if getattr(model, 'soft_role_bank', None) is not None:
        raise RuntimeError('a soft-role bank is already attached to this model')
    emb = model.get_input_embeddings()
    if emb is None or not hasattr(emb, 'weight'):
        raise RuntimeError('this model exposes no input-embedding matrix to insert vectors into')
    W = emb.weight.detach()
    d = int(W.shape[1])
    std = float(W.to(torch.float32).std())
    if not std > 0:
        raise RuntimeError('the base input-embedding matrix has zero spread, so there is no scale to initialize the role vectors at; refusing to insert zero vectors and call it the method.')
    bank = SoftRoleBank(d, std, slots=slots, seed=seed, role_dim=role_dim, param=param).to(next(model.parameters()).device)
    for p in bank.parameters():
        p.requires_grad_(True)
    model.soft_role_bank = bank
    return {MECHANISM: True, 'formula': FORMULA, 'slots': list(bank.slot_names), 'slots_not_allocated': [n for n in SLOT_NAMES if n not in bank.slot_names], 'slot_sources': {n: list(SLOT_SOURCES[n]) for n in bank.slot_names}, 'd_model': d, 'role_dim': bank.role_dim, 'n_slots': bank.n_slots, 'n_role_embed_params': bank.n_params()['role_embed'], 'n_encoder_params': bank.n_params()['encoder'], 'n_soft_params': bank.n_params()['total'], 'n_direct_params': bank.n_params()['direct'], 'param_count_formula': 'n_slots * d_model' if bank.param == PARAM_DIRECT else 'n_slots * d_role + d_model * d_role + d_model', 'init': 'prompt tuning: one d_model vector per slot ~ normal(0, emb_std), the base input-embedding element scale, so an inserted position starts out looking like an ordinary token embedding rather than a zero vector' if bank.param == PARAM_DIRECT else 'scale-balanced: encoder weight has ORTHONORMAL COLUMNS (unit gain both ways), bias 0, role_embed ~ normal(0, emb_std * sqrt(d_model / d_role)) -- so the PRODUCED vector starts at the base input-embedding scale while the gradient reaching the latents is not attenuated by a small projection', 'init_std': std, 'init_seed': int(seed), 'tasks_with_roles': list(ROLE_TASKS), 'insertion': 'one vector immediately before every target / context CELL, header cells included; inserted after the max_seq_len cap, never re-truncated', 'parameterization': PARAM_TAG[bank.param], 'param': bank.param}

def detach(model):
    if getattr(model, 'soft_role_bank', None) is not None:
        del model.soft_role_bank

def get_bank(model):
    bank = getattr(model, 'soft_role_bank', None)
    if bank is None:
        raise RuntimeError('no soft-role bank attached to this model')
    return bank

def is_attached(model):
    return getattr(model, 'soft_role_bank', None) is not None

def embed_inputs(model, input_ids, slot_ids):
    bank = get_bank(model)
    base = model.get_input_embeddings()(input_ids)
    return bank.apply_to(base, slot_ids)

@torch.no_grad()
def generation_batch(model, tok, raw_prompts, texts, task, pad_id=None, device=None):
    dev = model.device if device is None else device
    pad = tok.pad_token_id if pad_id is None else pad_id
    if pad is None:
        pad = tok.eos_token_id
    if pad is None:
        raise RuntimeError('the tokenizer has neither a pad nor an eos id to use as a placeholder')
    (seqs, slots, absorbed) = ([], [], 0)
    for (raw, txt) in zip(raw_prompts, texts):
        (ids, pts, ab) = insertion_points(tok, raw, txt, task)
        (a, _, s) = insert(ids, None, pts, pad)
        seqs.append(a)
        slots.append(s)
        absorbed += ab
    T = max((len(s) for s in seqs))
    ii = [[pad] * (T - len(s)) + list(s) for s in seqs]
    ss = pad_slots(slots, T, side='left')
    am = [[0] * (T - len(s)) + [1] * len(s) for s in seqs]
    ids_t = torch.tensor(ii, dtype=torch.long, device=dev)
    slot_t = torch.tensor(ss, dtype=torch.long, device=dev)
    return {'inputs_embeds': embed_inputs(model, ids_t, slot_t), 'attention_mask': torch.tensor(am, dtype=torch.long, device=dev)}

def snapshot(model):
    bank = get_bank(model)
    snap = {'vectors': bank.vectors.detach().to(torch.float32).cpu().clone()}
    if bank.param == PARAM_DIRECT:
        snap['direct'] = snap['vectors'].clone()
        return snap
    snap.update({'role_embed': bank.role_embed.detach().to(torch.float32).cpu().clone(), 'proj_weight': bank.proj.weight.detach().to(torch.float32).cpu().clone(), 'proj_bias': bank.proj.bias.detach().to(torch.float32).cpu().clone()})
    return snap

def stats(model, init_snapshot=None):
    bank = get_bank(model)
    direct = bank.param == PARAM_DIRECT
    v = bank.vectors.detach().to(torch.float32).cpu()
    e = None if direct else bank.role_embed.detach().to(torch.float32).cpu()
    init = init_snapshot or {}
    out = {}
    for (i, n) in enumerate(bank.slot_names):
        rec = {'norm': float(v[i].norm())}
        if e is not None:
            rec['latent_norm'] = float(e[i].norm())
        if 'vectors' in init:
            rec['move_from_init'] = float((v[i] - init['vectors'][i].to(torch.float32)).norm())
        if e is not None and 'role_embed' in init:
            rec['latent_move_from_init'] = float((e[i] - init['role_embed'][i].to(torch.float32)).norm())
        out[n] = rec
    for t in ROLE_TASKS:
        if not (bank.holds(f'{t}_target') and bank.holds(f'{t}_context')):
            continue
        a = v[bank.slot_names.index(f'{t}_target')]
        b = v[bank.slot_names.index(f'{t}_context')]
        out[f'{t}_target_context_cos'] = float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))
        if e is not None:
            la = e[bank.slot_names.index(f'{t}_target')]
            lb = e[bank.slot_names.index(f'{t}_context')]
            out[f'{t}_latent_target_context_cos'] = float(torch.dot(la, lb) / (la.norm() * lb.norm() + 1e-12))
    if direct:
        out['encoder'] = {'parameterization': PARAM_TAG[bank.param], 'd_model': int(bank.d_model), 'note': 'prompt tuning: no encoder, the vectors are the parameters'}
        return out
    w = bank.proj.weight.detach().to(torch.float32).cpu()
    bb = bank.proj.bias.detach().to(torch.float32).cpu()
    enc = {'d_role': int(bank.role_dim), 'd_model': int(bank.d_model), 'weight_norm': float(w.norm()), 'bias_norm': float(bb.norm())}
    if 'proj_weight' in init:
        enc['weight_move_from_init'] = float((w - init['proj_weight'].to(torch.float32)).norm())
    if 'proj_bias' in init:
        enc['bias_move_from_init'] = float((bb - init['proj_bias'].to(torch.float32)).norm())
    out['encoder'] = enc
    return out

def save(model, out_dir, extra=None):
    bank = get_bank(model)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if bank.param == PARAM_DIRECT:
        torch.save({'direct': bank.direct.detach().to('cpu', torch.float32).clone(), 'vectors': bank.vectors.detach().to('cpu', torch.float32).clone()}, out / STATE_FILE)
    else:
        torch.save({'role_embed': bank.role_embed.detach().to('cpu', torch.float32).clone(), 'proj_weight': bank.proj.weight.detach().to('cpu', torch.float32).clone(), 'proj_bias': bank.proj.bias.detach().to('cpu', torch.float32).clone(), 'vectors': bank.vectors.detach().to('cpu', torch.float32).clone()}, out / STATE_FILE)
    cfg = {MECHANISM: True, 'formula': FORMULA, 'slots': list(bank.slot_names), 'slots_not_allocated': [n for n in SLOT_NAMES if n not in bank.slot_names], 'slot_sources': {n: list(SLOT_SOURCES[n]) for n in bank.slot_names}, 'd_model': int(bank.d_model), 'role_dim': int(bank.role_dim), 'n_slots': bank.n_slots, 'n_role_embed_params': bank.n_params()['role_embed'], 'n_encoder_params': bank.n_params()['encoder'], 'n_soft_params': bank.n_params()['total'], 'n_direct_params': bank.n_params()['direct'], 'parameterization': PARAM_TAG[bank.param], 'param': bank.param, 'init_std': float(bank.init_std), 'init_seed': int(bank.seed), 'tasks_with_roles': list(ROLE_TASKS), 'state_file': STATE_FILE}
    if extra:
        cfg.update(extra)
    with open(out / CONFIG_FILE, 'w') as f:
        json.dump(cfg, f, indent=2)
    return str(out / STATE_FILE)

def load(model, adapter_dir, strict=True):
    bank = get_bank(model)
    d = Path(adapter_dir)
    (st_path, cfg_path) = (d / STATE_FILE, d / CONFIG_FILE)
    if not st_path.exists():
        raise RuntimeError(f'{st_path} not found; this adapter carries no soft-role vectors.')
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    saved_slots = cfg.get('slots')
    if saved_slots is not None and list(saved_slots) != list(bank.slot_names):
        raise RuntimeError(f"this checkpoint holds the slots {list(saved_slots)}, but the attached bank was built for {list(bank.slot_names)}; loading it would give cells another slot's vector. Attach with soft_prompt.slots_from_adapter(<adapter dir>).")
    state = torch.load(st_path, map_location='cpu')
    if not isinstance(state, dict):
        raise RuntimeError(f'{st_path} is not a soft-role state dict')
    saved_param = cfg.get('param') or (PARAM_DIRECT if 'direct' in state else PARAM_ENCODER)
    if saved_param != bank.param:
        raise RuntimeError(f"{st_path} was trained with the {saved_param!r} parameterization ({PARAM_TAG.get(saved_param, '?')}), but the attached bank is {bank.param!r} ({PARAM_TAG[bank.param]}). Attach with param=soft_prompt.param_from_adapter(<adapter dir>); refusing to guess.")
    if bank.param == PARAM_DIRECT:
        if 'direct' not in state:
            raise RuntimeError(f"{st_path} carries no 'direct' tensor, so there are no prompt-tuning vectors to restore.")
        want = {'direct': bank.direct}
    else:
        if 'role_embed' not in state:
            raise RuntimeError(f'{st_path} stores the soft-role prompt as direct d_model vectors, with no role embeddings and no encoder. This bank learns the vectors through a d_role -> d_model prompt encoder, so there is nothing here to restore it into.')
        want = {'role_embed': bank.role_embed, 'proj_weight': bank.proj.weight, 'proj_bias': bank.proj.bias}
    missing = [k for k in want if k not in state]
    if missing:
        raise RuntimeError(f'soft-role checkpoint is missing {missing}')
    for (k, p) in want.items():
        if tuple(state[k].shape) != tuple(p.shape):
            raise RuntimeError(f"soft-role checkpoint {k} {tuple(state[k].shape)} != this model's {tuple(p.shape)}")
    with torch.no_grad():
        for (k, p) in want.items():
            p.copy_(state[k].to(p.device, p.dtype))
    if strict and cfg.get('d_model') is not None and (int(cfg['d_model']) != int(bank.d_model)):
        raise RuntimeError(f"checkpoint d_model {cfg['d_model']} != this model's {bank.d_model}")
    if strict and bank.param == PARAM_ENCODER and (cfg.get('role_dim') is not None) and (int(cfg['role_dim']) != int(bank.role_dim)):
        raise RuntimeError(f"checkpoint role_dim {cfg['role_dim']} != this model's {bank.role_dim}")
    return {'loaded': bank.n_slots, 'slots': list(bank.slot_names), 'role_dim': int(bank.role_dim), 'state_file': str(st_path)}

def is_soft_role_adapter(adapter_dir):
    if not adapter_dir:
        return False
    return (Path(adapter_dir) / STATE_FILE).exists()

def param_from_adapter(adapter_dir):
    cfg = read_config(adapter_dir)
    if not cfg:
        return DEFAULT_PARAM
    if cfg.get('param') in PARAMETERIZATIONS:
        return cfg['param']
    for (k, v) in PARAM_TAG.items():
        if cfg.get('parameterization') == v:
            return k
    return DEFAULT_PARAM

def read_config(adapter_dir):
    if not adapter_dir:
        return None
    p = Path(adapter_dir) / CONFIG_FILE
    if not p.exists():
        return None
    return json.loads(p.read_text())
