#!/usr/bin/env python
"""
Bonus task: "Synthetic data to the rescue!"  (DD2412, unsupervised / semi-supervised practical)

FashionMNIST with only 0.1% of the training labels (60 images, 6 per class).

  Baselines (same 60 labels, same test set):
    * supervised   - small ResNet trained on the 60 labelled images only (strong augmentation)
    * fixmatch     - FixMatch (weak/strong augmentation, confidence threshold 0.95, EMA weights)
    * simclr       - SimCLR pre-training on all 60k unlabelled images + linear probe on the 60 labels
    * mae          - MAE (tiny ViT, patch 4, fixed sin-cos positions) pre-training + linear probe

  Generative improvement:
    * gen_train    - fine-tune the pretrained DDPM "google/ddpm-cifar10-32" (35.7M-param UNet,
                     Ho et al. 2020, from the Hugging Face hub) into a CLASS-CONDITIONAL diffusion
                     model on FashionMNIST, using
                       - the 60 true labels,
                       - confident FixMatch pseudo-labels of the unlabelled images (p >= 0.95),
                       - every other unlabelled image as "unconditional" (null class),
                     with classifier-free guidance dropout.
    * gen_sample   - DDIM + classifier-free guidance -> synthetic labelled images, filtered by
                     label consistency with the FixMatch classifier.
    * synth_sup    - discriminative model trained on 60 real + synthetic labelled images.
    * synth_fixmatch - FixMatch whose labelled set is real + synthetic images.

Everything runs data-parallel on all visible GPUs (DDP, one process per GPU):

    torchrun --standalone --nproc_per_node=gpu bonus_synthetic_fmnist.py --stage all

Stages can be run separately (--stage fixmatch, --stage gen_train, ...); finished stages are
skipped unless --force is given, so an interrupted job can simply be restarted.
Results are collected in <out_dir>/results.json and <out_dir>/results.md.
"""
import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

GEN_REPO = "google/ddpm-cifar10-32"
NULL_CLASS = 10
FMNIST_MEAN, FMNIST_STD = 0.2860, 0.3530
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

PRESETS = {
    # about 30-45 min in total on 4 modern GPUs (A100 / MI250X class)
    "standard": dict(fm_steps=8000, sup_steps=3000, simclr_epochs=60, mae_epochs=100,
                     gen_steps=8000, n_per_class=1000, ddim_steps=50),
    # roughly half of that
    "quick": dict(fm_steps=4000, sup_steps=2000, simclr_epochs=30, mae_epochs=50,
                  gen_steps=4000, n_per_class=500, ddim_steps=40),
    # CPU check that every stage runs (minutes); results are meaningless
    "smoke": dict(fm_steps=6, sup_steps=6, simclr_epochs=1, mae_epochs=1,
                  gen_steps=4, n_per_class=4, ddim_steps=4),
}


# =============================================================================== distributed
class Dist:
    """Thin wrapper so the same code runs with `python` (1 process) or `torchrun` (1 per GPU)."""

    def __init__(self):
        self.initialized = "RANK" in os.environ and "WORLD_SIZE" in os.environ
        if self.initialized:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend)
            self.rank, self.world = dist.get_rank(), dist.get_world_size()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        else:
            self.rank, self.world, self.local_rank = 0, 1, 0
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device("cpu")
        self.is_main = self.rank == 0

    def barrier(self):
        if self.initialized:
            if self.device.type == "cuda":
                dist.barrier(device_ids=[self.local_rank])
            else:
                dist.barrier()

    def wrap(self, model):
        if self.initialized and self.world > 1:
            ids = [self.local_rank] if self.device.type == "cuda" else None
            return nn.parallel.DistributedDataParallel(model, device_ids=ids)
        return model

    def mean(self, value):
        t = torch.tensor(float(value.detach()) if torch.is_tensor(value) else float(value), device=self.device)
        if self.initialized and self.world > 1:
            dist.all_reduce(t)
            t /= self.world
        return t.item()

    def print(self, *a, **k):
        if self.is_main:
            print(*a, **k, flush=True)

    def close(self):
        if self.initialized:
            dist.destroy_process_group()


def unwrap(model):
    return model.module if hasattr(model, "module") else model


class Amp:
    """bf16 autocast where supported, fp16 + GradScaler otherwise, fp32 on CPU."""

    def __init__(self, device):
        self.enabled = device.type == "cuda"
        self.dtype = torch.bfloat16 if self.enabled and torch.cuda.is_bf16_supported() else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.enabled and self.dtype == torch.float16)

    def autocast(self):
        return torch.autocast(device_type="cuda", dtype=self.dtype, enabled=self.enabled)

    def step(self, loss, opt, params=None, clip=None):
        self.scaler.scale(loss).backward()
        if clip is not None:
            self.scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, clip)
        self.scaler.step(opt)
        self.scaler.update()
        opt.zero_grad(set_to_none=True)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(unwrap(model)).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        src = unwrap(model).state_dict()
        for k, v in self.model.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(src[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(src[k])


def cosine_lr(opt, base_lr, step, total, warmup=0, fixmatch=False):
    if fixmatch:  # FixMatch schedule: lr * cos(7 pi k / 16 K)
        lr = base_lr * math.cos(7 * math.pi * step / (16 * total))
    elif step < warmup:
        lr = base_lr * (step + 1) / warmup
    else:
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    for g in opt.param_groups:
        g["lr"] = lr


# =============================================================================== data
def load_data(args, D):
    from torchvision.datasets import FashionMNIST
    if D.is_main:
        FashionMNIST(args.data_dir, train=True, download=True)
        FashionMNIST(args.data_dir, train=False, download=True)
    D.barrier()
    tr = FashionMNIST(args.data_dir, train=True)
    te = FashionMNIST(args.data_dir, train=False)
    to_t = lambda d: d.data.float().div(255).unsqueeze(1)  # (N, 1, 28, 28) in [0, 1]
    xtr, ytr = to_t(tr), tr.targets.clone()
    xte, yte = to_t(te), te.targets.clone()
    if args.preset == "smoke":
        xte, yte = xte[:500], yte[:500]

    split_file = Path(args.out_dir) / "label_split.pt"
    n_labels = max(10, int(round(args.label_frac * len(ytr))))  # 0.1% of 60k = 60
    if D.is_main and not split_file.exists():
        rng = np.random.RandomState(args.seed)
        idx = np.concatenate([rng.choice(np.where(ytr.numpy() == c)[0], n_labels // 10, replace=False)
                              for c in range(10)])
        torch.save(torch.from_numpy(np.sort(idx)), split_file)
    D.barrier()
    lab_idx = torch.load(split_file)
    dev = D.device
    data = dict(x_lab=xtr[lab_idx].to(dev), y_lab=ytr[lab_idx].to(dev),
                x_unl=xtr.to(dev),  # all 60k images (labelled ones included, labels unused)
                x_te=xte.to(dev), y_te=yte.to(dev), lab_idx=lab_idx)
    D.print(f"FashionMNIST: {len(lab_idx)} labelled ({len(lab_idx) / len(ytr):.2%}), "
            f"{len(xtr)} unlabelled, {len(xte)} test images; {D.world} process(es) on {dev.type}")
    return data


# =============================================================================== GPU augmentation
def _warp(x, theta, pad="zeros"):
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode=pad, align_corners=False)


def weak_aug(x):
    """FixMatch weak augmentation: random horizontal flip + translation by up to 12.5%."""
    B, dev = x.shape[0], x.device
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
    theta[:, 1, 1] = 1.0
    theta[:, :, 2] = (torch.rand(B, 2, device=dev) * 2 - 1) * 0.25  # 0.25 normalised = 12.5% of the size
    return _warp(x, theta, pad="reflection")


_SMOOTH = torch.tensor([[1., 1., 1.], [1., 5., 1.], [1., 1., 1.]]).div(13).view(1, 1, 3, 3)


def strong_aug(x, n_ops=2):
    """RandAugment-style strong augmentation on the GPU (2 random ops with random magnitude per image,
    as in FixMatch-RA), followed by Cutout. Ops: identity, autocontrast, brightness, contrast, sharpness,
    solarize, posterize, rotate, shear-x/y, translate-x/y (colour/equalize omitted for greyscale)."""
    B, dev = x.shape[0], x.device
    A = torch.eye(3, device=dev).repeat(B, 1, 1)
    kernel = _SMOOTH.to(dev, x.dtype)
    for _ in range(n_ops):
        op = torch.randint(0, 12, (B,), device=dev)
        m = torch.rand(B, device=dev)
        v = m * torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)  # signed magnitude
        sel = lambda k: (op == k).view(B, 1, 1, 1)
        m4, v4 = m.view(B, 1, 1, 1), v.view(B, 1, 1, 1)
        lo, hi = x.amin((1, 2, 3), keepdim=True), x.amax((1, 2, 3), keepdim=True)
        mean = x.mean((1, 2, 3), keepdim=True)
        blur = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), kernel)
        bits = 8 - (m4 * 4).round()
        out = x
        out = torch.where(sel(1), (x - lo) / (hi - lo).clamp_min(1e-3), out)        # autocontrast
        out = torch.where(sel(2), x * (1 + 0.9 * v4), out)                           # brightness
        out = torch.where(sel(3), mean + (x - mean) * (1 + 0.9 * v4), out)           # contrast
        out = torch.where(sel(4), blur + (x - blur) * (1 + 0.9 * v4), out)           # sharpness
        out = torch.where(sel(5), torch.where(x >= 1 - m4, 1 - x, x), out)           # solarize
        out = torch.where(sel(6), torch.floor(x * 2 ** bits) / 2 ** bits, out)       # posterize
        x = out.clamp(0, 1)
        # geometric ops -> one affine matrix per image
        M = torch.eye(3, device=dev).repeat(B, 1, 1)
        ang = v * math.radians(30)
        rot = op == 7
        M[rot, 0, 0], M[rot, 0, 1] = torch.cos(ang[rot]), -torch.sin(ang[rot])
        M[rot, 1, 0], M[rot, 1, 1] = torch.sin(ang[rot]), torch.cos(ang[rot])
        M[op == 8, 0, 1] = 0.3 * v[op == 8]                                            # shear x
        M[op == 9, 1, 0] = 0.3 * v[op == 9]                                            # shear y
        M[op == 10, 0, 2] = 0.6 * v[op == 10]                                          # translate x (30%)
        M[op == 11, 1, 2] = 0.6 * v[op == 11]                                          # translate y
        A = A @ M
    x = _warp(x, A[:, :2, :])
    # Cutout: a grey square with side 7-14 px (up to half the image) at a random position
    H, W = x.shape[-2:]
    s = torch.randint(7, 15, (B, 1, 1), device=dev)
    cy, cx = torch.randint(0, H, (B, 1, 1), device=dev), torch.randint(0, W, (B, 1, 1), device=dev)
    yy, xx = torch.arange(H, device=dev).view(1, H, 1), torch.arange(W, device=dev).view(1, 1, W)
    hole = ((yy - cy).abs() <= s // 2) & ((xx - cx).abs() <= s // 2)
    return torch.where(hole.unsqueeze(1), torch.full_like(x, 0.5), x)


def simclr_aug(x):
    """SimCLR augmentation: random resized crop (scale 0.2-1) + flip + brightness/contrast jitter."""
    B, dev = x.shape[0], x.device
    s = torch.empty(B, device=dev).uniform_(0.2, 1.0)
    r = torch.exp(torch.empty(B, device=dev).uniform_(math.log(3 / 4), math.log(4 / 3)))
    w, h = (s * r).sqrt().clamp(max=1), (s / r).sqrt().clamp(max=1)
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = w * torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
    theta[:, 1, 1] = h
    theta[:, 0, 2] = (torch.rand(B, device=dev) * 2 - 1) * (1 - w)
    theta[:, 1, 2] = (torch.rand(B, device=dev) * 2 - 1) * (1 - h)
    x = _warp(x, theta)
    apply = (torch.rand(B, 1, 1, 1, device=dev) < 0.8)
    b = 1 + 0.4 * (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1)
    c = 1 + 0.4 * (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1)
    xj = x * b
    mu = xj.mean((1, 2, 3), keepdim=True)
    xj = (xj - mu) * c + mu
    return torch.where(apply, xj, x).clamp(0, 1)


# =============================================================================== models
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.short = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
                      if stride != 1 or cin != cout else nn.Identity())

    def forward(self, x):
        out = F.relu(self.b1(self.c1(x)))
        return F.relu(self.b2(self.c2(out)) + self.short(x))


class SmallResNet(nn.Module):
    """ResNet for 28x28 greyscale images (~2.4M params): 28 -> 14 -> 7, 256-d features."""

    def __init__(self, num_classes=10, width=64):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, width, 3, 1, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU())
        self.layers = nn.Sequential(BasicBlock(width, width), BasicBlock(width, 2 * width, 2),
                                    BasicBlock(2 * width, 4 * width, 2), BasicBlock(4 * width, 4 * width))
        self.feat_dim = 4 * width
        self.fc = nn.Linear(self.feat_dim, num_classes)

    def features(self, x):
        x = (x - FMNIST_MEAN) / FMNIST_STD
        return self.layers(self.stem(x)).mean((2, 3))

    def forward(self, x):
        return self.fc(self.features(x))


def sincos_2d(dim, grid):
    """Fixed 2D sine-cosine positional encoding (grid*grid, dim) -- better than learned in our MAE run."""
    def one_d(d, pos):
        omega = 1.0 / 10000 ** (np.arange(d // 2, dtype=np.float64) / (d / 2))
        out = np.einsum("m,d->md", pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], 1)
    gh, gw = np.meshgrid(np.arange(grid, dtype=np.float64), np.arange(grid, dtype=np.float64), indexing="ij")
    return torch.from_numpy(np.concatenate([one_d(dim // 2, gh), one_d(dim // 2, gw)], 1)).float().unsqueeze(0)


def _blocks(dim, heads, depth):
    return nn.ModuleList([nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, activation="gelu",
                                                     batch_first=True, norm_first=True) for _ in range(depth)])


class TinyMAE(nn.Module):
    """MAE with a ViT-Tiny-like encoder (dim 192, depth 6) on 28x28 images, patch 4 -> 49 tokens."""

    def __init__(self, patch=4, dim=192, depth=6, heads=3, ddim=128, ddepth=2, dheads=4):
        super().__init__()
        self.patch, self.L = patch, (28 // patch) ** 2
        self.patch_embed = nn.Conv2d(1, dim, patch, patch)
        self.register_buffer("pos", sincos_2d(dim, 28 // patch))
        self.blocks, self.norm = _blocks(dim, heads, depth), nn.LayerNorm(dim)
        self.dec_embed = nn.Linear(dim, ddim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, ddim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.register_buffer("dec_pos", sincos_2d(ddim, 28 // patch))
        self.dec_blocks, self.dec_norm = _blocks(ddim, dheads, ddepth), nn.LayerNorm(ddim)
        self.dec_pred = nn.Linear(ddim, patch * patch)
        self.feat_dim = dim

    def tokens(self, x):
        x = (x - FMNIST_MEAN) / FMNIST_STD
        return self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos

    def features(self, x):  # no masking for downstream use: mean of all encoded tokens
        t = self.tokens(x)
        for b in self.blocks:
            t = b(t)
        return self.norm(t).mean(1)

    def forward(self, x, mask_ratio=0.75):
        t = self.tokens(x)
        N, L, D = t.shape
        keep = int(L * (1 - mask_ratio))
        ids_shuffle = torch.rand(N, L, device=x.device).argsort(1)
        ids_restore = ids_shuffle.argsort(1)
        t = torch.gather(t, 1, ids_shuffle[:, :keep, None].expand(-1, -1, D))
        for b in self.blocks:
            t = b(t)
        t = self.dec_embed(self.norm(t))
        t = torch.cat([t, self.mask_token.expand(N, L - keep, -1)], 1)
        t = torch.gather(t, 1, ids_restore[..., None].expand(-1, -1, t.shape[-1])) + self.dec_pos
        for b in self.dec_blocks:
            t = b(t)
        pred = self.dec_pred(self.dec_norm(t))
        mask = torch.ones(N, L, device=x.device)
        mask[:, :keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        p = self.patch
        target = x.unfold(2, p, p).unfold(3, p, p).reshape(N, L, p * p)
        # normalised-pixel targets (MAE paper: better representations)
        target = (target - target.mean(-1, keepdim=True)) / (target.var(-1, keepdim=True) + 1e-6).sqrt()
        loss = ((pred.float() - target) ** 2).mean(-1)
        return (loss * mask).sum() / mask.sum()


# =============================================================================== evaluation helpers
@torch.no_grad()
def evaluate(model, x, y, bs=2000):
    model.eval()
    correct = sum((model(x[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item() for i in range(0, len(x), bs))
    return correct / len(x)


@torch.no_grad()
def extract(fn, x, bs=2000):
    return torch.cat([fn(x[i:i + bs]).float() for i in range(0, len(x), bs)])


def linear_probe(f_lab, y_lab, f_te, y_te, f_unl, seed=0):
    """Logistic regression on frozen features. Features are standardised with statistics of the
    unlabelled set; weight decay is chosen by 3-fold cross-validation on the labelled images only."""
    mu, sd = f_unl.mean(0), f_unl.std(0) + 1e-6
    f_lab, f_te = (f_lab - mu) / sd, (f_te - mu) / sd

    def fit(f, y, wd):
        torch.manual_seed(seed)
        clf = nn.Linear(f.shape[1], 10).to(f.device)
        opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=wd)
        for _ in range(500):
            opt.zero_grad()
            F.cross_entropy(clf(f), y).backward()
            opt.step()
        return clf

    rng = np.random.RandomState(seed)
    folds = np.zeros(len(y_lab), dtype=int)
    for c in range(10):
        idx = np.where(y_lab.cpu().numpy() == c)[0]
        folds[rng.permutation(idx)] = np.arange(len(idx)) % 3
    folds = torch.from_numpy(folds).to(y_lab.device)
    best = None
    for wd in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        accs = []
        for k in range(3):
            tr, va = folds != k, folds == k
            clf = fit(f_lab[tr], y_lab[tr], wd)
            accs.append((clf(f_lab[va]).argmax(1) == y_lab[va]).float().mean().item())
        if best is None or np.mean(accs) > best[0]:
            best = (np.mean(accs), wd)
    clf = fit(f_lab, y_lab, best[1])
    with torch.no_grad():
        acc = (clf(f_te).argmax(1) == y_te).float().mean().item()
    return acc, best[1]


# =============================================================================== results bookkeeping
def results_path(args):
    return Path(args.out_dir) / "results.json"


def load_results(args):
    p = results_path(args)
    return json.loads(p.read_text()) if p.exists() else {}


def save_result(args, D, key, value):
    if D.is_main:
        r = load_results(args)
        r[key] = value
        results_path(args).write_text(json.dumps(r, indent=2))


def done(args, key):
    return (not args.force) and key in load_results(args)


# =============================================================================== supervised / FixMatch
def train_classifier(args, D, data, steps, tag, lambda_u=1.0, x_syn=None, y_syn=None, syn_frac=0.5,
                     lab_batch=None):
    """FixMatch (lambda_u > 0) or plain supervised training (lambda_u = 0) of SmallResNet.
    If synthetic images are given, every labelled batch contains a fraction `syn_frac` of synthetic
    and 1 - syn_frac of real labelled images (so the 60 real images are not drowned out)."""
    dev, amp = D.device, Amp(D.device)
    torch.manual_seed(args.seed + 1000 * D.rank)
    model = SmallResNet().to(dev)
    ddp = D.wrap(model)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim == 1 else decay).append(p)
    opt = torch.optim.SGD([{"params": decay, "weight_decay": 5e-4}, {"params": no_decay, "weight_decay": 0}],
                          lr=args.fm_lr, momentum=0.9, nesterov=True)
    ema = EMA(model, 0.999)
    B = max(2, (lab_batch or args.fm_batch) // D.world)  # per-GPU labelled batch
    U = args.mu * B
    x_lab, y_lab = data["x_lab"], data["y_lab"]
    n_syn = int(round(B * syn_frac)) if x_syn is not None and len(x_syn) else 0
    t0 = time.time()
    for step in range(steps):
        cosine_lr(opt, args.fm_lr, step, steps, fixmatch=True)
        ddp.train()
        il = torch.randint(len(x_lab), (B - n_syn,), device=dev)
        xl, yl = x_lab[il], y_lab[il]
        if n_syn:
            isyn = torch.randint(len(x_syn), (n_syn,), device=dev)
            xl, yl = torch.cat([xl, x_syn[isyn]]), torch.cat([yl, y_syn[isyn]])
        xl = weak_aug(xl) if lambda_u > 0 else strong_aug(weak_aug(xl))
        with amp.autocast():
            if lambda_u > 0:
                xu = data["x_unl"][torch.randint(len(data["x_unl"]), (U,), device=dev)]
                logits = ddp(torch.cat([xl, weak_aug(xu), strong_aug(xu)]))
                lg_l, lg_w, lg_s = logits[:len(xl)], logits[len(xl):len(xl) + U], logits[len(xl) + U:]
                loss_s = F.cross_entropy(lg_l.float(), yl)
                conf, pseudo = lg_w.detach().float().softmax(1).max(1)
                mask = (conf >= args.tau).float()
                loss_u = (F.cross_entropy(lg_s.float(), pseudo, reduction="none") * mask).mean()
            else:
                loss_s = F.cross_entropy(ddp(xl).float(), yl)
                loss_u, mask = torch.zeros((), device=dev), torch.zeros(1, device=dev)
            loss = loss_s + lambda_u * loss_u
        amp.step(loss, opt)
        ema.update(model)
        if (step + 1) % args.log_every == 0 or step + 1 == steps:
            msg = f"[{tag}] step {step + 1}/{steps} | L_s {D.mean(loss_s):.3f} | L_u {D.mean(loss_u):.3f} " \
                  f"| mask {D.mean(mask.mean()):.2f} | {time.time() - t0:.0f}s"
            if D.is_main and ((step + 1) % (args.log_every * 5) == 0 or step + 1 == steps):
                msg += f" | test acc (EMA) {evaluate(ema.model, data['x_te'], data['y_te']):.4f}"
            D.print(msg)
    acc = evaluate(ema.model, data["x_te"], data["y_te"]) if D.is_main else None
    if D.is_main:
        torch.save(ema.model.state_dict(), Path(args.out_dir) / f"{tag}_ema.pt")
    D.barrier()
    return acc, time.time() - t0


def stage_supervised(args, D, data):
    if done(args, "supervised_60"):
        return
    acc, sec = train_classifier(args, D, data, args.sup_steps, "supervised", lambda_u=0.0, lab_batch=64)
    D.print(f"==> supervised on {len(data['y_lab'])} labels: test acc {acc}")
    save_result(args, D, "supervised_60", {"test_acc": acc, "seconds": sec})


def stage_fixmatch(args, D, data):
    if done(args, "fixmatch"):
        return
    acc, sec = train_classifier(args, D, data, args.fm_steps, "fixmatch", lambda_u=args.lambda_u)
    D.print(f"==> FixMatch: test acc {acc}")
    save_result(args, D, "fixmatch", {"test_acc": acc, "seconds": sec})


# =============================================================================== SimCLR
class SimCLRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SmallResNet()
        self.encoder.fc = nn.Identity()  # unused during pre-training (DDP requires every parameter to get a gradient)
        d = self.encoder.feat_dim
        self.proj = nn.Sequential(nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(), nn.Linear(d, 128))

    def forward(self, x):
        return F.normalize(self.proj(self.encoder.features(x)).float(), dim=1)


def nt_xent_global(z1, z2, tau, D):
    """NT-Xent with negatives from ALL GPUs: local rows (2n) against the gathered columns (2N)."""
    n = z1.shape[0]
    if D.initialized and D.world > 1:
        from torch.distributed.nn.functional import all_gather  # differentiable all_gather
        g1, g2 = torch.cat(all_gather(z1)), torch.cat(all_gather(z2))
        off = D.rank * n
    else:
        g1, g2, off = z1, z2, 0
    N = g1.shape[0]
    zl, zg = torch.cat([z1, z2]), torch.cat([g1, g2])
    sim = zl @ zg.t() / tau
    ar = torch.arange(n, device=z1.device)
    self_idx = torch.cat([ar + off, ar + N + off])
    pos_idx = torch.cat([ar + N + off, ar + off])
    sim = sim.masked_fill(F.one_hot(self_idx, 2 * N).bool(), -1e9)
    return F.cross_entropy(sim, pos_idx)


def stage_simclr(args, D, data):
    if done(args, "simclr_linear"):
        return
    dev, amp = D.device, Amp(D.device)
    torch.manual_seed(args.seed + 1000 * D.rank)
    model = SimCLRNet().to(dev)
    if D.world > 1 and dev.type == "cuda":
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # global BN, as in SimCLR
    ddp = D.wrap(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.simclr_lr, weight_decay=1e-4)
    B = args.simclr_batch // D.world
    x = data["x_unl"]
    steps = max(1, args.simclr_epochs * len(x) // args.simclr_batch)
    if args.preset == "smoke":
        steps = 3
    t0 = time.time()
    for step in range(steps):
        cosine_lr(opt, args.simclr_lr, step, steps, warmup=max(1, steps // 20))
        ddp.train()
        xb = x[torch.randint(len(x), (B,), device=dev)]
        with amp.autocast():
            z = ddp(torch.cat([simclr_aug(xb), simclr_aug(xb)]))
        loss = nt_xent_global(z[:B], z[B:], args.simclr_tau, D)
        amp.step(loss, opt)
        if (step + 1) % args.log_every == 0 or step + 1 == steps:
            D.print(f"[simclr] step {step + 1}/{steps} | NT-Xent {D.mean(loss):.3f} | {time.time() - t0:.0f}s")
    if D.is_main:
        enc = unwrap(ddp).encoder.eval()
        with amp.autocast():
            f_lab, f_te, f_unl = (extract(enc.features, data[k]) for k in ("x_lab", "x_te", "x_unl"))
        acc, wd = linear_probe(f_lab, data["y_lab"], f_te, data["y_te"], f_unl)
        torch.save(enc.state_dict(), Path(args.out_dir) / "simclr_encoder.pt")
        D.print(f"==> SimCLR + linear probe: test acc {acc:.4f} (wd {wd})")
        save_result(args, D, "simclr_linear", {"test_acc": acc, "probe_wd": wd, "seconds": time.time() - t0})
    D.barrier()


# =============================================================================== MAE
def stage_mae(args, D, data):
    if done(args, "mae_linear"):
        return
    dev, amp = D.device, Amp(D.device)
    torch.manual_seed(args.seed + 1000 * D.rank)
    model = TinyMAE().to(dev)
    ddp = D.wrap(model)
    lr = args.mae_blr * args.mae_batch / 256
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    B = args.mae_batch // D.world
    x = data["x_unl"]
    steps = max(1, args.mae_epochs * len(x) // args.mae_batch)
    if args.preset == "smoke":
        steps = 3
    t0 = time.time()
    for step in range(steps):
        cosine_lr(opt, lr, step, steps, warmup=max(1, steps // 20))
        ddp.train()
        xb = weak_aug(x[torch.randint(len(x), (B,), device=dev)])
        with amp.autocast():
            loss = ddp(xb, mask_ratio=0.75)
        amp.step(loss, opt)
        if (step + 1) % args.log_every == 0 or step + 1 == steps:
            D.print(f"[mae] step {step + 1}/{steps} | masked-patch loss {D.mean(loss):.3f} | {time.time() - t0:.0f}s")
    if D.is_main:
        enc = unwrap(ddp).eval()
        with amp.autocast():
            f_lab, f_te, f_unl = (extract(enc.features, data[k]) for k in ("x_lab", "x_te", "x_unl"))
        acc, wd = linear_probe(f_lab, data["y_lab"], f_te, data["y_te"], f_unl)
        torch.save(enc.state_dict(), Path(args.out_dir) / "mae.pt")
        D.print(f"==> MAE + linear probe: test acc {acc:.4f} (wd {wd})")
        save_result(args, D, "mae_linear", {"test_acc": acc, "probe_wd": wd, "seconds": time.time() - t0})
    D.barrier()


# =============================================================================== diffusion model
def to_gen(x):    # (B,1,28,28) in [0,1] -> (B,3,32,32) in [-1,1], the pretrained model's input format
    return (F.pad(x, (2, 2, 2, 2)) * 2 - 1).repeat(1, 3, 1, 1)


def from_gen(x):  # inverse: average the 3 channels, crop the padding
    return ((x.float().clamp(-1, 1) + 1) / 2).mean(1, keepdim=True)[..., 2:30, 2:30]


def build_conditional_unet(D):
    """Pretrained unconditional DDPM UNet + a new class embedding (10 classes + null), initialised to
    zero so that the model starts exactly as the pretrained unconditional model."""
    from diffusers import UNet2DModel
    if D.is_main:
        UNet2DModel.from_pretrained(GEN_REPO)  # download once into the HF cache
    D.barrier()
    base = UNet2DModel.from_pretrained(GEN_REPO)
    cfg = {k: v for k, v in base.config.items() if not k.startswith("_")}  # "_use_default_values" would reset our change
    cfg["num_class_embeds"] = NULL_CLASS + 1
    unet = UNet2DModel.from_config(cfg)
    missing, unexpected = unet.load_state_dict(base.state_dict(), strict=False)
    assert set(missing) == {"class_embedding.weight"} and not unexpected, (missing, unexpected)
    nn.init.zeros_(unet.class_embedding.weight)
    return unet


def load_classifier(args, tag):
    p = Path(args.out_dir) / f"{tag}_ema.pt"
    if not p.exists():
        return None
    m = SmallResNet()
    m.load_state_dict(torch.load(p, map_location="cpu"))
    return m.eval()


def generator_labels(args, D, data):
    """Conditioning label for each of the 60k training images: the true label for the 60 labelled
    ones, a confident FixMatch pseudo-label (p >= gen_tau) if available, otherwise NULL (unconditional)."""
    labels = torch.full((len(data["x_unl"]),), NULL_CLASS, dtype=torch.long, device=D.device)
    clf = load_classifier(args, "fixmatch") if args.use_pseudo_labels else None
    stats = {"pseudo_labelled": 0}
    if clf is not None:
        clf = clf.to(D.device)
        with torch.no_grad():
            p = torch.cat([clf(data["x_unl"][i:i + 2000]).softmax(1) for i in range(0, len(data["x_unl"]), 2000)])
        conf, pred = p.max(1)
        keep = conf >= args.gen_tau
        labels[keep] = pred[keep]
        stats["pseudo_labelled"] = int(keep.sum())
    labels[data["lab_idx"].to(D.device)] = data["y_lab"]
    stats["true_labelled"] = len(data["lab_idx"])
    stats["unconditional"] = int((labels == NULL_CLASS).sum())
    return labels, stats


def stage_gen_train(args, D, data):
    ckpt = Path(args.out_dir) / "gen_unet_ema.pt"
    if ckpt.exists() and not args.force:
        return
    from diffusers import DDPMScheduler
    dev, amp = D.device, Amp(D.device)
    torch.manual_seed(args.seed + 1000 * D.rank)
    labels, stats = generator_labels(args, D, data)
    D.print(f"[gen] conditioning labels: {stats}")
    unet = build_conditional_unet(D).to(dev)
    ddp = D.wrap(unet)
    sched = DDPMScheduler.from_pretrained(GEN_REPO)
    opt = torch.optim.AdamW(unet.parameters(), lr=args.gen_lr, weight_decay=0.0)
    ema = EMA(unet, args.gen_ema)
    B = max(2, args.gen_batch // D.world)
    n_real = max(1, B // 8)  # 1/8 of every batch: the 60 true-labelled images (oversampled)
    x_all, x_lab, y_lab = data["x_unl"], data["x_lab"], data["y_lab"]
    T = sched.config.num_train_timesteps
    t0 = time.time()
    for step in range(args.gen_steps):
        cosine_lr(opt, args.gen_lr, step, args.gen_steps, warmup=min(500, max(1, args.gen_steps // 10)))
        ddp.train()
        iu = torch.randint(len(x_all), (B - n_real,), device=dev)
        il = torch.randint(len(x_lab), (n_real,), device=dev)
        x0 = torch.cat([x_all[iu], x_lab[il]])
        c = torch.cat([labels[iu], y_lab[il]])
        flip = torch.rand(len(x0), device=dev) < 0.5
        x0 = torch.where(flip.view(-1, 1, 1, 1), x0.flip(-1), x0)          # horizontal flips
        c = torch.where(torch.rand(len(c), device=dev) < args.p_uncond, torch.full_like(c, NULL_CLASS), c)  # CFG dropout
        x0 = to_gen(x0)
        noise = torch.randn_like(x0)
        t = torch.randint(0, T, (len(x0),), device=dev)
        xt = sched.add_noise(x0, noise, t)
        with amp.autocast():
            pred = ddp(xt, t, class_labels=c).sample
        loss = F.mse_loss(pred.float(), noise)
        amp.step(loss, opt, unet.parameters(), clip=1.0)
        ema.update(unet)
        if (step + 1) % args.log_every == 0 or step + 1 == args.gen_steps:
            D.print(f"[gen] step {step + 1}/{args.gen_steps} | diffusion loss {D.mean(loss):.4f} | "
                    f"lr {opt.param_groups[0]['lr']:.2e} | {time.time() - t0:.0f}s")
    if D.is_main:
        torch.save(ema.model.state_dict(), ckpt)
        save_result(args, D, "generator", {"base_model": GEN_REPO, "steps": args.gen_steps,
                                           "global_batch": B * D.world, "labels": stats,
                                           "seconds": time.time() - t0})
    D.barrier()


@torch.no_grad()
def stage_gen_sample(args, D, data):
    out = Path(args.out_dir) / "synthetic.pt"
    if out.exists() and not args.force:
        return
    from diffusers import DDIMScheduler
    dev, amp = D.device, Amp(D.device)
    unet = build_conditional_unet(D)
    unet.load_state_dict(torch.load(Path(args.out_dir) / "gen_unet_ema.pt", map_location="cpu"))
    unet = unet.to(dev).eval()
    sched = DDIMScheduler.from_pretrained(GEN_REPO)
    sched.set_timesteps(args.ddim_steps)
    total = args.n_per_class * 10
    my = torch.arange(total)[D.rank::D.world] % 10  # this GPU's share of the class labels
    g = torch.Generator(device=dev).manual_seed(args.seed + 7 * D.rank)
    xs, t0 = [], time.time()
    for i in range(0, len(my), args.gen_sample_batch):
        c = my[i:i + args.gen_sample_batch].to(dev)
        x = torch.randn(len(c), 3, 32, 32, device=dev, generator=g)
        null = torch.full_like(c, NULL_CLASS)
        for t in sched.timesteps:
            with amp.autocast():
                eps = unet(torch.cat([x, x]), t, class_labels=torch.cat([c, null])).sample.float()
            e_c, e_u = eps.chunk(2)
            x = sched.step(e_u + args.guidance * (e_c - e_u), t, x).prev_sample
        xs.append(from_gen(x).cpu())
        D.print(f"[sample] rank 0: {min(i + args.gen_sample_batch, len(my))}/{len(my)} images | {time.time() - t0:.0f}s")
    torch.save({"x": torch.cat(xs), "y": my}, Path(args.out_dir) / f"synthetic_rank{D.rank}.pt")
    D.barrier()
    if D.is_main:
        parts = [torch.load(Path(args.out_dir) / f"synthetic_rank{r}.pt") for r in range(D.world)]
        x, y = torch.cat([p["x"] for p in parts]), torch.cat([p["y"] for p in parts])
        stats = {"generated": len(y)}
        clf = load_classifier(args, "fixmatch") if args.filter_conf > 0 else None
        if clf is not None:  # label-consistency filter: keep samples the classifier agrees with
            clf = clf.to(dev)
            p = torch.cat([clf(x[i:i + 2000].to(dev)).softmax(1).cpu() for i in range(0, len(x), 2000)])
            score = p[torch.arange(len(y)), y]
            keep = score >= args.filter_conf
            # never empty a class: keep at least the most label-consistent `min_keep` fraction of each class
            for c in range(10):
                idx = torch.where(y == c)[0]
                k = int(math.ceil(args.min_keep * len(idx)))
                if len(idx) and int(keep[idx].sum()) < k:
                    keep[idx[score[idx].argsort(descending=True)[:k]]] = True
            stats["agreement_with_fixmatch"] = float((p.argmax(1) == y).float().mean())
            x, y = x[keep], y[keep]
        stats["kept"] = len(y)
        stats["kept_per_class"] = torch.bincount(y, minlength=10).tolist()
        torch.save({"x": x, "y": y}, out)
        save_grid(x, y, Path(args.out_dir) / "synthetic_samples.png")
        save_grid(data["x_lab"].cpu(), data["y_lab"].cpu(), Path(args.out_dir) / "real_labelled.png")
        for r in range(D.world):
            (Path(args.out_dir) / f"synthetic_rank{r}.pt").unlink()
        stats["seconds"] = time.time() - t0
        D.print(f"[sample] {stats}")
        save_result(args, D, "synthetic", stats)
    D.barrier()


def save_grid(x, y, path, per_class=10):
    from torchvision.utils import save_image
    rows = []
    for c in range(10):
        xc = x[y == c][:per_class]
        if len(xc) < per_class:
            xc = torch.cat([xc, torch.zeros(per_class - len(xc), *x.shape[1:])])
        rows.append(xc)
    save_image(torch.cat(rows), path, nrow=per_class, padding=2, pad_value=1.0)


def load_synthetic(args, D):
    s = torch.load(Path(args.out_dir) / "synthetic.pt")
    return s["x"].to(D.device), s["y"].to(D.device)


def stage_synth_sup(args, D, data):
    if done(args, "synthetic_supervised"):
        return
    x_syn, y_syn = load_synthetic(args, D)
    acc, sec = train_classifier(args, D, data, args.sup_steps, "synth_sup", lambda_u=0.0,
                                x_syn=x_syn, y_syn=y_syn, syn_frac=args.syn_frac, lab_batch=256)
    D.print(f"==> supervised on 60 real + {len(y_syn)} synthetic: test acc {acc}")
    save_result(args, D, "synthetic_supervised", {"test_acc": acc, "n_synthetic": len(y_syn), "seconds": sec})


def stage_synth_fixmatch(args, D, data):
    if done(args, "fixmatch_plus_synthetic"):
        return
    x_syn, y_syn = load_synthetic(args, D)
    acc, sec = train_classifier(args, D, data, args.fm_steps, "synth_fixmatch", lambda_u=args.lambda_u,
                                x_syn=x_syn, y_syn=y_syn, syn_frac=args.syn_frac)
    D.print(f"==> FixMatch with 60 real + {len(y_syn)} synthetic labelled: test acc {acc}")
    save_result(args, D, "fixmatch_plus_synthetic", {"test_acc": acc, "n_synthetic": len(y_syn), "seconds": sec})


def stage_report(args, D, data):
    if not D.is_main:
        return
    r = load_results(args)
    rows = [("Supervised, 60 labels only", "supervised_60"),
            ("FixMatch (baseline)", "fixmatch"),
            ("SimCLR + linear probe (baseline)", "simclr_linear"),
            ("MAE + linear probe (baseline)", "mae_linear"),
            ("Ours: supervised on real + synthetic", "synthetic_supervised"),
            ("Ours: FixMatch with real + synthetic labelled", "fixmatch_plus_synthetic")]
    lines = ["| method | test accuracy |", "|---|---|"]
    for name, k in rows:
        acc = r.get(k, {}).get("test_acc")
        lines.append(f"| {name} | {'-' if acc is None else f'{acc:.4f}'} |")
    base = {k: r[k]["test_acc"] for k in ("fixmatch", "simclr_linear", "mae_linear") if k in r}
    ours = {k: r[k]["test_acc"] for k in ("synthetic_supervised", "fixmatch_plus_synthetic") if k in r}
    if base and ours:
        best_k = max(ours, key=ours.get)
        beaten = [k for k, v in base.items() if ours[best_k] > v]
        lines.append(f"\nBest synthetic-data model ({best_k}, {ours[best_k]:.4f}) beats: "
                     f"{', '.join(beaten) if beaten else 'none of the baselines'}.")
    if "synthetic" in r:
        lines.append(f"\nSynthetic data: {r['synthetic']}")
    if "generator" in r:
        lines.append(f"\nGenerator: {r['generator']}")
    text = "\n".join(lines)
    (Path(args.out_dir) / "results.md").write_text(text + "\n")
    print("\n" + text, flush=True)


STAGES = {"supervised": stage_supervised, "fixmatch": stage_fixmatch, "simclr": stage_simclr, "mae": stage_mae,
          "gen_train": stage_gen_train, "gen_sample": stage_gen_sample, "synth_sup": stage_synth_sup,
          "synth_fixmatch": stage_synth_fixmatch, "report": stage_report}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", help="all | download | " + " | ".join(STAGES))
    ap.add_argument("--preset", default="standard", choices=list(PRESETS))
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--out_dir", default="./bonus_out")
    ap.add_argument("--force", action="store_true", help="recompute stages that already have results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label_frac", type=float, default=0.001)
    ap.add_argument("--log_every", type=int, default=100)
    # FixMatch / supervised
    ap.add_argument("--fm_steps", type=int); ap.add_argument("--sup_steps", type=int)
    ap.add_argument("--fm_batch", type=int, default=64, help="global labelled batch")
    ap.add_argument("--mu", type=int, default=7)
    ap.add_argument("--fm_lr", type=float, default=0.03)
    ap.add_argument("--tau", type=float, default=0.95)
    ap.add_argument("--lambda_u", type=float, default=1.0)
    # SimCLR / MAE
    ap.add_argument("--simclr_epochs", type=int); ap.add_argument("--simclr_batch", type=int, default=1024)
    ap.add_argument("--simclr_lr", type=float, default=2e-3); ap.add_argument("--simclr_tau", type=float, default=0.2)
    ap.add_argument("--mae_epochs", type=int); ap.add_argument("--mae_batch", type=int, default=512)
    ap.add_argument("--mae_blr", type=float, default=1.5e-4)
    # generator
    ap.add_argument("--gen_steps", type=int); ap.add_argument("--gen_batch", type=int, default=256)
    ap.add_argument("--gen_lr", type=float, default=1e-4); ap.add_argument("--gen_ema", type=float, default=0.999)
    ap.add_argument("--gen_tau", type=float, default=0.95, help="pseudo-label confidence for conditioning")
    ap.add_argument("--use_pseudo_labels", type=int, default=1)
    ap.add_argument("--p_uncond", type=float, default=0.15, help="classifier-free guidance label dropout")
    ap.add_argument("--n_per_class", type=int); ap.add_argument("--ddim_steps", type=int)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--gen_sample_batch", type=int, default=500)
    ap.add_argument("--filter_conf", type=float, default=0.5, help="0 disables the label-consistency filter")
    ap.add_argument("--min_keep", type=float, default=0.25, help="minimum fraction of each class kept by the filter")
    ap.add_argument("--syn_frac", type=float, default=0.5, help="share of synthetic images in labelled batches")
    args = ap.parse_args()
    for k, v in PRESETS[args.preset].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.preset == "smoke":
        args.log_every, args.gen_batch, args.simclr_batch, args.mae_batch = 2, 8, 32, 32
        args.gen_sample_batch, args.fm_batch = 8, 8

    D = Dist()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if D.is_main:
        print(f"devices: {D.world} x {D.device.type}"
              + (f" ({torch.cuda.get_device_name(D.local_rank)})" if D.device.type == "cuda" else ""), flush=True)
        (Path(args.out_dir) / "args.json").write_text(json.dumps(vars(args), indent=2))
    if args.stage == "download":  # for clusters whose compute nodes have no internet: run on a login node
        from torchvision.datasets import FashionMNIST
        from diffusers import UNet2DModel, DDPMScheduler
        FashionMNIST(args.data_dir, train=True, download=True); FashionMNIST(args.data_dir, train=False, download=True)
        UNet2DModel.from_pretrained(GEN_REPO); DDPMScheduler.from_pretrained(GEN_REPO)
        D.print("downloaded FashionMNIST and", GEN_REPO)
        D.close()
        return
    data = load_data(args, D)
    order = list(STAGES) if args.stage == "all" else args.stage.split(",")
    for name in order:
        t0 = time.time()
        D.print(f"\n===== stage: {name} =====")
        STAGES[name](args, D, data)
        D.print(f"===== {name} finished in {time.time() - t0:.0f}s =====")
    D.close()


if __name__ == "__main__":
    main()
