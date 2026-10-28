"""Train the crack U-Net locally (VS Code terminal).

Setup:
    pip install -r requirements-train.txt
    python prepare_data.py deepcrack                      # public pretraining data
    python prepare_data.py folder --images X --masks Y --site haveliA   # your labelled photos
    python prepare_data.py stats
Train:
    python train.py                       # good defaults
    python train.py --epochs 1 --limit 16 # 1-minute smoke test to check everything runs

What makes this stronger than a basic run:
  - crops at the same scale the app uses (1024 px long side) + scale jitter, so thin cracks stay thin
  - 70% of crops centred on cracks, 30% anywhere (so it also sees plenty of crack-free texture)
  - heritage photos oversampled vs public data (--own-boost)
  - BCE + Tversky loss (penalises missed cracks, still punishes false alarms)
  - pretrained encoder, OneCycle LR, mixed precision on GPU, early stopping
  - validation on whole images, by held-out SITE, with 2 px tolerance metrics, false-alarm rate on
    crack-free images, and an automatic threshold search that gets saved with the weights
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Dataset

import crack_data as cd

ROOT = Path(__file__).resolve().parent
WORK = 1024


class Crops(Dataset):
    def __init__(self, items, crop, per_image, thin_prob=0.0):
        self.items, self.crop, self.n, self.thin = items, crop, per_image, thin_prob

    def __len__(self):
        return len(self.items) * self.n

    def __getitem__(self, i):
        if random.random() < self.thin:
            x, y = cd.mosaic_crop(self.items, self.crop, random)
        else:
            img, msk = self.items[i % len(self.items)]
            x, y = cd.sample_crop(img, msk, self.crop, WORK, random, train=True)
        return torch.from_numpy(cd.normalize(x)), torch.from_numpy(y[None].astype(np.float32))


def tversky(logits, y, a=0.4, b=0.6, eps=1.0):
    p = torch.sigmoid(logits)
    tp = (p * y).sum((1, 2, 3)); fp = (p * (1 - y)).sum((1, 2, 3)); fn = ((1 - p) * y).sum((1, 2, 3))
    return 1 - ((tp + eps) / (tp + a * fp + b * fn + eps)).mean()


@torch.no_grad()
def predict_full(net, img, dev):
    h, w = img.shape[:2]
    ph, pw = (-h) % 32, (-w) % 32
    padded = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode='edge')
    x = torch.from_numpy(cd.normalize(padded)).unsqueeze(0).to(dev)
    return torch.sigmoid(net(x))[0, 0].cpu().numpy()[:h, :w]


def score(m):
    """Higher is better: crack F1, minus a penalty for false alarms on crack-free images."""
    return m['tol_f1'] - 0.05 * m['neg_fp_pct']


@torch.no_grad()
def evaluate(net, val_items, dev):
    net.eval()
    tot = {t: None for t in cd.THRESHOLDS}
    for img, msk in val_items:
        img, msk = cd.prep_val(img, msk, WORK)
        prob = predict_full(net, img, dev)
        for t in cd.THRESHOLDS:
            c = cd.counts(prob > t, msk)
            tot[t] = c if tot[t] is None else {k: tot[t][k] + c[k] for k in c}
    res = {t: cd.summarize(tot[t]) for t in cd.THRESHOLDS}
    best_t = max(res, key=lambda t: score(res[t]))
    return best_t, res[best_t], res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--crop', type=int, default=384)
    ap.add_argument('--crops-per-image', type=int, default=3)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--encoder', default='resnet34', help='resnet18 is faster on CPU')
    ap.add_argument('--own-boost', type=int, default=3, help='repeat your heritage images this many times')
    ap.add_argument('--patience', type=int, default=12)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0, help='use only N pairs (smoke test)')
    ap.add_argument('--no-pretrained', action='store_true')
    ap.add_argument('--thin-prob', type=float, default=0.4, help='share of crops built from shrunken tiles (thin cracks in wide shots)')
    ap.add_argument('--resume', action='store_true', help='continue from weights/crack_unet.pt (use a lower --lr, e.g. 2e-4)')
    a = ap.parse_args()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    pairs = cd.find_pairs(ROOT / 'data')
    if a.limit:
        random.shuffle(pairs); pairs = pairs[:a.limit]
    if len(pairs) < 10:
        raise SystemExit(f'Only {len(pairs)} pairs in data/. Run: python prepare_data.py deepcrack')
    tr_pairs, va_pairs, note = cd.split(pairs)
    print(note)
    print('loading images into RAM...')
    tr = [(cd.load(p), cd.is_public(p[0])) for p in tr_pairs]
    tr = [(x, pub) for x, pub in tr if x is not None]
    va = [x for x in (cd.load(p) for p in va_pairs) if x is not None] or [x for x, _ in tr[:max(4, len(tr) // 5)]]
    train_items = []
    for x, pub in tr:
        train_items += [x] * (1 if pub else a.own_boost)
    print(f'{len(tr)} train images ({len(train_items)} after oversampling) | {len(va)} validation images')

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', dev, '| encoder:', a.encoder)
    try:
        net = smp.Unet(a.encoder, encoder_weights=None if a.no_pretrained else 'imagenet', in_channels=3, classes=1)
    except Exception as exc:
        print('could not download pretrained weights, training from scratch:', exc)
        net = smp.Unet(a.encoder, encoder_weights=None, in_channels=3, classes=1)
    if a.resume:
        wp = ROOT / 'weights' / 'crack_unet.pt'
        ck = torch.load(wp, map_location='cpu')
        if ck.get('encoder') != a.encoder:
            raise SystemExit(f"Saved weights use encoder {ck.get('encoder')}; run with --encoder {ck.get('encoder')}")
        net.load_state_dict(ck['state_dict'])
        import shutil; shutil.copy(wp, ROOT / 'weights' / 'crack_unet_before_resume.pt')
        print('resumed from weights/crack_unet.pt (backup: crack_unet_before_resume.pt)')
    net.to(dev)

    loader = DataLoader(Crops(train_items, a.crop, a.crops_per_image, a.thin_prob), batch_size=a.batch, shuffle=True,
                        num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0)
    enc, dec = list(net.encoder.parameters()), list(net.decoder.parameters()) + list(net.segmentation_head.parameters())
    opt = torch.optim.AdamW([{'params': enc, 'lr': a.lr / 3}, {'params': dec, 'lr': a.lr}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr / 3, a.lr], total_steps=a.epochs * len(loader), pct_start=0.15)
    use_amp = dev == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0, device=dev))

    (ROOT / 'weights').mkdir(exist_ok=True)
    best, bad, log = -1, 0, []
    for ep in range(1, a.epochs + 1):
        t0 = time.time(); net.train(); total = 0
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                out = net(x)
            out = out.float()
            loss = bce(out, y) + tversky(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            total += loss.item()
        thr, m, _ = evaluate(net, va, dev)
        row = {'epoch': ep, 'loss': total / len(loader), 'threshold': thr, **m}
        log.append(row)
        print(f"ep {ep:02d} loss {row['loss']:.3f} | tol-F1 {m['tol_f1']:.3f} (P {m['tol_precision']:.3f} R {m['tol_recall']:.3f}) "
              f"IoU {m['iou']:.3f} | false-alarm on crack-free imgs {m['neg_fp_pct']:.3f}% | thr {thr} | {time.time() - t0:.0f}s")
        if score(m) > best:
            best, bad = score(m), 0
            torch.save({'state_dict': net.state_dict(), 'encoder': a.encoder, 'threshold': float(thr),
                        'metrics': {k: float(v) for k, v in m.items()}}, ROOT / 'weights' / 'crack_unet.pt')
            print('   saved weights/crack_unet.pt')
        else:
            bad += 1
            if bad >= a.patience:
                print('early stop'); break
    (ROOT / 'weights' / 'train_log.json').write_text(json.dumps(log, indent=2))
    print(f'best validation score (F1 minus false-alarm penalty): {best:.3f}  (log: weights/train_log.json)')


if __name__ == '__main__':
    main()
