"""Data loading, augmentation and metrics for crack training. Pure numpy/OpenCV (no torch)."""
import random
from pathlib import Path

import cv2
import numpy as np

EXTS = {'.jpg', '.jpeg', '.png', '.webp'}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
TOL_KERNEL = np.ones((5, 5), np.uint8)  # 2 px tolerance for thin-line matching
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)


# ---- pairs and splits ---------------------------------------------------
def find_pairs(root):
    root, out = Path(root), []
    img_dir = root / 'images'
    if not img_dir.exists():
        return out
    for ip in sorted(img_dir.iterdir()):
        if ip.suffix.lower() in EXTS:
            mp = root / 'masks' / (ip.stem + '.png')
            if mp.exists():
                out.append((ip, mp))
    return out


def site_of(path):
    return Path(path).stem.split('_')[0]


def is_public(path):
    return site_of(path).startswith('pub')


def split(pairs, val_frac=0.25, seed=0):
    """Validation = some held-out public chunks (they contain cracks, so F1 means something)
    + held-out heritage sites when you have >= 3 of them (they test false alarms on your walls)."""
    rng = random.Random(seed)
    pub = sorted({site_of(p[0]) for p in pairs if is_public(p[0])})
    own = sorted({site_of(p[0]) for p in pairs if not is_public(p[0])})
    val_sites = set()
    if pub:
        rng.shuffle(pub)
        val_sites |= set(pub[:max(1, round(len(pub) * 0.15))])
    note = 'validation includes held-out public images only (add your own crack-free photos to test false alarms on your walls)'
    if len(own) >= 3:
        rng.shuffle(own)
        own_val = set(own[:max(1, round(len(own) * val_frac))])
        val_sites |= own_val
        note = f'validation = held-out public images + {len(own_val)} held-out site(s) of your own photos'
    elif not pub:
        allsites = sorted({site_of(p[0]) for p in pairs}); rng.shuffle(allsites)
        val_sites = set(allsites[:max(1, round(len(allsites) * 0.2))]) if len(allsites) > 1 else set()
        note = 'WARNING: only your own photos found and fewer than 3 sites, so validation is weak.'
    tr = [p for p in pairs if site_of(p[0]) not in val_sites]
    va = [p for p in pairs if site_of(p[0]) in val_sites]
    return tr, va, note


# ---- loading and resizing -----------------------------------------------
def load(pair, max_side=1600):
    ip, mp = pair
    img = cv2.imread(str(ip))
    m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
    if img is None or m is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA)
    m = (m > 60).astype(np.uint8)
    if max(h, w) > max_side:  # keep RAM sane for big phone photos
        img, m = resize_to_long(img, m, max_side)
    return img, m


def resize_to_long(img, msk, long_side):
    h, w = msk.shape
    s = long_side / max(h, w)
    size = (max(8, round(w * s)), max(8, round(h * s)))
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    m = cv2.resize(msk * 255, size, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    return img, (m > (60 if s < 1 else 127)).astype(np.uint8)  # low threshold keeps thin cracks alive


# ---- augmentation and cropping ------------------------------------------
def augment(img, msk, rng=random):
    if rng.random() < .5: img, msk = img[:, ::-1], msk[:, ::-1]
    if rng.random() < .5: img, msk = img[::-1], msk[::-1]
    k = rng.randint(0, 3)
    img, msk = np.rot90(img, k), np.rot90(msk, k)
    img = np.ascontiguousarray(img).astype(np.float32)
    msk = np.ascontiguousarray(msk)
    img = img * rng.uniform(.7, 1.3) + rng.uniform(-25, 25)                       # contrast / brightness
    if rng.random() < .3:                                                          # gamma
        img = 255 * (np.clip(img, 0, 255) / 255) ** rng.uniform(.7, 1.4)
    if rng.random() < .3:                                                          # cast shadow / uneven light
        ramp = np.linspace(rng.uniform(.55, 1.0), 1.0, img.shape[0 if rng.random() < .5 else 1])
        ramp = ramp[::-1] if rng.random() < .5 else ramp
        img = img * (ramp[:, None, None] if len(ramp) == img.shape[0] else ramp[None, :, None])
    if rng.random() < .2: img = cv2.GaussianBlur(img, (0, 0), rng.uniform(.5, 1.5))  # soft focus
    if rng.random() < .2: img = img + np.random.normal(0, rng.uniform(2, 8), img.shape)  # sensor noise
    if rng.random() < .1:                                                          # grayscale
        g = img.mean(2, keepdims=True); img = np.repeat(g, 3, 2)
    return np.clip(img, 0, 255).astype(np.uint8), msk


def _pick_crop(img, msk, crop, rng, bias=0.7):
    h, w = msk.shape
    ph, pw = max(0, crop - h), max(0, crop - w)
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
        msk = cv2.copyMakeBorder(msk, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
        h, w = msk.shape
    ys, xs = np.nonzero(msk)
    if len(ys) and rng.random() < bias:  # centre near a crack most of the time, anywhere otherwise
        i = rng.randrange(len(ys))
        j = max(1, crop // 3)
        y0 = int(np.clip(ys[i] - crop // 2 + rng.randint(-j, j), 0, h - crop))
        x0 = int(np.clip(xs[i] - crop // 2 + rng.randint(-j, j), 0, w - crop))
    else:
        y0, x0 = rng.randint(0, h - crop), rng.randint(0, w - crop)
    return img[y0:y0 + crop, x0:x0 + crop], msk[y0:y0 + crop, x0:x0 + crop]


def sample_crop(img, msk, crop=384, work=1024, rng=random, train=True):
    if train:  # scale jitter around the size the app uses at inference
        img, msk = resize_to_long(img, msk, int(work * rng.uniform(.6, 1.4)))
    img, msk = _pick_crop(img, msk, crop, rng)
    if train:
        img, msk = augment(img, msk, rng)
    return img, msk


def mosaic_crop(items, crop=384, rng=random):
    """THIN-CRACK sampler: 3x3 grid of shrunken tiles from different photos.
    Shrinking makes cracks only a few pixels wide, like in wide shots of pillars and walls,
    which the normal close-up crops never show the model."""
    n = 3
    t = crop // n
    canvas = np.zeros((crop, crop, 3), np.uint8)
    out = np.zeros((crop, crop), np.uint8)
    for gy in range(n):
        for gx in range(n):
            img, msk = rng.choice(items)
            h, w = msk.shape
            s = max(rng.uniform(.12, .45), t / min(h, w))  # shrink hard, but keep tile filled when possible
            img, msk = resize_to_long(img, msk, max(8, int(max(h, w) * s)))
            ti, tm = _pick_crop(img, msk, t, rng, bias=.6)
            canvas[gy * t:(gy + 1) * t, gx * t:(gx + 1) * t] = ti
            out[gy * t:(gy + 1) * t, gx * t:(gx + 1) * t] = tm
    return augment(canvas, out, rng)


def normalize(img):
    return ((img.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).astype(np.float32)


def prep_val(img, msk, work=1024):
    return resize_to_long(img, msk, work)


# ---- metrics -------------------------------------------------------------
def counts(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    dg = cv2.dilate(gt.astype(np.uint8), TOL_KERNEL) > 0
    dp = cv2.dilate(pred.astype(np.uint8), TOL_KERNEL) > 0
    neg = not gt.any()
    return {'tp': int((pred & gt).sum()), 'fp': int((pred & ~gt).sum()), 'fn': int((~pred & gt).sum()),
            'tp_p': int((pred & dg).sum()), 'n_pred': int(pred.sum()),
            'tp_r': int((gt & dp).sum()), 'n_gt': int(gt.sum()),
            'neg_fp': int(pred.sum()) if neg else 0, 'neg_px': int(gt.size) if neg else 0}


def summarize(t):
    d = lambda a, b: a / b if b else 0.0
    p, r = d(t['tp_p'], t['n_pred']), d(t['tp_r'], t['n_gt'])
    return {'tol_precision': p, 'tol_recall': r, 'tol_f1': d(2 * p * r, p + r),
            'iou': d(t['tp'], t['tp'] + t['fp'] + t['fn']),
            'neg_fp_pct': 100 * d(t['neg_fp'], t['neg_px'])}
