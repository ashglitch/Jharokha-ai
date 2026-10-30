"""Crack segmentation + measurement + severity for JHAROKHA-AI.

Segmentation uses a trained U-Net if weights/crack_unet.pt exists (and torch is
installed). Otherwise it falls back to a ridge-filter baseline (Sato filter),
which is much better than raw Canny at finding thin dark lines.
All severity / interval numbers are screening heuristics, not engineering values.
"""
from pathlib import Path
import cv2
import numpy as np
from scipy import ndimage
from skimage.filters import sato
from skimage.morphology import skeletonize

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS = BASE_DIR / 'weights' / 'crack_unet.pt'
WORK_SIZE = 1024  # longest side used for analysis

# ---- tunable heuristics -------------------------------------------------
# max crack width -> severity level (0 none, 1 hairline, 2 moderate, 3 severe)
WIDTH_MM_LIMITS = (0.3, 1.0)    # used when scale is known
WIDTH_PX_LIMITS = (4, 10)        # used when scale is unknown (at WORK_SIZE)
LEVEL_NAMES = ['No cracks found', 'Hairline / minor', 'Moderate', 'Severe']
LEVEL_PENALTY = {0: 0, 1: 15, 2: 35, 3: 55}
# condition score -> (risk tier, months until re-inspection)
TIERS = [(80, 'Low', 12), (60, 'Watch', 6), (40, 'Elevated', 3), (0, 'High', 1)]

_model = {'obj': None, 'tried': False, 'thr': 0.5}


# ---- segmentation -------------------------------------------------------
def _load_model():
    if _model['tried']:
        return _model['obj']
    _model['tried'] = True
    if not WEIGHTS.exists():
        return None
    try:
        import torch
        import segmentation_models_pytorch as smp
        ckpt = torch.load(WEIGHTS, map_location='cpu')
        net = smp.Unet(ckpt.get('encoder', 'resnet18'), encoder_weights=None,
                       in_channels=3, classes=1)
        net.load_state_dict(ckpt['state_dict'])
        net.eval()
        import os
        _model['thr'] = float(os.environ.get('JHAROKHA_THR', min(float(ckpt.get('threshold', 0.5)), 0.5)))
        _model['obj'] = net
    except Exception as exc:  # torch missing, bad file, etc.
        print(f'[jharokha] model not loaded, using baseline: {exc}')
    return _model['obj']


def _net_prob(net, bgr, tta):
    import torch
    h, w = bgr.shape[:2]
    ph, pw = (-h) % 32, (-w) % 32
    padded = cv2.copyMakeBorder(bgr, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    rgb = ((rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).astype(np.float32)
    x = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        if tta:  # average original + horizontal flip + vertical flip
            p = torch.sigmoid(net(torch.cat([x, x.flip(3), x.flip(2)])))
            prob = ((p[0] + p[1].flip(2) + p[2].flip(1)) / 3)[0].numpy()
        else:
            prob = torch.sigmoid(net(x))[0, 0].numpy()
    return prob[:h, :w]


import os
TILE = int(os.environ.get('JHAROKHA_TILE', '0') or 0)  # >0 = low-memory mode (small tiles, no flip-averaging)
USE_TTA = os.environ.get('JHAROKHA_TTA', '1') != '0'
SCALES = tuple(float(x) for x in __import__('os').environ.get('JHAROKHA_SCALES', '1.0,3.0').split(','))  # the model is run at several zooms; thin cracks in wide shots only show up when zoomed ~3x


def _starts(n, tile, step):
    if n <= tile:
        return [0]
    s = list(range(0, n - tile + 1, step))
    if s[-1] != n - tile:
        s.append(n - tile)
    return s


def _tile_grid(h, w, tile=1024, overlap=96):
    step = tile - overlap
    return len(_starts(h, tile, step)), len(_starts(w, tile, step))


def _net_prob_tiled(net, bgr, tile=1024, overlap=96, cb=None):
    """Run big (zoomed) images in tiles so memory stays small; overlaps are merged with max().
    cb(done, total) is called after each tile (row by row)."""
    h, w = bgr.shape[:2]
    out = np.zeros((h, w), np.float32)
    step = tile - overlap
    ys, xs = _starts(h, tile, step), _starts(w, tile, step)
    total, done = len(ys) * len(xs), 0
    for y0 in ys:
        for x0 in xs:
            p = _net_prob(net, bgr[y0:y0 + tile, x0:x0 + tile], False)
            ph, pw = p.shape
            out[y0:y0 + ph, x0:x0 + pw] = np.maximum(out[y0:y0 + ph, x0:x0 + pw], p)
            done += 1
            if cb:
                cb(done, total)
    return out


def multiscale_prob(prob_fn, bgr, scales=SCALES):
    """prob_fn(image) -> probability map. Zoomed passes are shrunk back and combined with max()."""
    h, w = bgr.shape[:2]
    best = None
    for s in scales:
        im = bgr if s == 1.0 else cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        p = prob_fn(im, s == 1.0)
        if p.shape != (h, w):
            p = cv2.resize(p, (w, h), interpolation=cv2.INTER_AREA)
        best = p if best is None else np.maximum(best, p)
    return best


def model_probs(bgr):
    """Per-scale probability maps, for diagnose.py."""
    net = _load_model()
    out = {}
    for s in (1.0, 2.0, 3.0, 4.0):
        im = bgr if s == 1.0 else cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        p = _net_prob(net, im, True) if s == 1.0 else _net_prob_tiled(net, im)
        out[s] = cv2.resize(p, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_AREA) if p.shape != bgr.shape[:2] else p
    return out


SKIP_ZOOM_AT = 300  # if the normal-size pass already finds this many px of crack line, the slow zoom pass is skipped


def _segment_model(bgr, net, progress=None):
    P = progress or (lambda *a, **k: None)
    thr = _model['thr']
    h, w = bgr.shape[:2]
    P('detect', 0.0)
    best = _net_prob_tiled(net, bgr, tile=TILE) if TILE else _net_prob(net, bgr, USE_TTA)
    P('detect', 1.0)
    first = _clean(best > thr, min_len=30, max_thickness=40)
    if (int(skeletonize(first).sum()) if first.any() else 0) >= SKIP_ZOOM_AT:
        P('zoom_skipped', 1.0)  # obvious cracks already found: no need for the slow pass
    else:
        for sc in SCALES:
            if sc == 1.0:
                continue
            im = cv2.resize(bgr, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
            tl = TILE or 1024
            rows, cols = _tile_grid(im.shape[0], im.shape[1], tile=tl)
            P('zoom', 0.0, rows=rows, cols=cols, done=0, total=rows * cols)
            pz = _net_prob_tiled(net, im, tile=tl, cb=lambda d, t: P('zoom', d / t, rows=rows, cols=cols, done=d, total=t))
            best = np.maximum(best, cv2.resize(pz, (w, h), interpolation=cv2.INTER_AREA))
    mask = _clean(best > thr, min_len=30, max_thickness=40)
    return mask, float(best[mask].mean()) if mask.any() else 0.0


def _clean(mask, min_len=50, max_thickness=14):
    """Keep only thin, elongated components (crack-like)."""
    mask = mask.astype(bool)
    labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask
    skel = skeletonize(mask)
    area = np.bincount(labels.ravel(), minlength=n + 1)
    length = np.bincount(labels[skel], minlength=n + 1)
    keep = (length >= min_len) & (area / np.maximum(length, 1) <= max_thickness)
    keep[0] = False
    return keep[labels]


def _segment_baseline(bgr):
    """Thin DARK lines only: ridge filter AND local-contrast (black-hat). Step edges and soft shading drop out."""
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    smooth = cv2.GaussianBlur(gray, (0, 0), 1.0)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    blackhat = cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, se).astype(np.float32)
    ridge = sato(gray.astype(np.float32) / 255, sigmas=[1, 1.5, 2.2], black_ridges=True)
    r8 = cv2.normalize(ridge, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    otsu, _ = cv2.threshold(r8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(otsu, float(r8.mean() + 1.5 * r8.std()))
    return _clean((blackhat >= 35) & (r8 > thr), min_len=60, max_thickness=9), None


def segment(bgr, progress=None):
    net = _load_model()
    if net is not None:
        mask, conf = _segment_model(bgr, net, progress)
        return mask, 'Trained U-Net segmentation', conf
    if progress:
        progress('detect', 0.0)
    mask, conf = _segment_baseline(bgr)
    if progress:
        progress('detect', 1.0)
    return mask, 'Baseline ridge filter (no trained weights found)', conf


# ---- measurement --------------------------------------------------------
def _orientation_split(skel):
    labels, n = ndimage.label(skel, structure=np.ones((3, 3)))
    out = {'Horizontal': 0, 'Vertical': 0, 'Diagonal': 0}
    for i in range(1, n + 1):
        ys, xs = np.nonzero(labels == i)
        if len(xs) < 15:
            continue
        pts = np.column_stack([xs, ys]).astype(np.float32)
        pts -= pts.mean(0)
        _, vecs = np.linalg.eigh(np.cov(pts.T))
        vx, vy = vecs[:, -1]
        ang = np.degrees(np.arctan2(abs(vy), abs(vx)))  # 0 = horizontal, 90 = vertical
        key = 'Horizontal' if ang < 22.5 else 'Vertical' if ang > 67.5 else 'Diagonal'
        out[key] += len(xs)
    total = sum(out.values()) or 1
    return {k: round(v / total * 100, 1) for k, v in out.items()}


def _branch_points(skel):
    k = np.ones((3, 3), np.uint8)
    nb = cv2.filter2D(skel.astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT) - 1
    return int(np.count_nonzero(skel & (nb >= 3)))


MIN_TOTAL_MODEL = 150  # px of centreline (at 1024 px scale) needed before the trained model's output counts as a crack
MIN_TOTAL_BASELINE = 60


def measure(mask, mm_per_px=None, min_total=MIN_TOTAL_BASELINE):
    h, w = mask.shape
    sk_len = int(np.count_nonzero(skeletonize(mask))) if mask.any() else 0
    if sk_len < min_total:
        return {'crack_found': False, 'ignored_px': sk_len, 'level': 0, 'severity': LEVEL_NAMES[0],
                'crack_area_pct': 0.0, 'length_px': 0, 'length_mm': None,
                'mean_width': 0.0, 'max_width': 0.0, 'branch_points': 0,
                'orientation': {'Horizontal': 0, 'Vertical': 0, 'Diagonal': 0},
                'width_hist': {'labels': [], 'counts': []}, 'unit': 'mm' if mm_per_px else 'px',
                'condition_score': 100, 'risk_tier': 'Low', 'reinspect_months': 12,
                'components': 0}
    skel = skeletonize(mask)
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    widths_px = dist[skel] * 2
    length_px = int(np.count_nonzero(skel))
    scale = mm_per_px or 1.0
    widths = widths_px * scale
    max_w, mean_w = float(np.percentile(widths, 98)), float(widths.mean())  # p98 ignores tiny spikes
    limits = WIDTH_MM_LIMITS if mm_per_px else WIDTH_PX_LIMITS
    level = 1 + sum(max_w > lim for lim in limits)
    area_pct = float(np.count_nonzero(mask)) / mask.size * 100
    branches = _branch_points(skel)
    penalty = (LEVEL_PENALTY[level] + min(25, area_pct * 5) + min(20, branches * 2))
    score = int(round(max(0, 100 - penalty)))
    tier, months = next((t, m) for lo, t, m in TIERS if score >= lo)
    edges = np.linspace(0, max(widths.max(), 1e-6), 7)
    counts, _ = np.histogram(widths, bins=edges)
    labels = [f'{edges[i]:.1f}-{edges[i + 1]:.1f}' for i in range(6)]
    return {
        'crack_found': True, 'level': level, 'severity': LEVEL_NAMES[level],
        'crack_area_pct': round(area_pct, 2), 'length_px': length_px,
        'length_mm': round(length_px * mm_per_px, 1) if mm_per_px else None,
        'mean_width': round(mean_w, 2), 'max_width': round(max_w, 2),
        'branch_points': branches, 'orientation': _orientation_split(skel),
        'width_hist': {'labels': labels, 'counts': [int(c) for c in counts]},
        'unit': 'mm' if mm_per_px else 'px', 'condition_score': score,
        'risk_tier': tier, 'reinspect_months': months,
        'components': int(ndimage.label(mask, structure=np.ones((3, 3)))[1]),
    }, skel


def analyze_image(image_path, output_path, width_cm=None, progress=None):
    P = progress or (lambda *a, **k: None)
    P('prepare', 0.0)
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError('The uploaded file could not be read as an image.')
    h, w = bgr.shape[:2]
    if max(h, w) > WORK_SIZE:
        s = WORK_SIZE / max(h, w)
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    h, w = bgr.shape[:2]
    mm_per_px = (width_cm * 10 / w) if width_cm else None
    P('prepare', 1.0)

    mask, method, conf = segment(bgr, progress)
    P('measure', 0.0)
    res = measure(mask, mm_per_px, MIN_TOTAL_BASELINE if conf is None else MIN_TOTAL_MODEL)
    skel = None
    if isinstance(res, tuple):
        res, skel = res

    P('draw', 0.0)
    overlay = bgr.copy()
    overlay[mask] = (35, 65, 220)
    blended = cv2.addWeighted(bgr, 0.55, overlay, 0.45, 0)
    if skel is not None:
        blended[cv2.dilate(skel.astype(np.uint8), np.ones((2, 2))) > 0] = (0, 215, 255)
    cv2.imwrite(str(output_path), blended)

    res['preliminary'] = conf is None and 'Baseline' in method
    res['hint_lines'] = 0
    if not res['preliminary'] and not res['crack_found']:
        # trained model saw nothing: cross-check with the simple line detector so a miss is not silent
        bm, _ = _segment_baseline(bgr)
        res['hint_lines'] = int(skeletonize(bm).sum()) if bm.any() else 0
    if res['preliminary']:  # no trained model: do not present a confident risk verdict
        res['risk_tier'], res['reinspect_months'] = 'Preliminary', None
    res.update({'width': w, 'height': h, 'method': method, 'mm_per_px': mm_per_px,
                'model_confidence': round(conf * 100, 1) if conf else None})
    return res


# ---- growth projection (needs repeat photos of the same spot) -----------
def project_growth(points, horizon_months=12):
    """points: list of (days_since_first, length). Linear fit, honest about limits."""
    if len(points) < 2:
        return None
    x = np.array([p[0] for p in points], float)
    y = np.array([p[1] for p in points], float)
    if np.ptp(x) < 1:
        return None
    slope, intercept = np.polyfit(x, y, 1)  # length units per day
    last_x = x.max()
    future = [last_x + d for d in range(0, int(horizon_months * 30.4) + 1, 30)]
    proj = [max(0.0, float(slope * d + intercept)) for d in future]
    double_in = None
    if slope > 0 and y[-1] > 0:
        double_in = round(float(y[-1] / slope / 30.4), 1)  # months for length to grow by another 100%
    return {'slope_per_month': round(float(slope) * 30.4, 2), 'n': len(points),
            'fit_days': [round(float(d), 1) for d in x], 'fit_vals': [round(float(v), 1) for v in y],
            'proj_days': [round(float(d), 1) for d in future], 'proj_vals': [round(v, 1) for v in proj],
            'double_in_months': double_in, 'reliable': len(points) >= 4}
