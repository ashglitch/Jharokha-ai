"""Build data/images + data/masks.

  python prepare_data.py deepcrack                       # downloads public DeepCrack (537 images) from GitHub
  python prepare_data.py deepcrack --zip path\\to\\file.zip # if you downloaded the GitHub zip yourself
  python prepare_data.py folder --images D:\\photos --masks D:\\masks --site haveliA
  python prepare_data.py all                             # DeepCrack + masonry + your my_photos/no_cracks, in one go
  python prepare_data.py stats                           # what do I have, and is validation trustworthy?

Naming rule: <site>_<anything>. Splits are made per site, so use one site name per building/wall.
Public sets are named pub*... and are always training-only.
"""
import argparse
import re
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

import crack_data

ROOT = Path(__file__).resolve().parent
URL = 'https://codeload.github.com/yhlleo/DeepCrack/zip/refs/heads/master'


def clean(s):
    return re.sub(r'[^A-Za-z0-9-]+', '-', s)


def save_pair(ip, mp, name, out):
    img, m = cv2.imread(str(ip)), cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
    if img is None or m is None:
        return False
    cv2.imwrite(str(out / 'images' / f'{name}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(out / 'masks' / f'{name}.png'), ((m > 127) * 255).astype(np.uint8))
    return True


def deepcrack(a, out):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = Path(a.zip) if a.zip else tmp / 'dc.zip'
        if not a.zip:
            print('downloading DeepCrack from GitHub (~70 MB)...')
            urllib.request.urlretrieve(URL, src)
        with zipfile.ZipFile(src) as z:
            inner = next(n for n in z.namelist() if n.endswith('dataset/DeepCrack.zip'))
            z.extract(inner, tmp)
        with zipfile.ZipFile(tmp / inner) as z:
            z.extractall(tmp / 'dc')
        n = 0
        for split in ('train', 'test'):
            imgs = sorted((tmp / 'dc' / f'{split}_img').glob('*.jpg'))
            for ip in imgs:
                mp = tmp / 'dc' / f'{split}_lab' / (ip.stem + '.png')
                if mp.exists() and save_pair(ip, mp, f'pubdeepcrack{n // 25}_{clean(split + ip.stem)}', out):
                    n += 1
        print(f'added {n} DeepCrack pairs (training-only)')


MASONRY_URL = 'https://codeload.github.com/dimitrisdais/crack_detection_CNN_masonry/zip/refs/heads/main'


def masonry(a, out):
    """240 real brick-masonry crack images + masks (Dais et al., Automation in Construction)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = Path(getattr(a, 'masonry_zip', None) or '') if getattr(a, 'masonry_zip', None) else tmp / 'm.zip'
        if not getattr(a, 'masonry_zip', None):
            print('downloading masonry crack dataset from GitHub (~40 MB)...')
            urllib.request.urlretrieve(MASONRY_URL, src)
        with zipfile.ZipFile(src) as z:
            z.extractall(tmp / 'x')
        base = next((tmp / 'x').glob('*/dataset'))
        n = 0
        for ip in sorted((base / 'crack_detection_224_images').glob('*.png')):
            mp = base / 'crack_detection_224_masks' / ip.name
            if mp.exists() and save_pair(ip, mp, f'pubmasonry{n // 25}_{clean(ip.stem)}', out):
                n += 1
        print(f'added {n} masonry pairs (training-only)')


def mine(a, out):
    """Your crack-free photos in my_photos/no_cracks become hard negatives. Every 10 photos = one 'site'."""
    src = ROOT / 'my_photos' / 'no_cracks'
    files = sorted(p for p in src.glob('*') if p.suffix.lower() in crack_data.EXTS) if src.exists() else []
    n = 0
    for i, ip in enumerate(files):
        img = cv2.imread(str(ip))
        if img is None:
            continue
        name = f'mine{i // 10}_{clean(ip.stem)}'
        cv2.imwrite(str(out / 'images' / f'{name}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out / 'masks' / f'{name}.png'), np.zeros(img.shape[:2], np.uint8))
        n += 1
    print(f'added {n} of your crack-free photos from my_photos/no_cracks' + ('' if n else '  (folder is empty, that is fine but add some for better results)'))


def folder(a, out):
    if '_' in a.site:
        raise SystemExit('--site must not contain "_" (it separates site from filename).')
    masks = {p.stem: p for p in Path(a.masks).iterdir() if p.suffix.lower() in crack_data.EXTS} if a.masks else {}
    if not a.masks and not a.negatives:
        raise SystemExit('Give --masks, or use --negatives for crack-free photos.')
    n = miss = 0
    for ip in sorted(Path(a.images).iterdir()):
        if ip.suffix.lower() not in crack_data.EXTS:
            continue
        mp = masks.get(ip.stem)
        if mp is None:
            if a.negatives:  # crack-free hard negative: create an all-black mask
                img = cv2.imread(str(ip))
                if img is None:
                    continue
                cv2.imwrite(str(out / 'images' / f'{a.site}_{clean(ip.stem)}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(out / 'masks' / f'{a.site}_{clean(ip.stem)}.png'), np.zeros(img.shape[:2], np.uint8))
                n += 1
            else:
                miss += 1
            continue
        n += save_pair(ip, mp, f'{a.site}_{clean(ip.stem)}', out)
    print(f'added {n} pairs for site "{a.site}"; {miss} images skipped (no mask found)')


def stats(out):
    pairs = crack_data.find_pairs(out)
    own = [p for p in pairs if not crack_data.is_public(p[0])]
    sites = sorted({crack_data.site_of(p[0]) for p in own})
    neg = sum(1 for _, m in own if cv2.imread(str(m), cv2.IMREAD_GRAYSCALE).max() == 0)
    print(f'total pairs: {len(pairs)} | public: {len(pairs) - len(own)} | your heritage images: {len(own)}')
    print(f'your sites ({len(sites)}): {sites}')
    print(f'crack-free hard negatives among yours: {neg}')
    _, _, note = crack_data.split(pairs)
    print(note)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('deepcrack'); d.add_argument('--zip')
    f = sub.add_parser('folder'); f.add_argument('--images', required=True); f.add_argument('--masks', help='folder of mask PNGs (not needed with --negatives)')
    f.add_argument('--site', required=True); f.add_argument('--negatives', action='store_true',
                   help='images with no matching mask become crack-free hard negatives')
    sub.add_parser('masonry')
    sub.add_parser('mine')
    al = sub.add_parser('all'); al.add_argument('--zip')
    sub.add_parser('stats')
    ap.add_argument('--out', default=str(ROOT / 'data'))
    a = ap.parse_args()
    out = Path(a.out)
    (out / 'images').mkdir(parents=True, exist_ok=True); (out / 'masks').mkdir(parents=True, exist_ok=True)
    if a.cmd == 'all':
        deepcrack(a, out); masonry(a, out); mine(a, out); stats(out)
    elif a.cmd == 'masonry':
        masonry(a, out)
    elif a.cmd == 'mine':
        mine(a, out)
    elif a.cmd == 'deepcrack':
        deepcrack(a, out)
    elif a.cmd == 'folder':
        folder(a, out)
    else:
        stats(out)
