from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from pathlib import Path
from datetime import datetime
from werkzeug.utils import secure_filename
import cv2
import json
import os
import threading
import time
import uuid
from collections import OrderedDict
from analysis import analyze_image, project_growth

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / 'static' / 'uploads'
RESULT_DIR = BASE_DIR / 'static' / 'results'
HISTORY_FILE = BASE_DIR / 'history.json'
PUBLIC = os.environ.get('JHAROKHA_PUBLIC') == '1'  # online demo: never store or show other people's photos
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MAX_IMAGE_BYTES = 8 * 1024 * 1024

JOBS = OrderedDict()      # record_id -> progress info (shown on the loading page)
RESULTS = OrderedDict()   # record_id -> (record, projection, note) kept in memory
ANALYSIS_LOCK = threading.Lock()  # one analysis at a time keeps memory use low
STAGE_SPAN = {'prepare': (0, 4), 'detect': (4, 25), 'zoom': (25, 90), 'zoom_skipped': (90, 90),
              'measure': (90, 96), 'draw': (96, 99)}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or os.urandom(24).hex()
app.config['MAX_CONTENT_LENGTH'] = MAX_IMAGE_BYTES


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def read_history():
    if PUBLIC:
        return []
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        return []


def save_history(items):
    if PUBLIC:
        return
    HISTORY_FILE.write_text(json.dumps(items[:50], indent=2), encoding='utf-8')


def build_projection(record, history):
    """Growth projection from repeat photos of the same labelled site."""
    label = record['building_label'].lower()
    if label == 'unlabelled site':
        return None, 'Give the site a label and re-upload the same spot over time to unlock the growth projection.'
    same = [r for r in history if r['building_label'].lower() == label] + [record]
    use_mm = all(r.get('length_mm') for r in same)
    use_px = all(not r.get('length_mm') for r in same)
    if not (use_mm or use_px):
        return None, 'Earlier photos of this site mix scaled and unscaled measurements, so they cannot be compared.'
    pts = []
    for r in same:
        d = datetime.strptime(r['capture_date'], '%Y-%m-%d')
        pts.append((d, r['length_mm'] if use_mm else r['length_px']))
    pts.sort(key=lambda p: p[0])
    base = pts[0][0]
    proj = project_growth([((d - base).days, v) for d, v in pts])
    if not proj:
        return None, 'Need at least two photos of this site taken on different dates.'
    proj['unit'] = 'mm' if use_mm else 'px'
    proj['start'] = base.strftime('%Y-%m-%d')
    return proj, None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    uploaded = request.files.get('image')
    building_label = (request.form.get('building_label') or '').strip()[:100]
    surface_type = (request.form.get('surface_type') or 'Unknown / not sure')[:80]
    try:
        width_cm = float(request.form.get('width_cm') or 0) or None
        if width_cm is not None and not (1 <= width_cm <= 5000):
            width_cm = None
    except ValueError:
        width_cm = None
    try:
        capture_date = datetime.strptime(request.form.get('capture_date', ''), '%Y-%m-%d').strftime('%Y-%m-%d')
    except ValueError:
        capture_date = datetime.now().strftime('%Y-%m-%d')

    if not uploaded or uploaded.filename == '':
        flash('Choose a photo first, then run the screening.', 'error')
        return redirect(url_for('index'))
    if not allowed_file(uploaded.filename):
        flash('Use a PNG, JPG, JPEG, or WEBP image.', 'error')
        return redirect(url_for('index'))

    safe_name = secure_filename(uploaded.filename)
    record_id = uuid.uuid4().hex[:10]
    input_name = f'{record_id}_{safe_name}'
    output_name = f'{record_id}_overlay.jpg'
    input_path = UPLOAD_DIR / input_name
    output_path = RESULT_DIR / output_name
    uploaded.save(input_path)

    JOBS[record_id] = {'status': 'queued', 'stage': 'queued', 'pct': 0, 'started': time.time()}
    while len(JOBS) > 100:
        JOBS.popitem(last=False)
    meta = {'building_label': building_label, 'surface_type': surface_type, 'capture_date': capture_date,
            'input_name': input_name, 'output_name': output_name}
    threading.Thread(target=run_job, args=(record_id, input_path, output_path, width_cm, meta), daemon=True).start()
    return render_template('loading.html', job_id=record_id, image=f'uploads/{input_name}')


def run_job(record_id, input_path, output_path, width_cm, meta):
    job = JOBS[record_id]

    def progress(stage, frac=None, **extra):
        lo, hi = STAGE_SPAN.get(stage, (0, 0))
        job.update(stage=stage, status='running', pct=round(lo + (hi - lo) * (frac or 0), 1), **extra)

    try:
        with ANALYSIS_LOCK:
            job.update(status='running', stage='prepare', pct=0)
            metrics = analyze_image(input_path, output_path, width_cm, progress)
        record = {
            'id': record_id,
            'created_at': datetime.now().strftime('%d %b %Y, %I:%M %p'),
            'building_label': meta['building_label'] or 'Unlabelled site',
            'surface_type': meta['surface_type'],
            'capture_date': meta['capture_date'],
            'original_image': f"uploads/{meta['input_name']}",
            'overlay_image': f"results/{meta['output_name']}",
            **metrics,
        }
        history = read_history()
        projection, note = build_projection(record, history)
        history.insert(0, record)
        save_history(history)
        RESULTS[record_id] = (record, projection, note)
        while len(RESULTS) > 100:
            RESULTS.popitem(last=False)
        job.update(status='done', stage='done', pct=100, result_id=record_id)
    except Exception as exc:  # shown on the loading page instead of failing silently
        input_path.unlink(missing_ok=True)
        job.update(status='error', error=f'Could not analyze this image: {exc}')


@app.route('/status/<record_id>')
def status(record_id):
    job = JOBS.get(record_id)
    if job is None:
        return jsonify({'status': 'error', 'error': 'This analysis was not found (the server may have restarted). Please upload again.'})
    return jsonify({**job, 'elapsed': round(time.time() - job['started'], 1)})


@app.route('/result/<record_id>')
def view_result(record_id):
    if record_id in RESULTS:
        record, projection, note = RESULTS[record_id]
        return render_template('result.html', result=record, projection=projection, projection_note=note)
    history = read_history()
    record = next((r for r in history if r['id'] == record_id), None)
    if record is None:
        flash('That report was not found.', 'error')
        return redirect(url_for('history'))
    older = [r for r in history if r['id'] != record_id]
    projection, note = build_projection(record, older)
    return render_template('result.html', result=record, projection=projection, projection_note=note)


@app.route('/history')
def history():
    if PUBLIC:
        flash('History is turned off in the public demo, so nobody else sees your photos.', 'error')
        return redirect(url_for('index'))
    return render_template('history.html', records=read_history())


@app.route('/about')
def about():
    return render_template('about.html')


@app.errorhandler(413)
def too_large(_error):
    flash('That file is too large. Please use an image smaller than 8 MB.', 'error')
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
