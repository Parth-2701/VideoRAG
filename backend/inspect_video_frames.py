import subprocess as _subprocess
_original_subprocess_run = _subprocess.run

def _patched_subprocess_run(*args, **kwargs):
    cmd = args[0] if args else kwargs.get('args', '')
    cmd_str = ' '.join(cmd) if isinstance(cmd, (list, tuple)) else str(cmd)
    if 'pg_ctl' in cmd_str and kwargs.get('timeout') is not None:
        kwargs['timeout'] = max(kwargs['timeout'], 90)
    return _original_subprocess_run(*args, **kwargs)

_subprocess.run = _patched_subprocess_run

import pixeltable as pxt
import os

VIDEO_FILENAME = "test_bunny.mp4"  # change to match your uploaded filename
FPS = 1.0  # must match schema.py's frame_iterator fps

frames = pxt.get_table('video_rag.frames')

print("Actual distinct frames.video values stored in the database:")
print("=" * 70)
distinct_paths = frames.select(frames.video).distinct().collect()
for r in distinct_paths:
    print(f"  {r['video']!r}")
print("=" * 70)

# Use __file__'s directory, NOT os.getcwd() — matches how main.py computes
# BASE_DIR, which is stable regardless of what directory you run this from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "storage"))
VIDEOS_DIR = os.path.join(STORAGE_DIR, "videos")
target_path = os.path.join(VIDEOS_DIR, VIDEO_FILENAME)
print(f"\nComputed target_path: {target_path!r}")
print("Compare this against the distinct values printed above.\n")

# 1. Unfiltered count — confirms the table genuinely has data at all
total_unfiltered = frames.count()
print(f"Total rows in frames table (no filter): {total_unfiltered}")

# 2. Try the Pixeltable-expression filter (what main.py currently does)
pxt_filtered_count = frames.where(frames.title == VIDEO_FILENAME).count()
print(f"Rows matching frames.video == target_path (Pixeltable expression): {pxt_filtered_count}")

# 3. Try casting to String explicitly, in case Video-typed equality doesn't
# behave like plain string equality inside a Pixeltable query expression
try:
    cast_filtered_count = frames.where(frames.video.astype(pxt.String) == target_path).count()
    print(f"Rows matching with explicit .astype(pxt.String) cast: {cast_filtered_count}")
except Exception as e:
    print(f"astype(pxt.String) approach failed: {e}")

# 4. Reliable fallback: fetch everything, filter in plain Python
print("\nFalling back to Python-side filtering (fetch all, compare manually)...")
all_rows = frames.select(frames.pos, frames.video, frames.clean_caption).collect()
python_filtered = [r for r in all_rows if str(r['video']) == target_path]
print(f"Rows matching via Python-side str() comparison: {len(python_filtered)}")

results = python_filtered

print(f"Total sampled frames for {VIDEO_FILENAME}: {len(results)}")
print(f"(at fps={FPS}, that's one frame every {1/FPS:.1f}s)")
print("=" * 70)
for r in sorted(results, key=lambda x: x['pos']):
    real_seconds = r['pos'] / FPS
    print(f"  t={real_seconds:.1f}s (pos={r['pos']}): {r['clean_caption']}")
    print("-" * 70)