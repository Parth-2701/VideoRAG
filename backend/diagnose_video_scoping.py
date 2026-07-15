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

videos = pxt.get_table('video_rag.videos')
frames = pxt.get_table('video_rag.frames')

print("=" * 70)
print("VIDEOS TABLE:")
print("=" * 70)
video_rows = videos.select(videos.title, videos.video).collect()
for r in video_rows:
    print(f"  title={r['title']!r}")
    print(f"  video path={r['video']!r}")
    print("-" * 70)

print("\n" + "=" * 70)
print("DISTINCT frames.video VALUES (what main.py's filter compares against):")
print("=" * 70)
distinct_video_paths = frames.select(frames.video).distinct().collect()
for r in distinct_video_paths:
    print(f"  {r['video']!r}")

print("\n" + "=" * 70)
print("FRAME COUNTS PER VIDEO + caption status:")
print("=" * 70)
all_frames = frames.select(frames.pos, frames.video, frames.caption_text, frames.clean_caption).collect()
by_video = {}
for r in all_frames:
    v = r['video']
    by_video.setdefault(v, {'total': 0, 'null_caption': 0})
    by_video[v]['total'] += 1
    if r['clean_caption'] is None:
        by_video[v]['null_caption'] += 1

for v, stats in by_video.items():
    print(f"  {v!r}: {stats['total']} frames, {stats['null_caption']} with null clean_caption")

print("\n" + "=" * 70)
print("What main.py would compute for target_full_path, given VIDEOS_DIR:")
print("=" * 70)
BASE_DIR = os.getcwd()  # run this from the backend/ dir, same as main.py
STORAGE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "storage"))
VIDEOS_DIR = os.path.join(STORAGE_DIR, "videos")
target_filename = "test_bunny.mp4"
target_full_path = os.path.join(VIDEOS_DIR, target_filename)
print(f"  computed target_full_path = {target_full_path!r}")
print(f"  Does this exact string appear in the distinct frames.video list above? Compare manually.")