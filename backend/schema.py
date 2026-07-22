#schema.py
import os
import sys


import subprocess as _subprocess
_original_subprocess_run = _subprocess.run
 
def _patched_subprocess_run(*args, **kwargs):
    cmd = args[0] if args else kwargs.get('args', '')
    cmd_str = ' '.join(cmd) if isinstance(cmd, (list, tuple)) else str(cmd)
    if 'pg_ctl' in cmd_str and kwargs.get('timeout') is not None:
        kwargs['timeout'] = max(kwargs['timeout'], 90)  # covers the ~35s AV lock delay with margin
    return _original_subprocess_run(*args, **kwargs)
 
_subprocess.run = _patched_subprocess_run
print("🔧 Patched subprocess.run: pg_ctl timeout extended to >=90s (Windows AV lock workaround)")

import pixeltable as pxt
from pixeltable.functions.gemini import embed_content
from pixeltable.functions.video import frame_iterator
from dotenv import load_dotenv

# 1. Load environment variables from .env file
load_dotenv()

# 2. API key
#    GEMINI_API_KEY is the ONLY external key this file needs now. It's used for:
#      - captioning (gemini_caption below, primary path)
#      - embeddings (add_embedding_index below)
#      - and separately by main.py for chat response synthesis
#    Everything runs on Google AI Studio's free tier (no credit card, doesn't
#    expire) — you're just subject to per-minute/per-day rate limits rather than
#    the pay-per-request credits that HF Inference Providers uses.
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key or api_key == "your-gemini-api-key-here":
    print("⚠️  GEMINI_API_KEY not set — needed for captioning, embeddings, "
          "and main.py's chat endpoint.")

# 3. Local fallback vision-language model: Florence-2 (microsoft/Florence-2-base)
#    ~0.23B params, ~460MB on disk, runs comfortably on CPU.
#    Loaded once at module import time so Pixeltable doesn't reload it per frame.
#    This only runs when the remote Gemini call fails (rate limit, network blip,
#    empty response, etc.), so captioning never hard-fails.
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
FLORENCE_MODEL_ID = "microsoft/Florence-2-base"  # swap to "-large" (0.77B) for a quality bump

print(f"Loading Florence-2 ({FLORENCE_MODEL_ID}) on {DEVICE} as fallback captioner...")
_florence_model = AutoModelForCausalLM.from_pretrained(
    FLORENCE_MODEL_ID,
    torch_dtype=TORCH_DTYPE,
    trust_remote_code=True,
).to(DEVICE).eval()
_florence_processor = AutoProcessor.from_pretrained(
    FLORENCE_MODEL_ID, trust_remote_code=True
)
print("✅ Florence-2 fallback loaded.")

FLORENCE_TASK_PROMPT = "<MORE_DETAILED_CAPTION>"


def _run_florence(frame) -> str:
    """Raw Florence-2 inference on a single frame. Plain function (not a UDF) so it
    can be called both as its own UDF and as the fallback from inside
    gemini_caption without any cross-UDF-invocation weirdness."""
    if frame is None:
        return "unlabeled visual content"
    try:
        if frame.mode != "RGB":
            frame = frame.convert("RGB")

        inputs = _florence_processor(
            text=FLORENCE_TASK_PROMPT, images=frame, return_tensors="pt"
        ).to(DEVICE, TORCH_DTYPE)

        with torch.no_grad():
            generated_ids = _florence_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                num_beams=3,
                do_sample=False,
            )

        generated_text = _florence_processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed = _florence_processor.post_process_generation(
            generated_text,
            task=FLORENCE_TASK_PROMPT,
            image_size=(frame.width, frame.height),
        )

        caption = parsed.get(FLORENCE_TASK_PROMPT, "").strip()
        return caption if caption else "unlabeled visual content"
    except Exception as e:
        print(f"⚠️  Florence-2 fallback also failed for one frame: {e}")
        return "unlabeled visual content"


@pxt.udf
def florence_caption(frame: pxt.Image) -> str:
    """Standalone Florence-2 UDF, kept for direct use / debugging if you ever want
    to add it back as its own computed column."""
    return _run_florence(frame)


# 3b. Primary captioner: a cascade of Gemini models (all free tier). Free of
#     charge on Google AI Studio — no credit card, doesn't expire — but
#     rate-limited (RPM/RPD, not a spend-based credit pool like HF). Crucially,
#     rate limits are tracked PER MODEL, not shared across models, so trying
#     several models in sequence genuinely multiplies your effective free
#     throughput rather than just reordering the same quota.
#
#     Order below is best-quality-first, then models with progressively more
#     free-tier headroom, so you get the best caption whenever quota allows and
#     only degrade in quality when you have to — Florence-2 (local, unlimited)
#     is the very last resort after every Gemini option is exhausted.
from google import genai

GEMINI_CAPTION_MODELS = [
    # --- PRO TIER: Elite visual analysis, but tightest rate limits ---
    "gemini-3.1-pro-preview",           # Deepest reasoning, highest overall quality
    "gemini-3-pro-image",                # Dedicated Pro vision model, supreme pixel details
    "gemini-2.5-pro",                    # Stable generation Pro tier

    # --- FLASH TIER: High quality, standard production headroom ---
    "gemini-3.5-flash",                  # Best balance of new capability & speed
    "gemini-3.1-flash-image",            # State-of-the-art specialized flash vision
    "gemini-3-flash-preview",            # Strong 3-series fallback
    "gemini-2.5-flash",                  # Highly stable GA version
    "gemini-2.5-flash-image",            # 2.5-era specialized vision
    "gemini-2.0-flash",                  # Fast, classic 2.0 engine

    # --- LITE TIER: Maximum free-tier RPM/RPD headroom ---
    "gemini-3.1-flash-lite",             # Highest quality of the low-latency tiers
    "gemini-3.1-flash-lite-image",       # Specialized fast-vision endpoint
    "gemini-2.5-flash-lite",             # Most generous daily request quota (RPD)
    "gemini-2.0-flash-lite"              # Base baseline model for high-throughput safety net
]

# GEMINI_CAPTION_MODELS = [

#     "gemini-3.5-flash",       # newest/highest quality; tightest free-tier limits

#     "gemini-3-flash-preview", # still strong if 3.5 is rate-limited/unavailable

#     "gemini-2.5-flash",       # stable GA, more RPM/RPD headroom than the above

#     "gemini-2.0-flash",  

#     "gemini-3.1-flash-lite",  # much higher free-tier RPM/RPD, still solid quality

#     "gemini-2.5-flash-lite",  # most generous free-tier limits of the bunch

# ]
# GEMINI_PROMPT = "Describe this image in detail, focusing on concrete objects, actions, and setting."

GEMINI_PROMPT = """
You are generating descriptions for frames extracted from a continuous video.

Treat this image as one moment in an ongoing sequence rather than as an isolated photograph. The video may belong to any domain, including educational content, lectures, tutorials, movies, animation, sports, gameplay, documentaries, news, surveillance, vlogs, advertisements, or presentations.

Write a detailed factual description that captures everything useful for video retrieval and question answering.

Include:
- The overall scene or environment.
- Every significant person, character, animal, vehicle, or object.
- What each visible subject is doing.
- Interactions between subjects or objects.
- Relative positions (foreground/background, left/right/center).
- Any visible motion or changes occurring in this moment.
- Readable text, subtitles, labels, signs, or UI elements.
- Visual attributes such as colors, clothing, appearance, lighting, weather, and time of day if evident.
- Camera viewpoint if obvious (close-up, wide shot, aerial view, over-the-shoulder, etc.).

Describe only what is directly observable in this frame.
Do not infer identities, emotions, intentions, dialogue, or events that are not visually evident.
Do not speculate about what happened before or after this frame.
Write one coherent paragraph in natural language.
"""


_genai_client = genai.Client(api_key=api_key) if api_key else None


def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg


@pxt.udf
def gemini_caption(frame: pxt.Image) -> str:
    """Generates a dense visual caption for a single video frame by trying each
    model in GEMINI_CAPTION_MODELS in order (each has its own separate free-tier
    quota), falling back to local Florence-2 only if every Gemini model fails."""
    if frame is None:
        return "unlabeled visual content"

    if _genai_client is None:
        # No Gemini key configured — go straight to local fallback.
        return _run_florence(frame)

    if frame.mode != "RGB":
        frame = frame.convert("RGB")

    for model_id in GEMINI_CAPTION_MODELS:
        try:
            response = _genai_client.models.generate_content(
                model=model_id,
                contents=[frame, GEMINI_PROMPT],
            )
            caption = (response.text or "").strip()
            if caption:
                return caption
            # Empty response from this model — try the next one rather than
            # assuming the whole pipeline is broken.
            print(f"⚠️  {model_id} returned an empty caption, trying next model...")
        except Exception as e:
            reason = "rate-limited" if _is_rate_limit_error(e) else f"{type(e).__name__}: {e!r}"
            print(f"⚠️  {model_id} failed ({reason}), trying next model...")
            continue

    # Never let a single bad/rate-limited frame poison the row: a None/error
    # caption_text cascades into a None clean_caption, which then breaks
    # embedding-index similarity queries for the WHOLE table (not just this
    # row) at query time.
    print("⚠️  All Gemini caption models failed for one frame, falling back to Florence-2.")
    return _run_florence(frame)


# 4. Context-engineering UDF (unchanged) — top-level so Pixeltable can import it
@pxt.udf
def compress_text(text: str) -> str:
    """Removes conversational filler words to reduce embedding noise and token costs."""
    if not text or str(text).strip() == "":
        return "unlabeled visual content"

    fillers = [' um ', ' uh ', ' you know ', ' like ', ' basically ', ' actually ']
    clean_text = " " + text.lower() + " "
    for filler in fillers:
        clean_text = clean_text.replace(filler, ' ')
    return clean_text.strip()


# 5. Wrap schema creation in a function to prevent transaction locks on import
def setup_db():
    print("Initializing Database Schema...")
    pxt.create_dir('video_rag', if_exists='ignore')

    # Define the primary Video Table
    videos = pxt.create_table(
        'video_rag.videos',
        {
            'video': pxt.Video,
            'title': pxt.String
        },
        if_exists='ignore'
    )

    # Create a Frame Extraction View (samples video at 1 frame every 2 seconds).
    # NOTE: this directly drives how many Gemini requests you burn per video.
    # A 2-minute video at fps=0.5 = 60 frames = 60 requests. If you're on the
    # free tier and testing with several videos, consider lowering this (e.g.
    # fps=0.2, one frame every 5s) to leave headroom under the RPM/RPD caps.
    frames_view = pxt.create_view(
        'video_rag.frames',
        videos,
        iterator=frame_iterator(video=videos.video, fps=1.0),
        if_exists='ignore'
    )

    # Primary captioning pipeline: a cascade of Gemini models (free tier, see
    # GEMINI_CAPTION_MODELS above), with Florence-2 (local) as the final
    # automatic fallback — all baked into the UDF itself, so there's still
    # just one computed column here.
    frames_view.add_computed_column(
        caption_text=gemini_caption(frames_view.frame),
        if_exists='ignore'
    )

    # Apply Context Engineering compression to the caption text
    frames_view.add_computed_column(
        clean_caption=compress_text(frames_view.caption_text),
        if_exists='ignore'
    )

    # Dense Semantic Embedding Index — Gemini embeddings (remote, via Google's
    # Gemini API, also free tier). Uses the same GEMINI_API_KEY already loaded
    # above; Pixeltable's gemini module reads GOOGLE_API_KEY / GEMINI_API_KEY
    # from the environment automatically, so nothing else needs to be passed in.
    # Requires: pip install google-genai
    frames_view.add_embedding_index(
        'clean_caption',
        idx_name='clean_caption_idx',
        embedding=embed_content.using(model='gemini-embedding-001'),
        if_exists='replace'
    )
    print("✅ Schema setup complete.")


if __name__ == "__main__":
    setup_db()