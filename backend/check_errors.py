import pixeltable as pxt
import schema # We import this so Pixeltable can find our compress_text function!

# 1. Grab our persistent frame view
frames = pxt.get_table('video_rag.frames')

# 2. Query the hidden '.errormsg' metadata column using Aliases (error=...)
crash_log = (
    frames.where(frames.raw_caption.errormsg != None)
    .select(pos=frames.pos, error=frames.raw_caption.errormsg)
    .limit(1)
    .collect()
)

if len(crash_log) > 0:
    print("🚨 THE EXACT ERROR THROWN BY GOOGLE:")
    print("-" * 60)
    print(crash_log[0]["error"])
    print("-" * 60)
else:
    print("No error message captured.")