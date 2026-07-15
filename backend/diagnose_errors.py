import pixeltable as pxt

frames = pxt.get_table('frames')

results = frames.select(
    pos=frames.pos,
    caption_text=frames.caption_text,
    caption_text_err=frames.caption_text.errormsg,
    clean_caption=frames.clean_caption,
    clean_caption_err=frames.clean_caption.errormsg,
).collect()

print("=" * 70)
for r in results:
    print(f"pos: {r['pos']}")
    print(f"  caption_text:      {r['caption_text']!r}")
    print(f"  caption_text_err:  {r['caption_text_err']}")
    print(f"  clean_caption:     {r['clean_caption']!r}")
    print(f"  clean_caption_err: {r['clean_caption_err']}")
    print("-" * 70)