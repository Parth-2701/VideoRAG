"""
Run this after: uv add "transformers==4.51.3" "sentence-transformers>=2.7.0"

Verifies both models load and run under the same transformers version,
before touching schema.py. If either section fails, you'll know exactly
which model is the problem.
"""
import torch

print(f"transformers/sentence-transformers version check...")
import transformers
import sentence_transformers
print(f"  transformers: {transformers.__version__}")
print(f"  sentence-transformers: {sentence_transformers.__version__}")

# --- 1. Florence-2 ---
print("\n[1/2] Loading Florence-2-base...")
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import numpy as np

try:
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base", torch_dtype=torch.float32, trust_remote_code=True
    ).eval()
    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-base", trust_remote_code=True
    )
    # Dummy RGB image so we don't need a real file
    dummy_img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8), mode="RGB")
    inputs = processor(text="<MORE_DETAILED_CAPTION>", images=dummy_img, return_tensors="pt")
    with torch.no_grad():
        out_ids = model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
            max_new_tokens=32, num_beams=1, do_sample=False,
        )
    decoded = processor.batch_decode(out_ids, skip_special_tokens=False)[0]
    print(f"✅ Florence-2 OK — sample output: {decoded[:60]}...")
except Exception as e:
    print(f"❌ Florence-2 FAILED: {e}")

# --- 2. Qwen3-Embedding-0.6B ---
print("\n[2/2] Loading Qwen3-Embedding-0.6B (larger download, ~1.2GB, be patient)...")
from sentence_transformers import SentenceTransformer

try:
    embed_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    vecs = embed_model.encode(["a test caption about the solar system"])
    print(f"✅ Qwen3-Embedding OK — output shape: {vecs.shape}")
except Exception as e:
    print(f"❌ Qwen3-Embedding FAILED: {e}")

print("\nDone. If both show ✅, transformers==4.51.3 is safe to use in schema.py.")