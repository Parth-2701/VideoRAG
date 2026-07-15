import os
import traceback
from dotenv import load_dotenv
from huggingface_hub import InferenceClient, model_info

load_dotenv()

hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
print(f"Token present: {bool(hf_token)}  (starts with: {hf_token[:6] if hf_token else None})")

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

# Step 1: ask HF which providers actually serve this model right now
print(f"\n--- Checking live Inference Providers for {MODEL_ID} ---")
info = model_info(MODEL_ID, token=hf_token, expand="inferenceProviderMapping")
mapping = info.inference_provider_mapping or {}
if not mapping:
    print("No provider mapping returned at all — this model may not be served by any "
          "Inference Provider right now. Check the model page's 'Deploy' tab in a "
          "browser to confirm current status.")
    raise SystemExit(1)

for name, m in mapping.items():
    print(f"  provider={name!r}  status={getattr(m, 'status', None)!r}  task={getattr(m, 'task', None)!r}")

live_providers = [name for name, m in mapping.items() if getattr(m, "status", None) == "live"]
provider = live_providers[0] if live_providers else next(iter(mapping))
print(f"\nUsing provider: {provider!r}")

# Step 2: actually try a chat completion through that provider
client = InferenceClient(api_key=hf_token, provider=provider)

TEST_IMG = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lE"
    "QVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)

try:
    completion = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail."},
                    {"type": "image_url", "image_url": {"url": TEST_IMG}},
                ],
            }
        ],
        max_tokens=64,
    )
    print("\nSUCCESS:")
    print(completion.choices[0].message.content)
except Exception as e:
    print(f"\ntype: {type(e)}")
    print(f"repr: {e!r}")
    print("--- full traceback ---")
    traceback.print_exc()