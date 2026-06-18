import os
from huggingface_hub import snapshot_download
import sys

print("Starting background download of Gemma text encoder (approx 24 GB)...")
print("This may take 10-30 minutes depending on your internet connection.")

try:
    snapshot_download(
        repo_id='Lightricks/gemma-3-12b-it-qat-q4_0-unquantized',
        local_dir=r'C:\Users\PC\Desktop\VideoMaker\ai-director\assets_generated\models\text_encoders\gemma3',
        allow_patterns=['*.safetensors', 'tokenizer.model', 'preprocessor_config.json', 'config.json']
    )
    print("Download completed successfully!")
except Exception as e:
    print(f"Download failed: {e}", file=sys.stderr)
