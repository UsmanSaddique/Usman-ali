import sys
import os
sys.path.insert(0, r"C:\Users\PC\Desktop\VideoMaker\ai-director")

from app.config import settings
from app.services.model_manager import ModelManager, ModelType, register_all_loaders

print("Initializing ModelManager...")
manager = ModelManager()
register_all_loaders(manager, settings)

print("Loading LLM model...")
loaded = manager.load(ModelType.LLM)
print("Model loaded successfully!")
print("Model details: name =", loaded.name, "type =", loaded.model_type)

print("Sending completion request...")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Say 'Hello, World!' in exactly 3 words."}
]
response = loaded.model.create_chat_completion(
    messages=messages,
    temperature=0.7,
    max_tokens=50
)
print("Response:")
print(response["choices"][0]["message"]["content"])

print("Unloading model...")
manager.unload()
print("Done!")
