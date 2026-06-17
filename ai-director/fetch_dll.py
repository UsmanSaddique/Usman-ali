import urllib.request
import zipfile
import io
import os
import shutil

dll_url = 'https://github.com/ggml-org/llama.cpp/releases/download/b3600/llama-b3600-bin-win-noavx-x64.zip'

print("Downloading:", dll_url)
req = urllib.request.Request(dll_url, headers={'User-Agent': 'Mozilla/5.0'})
zip_resp = urllib.request.urlopen(req)

with zipfile.ZipFile(io.BytesIO(zip_resp.read())) as z:
    for info in z.infolist():
        if info.filename.endswith('llama.dll'):
            info.filename = 'llama.dll'
            z.extract(info, 'C:/Users/PC/Desktop/VideoMaker/ai-director/')
            break

target_path = r'C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\python_embeded\Lib\site-packages\llama_cpp\lib\llama.dll'
if os.path.exists(target_path):
    os.remove(target_path)
    
shutil.copy('C:/Users/PC/Desktop/VideoMaker/ai-director/llama.dll', target_path)
print("Success! Replaced llama.dll with the basic (no-AVX2) version.")
