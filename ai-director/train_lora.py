import sys, os, time, glob
sys.path.insert(0, os.getcwd())
from app.config import settings
from app.services.comfyui_client import ComfyUIClient
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
PREFIX = "yusuf_v1"
c = ComfyUIClient(); c.free_vram(); time.sleep(4)
wf = {
 "1": {"class_type":"CheckpointLoaderSimple","inputs":{"ckpt_name":"sd_xl_base_1.0.safetensors"}},
 "2": {"class_type":"LoadImageDataSetFromFolder","inputs":{"folder":"yusuf_train"}},
 "3": {"class_type":"MakeTrainingDataset","inputs":{"images":["2",0],"vae":["1",2],"clip":["1",1]}},
 "4": {"class_type":"TrainLoraNode","inputs":{
        "model":["1",0],"latents":["3",0],"positive":["3",1],
        "batch_size":1,"grad_accumulation_steps":1,"steps":STEPS,"learning_rate":0.0003,
        "rank":16,"optimizer":"AdamW","loss_function":"MSE","seed":42,
        "training_dtype":"bf16","lora_dtype":"bf16","quantized_backward":False,
        "algorithm":"LoRA","gradient_checkpointing":True,"checkpoint_depth":1,
        "offloading":False,"existing_lora":"[None]","bucket_mode":False,"bypass_mode":False}},
 "5": {"class_type":"SaveLoRA","inputs":{"lora":["4",0],"prefix":PREFIX}},
}
print(f"submitting LoRA training: {STEPS} steps...")
t0=time.time()
pid=c.submit(wf)
h=c.wait_for_completion(pid, timeout=3600, poll=3.0)
print(f"training done in {time.time()-t0:.0f}s")
loras="/c/ComfyUI_windows_portable_nvidia_cu126/ComfyUI_windows_portable/ComfyUI/models/loras".replace("/c/","C:/").replace("/","\\")
found=glob.glob(loras+"\\"+PREFIX+"*")
print("LoRA files:", [os.path.basename(f) for f in found])
