"""
AI Director — ComfyUI Workflow Validator (no generation)

Validates the LTX/Wan workflow JSON against the LIVE ComfyUI install by
querying /object_info. Confirms — without spending GPU time:
  - every class_type used in the workflow exists as a node in this ComfyUI
  - every model file the loaders reference (unet, clip, vae, lora) is actually
    present in ComfyUI's combo lists

This is the cheapest way to catch the #1 unknown flagged in the project report:
"does the workflow match the real node versions / filenames?"

Run with ComfyUI running:  python_embeded\\python.exe test_workflow_validate.py
"""
import sys, os, json, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import settings
from app.services.comfyui_client import (
    ComfyUIClient, detect_family, get_defaults_for_model,
    build_ltx_workflow, build_wan_workflow,
)

BASE = "http://127.0.0.1:8188"


def object_info():
    with urllib.request.urlopen(f"{BASE}/object_info", timeout=30) as r:
        return json.loads(r.read())


# which loader input holds the filename combo, per node class
FILE_INPUT = {
    "UnetLoaderGGUF": "unet_name",
    "UNETLoader": "unet_name",
    "VAELoader": "vae_name",
    "LoraLoader": "lora_name",
    "CLIPLoader": "clip_name",
    "DualCLIPLoader": None,  # has clip_name1 / clip_name2, handled specially
}


def combo_values(info, cls, input_name):
    """Return the allowed values list for a combo input, or None."""
    try:
        req = info[cls]["input"]["required"]
        if input_name in req:
            v = req[input_name][0]
            return v if isinstance(v, list) else None
        opt = info[cls].get("input", {}).get("optional", {})
        if input_name in opt:
            v = opt[input_name][0]
            return v if isinstance(v, list) else None
    except Exception:
        return None
    return None


def validate(wf, info, label):
    print(f"\n--- {label} ---")
    problems = []
    for node_id, node in wf.items():
        cls = node["class_type"]
        if cls not in info:
            problems.append(f"  [MISSING NODE] '{cls}' (node {node_id}) not in this ComfyUI")
            continue
        inp = node.get("inputs", {})
        # check file references
        if cls == "DualCLIPLoader":
            for k in ("clip_name1", "clip_name2"):
                if k in inp:
                    vals = combo_values(info, cls, k)
                    if vals is not None and inp[k] not in vals:
                        problems.append(f"  [MISSING FILE] {cls}.{k} = '{inp[k]}' not available")
        else:
            fk = FILE_INPUT.get(cls)
            if fk and fk in inp:
                vals = combo_values(info, cls, fk)
                if vals is not None and inp[fk] not in vals:
                    problems.append(f"  [MISSING FILE] {cls}.{fk} = '{inp[fk]}' not available")
    if not problems:
        print(f"  OK — all {len(wf)} nodes valid, all referenced files present")
    else:
        for p in problems:
            print(p)
    return problems


if __name__ == "__main__":
    client = ComfyUIClient()
    if not client.ping():
        print("[XX] ComfyUI not reachable at 127.0.0.1:8188 — start it first.")
        sys.exit(2)

    info = object_info()
    print(f"ComfyUI reports {len(info)} available node types")

    model = settings.video.model_path.name  # LTX distilled GGUF
    d = get_defaults_for_model(model)
    family = detect_family(model)
    print(f"Validating workflow for: {model} (family={family})")

    if family == "ltx":
        wf = build_ltx_workflow(model_filename=model, prompt="a calm fairy garden",
                                negative_prompt="blurry", **d)
    else:
        wf = build_wan_workflow(model_filename=model, prompt="a calm fairy garden",
                                negative_prompt="blurry", **d)

    probs = validate(wf, info, f"{family} workflow ({model})")

    print("\n" + "=" * 60)
    if probs:
        print(f"{len(probs)} issue(s) found — workflow needs node/filename fixes.")
        print("Tip: open ComfyUI, check the exact node names + model filenames,")
        print("then update app/services/comfyui_client.py builders.")
        sys.exit(1)
    print("Workflow is valid against the live ComfyUI. Safe to generate.")
    sys.exit(0)
