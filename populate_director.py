"""
LTX Director Auto-Populator
============================
Reads your prompts, injects them into the Director workflow,
saves a ready-to-run workflow. Open it in ComfyUI and hit Run.

Usage:
  python populate_director.py prompts.txt director_workflow.json [output.json]
  
  Or put prompts.txt + Director*.json in same folder and just run the script.
"""

import json, re, sys, time, random, string, os

SECONDS_PER_CLIP = 5.0
FRAME_RATE = 24
GUIDE_STRENGTH = 0.99

def generate_id():
    ts = str(int(time.time() * 1000))
    chars = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
    return f"{ts}{chars}"

def parse_prompts(text):
    text = text.strip()
    if "---" in text:
        parts = re.split(r'\n?---+\n?', text)
        prompts = [p.strip().replace("|",",") for p in parts if p.strip()]
        if len(prompts) > 1: return prompts
    parts = re.split(r'(?:^|\n)###?\s*Clip\s*\d+[^\n]*\n', text, flags=re.IGNORECASE)
    parts = [p.strip().replace("|",",") for p in parts if p.strip()]
    if len(parts) > 1: return parts
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip().replace('\n',' ').replace("|",",") for p in paragraphs if p.strip()] or [text]

def populate(workflow_path, prompts_text, output_path=None):
    with open(workflow_path, encoding='utf-8') as f:
        wf = json.load(f)
    
    prompts = parse_prompts(prompts_text)
    clip_count = len(prompts)
    frames_per_clip = int(SECONDS_PER_CLIP * FRAME_RATE)
    total_frames = frames_per_clip * clip_count
    total_seconds = total_frames / FRAME_RATE
    
    print(f"\n  Clips: {clip_count} x {SECONDS_PER_CLIP}s = {total_seconds:.0f}s ({total_seconds/60:.1f} min)")
    for i, p in enumerate(prompts):
        print(f"  Clip {i+1:>2}: {p[:70]}{'...' if len(p)>70 else ''}")
    
    # Build timeline segments
    segments = []
    for i, prompt in enumerate(prompts):
        segments.append({"id": generate_id(), "start": i * frames_per_clip,
                        "length": frames_per_clip, "prompt": prompt, "type": "text"})
        time.sleep(0.002)
    
    timeline_data = json.dumps({"segments": segments, "audioSegments": []})
    local_prompts = " | ".join(prompts)
    segment_lengths = ",".join([str(frames_per_clip)] * clip_count)
    
    # Update LTXDirector
    for node in wf['nodes']:
        if node['type'] == 'LTXDirector':
            wv = node.get('widgets_values', [])
            while len(wv) < 17: wv.append('')
            wv[1] = total_frames
            wv[2] = total_seconds
            wv[3] = timeline_data
            wv[4] = local_prompts
            wv[5] = segment_lengths
            wv[6] = GUIDE_STRENGTH
            node['widgets_values'] = wv
            
            # Clear any broken input connections
            for inp in node.get('inputs', []):
                wname = inp.get('widget',{}).get('name','')
                if wname in ['local_prompts','segment_lengths','timeline_data',
                            'guide_strength','global_prompt','duration_frames','duration_seconds']:
                    inp['link'] = None
            break
    
    # Remove ScriptToDirector nodes (not needed with this approach)
    wf['nodes'] = [n for n in wf['nodes'] if n['type'] not in ('ScriptToDirector','ScriptPreview')]
    # Clean orphan links
    node_ids = {n['id'] for n in wf['nodes']}
    wf['links'] = [l for l in wf['links'] if l[1] in node_ids and l[3] in node_ids]
    
    if not output_path:
        output_path = os.path.splitext(workflow_path)[0] + f"_{clip_count}clips.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(wf, f, indent=2)
    
    print(f"\n  SAVED: {output_path}")
    print(f"  Open in ComfyUI → Hit Run!")
    return output_path

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        with open(sys.argv[1], encoding='utf-8') as f: pt = f.read()
        populate(sys.argv[2], pt, sys.argv[3] if len(sys.argv)>3 else None)
    elif len(sys.argv) == 2:
        populate(sys.argv[1], open('prompts.txt',encoding='utf-8').read() if os.path.exists('prompts.txt') else "test prompt 1\n---\ntest prompt 2")
    else:
        sd = os.path.dirname(os.path.abspath(__file__))
        pf = os.path.join(sd,'prompts.txt')
        wf = next((os.path.join(sd,f) for f in os.listdir(sd) if f.endswith('.json') and 'director' in f.lower() and 'clip' not in f.lower()), None)
        if not wf: print("Put prompts.txt + Director*.json in same folder"); sys.exit(1)
        pt = open(pf,encoding='utf-8').read() if os.path.exists(pf) else "test1\n---\ntest2"
        populate(wf, pt)
    input("\nPress Enter to exit...")
