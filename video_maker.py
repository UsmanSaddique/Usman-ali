"""
VideoMaker — AI-Powered YouTube Video Pipeline
================================================
Full web UI for managing YouTube channels, generating AI video prompts,
creating ComfyUI workflows, and launching renders.

Usage:  python video_maker.py
        Open http://localhost:5000
"""

import json, re, sys, time, random, string, os, subprocess, sqlite3, webbrowser, uuid, threading
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, g

app = Flask(__name__)

SCRIPT_DIR = Path(__file__).parent.resolve()
DB_PATH = SCRIPT_DIR / "videomaker.db"
DEFAULT_WORKFLOW = SCRIPT_DIR / "director.json"
DEFAULT_OUTPUT_DIR = Path("D:/assets_genrated/ltx_output")
COMFYUI_DIR = Path(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable")

ASPECT_RATIOS = {
    "16:9 (1024x576)": (1024, 576),
    "16:9 (768x432)": (768, 432),
    "9:16 (576x1024)": (576, 1024),
    "1:1 (576x576)": (576, 576),
    "1:1 (768x768)": (768, 768),
    "4:3 (768x576)": (768, 576),
    "21:9 (1024x432)": (1024, 432),
}

# ═══════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(str(DB_PATH))
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA journal_mode=WAL")
        g.db.execute("PRAGMA foreign_keys=ON")
    return g.db

@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db:
        db.close()

def init_db():
    db = sqlite3.connect(str(DB_PATH))
    db.executescript("""
        CREATE TABLE IF NOT EXISTS channels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            niche TEXT DEFAULT '',
            style_prompt TEXT DEFAULT '',
            target_audience TEXT DEFAULT '',
            voice_style TEXT DEFAULT '',
            language TEXT DEFAULT 'English',
            keywords TEXT DEFAULT '',
            for_kids INTEGER DEFAULT 0,
            negative_prompt TEXT DEFAULT 'blurry, low res, low quality, bad anatomy, worst quality, text, watermark, ugly, deformed, morphed, bad proportions, unnatural movement, jittery, still frame',
            global_prompt TEXT DEFAULT '',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_id INTEGER REFERENCES channels(id),
            topic TEXT DEFAULT '',
            title TEXT DEFAULT '',
            status TEXT DEFAULT 'draft',
            prompts_text TEXT DEFAULT '',
            config_json TEXT DEFAULT '{}',
            seo_json TEXT DEFAULT '{}',
            workflow_path TEXT DEFAULT '',
            output_path TEXT DEFAULT '',
            yt_url TEXT DEFAULT '',
            clip_count INTEGER DEFAULT 0,
            total_seconds REAL DEFAULT 0,
            resolution TEXT DEFAULT '',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            published_at TEXT
        );
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT DEFAULT ''
        );
    """)
    if not db.execute("SELECT COUNT(*) FROM channels").fetchone()[0]:
        db.execute("""INSERT INTO channels (name, niche, style_prompt, target_audience, voice_style, language, keywords, for_kids, negative_prompt)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
            "Little Fairy Dreams",
            "Kids Fantasy / Fairy Tales",
            "Colorful fantasy fairy scenes: glowing fairy gardens, unicorns, butterflies, enchanted castles, rainbow waterfalls, glowing mushrooms, sparkling forests. Pastel colors, dreamy atmosphere, children's storybook illustration style, soft glow effects. No human faces, no complex motion, no text in frames.",
            "Kids under 5, toddlers, girls",
            "Soft, slow, warm female",
            "English",
            "fairy tales for kids, toddler videos, calming videos for babies, unicorn videos, fairy garden, magical world for kids, bedtime videos toddlers, relaxing kids videos",
            1,
            "blurry, low res, low quality, bad anatomy, worst quality, text, watermark, ugly, deformed, morphed, bad proportions, unnatural movement, jittery, still frame, human face, realistic, scary, dark, horror",
        ))
    db.commit()
    db.close()

def get_setting(key, default=""):
    try:
        row = get_db().execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default
    except Exception:
        return default

def set_setting(key, value):
    get_db().execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, str(value)))
    get_db().commit()


# ═══════════════════════════════════════════════════════════════════
# WORKFLOW GENERATION (from original)
# ═══════════════════════════════════════════════════════════════════

def generate_id():
    ts = str(int(time.time() * 1000))
    chars = "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
    return f"{ts}{chars}"

def parse_prompts(text):
    text = text.strip()
    if "---" in text:
        parts = re.split(r"\n?---+\n?", text)
        prompts = [p.strip().replace("|", ",") for p in parts if p.strip()]
        if len(prompts) > 1:
            return prompts
    parts = re.split(r"(?:^|\n)###?\s*Clip\s*\d+[^\n]*\n", text, flags=re.IGNORECASE)
    parts = [p.strip().replace("|", ",") for p in parts if p.strip()]
    if len(parts) > 1:
        return parts
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip().replace("\n", " ").replace("|", ",") for p in paragraphs if p.strip()] or [text]

def populate(workflow_path, prompts_text, config):
    with open(workflow_path, encoding="utf-8") as f:
        wf = json.load(f)
    spc = config["seconds_per_clip"]
    fps = config["frame_rate"]
    prompts = [p.strip() for p in prompts_text.split("---") if p.strip()]
    if not prompts:
        prompts = [p.strip() for p in prompts_text.split("\n") if p.strip()]
    clip_count = len(prompts)
    gs = config["guide_strength"]
    w, h = config["width"], config["height"]
    neg = config.get("negative_prompt", "")
    gp = config.get("global_prompt", "")
    prefix = config.get("filename_prefix", "%date:yyyy-MM-dd%/LTX-2_3")
    
    chunk_size = 4
    chunks = [prompts[i:i + chunk_size] for i in range(0, len(prompts), chunk_size)]
    
    workflows = []
    import copy
    import time
    for chunk_idx, chunk_prompts in enumerate(chunks):
        cwf = copy.deepcopy(wf)
        c_clip_count = len(chunk_prompts)
        c_fpc = int(spc * fps)
        c_total_frames = c_fpc * c_clip_count
        c_total_seconds = c_total_frames / fps
        segments = []
        for i, prompt in enumerate(chunk_prompts):
            segments.append({"id": generate_id(), "start": i * c_fpc, "length": c_fpc, "prompt": prompt, "type": "text"})
            time.sleep(0.002)
        timeline = json.dumps({"segments": segments, "audioSegments": []})
        local_prompts = " | ".join(chunk_prompts)
        seg_lengths = ",".join([str(c_fpc)] * c_clip_count)
        
        c_prefix = f"{prefix}_chunk{chunk_idx}"
        
        for node in cwf["nodes"]:
            if node["type"] == "ScriptToDirector":
                wv = node.get("widgets_values", [])
                while len(wv) < 6: wv.append("")
                wv[0] = "\n---\n".join(chunk_prompts)
                wv[1], wv[2], wv[3], wv[4], wv[5] = spc, fps, gs, neg, gp
                node["widgets_values"] = wv
            if node["type"] == "LTXDirector":
                wv = node.get("widgets_values", [])
                while len(wv) < 17: wv.append("")
                wv[9], wv[11], wv[12] = fps, w, h
                node["widgets_values"] = wv
            if node["type"] == "SaveVideo":
                wv = node.get("widgets_values", [])
                if wv:
                    wv[0] = c_prefix
                    node["widgets_values"] = wv
        node_ids = {n["id"] for n in cwf["nodes"]}
        cwf["links"] = [l for l in cwf["links"] if l[1] in node_ids and l[3] in node_ids]
        workflows.append(cwf)
        
    return workflows, {"clip_count": clip_count, "total_frames": int(spc * fps * clip_count), "total_seconds": spc * clip_count, "resolution": f"{w}x{h}"}


# ═══════════════════════════════════════════════════════════════════
# AI ENGINE INTEGRATION
# ═══════════════════════════════════════════════════════════════════

def get_ai_engine():
    from ai_engine import AIEngine
    provider = get_setting("ai_provider", "claude")
    return AIEngine(
        provider=provider,
        claude_key=get_setting("claude_api_key"),
        gemini_key=get_setting("gemini_api_key"),
    )

def get_channel_history(channel_id):
    rows = get_db().execute(
        "SELECT topic, title FROM videos WHERE channel_id=? ORDER BY created_at DESC LIMIT 30",
        (channel_id,),
    ).fetchall()
    return [r["topic"] or r["title"] or "Untitled" for r in rows]


# ═══════════════════════════════════════════════════════════════════
# COMFYUI INTEGRATION
# ═══════════════════════════════════════════════════════════════════

COMFYUI_URL = "http://127.0.0.1:8188"

def comfyui_running():
    try:
        import urllib.request
        urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=2)
        return True
    except Exception:
        return False

def ensure_ffmpeg():
    ffmpeg_path = Path("ffmpeg.exe")
    if ffmpeg_path.exists():
        return str(ffmpeg_path.absolute())
    import urllib.request
    import zipfile
    print("Downloading FFmpeg...")
    url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
    zip_path = Path("ffmpeg.zip")
    if not zip_path.exists():
        urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as z:
        for info in z.infolist():
            if info.filename.endswith("ffmpeg.exe"):
                info.filename = "ffmpeg.exe"
                z.extract(info, ".")
                break
    try:
        zip_path.unlink()
    except Exception:
        pass
    return str(ffmpeg_path.absolute())

def start_comfyui():
    if comfyui_running():
        return True, "ComfyUI already running"
    bat = COMFYUI_DIR / "run_nvidia_gpu.bat"
    if not bat.exists():
        return False, f"ComfyUI not found at {bat}"
    cwd_bak = os.getcwd()
    os.chdir(str(COMFYUI_DIR))
    try:
        os.startfile(bat.name)
    finally:
        os.chdir(cwd_bak)
    for _ in range(120):
        if comfyui_running():
            return True, "ComfyUI started"
        time.sleep(1)
    return False, "ComfyUI failed to start within 120s"

def fetch_object_info():
    import urllib.request
    resp = urllib.request.urlopen(f"{COMFYUI_URL}/object_info", timeout=10)
    return json.loads(resp.read())

def convert_webui_to_api(workflow):
    """Convert ComfyUI web UI format (nodes+links) to API format (flat dict).

    The tricky part: widgets_values is a flat positional array that includes
    hidden control widgets (e.g. control_after_generate after seed inputs,
    image upload widgets after image paths). We must skip those to keep
    the index aligned with actual input names from object_info.
    """
    obj_info = fetch_object_info()

    link_map = {}
    for link in workflow.get("links", []):
        if not link: continue
        link_id = link[0]
        src_id, src_slot = link[1], link[2]
        link_map[link_id] = [str(src_id), src_slot]

    # Flatten subgraphs
    subgraphs_def = {sg["id"]: sg for sg in workflow.get("definitions", {}).get("subgraphs", [])}
    flat_nodes = []
    
    for node in workflow.get("nodes", []):
        ntype = node.get("type", "")
        if ntype in subgraphs_def:
            sg = subgraphs_def[ntype]
            parent_id = str(node["id"])
            
            # Map inner links
            for link in sg.get("links", []):
                if not link: continue
                if isinstance(link, dict):
                    link_id = link.get("id")
                    src_id = link.get("origin_id")
                    src_slot = link.get("origin_slot")
                else:
                    link_id = link[0]
                    src_id, src_slot = link[1], link[2]
                link_map[link_id] = [f"{parent_id}:{src_id}", src_slot]
                
            # Map external inputs to inner links
            ext_input_links = {}
            for inp in node.get("inputs", []):
                if inp.get("link") is not None:
                    ext_input_links[inp["name"]] = inp["link"]
                    
            for sg_inp in sg.get("inputs", []):
                name = sg_inp.get("name")
                if name in ext_input_links:
                    ext_lid = ext_input_links[name]
                    if ext_lid in link_map:
                        ext_src = link_map[ext_lid]
                        for inner_lid in sg_inp.get("linkIds", []):
                            link_map[inner_lid] = ext_src
                            
            # Map inner outputs to external links
            ext_output_links = {}
            for out in node.get("outputs", []):
                name = out.get("name")
                links = out.get("links", [])
                if links is not None:
                    ext_output_links[name] = links
                    
            for sg_out in sg.get("outputs", []):
                name = sg_out.get("name")
                inner_links = sg_out.get("linkIds", [])
                if inner_links and inner_links[0] in link_map:
                    inner_src = link_map[inner_links[0]]
                    if name in ext_output_links:
                        for ext_lid in ext_output_links[name]:
                            link_map[ext_lid] = inner_src
                            
            # Add inner nodes to flat_nodes
            for inner_node in sg.get("nodes", []):
                inner_node_copy = dict(inner_node)
                inner_node_copy["id"] = f"{parent_id}:{inner_node['id']}"
                flat_nodes.append(inner_node_copy)
        else:
            flat_nodes.append(node)

    api = {}
    for node in flat_nodes:
        ntype = node.get("type", "")
        if ntype in ("Reroute", "Note", "PrimitiveNode"):
            continue
        if ntype not in obj_info:
            continue
        info = obj_info[ntype]
        inp_def = info.get("input", {})
        required = inp_def.get("required", {})
        optional = inp_def.get("optional", {})

        input_links = {}
        for inp in node.get("inputs", []):
            if inp.get("link") is not None:
                input_links[inp["name"]] = inp["link"]

        api_inputs = {}
        wv = node.get("widgets_values", [])
        wi = 0

        def is_widget_type(spec):
            t = spec[0]
            if isinstance(t, list):
                return True
            return t in ("STRING", "INT", "FLOAT", "BOOLEAN", "COMBO")

        def has_extra_widget(spec):
            """Check if this input has a hidden trailing widget in widgets_values."""
            if len(spec) < 2 or not isinstance(spec[1], dict):
                return False
            cfg = spec[1]
            if cfg.get("control_after_generate"):
                return True
            if cfg.get("image_upload"):
                return True
            return False

        all_inputs = list(required.items()) + list(optional.items())
        for name, spec in all_inputs:
            if name in input_links:
                lid = input_links[name]
                if lid in link_map:
                    api_inputs[name] = link_map[lid]
                if is_widget_type(spec):
                    wi += 1
                    if has_extra_widget(spec):
                        wi += 1
            elif is_widget_type(spec):
                if wi < len(wv):
                    val = wv[wi]
                    if val is not None:
                        api_inputs[name] = val
                wi += 1
                if has_extra_widget(spec):
                    wi += 1

        api[str(node["id"])] = {
            "class_type": ntype,
            "inputs": api_inputs,
        }
    return api

def queue_prompt(api_workflow, workflow_webui=None):
    """Queue a prompt on ComfyUI. Returns prompt_id."""
    import urllib.request
    payload = {"prompt": api_workflow}
    if workflow_webui:
        payload["extra_data"] = {"extra_pnginfo": {"workflow": workflow_webui}}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt", data=data,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=15)
    result = json.loads(resp.read())
    return result

def get_queue_status():
    import urllib.request
    resp = urllib.request.urlopen(f"{COMFYUI_URL}/queue", timeout=5)
    return json.loads(resp.read())


# ═══════════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/config")
def get_config():
    return jsonify({
        "workflow_path": str(DEFAULT_WORKFLOW).replace("\\", "/"),
        "output_dir": get_setting("output_dir", str(DEFAULT_OUTPUT_DIR)),
    })

# --- Channels ---
@app.route("/api/channels", methods=["GET"])
def list_channels():
    rows = get_db().execute("""
        SELECT c.*, (SELECT COUNT(*) FROM videos WHERE channel_id=c.id) as video_count,
               (SELECT COUNT(*) FROM videos WHERE channel_id=c.id AND status='published') as published_count
        FROM channels c ORDER BY c.name
    """).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/api/channels", methods=["POST"])
def create_channel():
    d = request.get_json()
    db = get_db()
    db.execute("""INSERT INTO channels (name,niche,style_prompt,target_audience,voice_style,language,keywords,for_kids,negative_prompt,global_prompt)
        VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (d["name"], d.get("niche",""), d.get("style_prompt",""), d.get("target_audience",""),
         d.get("voice_style",""), d.get("language","English"), d.get("keywords",""),
         1 if d.get("for_kids") else 0, d.get("negative_prompt",""), d.get("global_prompt","")))
    db.commit()
    return jsonify({"id": db.execute("SELECT last_insert_rowid()").fetchone()[0]})

@app.route("/api/channels/<int:cid>", methods=["PUT"])
def update_channel(cid):
    d = request.get_json()
    db = get_db()
    db.execute("""UPDATE channels SET name=?,niche=?,style_prompt=?,target_audience=?,voice_style=?,language=?,keywords=?,for_kids=?,negative_prompt=?,global_prompt=? WHERE id=?""",
        (d["name"], d.get("niche",""), d.get("style_prompt",""), d.get("target_audience",""),
         d.get("voice_style",""), d.get("language","English"), d.get("keywords",""),
         1 if d.get("for_kids") else 0, d.get("negative_prompt",""), d.get("global_prompt",""), cid))
    db.commit()
    return jsonify({"ok": True})

@app.route("/api/channels/<int:cid>", methods=["DELETE"])
def delete_channel(cid):
    db = get_db()
    db.execute("DELETE FROM videos WHERE channel_id=?", (cid,))
    db.execute("DELETE FROM channels WHERE id=?", (cid,))
    db.commit()
    return jsonify({"ok": True})

# --- Videos ---
@app.route("/api/videos", methods=["GET"])
def list_videos():
    cid = request.args.get("channel_id")
    status = request.args.get("status")
    q = "SELECT v.*, c.name as channel_name FROM videos v JOIN channels c ON v.channel_id=c.id WHERE 1=1"
    params = []
    if cid:
        q += " AND v.channel_id=?"
        params.append(cid)
    if status:
        q += " AND v.status=?"
        params.append(status)
    q += " ORDER BY v.created_at DESC"
    return jsonify([dict(r) for r in get_db().execute(q, params).fetchall()])

@app.route("/api/videos/<int:vid>", methods=["GET"])
def get_video(vid):
    r = get_db().execute("SELECT v.*, c.name as channel_name FROM videos v JOIN channels c ON v.channel_id=c.id WHERE v.id=?", (vid,)).fetchone()
    return jsonify(dict(r)) if r else ("", 404)

@app.route("/api/videos/<int:vid>", methods=["PUT"])
def update_video(vid):
    d = request.get_json()
    db = get_db()
    sets, vals = [], []
    for col in ("title", "status", "yt_url", "topic", "prompts_text", "seo_json"):
        if col in d:
            sets.append(f"{col}=?")
            vals.append(d[col] if not isinstance(d[col], dict) else json.dumps(d[col]))
    if d.get("status") == "published":
        sets.append("published_at=?")
        vals.append(datetime.now().isoformat())
    if sets:
        vals.append(vid)
        db.execute(f"UPDATE videos SET {','.join(sets)} WHERE id=?", vals)
        db.commit()
    return jsonify({"ok": True})

@app.route("/api/videos/<int:vid>", methods=["DELETE"])
def delete_video(vid):
    get_db().execute("DELETE FROM videos WHERE id=?", (vid,))
    get_db().commit()
    return jsonify({"ok": True})

@app.route("/api/videos/<int:vid>/stitch", methods=["POST"])
def stitch_video(vid):
    video = get_db().execute("SELECT * FROM videos WHERE id=?", (vid,)).fetchone()
    if not video or not video["workflow_path"]:
        return jsonify({"error": "Video or master workflow not found"}), 404
        
    master_path = Path(video["workflow_path"])
    if not master_path.exists():
        return jsonify({"error": "Master workflow file not found"}), 404
        
    with open(master_path, encoding="utf-8") as f:
        master_data = json.load(f)
        
    if not master_data.get("is_master"):
        return jsonify({"error": "Not a chunked master workflow"}), 400
        
    config = json.loads(video["config_json"])
    prefix = config.get("filename_prefix", "%date:yyyy-MM-dd%/LTX-2_3")
    base_prefix = Path(prefix).name
    
    import glob
    comfy_out = COMFYUI_DIR / "output"
    chunk_paths = []
    
    for i in range(len(master_data["chunks"])):
        pattern = str(comfy_out / f"**/*{base_prefix}_chunk{i}*.mp4")
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            return jsonify({"error": f"Output for chunk {i} not found. Ensure ComfyUI finished rendering."}), 404
        matches.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        chunk_paths.append(matches[0])
        
    ffmpeg_exe = ensure_ffmpeg()
    concat_file = Path("concat.txt")
    with open(concat_file, "w", encoding="utf-8") as f:
        for p in chunk_paths:
            f.write(f"file '{Path(p).absolute().as_posix()}'\n")
            
    out_video = Path("static") / "videos" / f"stitched_{vid}.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [ffmpeg_exe, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(out_video)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        return jsonify({"error": "FFmpeg failed", "details": res.stderr.decode()}), 500
        
    db = get_db()
    db.execute("UPDATE videos SET status='stitched' WHERE id=?", (vid,))
    db.commit()
    
    return jsonify({"ok": True, "video_url": f"/static/videos/{out_video.name}"})

# --- AI ---
@app.route("/api/ai/suggest-topics", methods=["POST"])
def ai_suggest_topics():
    d = request.get_json()
    ch = get_db().execute("SELECT * FROM channels WHERE id=?", (d["channel_id"],)).fetchone()
    if not ch:
        return jsonify({"error": "Channel not found"}), 404
    try:
        engine = get_ai_engine()
        history = get_channel_history(ch["id"])
        result = engine.suggest_topics(dict(ch), history)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/ai/generate-prompts", methods=["POST"])
def ai_generate_prompts():
    d = request.get_json()
    ch = get_db().execute("SELECT * FROM channels WHERE id=?", (d["channel_id"],)).fetchone()
    if not ch:
        return jsonify({"error": "Channel not found"}), 404
    try:
        engine = get_ai_engine()
        history = get_channel_history(ch["id"])
        result = engine.generate_prompts(
            dict(ch), history, d.get("topic", ""),
            num_clips=d.get("num_clips", 6),
            seconds_per_clip=d.get("seconds_per_clip", 5),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/ai/generate-seo", methods=["POST"])
def ai_generate_seo():
    d = request.get_json()
    ch = get_db().execute("SELECT * FROM channels WHERE id=?", (d["channel_id"],)).fetchone()
    if not ch:
        return jsonify({"error": "Channel not found"}), 404
    try:
        engine = get_ai_engine()
        result = engine.generate_seo(dict(ch), d.get("prompts_text", ""))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Settings ---
@app.route("/api/settings", methods=["GET"])
def get_settings():
    from ai_engine import get_available_providers
    return jsonify({
        "ai_provider": get_setting("ai_provider", "claude"),
        "claude_api_key_set": bool(get_setting("claude_api_key")),
        "gemini_api_key_set": bool(get_setting("gemini_api_key")),
        "comfyui_dir": get_setting("comfyui_dir", str(COMFYUI_DIR)),
        "output_dir": get_setting("output_dir", str(DEFAULT_OUTPUT_DIR)),
        "available_providers": get_available_providers(),
    })

@app.route("/api/settings", methods=["POST"])
def save_settings():
    d = request.get_json()
    for key in ("ai_provider", "claude_api_key", "gemini_api_key", "comfyui_dir", "output_dir"):
        if key in d:
            set_setting(key, d[key])
    return jsonify({"ok": True})

# --- Generate Workflow ---
@app.route("/api/generate", methods=["POST"])
def generate():
    d = request.get_json()
    wf_path = Path(d.get("workflow_path", str(DEFAULT_WORKFLOW)))
    if not wf_path.exists():
        return jsonify({"error": f"Workflow template not found: {wf_path}"}), 400
    prompts_text = d.get("prompts_text", "").strip()
    if not prompts_text:
        return jsonify({"error": "No prompts"}), 400
    out_dir = Path(d.get("output_dir", get_setting("output_dir", str(DEFAULT_OUTPUT_DIR))))
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "seconds_per_clip": d.get("seconds_per_clip", 5),
        "frame_rate": d.get("frame_rate", 24),
        "guide_strength": d.get("guide_strength", 0.99),
        "width": d.get("width", 1024),
        "height": d.get("height", 576),
        "negative_prompt": d.get("negative_prompt", ""),
        "global_prompt": d.get("global_prompt", ""),
        "filename_prefix": d.get("filename_prefix", "%date:yyyy-MM-dd%/LTX-2_3"),
    }
    try:
        workflows, info = populate(wf_path, prompts_text, config)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    chunk_files = []
    for i, wf in enumerate(workflows):
        chunk_file = out_dir / f"director_{info['clip_count']}clips_{ts}_chunk{i}.json"
        with open(chunk_file, "w", encoding="utf-8") as f:
            json.dump(wf, f, indent=2)
        chunk_files.append(str(chunk_file))
        
    out_file = out_dir / f"director_{info['clip_count']}clips_{ts}_master.json"
    master_data = {
        "is_master": True,
        "chunks": chunk_files
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(master_data, f, indent=2)
    # save video record
    channel_id = d.get("channel_id")
    db = get_db()
    if channel_id:
        cur = db.execute("""INSERT INTO videos (channel_id, topic, title, status, prompts_text, config_json, workflow_path, clip_count, total_seconds, resolution)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (channel_id, d.get("topic",""), d.get("title",""), "generated", prompts_text,
             json.dumps(config), str(out_file), info["clip_count"], info["total_seconds"], info["resolution"]))
        db.commit()
        video_id = cur.lastrowid
    else:
        video_id = None
    return jsonify({"output_file": str(out_file), "video_id": video_id, **info})

# --- ComfyUI ---
@app.route("/api/comfyui/status")
def comfyui_status():
    return jsonify({"running": comfyui_running()})

@app.route("/api/comfyui/start", methods=["POST"])
def comfyui_start_route():
    def _start():
        ok, msg = start_comfyui()
        return ok, msg
    ok, msg = _start()
    if ok:
        webbrowser.open("http://127.0.0.1:8188")
    return jsonify({"ok": ok, "message": msg})

@app.route("/api/comfyui/run", methods=["POST"])
def comfyui_run():
    import urllib.request
    import urllib.error
    d = request.get_json()
    wf_path = d.get("workflow_path", "")
    if not wf_path or not Path(wf_path).exists():
        return jsonify({"error": "Workflow file not found"}), 400
    ok, msg = start_comfyui() if not comfyui_running() else (True, "Already running")
    if not ok:
        return jsonify({"error": msg}), 500
    with open(wf_path, encoding="utf-8") as f:
        wf_data = json.load(f)
        
    try:
        prompt_ids = []
        if wf_data.get("is_master"):
            for chunk_path in wf_data["chunks"]:
                with open(chunk_path, encoding="utf-8") as cf:
                    chunk_wf = json.load(cf)
                api_wf = convert_webui_to_api(chunk_wf)
                result = queue_prompt(api_wf, workflow_webui=chunk_wf)
                prompt_ids.append(result.get("prompt_id", ""))
            msg = f"{len(prompt_ids)} chunks queued in ComfyUI!"
            prompt_id = prompt_ids[0] if prompt_ids else ""
        else:
            api_wf = convert_webui_to_api(wf_data)
            result = queue_prompt(api_wf, workflow_webui=wf_data)
            prompt_id = result.get("prompt_id", "")
            msg = f"Workflow queued in ComfyUI! Prompt ID: {prompt_id}"
            
        webbrowser.open(COMFYUI_URL)
        return jsonify({"ok": True, "queued": True, "prompt_id": prompt_id,
                        "message": msg})
    except urllib.error.HTTPError as e:
        err_data = e.read().decode()
        try:
            err_json = json.loads(err_data)
            err_msg = err_json.get("error", {}).get("message", "")
            err_details = err_json.get("error", {}).get("details", "")
            err_str = f"{err_msg}: {err_details}"
            if "node_errors" in err_json.get("error", {}):
                err_str += " | " + str(err_json["error"]["node_errors"])
        except Exception:
            err_str = err_data
        webbrowser.open(COMFYUI_URL)
        return jsonify({"ok": False, "queued": False,
                        "message": f"ComfyUI Validation Error: {err_str}",
                        "workflow_path": wf_path}), 400
    except Exception as e:
        webbrowser.open(COMFYUI_URL)
        return jsonify({"ok": False, "queued": False,
                        "message": f"ComfyUI running but auto-queue failed: {e}. Load workflow manually.",
                        "workflow_path": wf_path})

@app.route("/api/comfyui/queue-status")
def comfyui_queue_status():
    try:
        qs = get_queue_status()
        running = len(qs.get("queue_running", []))
        pending = len(qs.get("queue_pending", []))
        return jsonify({"running": running, "pending": pending, "total": running + pending})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════
# HTML TEMPLATE
# ═══════════════════════════════════════════════════════════════════

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VideoMaker</title>
<style>
:root{--bg:#0f1117;--s1:#1a1d27;--s2:#242836;--bd:#2e3347;--tx:#e4e6ed;--tx2:#8b8fa3;--ac:#6c5ce7;--ac2:#a29bfe;--ok:#00b894;--warn:#fdcb6e;--err:#ff7675;--blue:#0984e3}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--tx);min-height:100vh}
.nav{background:var(--s1);border-bottom:1px solid var(--bd);display:flex;align-items:center;padding:0 24px;height:56px;gap:8px}
.nav .logo{font-size:20px;font-weight:700;margin-right:24px}
.nav .logo span{color:var(--ac2);font-weight:400;font-size:14px;margin-left:8px}
.tab-btn{background:none;border:none;color:var(--tx2);font:inherit;font-size:14px;padding:16px 16px;cursor:pointer;border-bottom:2px solid transparent;transition:.2s}
.tab-btn:hover{color:var(--tx)}
.tab-btn.active{color:var(--ac2);border-bottom-color:var(--ac)}
.nav-right{margin-left:auto;display:flex;gap:12px;align-items:center}
.provider-badge{font-size:11px;padding:4px 10px;border-radius:12px;background:var(--s2);border:1px solid var(--bd)}
.comfy-dot{width:8px;height:8px;border-radius:50%;display:inline-block}
.comfy-dot.on{background:var(--ok)}.comfy-dot.off{background:var(--err)}
.page{display:none;padding:24px;max-width:1400px;margin:0 auto}.page.active{display:block}
.panel{background:var(--s1);border:1px solid var(--bd);border-radius:12px;padding:20px;margin-bottom:16px}
.panel-title{font-size:15px;font-weight:600;margin-bottom:16px;display:flex;align-items:center;gap:8px}
.badge{font-size:11px;padding:2px 8px;border-radius:10px;font-weight:500}
.badge-purple{background:var(--ac);color:#fff}.badge-green{background:var(--ok);color:#fff}
.badge-yellow{background:var(--warn);color:#000}.badge-red{background:var(--err);color:#fff}
.badge-blue{background:var(--blue);color:#fff}
label{display:block;font-size:13px;color:var(--tx2);margin-bottom:5px;font-weight:500}
input[type=text],input[type=number],input[type=password],select,textarea{width:100%;background:var(--s2);border:1px solid var(--bd);border-radius:8px;padding:9px 12px;color:var(--tx);font:14px/1.5 inherit;transition:border-color .2s}
input:focus,select:focus,textarea:focus{outline:none;border-color:var(--ac)}
textarea{resize:vertical;font-family:'Cascadia Code','Consolas',monospace;font-size:13px}
.row{display:grid;gap:12px;margin-bottom:12px}
.row-2{grid-template-columns:1fr 1fr}.row-3{grid-template-columns:1fr 1fr 1fr}.row-4{grid-template-columns:1fr 1fr 1fr 1fr}
.fg{margin-bottom:12px}
.btn{padding:10px 20px;border:none;border-radius:8px;font:600 14px/1 inherit;cursor:pointer;transition:.2s;white-space:nowrap}
.btn-p{background:var(--ac);color:#fff}.btn-p:hover{background:var(--ac2)}
.btn-s{background:var(--s2);color:var(--tx);border:1px solid var(--bd)}.btn-s:hover{background:var(--bd)}
.btn-ok{background:var(--ok);color:#fff}.btn-ok:hover{opacity:.9}
.btn-err{background:var(--err);color:#fff}
.btn-sm{padding:6px 14px;font-size:12px}
.btn-lg{padding:14px 32px;font-size:16px}
.btn:disabled{opacity:.5;cursor:not-allowed}
.btn-row{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px}
.card{background:var(--s2);border:1px solid var(--bd);border-radius:10px;padding:16px;cursor:pointer;transition:.2s}
.card:hover{border-color:var(--ac);transform:translateY(-2px)}
.card h3{font-size:15px;margin-bottom:6px}.card .meta{font-size:12px;color:var(--tx2)}
.status-pill{font-size:11px;padding:3px 10px;border-radius:12px;font-weight:600;text-transform:uppercase;letter-spacing:.3px}
.st-draft{background:#2d3436;color:var(--tx2)}.st-generated{background:#0a3d62;color:#74b9ff}
.st-rendering{background:#6c3483;color:#d2b4de}.st-rendered{background:#0e6655;color:#82e0aa}.st-published{background:#1e8449;color:#a9dfbf}
.tbl{width:100%;border-collapse:collapse}
.tbl th{text-align:left;font-size:12px;color:var(--tx2);padding:8px 12px;border-bottom:1px solid var(--bd);text-transform:uppercase;letter-spacing:.5px}
.tbl td{padding:10px 12px;border-bottom:1px solid var(--bd);font-size:13px;vertical-align:middle}
.tbl tr:hover td{background:var(--s2)}
.ar-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:10px}
.ar-btn{background:var(--s2);border:2px solid var(--bd);border-radius:6px;padding:6px 4px;color:var(--tx2);font-size:11px;cursor:pointer;text-align:center;transition:.2s}
.ar-btn:hover{border-color:var(--ac);color:var(--tx)}.ar-btn.active{border-color:var(--ac);background:rgba(108,92,231,.15);color:var(--ac2)}
.idea-card{background:var(--s2);border:1px solid var(--bd);border-radius:8px;padding:12px;margin-bottom:8px;cursor:pointer;transition:.2s}
.idea-card:hover{border-color:var(--ac)}
.idea-card h4{font-size:14px;margin-bottom:4px;color:var(--ac2)}.idea-card p{font-size:12px;color:var(--tx2)}
.seo-title-opt{background:var(--s2);border:1px solid var(--bd);border-radius:8px;padding:10px 14px;margin-bottom:6px;cursor:pointer;font-size:14px;transition:.2s}
.seo-title-opt:hover,.seo-title-opt.selected{border-color:var(--ac);background:rgba(108,92,231,.1)}
.tag-cloud{display:flex;flex-wrap:wrap;gap:6px}.tag-chip{background:var(--s2);border:1px solid var(--bd);border-radius:14px;padding:4px 12px;font-size:12px;color:var(--tx2)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(100px,1fr));gap:10px}
.stat{background:var(--s2);border-radius:8px;padding:10px;text-align:center}
.stat .v{font-size:22px;font-weight:700;color:var(--ac2)}.stat .l{font-size:10px;color:var(--tx2);text-transform:uppercase;letter-spacing:.5px;margin-top:2px}
.loading{display:inline-block;width:16px;height:16px;border:2px solid var(--bd);border-top-color:var(--ac);border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.toast{position:fixed;bottom:24px;right:24px;padding:14px 20px;border-radius:10px;font-size:14px;z-index:9999;max-width:500px;animation:slideIn .3s}
.toast-ok{background:#27ae60;color:#fff}.toast-err{background:#e74c3c;color:#fff}.toast-info{background:var(--s1);color:var(--tx);border:1px solid var(--bd)}
@keyframes slideIn{from{transform:translateY(20px);opacity:0}to{transform:translateY(0);opacity:1}}
.hidden{display:none!important}
.modal-bg{position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:100;display:flex;align-items:center;justify-content:center}
.modal{background:var(--s1);border:1px solid var(--bd);border-radius:14px;padding:24px;width:600px;max-width:90vw;max-height:80vh;overflow-y:auto}
.modal h2{font-size:18px;margin-bottom:16px}
.split{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:900px){.split{grid-template-columns:1fr}.row-3,.row-4{grid-template-columns:1fr 1fr}}
</style>
</head>
<body>

<!-- NAV -->
<div class="nav">
  <div class="logo">VideoMaker<span>AI Pipeline</span></div>
  <button class="tab-btn active" onclick="showTab('dashboard')">Dashboard</button>
  <button class="tab-btn" onclick="showTab('create')">Create Video</button>
  <button class="tab-btn" onclick="showTab('videos')">Video Library</button>
  <button class="tab-btn" onclick="showTab('settings')">Settings</button>
  <div class="nav-right">
    <span class="provider-badge" id="providerBadge">AI: —</span>
    <span title="ComfyUI status"><span class="comfy-dot off" id="comfyDot"></span> ComfyUI</span>
  </div>
</div>

<!-- ═══ DASHBOARD ═══ -->
<div class="page active" id="page-dashboard">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px">
    <h2>Channels</h2>
    <button class="btn btn-p" onclick="openChannelModal()">+ New Channel</button>
  </div>
  <div class="cards" id="channelCards"></div>
  <h2 style="margin:28px 0 16px">Recent Videos</h2>
  <div id="recentVideos"></div>
</div>

<!-- ═══ CREATE VIDEO ═══ -->
<div class="page" id="page-create">
  <div class="split">
    <!-- LEFT -->
    <div>
      <!-- Channel + Topic -->
      <div class="panel">
        <div class="panel-title">Channel & Topic</div>
        <div class="fg">
          <label>Channel</label>
          <select id="cChannel" onchange="onChannelChange()"><option value="">Select channel...</option></select>
        </div>
        <div class="fg">
          <label>Topic / Video Idea</label>
          <div style="display:flex;gap:8px">
            <input type="text" id="cTopic" placeholder="Enter topic or click AI Suggest...">
            <button class="btn btn-p btn-sm" onclick="aiSuggestTopics()" id="btnSuggest">AI Suggest</button>
          </div>
        </div>
        <div id="topicIdeas" class="hidden"></div>
      </div>

      <!-- Aspect Ratio -->
      <div class="panel">
        <div class="panel-title">Resolution</div>
        <div class="ar-grid" id="arGrid"></div>
        <div class="row row-2 hidden" id="customRes">
          <div class="fg"><label>Width</label><input type="number" id="cW" value="1024" min="64" max="2048" step="32"></div>
          <div class="fg"><label>Height</label><input type="number" id="cH" value="576" min="64" max="2048" step="32"></div>
        </div>
      </div>

      <!-- Gen Settings -->
      <div class="panel">
        <div class="panel-title">Generation Settings</div>
        <div class="row row-3">
          <div class="fg"><label>Sec / Clip</label><input type="number" id="cSpc" value="5" min="1" max="30" step="0.5"></div>
          <div class="fg"><label>FPS</label><input type="number" id="cFps" value="24" min="8" max="60"></div>
          <div class="fg"><label>Guide Strength</label><input type="number" id="cGs" value="0.99" min="0" max="1" step="0.01"></div>
        </div>
        <div class="fg"><label>Negative Prompt</label><input type="text" id="cNeg" value="blurry, low res, low quality, bad anatomy, worst quality, text, watermark, ugly, deformed, morphed, bad proportions, unnatural movement, jittery, still frame"></div>
        <div class="fg"><label>Global Prompt</label><input type="text" id="cGlobal" placeholder="Optional..."></div>
        <div class="row row-2">
          <div class="fg"><label>Output Dir</label><input type="text" id="cOutDir" value="D:/assets_genrated/ltx_output"></div>
          <div class="fg"><label>Filename Prefix</label><input type="text" id="cPrefix" value="%date:yyyy-MM-dd%/LTX-2_3"></div>
        </div>
      </div>

      <!-- SEO -->
      <div class="panel">
        <div class="panel-title">SEO Metadata <span class="badge badge-blue">AI</span></div>
        <button class="btn btn-p btn-sm" onclick="aiGenerateSeo()" id="btnSeo">Generate SEO</button>
        <div id="seoResults" class="hidden" style="margin-top:12px">
          <label>Pick a Title</label>
          <div id="seoTitles"></div>
          <div class="fg" style="margin-top:10px"><label>Description</label><textarea id="seoDesc" rows="6"></textarea></div>
          <div class="fg"><label>Tags</label><div class="tag-cloud" id="seoTags"></div></div>
        </div>
      </div>
    </div>

    <!-- RIGHT -->
    <div>
      <!-- Prompts -->
      <div class="panel" style="display:flex;flex-direction:column">
        <div class="panel-title">Clip Prompts <span class="badge badge-purple" id="clipBadge">0</span></div>
        <div class="btn-row" style="margin-top:0;margin-bottom:10px">
          <button class="btn btn-p btn-sm" onclick="aiGeneratePrompts()" id="btnGenPrompts">AI Generate</button>
          <button class="btn btn-s btn-sm" onclick="loadFairyTemplate()">Fairy Template</button>
          <button class="btn btn-s btn-sm" onclick="document.getElementById('cPrompts').value='';updateStats()">Clear</button>
        </div>
        <textarea id="cPrompts" rows="22" placeholder="Enter clip prompts separated by ---" oninput="updateStats()"></textarea>
      </div>

      <!-- Stats + Actions -->
      <div class="panel">
        <div class="stats" style="margin-bottom:14px">
          <div class="stat"><div class="v" id="stClips">0</div><div class="l">Clips</div></div>
          <div class="stat"><div class="v" id="stDur">0s</div><div class="l">Duration</div></div>
          <div class="stat"><div class="v" id="stFrames">0</div><div class="l">Frames</div></div>
          <div class="stat"><div class="v" id="stRes">—</div><div class="l">Resolution</div></div>
        </div>
        <div class="btn-row" style="justify-content:flex-end">
          <button class="btn btn-p btn-lg" onclick="generateWorkflow()">Generate Workflow</button>
          <button class="btn btn-ok btn-lg" onclick="generateAndRun()">Generate + Run ComfyUI</button>
        </div>
        <div id="genStatus" class="hidden" style="margin-top:12px;padding:12px;border-radius:8px;background:var(--s2);font-size:13px"></div>
      </div>
    </div>
  </div>
</div>

<!-- ═══ VIDEO LIBRARY ═══ -->
<div class="page" id="page-videos">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
    <h2>Video Library</h2>
    <div style="display:flex;gap:10px">
      <select id="vFilterCh" onchange="loadVideos()" style="width:180px"><option value="">All channels</option></select>
      <select id="vFilterSt" onchange="loadVideos()" style="width:140px">
        <option value="">All statuses</option>
        <option value="draft">Draft</option><option value="generated">Generated</option>
        <option value="rendering">Rendering</option><option value="rendered">Rendered</option>
        <option value="published">Published</option>
      </select>
    </div>
  </div>
  <table class="tbl">
    <thead><tr><th>Title / Topic</th><th>Channel</th><th>Clips</th><th>Duration</th><th>Status</th><th>Created</th><th>Actions</th></tr></thead>
    <tbody id="videoRows"></tbody>
  </table>
</div>

<!-- ═══ SETTINGS ═══ -->
<div class="page" id="page-settings">
  <h2 style="margin-bottom:20px">Settings</h2>
  <div class="split">
    <div class="panel">
      <div class="panel-title">AI Provider</div>
      <div class="fg"><label>Active Provider</label>
        <select id="sProvider" onchange="saveProviderSetting()"><option value="claude">Claude (Anthropic)</option><option value="gemini">Gemini (Google)</option></select>
      </div>
      <div class="fg"><label>Claude API Key</label>
        <div style="display:flex;gap:8px"><input type="password" id="sClaudeKey" placeholder="sk-ant-..."><button class="btn btn-s btn-sm" onclick="saveApiKey('claude')">Save</button></div>
      </div>
      <div class="fg"><label>Gemini API Key</label>
        <div style="display:flex;gap:8px"><input type="password" id="sGeminiKey" placeholder="AI..."><button class="btn btn-s btn-sm" onclick="saveApiKey('gemini')">Save</button></div>
      </div>
      <div id="settingsStatus" style="margin-top:8px;font-size:13px;color:var(--tx2)"></div>
    </div>
    <div class="panel">
      <div class="panel-title">Paths</div>
      <div class="fg"><label>ComfyUI Directory</label><input type="text" id="sComfyDir" value=""></div>
      <div class="fg"><label>Default Output Directory</label><input type="text" id="sOutDir" value=""></div>
      <div class="btn-row">
        <button class="btn btn-p btn-sm" onclick="savePaths()">Save Paths</button>
        <button class="btn btn-err btn-sm" onclick="killComfyUI()">Kill ComfyUI</button>
      </div>
    </div>
  </div>
</div>

<!-- ═══ CHANNEL MODAL ═══ -->
<div class="modal-bg hidden" id="channelModal" onclick="if(event.target===this)this.classList.add('hidden')">
  <div class="modal">
    <h2 id="chModalTitle">New Channel</h2>
    <input type="hidden" id="chId">
    <div class="row row-2">
      <div class="fg"><label>Name</label><input type="text" id="chName"></div>
      <div class="fg"><label>Niche</label><input type="text" id="chNiche"></div>
    </div>
    <div class="fg"><label>Visual Style Prompt</label><textarea id="chStyle" rows="3"></textarea></div>
    <div class="row row-2">
      <div class="fg"><label>Target Audience</label><input type="text" id="chAudience"></div>
      <div class="fg"><label>Voice Style</label><input type="text" id="chVoice"></div>
    </div>
    <div class="row row-2">
      <div class="fg"><label>Language</label><input type="text" id="chLang" value="English"></div>
      <div class="fg"><label>Made for Kids</label><select id="chKids"><option value="1">Yes</option><option value="0">No</option></select></div>
    </div>
    <div class="fg"><label>Keywords (comma-separated)</label><input type="text" id="chKeywords"></div>
    <div class="fg"><label>Default Negative Prompt</label><input type="text" id="chNegPrompt"></div>
    <div class="btn-row">
      <button class="btn btn-p" onclick="saveChannel()">Save Channel</button>
      <button class="btn btn-s" onclick="document.getElementById('channelModal').classList.add('hidden')">Cancel</button>
      <button class="btn btn-err btn-sm hidden" id="chDeleteBtn" onclick="deleteChannel()">Delete</button>
    </div>
  </div>
</div>

<!-- ═══ VIDEO DETAIL MODAL ═══ -->
<div class="modal-bg hidden" id="videoModal" onclick="if(event.target===this)this.classList.add('hidden')">
  <div class="modal" style="width:700px">
    <h2 id="vmTitle">Video Details</h2>
    <div id="vmContent"></div>
  </div>
</div>

<script>
// ═══ State ═══
let channels = [], currentAR = '16:9 (1024x576)', lastWorkflowPath = '', WORKFLOW_PATH = '';

// ═══ Tabs ═══
function showTab(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('page-'+name).classList.add('active');
  document.querySelectorAll('.tab-btn').forEach(b => { if(b.textContent.toLowerCase().includes(name.replace('dashboard','dash'))) b.classList.add('active'); });
  // find exact match
  document.querySelectorAll('.tab-btn').forEach(b => {
    const map = {dashboard:'Dashboard',create:'Create Video',videos:'Video Library',settings:'Settings'};
    if(b.textContent.trim()===map[name]) { b.classList.add('active'); }
  });
  if(name==='dashboard') { loadChannels(); loadRecentVideos(); }
  if(name==='create') loadChannelDropdown();
  if(name==='videos') { loadVideoFilters(); loadVideos(); }
  if(name==='settings') loadSettings();
}

// ═══ Toast ═══
function toast(msg, type='info') {
  const d = document.createElement('div');
  d.className = 'toast toast-'+type;
  d.textContent = msg;
  document.body.appendChild(d);
  setTimeout(() => d.remove(), 4000);
}

// ═══ API helpers ═══
async function api(url, opts={}) {
  if(opts.body && typeof opts.body==='object') {
    opts.headers = {'Content-Type':'application/json',...(opts.headers||{})};
    opts.body = JSON.stringify(opts.body);
  }
  const r = await fetch(url, opts);
  const j = await r.json();
  if(j.error) throw new Error(j.error);
  return j;
}

// ═══ Dashboard ═══
async function loadChannels() {
  channels = await api('/api/channels');
  const el = document.getElementById('channelCards');
  if(!channels.length) { el.innerHTML='<p style="color:var(--tx2)">No channels yet. Create one!</p>'; return; }
  el.innerHTML = channels.map(c => `
    <div class="card" onclick="editChannel(${c.id})">
      <h3>${esc(c.name)}</h3>
      <div class="meta">${esc(c.niche||'')}</div>
      <div class="meta" style="margin-top:6px">${c.video_count} videos &middot; ${c.published_count} published</div>
      <div class="meta">${c.for_kids?'Made for Kids':''} ${esc(c.language||'')}</div>
      <div class="btn-row"><button class="btn btn-p btn-sm" onclick="event.stopPropagation();startVideoForChannel(${c.id})">+ New Video</button></div>
    </div>
  `).join('');
}

async function loadRecentVideos() {
  try {
    const vids = await api('/api/videos');
    const el = document.getElementById('recentVideos');
    if(!vids.length) { el.innerHTML='<p style="color:var(--tx2)">No videos generated yet.</p>'; return; }
    el.innerHTML = '<table class="tbl"><thead><tr><th>Topic</th><th>Channel</th><th>Status</th><th>Created</th></tr></thead><tbody>' +
      vids.slice(0,10).map(v => `<tr style="cursor:pointer" onclick="openVideoDetail(${v.id})">
        <td>${esc(v.topic||v.title||'Untitled')}</td><td>${esc(v.channel_name||'')}</td>
        <td><span class="status-pill st-${v.status}">${v.status}</span></td>
        <td>${v.created_at?.slice(0,16)||''}</td></tr>`).join('') + '</tbody></table>';
  } catch(e) {}
}

function startVideoForChannel(cid) {
  showTab('create');
  setTimeout(() => { document.getElementById('cChannel').value = cid; onChannelChange(); }, 100);
}

// ═══ Channel Modal ═══
function openChannelModal(ch=null) {
  document.getElementById('chModalTitle').textContent = ch ? 'Edit Channel' : 'New Channel';
  document.getElementById('chId').value = ch?.id || '';
  document.getElementById('chName').value = ch?.name || '';
  document.getElementById('chNiche').value = ch?.niche || '';
  document.getElementById('chStyle').value = ch?.style_prompt || '';
  document.getElementById('chAudience').value = ch?.target_audience || '';
  document.getElementById('chVoice').value = ch?.voice_style || '';
  document.getElementById('chLang').value = ch?.language || 'English';
  document.getElementById('chKids').value = ch?.for_kids ? '1' : '0';
  document.getElementById('chKeywords').value = ch?.keywords || '';
  document.getElementById('chNegPrompt').value = ch?.negative_prompt || '';
  document.getElementById('chDeleteBtn').classList.toggle('hidden', !ch);
  document.getElementById('channelModal').classList.remove('hidden');
}

function editChannel(id) { const ch = channels.find(c=>c.id===id); if(ch) openChannelModal(ch); }

async function saveChannel() {
  const id = document.getElementById('chId').value;
  const data = {
    name: document.getElementById('chName').value,
    niche: document.getElementById('chNiche').value,
    style_prompt: document.getElementById('chStyle').value,
    target_audience: document.getElementById('chAudience').value,
    voice_style: document.getElementById('chVoice').value,
    language: document.getElementById('chLang').value,
    for_kids: document.getElementById('chKids').value === '1',
    keywords: document.getElementById('chKeywords').value,
    negative_prompt: document.getElementById('chNegPrompt').value,
  };
  try {
    if(id) await api(`/api/channels/${id}`, {method:'PUT', body:data});
    else await api('/api/channels', {method:'POST', body:data});
    document.getElementById('channelModal').classList.add('hidden');
    toast('Channel saved!', 'ok');
    loadChannels();
  } catch(e) { toast(e.message, 'err'); }
}

async function deleteChannel() {
  const id = document.getElementById('chId').value;
  if(!id || !confirm('Delete this channel and all its videos?')) return;
  await api(`/api/channels/${id}`, {method:'DELETE'});
  document.getElementById('channelModal').classList.add('hidden');
  toast('Channel deleted', 'ok');
  loadChannels();
}

// ═══ Create Video ═══
async function loadChannelDropdown() {
  if(!channels.length) channels = await api('/api/channels');
  const sel = document.getElementById('cChannel');
  const cur = sel.value;
  sel.innerHTML = '<option value="">Select channel...</option>' + channels.map(c => `<option value="${c.id}">${esc(c.name)}</option>`).join('');
  if(cur) sel.value = cur;
  buildARGrid();
}

function onChannelChange() {
  const cid = document.getElementById('cChannel').value;
  const ch = channels.find(c => c.id == cid);
  if(ch) {
    document.getElementById('cNeg').value = ch.negative_prompt || '';
    document.getElementById('cGlobal').value = ch.global_prompt || '';
  }
}

function buildARGrid() {
  const grid = document.getElementById('arGrid');
  grid.innerHTML = Object.entries(ASPECT_RATIOS).map(([name, [w,h]]) =>
    `<button class="ar-btn${name===currentAR?' active':''}" onclick="selectAR(this,'${name}',${w},${h})">${name}</button>`
  ).join('');
}
const ASPECT_RATIOS = {"16:9 (1024x576)":[1024,576],"16:9 (768x432)":[768,432],"9:16 (576x1024)":[576,1024],"1:1 (576x576)":[576,576],"1:1 (768x768)":[768,768],"4:3 (768x576)":[768,576],"21:9 (1024x432)":[1024,432]};

function selectAR(btn, name, w, h) {
  document.querySelectorAll('.ar-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  currentAR = name;
  document.getElementById('cW').value = w;
  document.getElementById('cH').value = h;
  updateStats();
}

function getRes() {
  return { w: parseInt(document.getElementById('cW').value)||1024, h: parseInt(document.getElementById('cH').value)||576 };
}

function countClips() {
  const t = document.getElementById('cPrompts').value.trim();
  if(!t) return 0;
  return t.includes('---') ? t.split(/\n?---+\n?/).filter(p=>p.trim()).length : t.split(/\n\s*\n/).filter(p=>p.trim()).length;
}

function updateStats() {
  const clips = countClips();
  const spc = parseFloat(document.getElementById('cSpc').value)||5;
  const fps = parseInt(document.getElementById('cFps').value)||24;
  const res = getRes();
  const dur = clips*spc;
  document.getElementById('stClips').textContent = clips;
  document.getElementById('stDur').textContent = dur>=60 ? (dur/60).toFixed(1)+'m' : dur+'s';
  document.getElementById('stFrames').textContent = clips*Math.round(spc*fps);
  document.getElementById('stRes').textContent = res.w+'x'+res.h;
  document.getElementById('clipBadge').textContent = clips;
}

// ═══ AI Functions ═══
async function aiSuggestTopics() {
  const cid = document.getElementById('cChannel').value;
  if(!cid) { toast('Select a channel first', 'err'); return; }
  const btn = document.getElementById('btnSuggest');
  btn.disabled = true; btn.innerHTML = '<span class="loading"></span>';
  try {
    const r = await api('/api/ai/suggest-topics', {method:'POST', body:{channel_id:parseInt(cid)}});
    const el = document.getElementById('topicIdeas');
    el.classList.remove('hidden');
    el.innerHTML = (r.ideas||[]).map(i => `
      <div class="idea-card" onclick="document.getElementById('cTopic').value='${esc(i.topic)}';document.getElementById('topicIdeas').classList.add('hidden')">
        <h4>${esc(i.topic)}</h4>
        <p>${esc(i.hook||'')} &mdash; ${esc(i.why||'')}</p>
      </div>
    `).join('');
  } catch(e) { toast(e.message, 'err'); }
  btn.disabled = false; btn.textContent = 'AI Suggest';
}

async function aiGeneratePrompts() {
  const cid = document.getElementById('cChannel').value;
  const topic = document.getElementById('cTopic').value;
  if(!cid) { toast('Select a channel first', 'err'); return; }
  if(!topic) { toast('Enter a topic first', 'err'); return; }
  const btn = document.getElementById('btnGenPrompts');
  btn.disabled = true; btn.innerHTML = '<span class="loading"></span> Generating...';
  try {
    const r = await api('/api/ai/generate-prompts', {method:'POST', body:{
      channel_id: parseInt(cid), topic,
      num_clips: parseInt(document.getElementById('cSpc').value) <= 3 ? 10 : 6,
      seconds_per_clip: parseFloat(document.getElementById('cSpc').value)||5,
    }});
    document.getElementById('cPrompts').value = (r.prompts||[]).join('\n---\n');
    if(r.topic) document.getElementById('cTopic').value = r.topic;
    updateStats();
    toast('Prompts generated!', 'ok');
  } catch(e) { toast(e.message, 'err'); }
  btn.disabled = false; btn.textContent = 'AI Generate';
}

async function aiGenerateSeo() {
  const cid = document.getElementById('cChannel').value;
  const prompts = document.getElementById('cPrompts').value;
  if(!cid) { toast('Select a channel first', 'err'); return; }
  if(!prompts.trim()) { toast('Generate prompts first', 'err'); return; }
  const btn = document.getElementById('btnSeo');
  btn.disabled = true; btn.innerHTML = '<span class="loading"></span> Generating...';
  try {
    const r = await api('/api/ai/generate-seo', {method:'POST', body:{channel_id:parseInt(cid), prompts_text:prompts}});
    const el = document.getElementById('seoResults');
    el.classList.remove('hidden');
    document.getElementById('seoTitles').innerHTML = (r.titles||[]).map((t,i) =>
      `<div class="seo-title-opt${i===0?' selected':''}" onclick="selectSeoTitle(this)">${esc(t)}</div>`
    ).join('');
    document.getElementById('seoDesc').value = r.description || '';
    document.getElementById('seoTags').innerHTML = (r.tags||[]).map(t => `<span class="tag-chip">${esc(t)}</span>`).join('');
    toast('SEO metadata generated!', 'ok');
  } catch(e) { toast(e.message, 'err'); }
  btn.disabled = false; btn.textContent = 'Generate SEO';
}

function selectSeoTitle(el) {
  document.querySelectorAll('.seo-title-opt').forEach(e=>e.classList.remove('selected'));
  el.classList.add('selected');
}

function loadFairyTemplate() {
  document.getElementById('cPrompts').value = `Cute magical fairy flying over glowing pink flower garden, sparkling particles, soft pastel colors, children's storybook illustration style, dreamy atmosphere
---
Slow cinematic drift through enchanted crystal forest, glowing mushrooms, fireflies, pastel purple and blue, soft volumetric light
---
Unicorn walking through rainbow flower field, butterflies, sparkles, pastel colors, gentle movement, fantasy world, children's book style
---
Magical castle floating in cotton candy clouds, rainbow waterfalls, tiny fairies dancing, warm golden sunlight, whimsical, dreamy
---
Gentle butterfly garden with oversized colorful flowers, dewdrops sparkling like diamonds, soft breeze, pastel sky at golden hour
---
Sparkling fairy dust trail swirling through moonlit enchanted garden, glowing flowers closing for night, peaceful blue and purple tones, calming bedtime atmosphere`;
  updateStats();
}

// ═══ Workflow Generation ═══
async function generateWorkflow(andRun=false) {
  const prompts = document.getElementById('cPrompts').value.trim();
  if(!prompts) { toast('Enter prompts first', 'err'); return; }
  const res = getRes();
  const selectedTitle = document.querySelector('.seo-title-opt.selected')?.textContent || '';
  const body = {
    prompts_text: prompts,
    seconds_per_clip: parseFloat(document.getElementById('cSpc').value),
    frame_rate: parseInt(document.getElementById('cFps').value),
    guide_strength: parseFloat(document.getElementById('cGs').value),
    width: res.w, height: res.h,
    negative_prompt: document.getElementById('cNeg').value,
    global_prompt: document.getElementById('cGlobal').value,
    filename_prefix: document.getElementById('cPrefix').value,
    output_dir: document.getElementById('cOutDir').value,
    workflow_path: WORKFLOW_PATH,
    channel_id: parseInt(document.getElementById('cChannel').value) || null,
    topic: document.getElementById('cTopic').value,
    title: selectedTitle,
  };
  const el = document.getElementById('genStatus');
  el.classList.remove('hidden');
  el.innerHTML = '<span class="loading"></span> Generating workflow...';
  try {
    const r = await api('/api/generate', {method:'POST', body});
    lastWorkflowPath = r.output_file;
    el.innerHTML = `<span style="color:var(--ok)">Workflow saved:</span> ${esc(r.output_file)}<br>
      ${r.clip_count} clips, ${r.total_seconds.toFixed(0)}s, ${r.resolution}
      <br><button class="btn btn-s btn-sm" style="margin-top:8px" onclick="copyPath('${r.output_file.replace(/\\/g,'\\\\')}')">Copy Path</button>`;
    toast('Workflow generated!', 'ok');
    if(andRun) runComfyUI(r.output_file);
  } catch(e) {
    el.innerHTML = `<span style="color:var(--err)">Error:</span> ${esc(e.message)}`;
    toast(e.message, 'err');
  }
}

async function generateAndRun() { await generateWorkflow(true); }

async function runComfyUI(wfPath) {
  const el = document.getElementById('genStatus');
  el.classList.remove('hidden');
  el.innerHTML += '<br><span class="loading"></span> Starting ComfyUI & queuing workflow...';
  try {
    const r = await api('/api/comfyui/run', {method:'POST', body:{workflow_path: wfPath||lastWorkflowPath}});
    if(r.queued) {
      el.innerHTML += `<br><span style="color:var(--ok)">Workflow queued and rendering!</span> Prompt ID: ${esc(r.prompt_id||'')}
        <br><button class="btn btn-s btn-sm" style="margin-top:6px" onclick="checkQueueStatus()">Check Progress</button>`;
      toast('Workflow queued in ComfyUI!', 'ok');
      checkComfyUI();
    } else {
      el.innerHTML += `<br><span style="color:var(--warn)">${esc(r.message)}</span>
        <br><button class="btn btn-s btn-sm" style="margin-top:6px" onclick="copyPath('${(wfPath||lastWorkflowPath).replace(/\\/g,'\\\\')}')">Copy Workflow Path</button>`;
      toast('ComfyUI running — load workflow manually', 'info');
    }
  } catch(e) { el.innerHTML += `<br><span style="color:var(--err)">${esc(e.message)}</span>`; }
}

async function checkQueueStatus() {
  try {
    const r = await api('/api/comfyui/queue-status');
    const el = document.getElementById('genStatus');
    if(r.total===0) {
      el.innerHTML += `<br><span style="color:var(--ok)">Queue empty — rendering complete!</span>`;
      toast('Rendering complete!', 'ok');
    } else {
      el.innerHTML += `<br>Queue: ${r.running} running, ${r.pending} pending`;
      toast(`Queue: ${r.running} running, ${r.pending} pending`, 'info');
    }
  } catch(e) { toast(e.message, 'err'); }
}

function copyPath(p) { navigator.clipboard.writeText(p); toast('Path copied!', 'ok'); }

// ═══ Video Library ═══
async function loadVideoFilters() {
  if(!channels.length) channels = await api('/api/channels');
  const sel = document.getElementById('vFilterCh');
  sel.innerHTML = '<option value="">All channels</option>' + channels.map(c => `<option value="${c.id}">${esc(c.name)}</option>`).join('');
}

async function loadVideos() {
  const cid = document.getElementById('vFilterCh').value;
  const st = document.getElementById('vFilterSt').value;
  let url = '/api/videos?';
  if(cid) url += `channel_id=${cid}&`;
  if(st) url += `status=${st}&`;
  const vids = await api(url);
  const el = document.getElementById('videoRows');
  if(!vids.length) { el.innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--tx2);padding:40px">No videos found</td></tr>'; return; }
  el.innerHTML = vids.map(v => `<tr>
    <td style="cursor:pointer" onclick="openVideoDetail(${v.id})">${esc(v.topic||v.title||'Untitled')}</td>
    <td>${esc(v.channel_name||'')}</td>
    <td>${v.clip_count||0}</td>
    <td>${v.total_seconds ? v.total_seconds.toFixed(0)+'s' : '—'}</td>
    <td><span class="status-pill st-${v.status}">${v.status}</span></td>
    <td>${v.created_at?.slice(0,16)||''}</td>
    <td>
      <div style="display:flex;gap:4px">
        ${v.status==='generated'?`<button class="btn btn-ok btn-sm" onclick="event.stopPropagation();markRendered(${v.id})">Rendered</button>`:''}
        ${v.status==='rendered'?`<button class="btn btn-ok btn-sm" onclick="event.stopPropagation();markPublished(${v.id})">Publish</button>`:''}
        ${v.workflow_path?`<button class="btn btn-s btn-sm" onclick="event.stopPropagation();runComfyUI('${v.workflow_path.replace(/\\/g,'\\\\')}')">Run</button>`:''}
        <button class="btn btn-err btn-sm" onclick="event.stopPropagation();deleteVideo(${v.id})">Del</button>
      </div>
    </td>
  </tr>`).join('');
}

async function markRendered(id) { await api(`/api/videos/${id}`,{method:'PUT',body:{status:'rendered'}}); loadVideos(); toast('Marked as rendered','ok'); }
async function markPublished(id) {
  const url = prompt('YouTube URL (optional):','');
  await api(`/api/videos/${id}`,{method:'PUT',body:{status:'published', yt_url:url||''}});
  loadVideos(); toast('Published!','ok');
}
async function deleteVideo(id) { if(!confirm('Delete this video?')) return; await api(`/api/videos/${id}`,{method:'DELETE'}); loadVideos(); }

async function openVideoDetail(id) {
  const v = await api(`/api/videos/${id}`);
  document.getElementById('vmTitle').textContent = v.topic || v.title || 'Video Details';
  let seo = {};
  try { seo = JSON.parse(v.seo_json||'{}'); } catch(e) {}
  document.getElementById('vmContent').innerHTML = `
    <div class="row row-2" style="margin-bottom:12px">
      <div><strong>Channel:</strong> ${esc(v.channel_name)}</div>
      <div><strong>Status:</strong> <span class="status-pill st-${v.status}">${v.status}</span></div>
    </div>
    <div class="row row-3" style="margin-bottom:12px">
      <div><strong>Clips:</strong> ${v.clip_count}</div>
      <div><strong>Duration:</strong> ${v.total_seconds?v.total_seconds.toFixed(0)+'s':'—'}</div>
      <div><strong>Resolution:</strong> ${v.resolution||'—'}</div>
    </div>
    ${v.title?`<div class="fg"><strong>Title:</strong> ${esc(v.title)}</div>`:''}
    ${v.yt_url?`<div class="fg"><strong>YouTube:</strong> <a href="${esc(v.yt_url)}" target="_blank" style="color:var(--ac2)">${esc(v.yt_url)}</a></div>`:''}
    ${v.workflow_path?`<div class="fg"><strong>Workflow:</strong> <code style="font-size:12px;color:var(--tx2)">${esc(v.workflow_path)}</code> <button class="btn btn-s btn-sm" onclick="copyPath('${v.workflow_path.replace(/\\/g,'\\\\')}')">Copy</button></div>`:''}
    <div class="fg"><strong>Prompts:</strong><pre style="background:var(--s2);padding:12px;border-radius:8px;font-size:12px;max-height:200px;overflow-y:auto;white-space:pre-wrap">${esc(v.prompts_text||'')}</pre></div>
    ${seo.description?`<div class="fg"><strong>Description:</strong><pre style="background:var(--s2);padding:12px;border-radius:8px;font-size:12px;max-height:150px;overflow-y:auto;white-space:pre-wrap">${esc(seo.description)}</pre></div>`:''}
    ${seo.tags?`<div class="fg"><strong>Tags:</strong><div class="tag-cloud">${seo.tags.map(t=>`<span class="tag-chip">${esc(t)}</span>`).join('')}</div></div>`:''}
    <div class="btn-row">
      <input type="text" id="vmYtUrl" value="${esc(v.yt_url||'')}" placeholder="YouTube URL" style="flex:1">
      <button class="btn btn-p btn-sm" onclick="updateVideoUrl(${v.id})">Save URL</button>
      ${v.status==='rendered'?`<button class="btn btn-s btn-sm" onclick="api('/api/videos/${v.id}/stitch',{method:'POST'}).then(r=>{toast('Stitched!','ok');loadVideos();openVideoDetail(${v.id});}).catch(e=>toast(e.message,'err'))">Stitch Chunks</button>`:''}
      ${v.status!=='published'?`<button class="btn btn-ok btn-sm" onclick="api('/api/videos/${v.id}',{method:'PUT',body:{status:'published',yt_url:document.getElementById('vmYtUrl').value}}).then(()=>{toast('Published!','ok');document.getElementById('videoModal').classList.add('hidden');loadVideos()})">Mark Published</button>`:''}
    </div>
  `;
  document.getElementById('videoModal').classList.remove('hidden');
}

async function updateVideoUrl(id) {
  await api(`/api/videos/${id}`,{method:'PUT',body:{yt_url:document.getElementById('vmYtUrl').value}});
  toast('URL saved','ok');
}

// ═══ Settings ═══
async function loadSettings() {
  const s = await api('/api/settings');
  document.getElementById('sProvider').value = s.ai_provider || 'claude';
  document.getElementById('sClaudeKey').value = s.claude_api_key_set ? '••••••••' : '';
  document.getElementById('sGeminiKey').value = s.gemini_api_key_set ? '••••••••' : '';
  document.getElementById('sComfyDir').value = s.comfyui_dir || '';
  document.getElementById('sOutDir').value = s.output_dir || '';
  document.getElementById('settingsStatus').innerHTML =
    `Claude: ${s.claude_api_key_set?'<span style="color:var(--ok)">Key set</span>':'<span style="color:var(--err)">Not set</span>'} &middot; ` +
    `Gemini: ${s.gemini_api_key_set?'<span style="color:var(--ok)">Key set</span>':'<span style="color:var(--err)">Not set</span>'} &middot; ` +
    `Available: ${s.available_providers.join(', ')||'none (install anthropic or google-generativeai)'}`;
  updateProviderBadge(s.ai_provider);
}

async function saveProviderSetting() {
  const p = document.getElementById('sProvider').value;
  await api('/api/settings', {method:'POST', body:{ai_provider:p}});
  updateProviderBadge(p);
  toast('Provider switched to '+p, 'ok');
}

async function saveApiKey(provider) {
  const key = document.getElementById(provider==='claude'?'sClaudeKey':'sGeminiKey').value;
  if(!key || key.includes('••')) { toast('Enter a valid key', 'err'); return; }
  await api('/api/settings', {method:'POST', body:{[provider+'_api_key']:key}});
  toast(provider+' key saved!', 'ok');
  loadSettings();
}

async function savePaths() {
  await api('/api/settings', {method:'POST', body:{
    comfyui_dir: document.getElementById('sComfyDir').value,
    output_dir: document.getElementById('sOutDir').value,
  }});
  toast('Paths saved!', 'ok');
}

function updateProviderBadge(p) {
  document.getElementById('providerBadge').textContent = 'AI: ' + (p||'—').charAt(0).toUpperCase() + (p||'').slice(1);
}

// ═══ ComfyUI Status ═══
async function checkComfyUI() {
  try {
    const r = await api('/api/comfyui/status');
    document.getElementById('comfyDot').className = 'comfy-dot ' + (r.running?'on':'off');
  } catch(e) { document.getElementById('comfyDot').className = 'comfy-dot off'; }
}


async function killComfyUI() {
  if(!confirm('Forcefully kill ComfyUI?')) return;
  try {
    const r = await api('/api/comfyui/kill', {method:'POST'});
    toast(r.message, 'ok');
    setTimeout(checkComfyUI, 1000);
  } catch(e) { toast(e.message, 'err'); }
}

// ═══ Utils ═══
function esc(s) { const d=document.createElement('div'); d.textContent=s||''; return d.innerHTML; }

// ═══ Init ═══
api('/api/config').then(c => { WORKFLOW_PATH = c.workflow_path; document.getElementById('cOutDir').value = c.output_dir || document.getElementById('cOutDir').value; }).catch(()=>{});
loadChannels();
loadRecentVideos();
checkComfyUI();
setInterval(checkComfyUI, 15000);
loadSettings().catch(()=>{});
document.getElementById('cSpc').addEventListener('input', updateStats);
document.getElementById('cFps').addEventListener('input', updateStats);
updateStats();
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  VideoMaker AI Pipeline")
    print(f"  ======================")
    print(f"  Database: {DB_PATH}")
    print(f"  ComfyUI:  {COMFYUI_DIR}")
    print(f"\n  Open http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=True)


HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VideoMaker</title>
<style>
:root{--bg:#0f1117;--s1:#1a1d27;--s2:#242836;--bd:#2e3347;--tx:#e4e6ed;--tx2:#8b8fa3;--ac:#6c5ce7;--ac2:#a29bfe;--ok:#00b894;--warn:#fdcb6e;--err:#ff7675;--blue:#0984e3}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--tx);min-height:100vh}
.nav{background:var(--s1);border-bottom:1px solid var(--bd);display:flex;align-items:center;padding:0 24px;height:56px;gap:8px}
.nav .logo{font-size:20px;font-weight:700;margin-right:24px}
.nav .logo span{color:var(--ac2);font-weight:400;font-size:14px;margin-left:8px}
.tab-btn{background:none;border:none;color:var(--tx2);font:inherit;font-size:14px;padding:16px 16px;cursor:pointer;border-bottom:2px solid transparent;transition:.2s}
.tab-btn:hover{color:var(--tx)}
.tab-btn.active{color:var(--ac2);border-bottom-color:var(--ac)}
.nav-right{margin-left:auto;display:flex;gap:12px;align-items:center}
.provider-badge{font-size:11px;padding:4px 10px;border-radius:12px;background:var(--s2);border:1px solid var(--bd)}
.comfy-dot{width:8px;height:8px;border-radius:50%;display:inline-block}
.comfy-dot.on{background:var(--ok)}.comfy-dot.off{background:var(--err)}
.page{display:none;padding:24px;max-width:1400px;margin:0 auto}.page.active{display:block}
.panel{background:var(--s1);border:1px solid var(--bd);border-radius:12px;padding:20px;margin-bottom:16px}
.panel-title{font-size:15px;font-weight:600;margin-bottom:16px;display:flex;align-items:center;gap:8px}
.badge{font-size:11px;padding:2px 8px;border-radius:10px;font-weight:500}
.badge-purple{background:var(--ac);color:#fff}.badge-green{background:var(--ok);color:#fff}
.badge-yellow{background:var(--warn);color:#000}.badge-red{background:var(--err);color:#fff}
.badge-blue{background:var(--blue);color:#fff}
label{display:block;font-size:13px;color:var(--tx2);margin-bottom:5px;font-weight:500}
input[type=text],input[type=number],input[type=password],select,textarea{width:100%;background:var(--s2);border:1px solid var(--bd);border-radius:8px;padding:9px 12px;color:var(--tx);font:14px/1.5 inherit;transition:border-color .2s}
input:focus,select:focus,textarea:focus{outline:none;border-color:var(--ac)}
textarea{resize:vertical;font-family:'Cascadia Code','Consolas',monospace;font-size:13px}
.row{display:grid;gap:12px;margin-bottom:12px}
.row-2{grid-template-columns:1fr 1fr}.row-3{grid-template-columns:1fr 1fr 1fr}.row-4{grid-template-columns:1fr 1fr 1fr 1fr}
.fg{margin-bottom:12px}
.btn{padding:10px 20px;border:none;border-radius:8px;font:600 14px/1 inherit;cursor:pointer;transition:.2s;white-space:nowrap}
.btn-p{background:var(--ac);color:#fff}.btn-p:hover{background:var(--ac2)}
.btn-s{background:var(--s2);color:var(--tx);border:1px solid var(--bd)}.btn-s:hover{background:var(--bd)}
.btn-ok{background:var(--ok);color:#fff}.btn-ok:hover{opacity:.9}
.btn-err{background:var(--err);color:#fff}
.btn-sm{padding:6px 14px;font-size:12px}
.btn-lg{padding:14px 32px;font-size:16px}
.btn:disabled{opacity:.5;cursor:not-allowed}
.btn-row{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px}
.card{background:var(--s2);border:1px solid var(--bd);border-radius:10px;padding:16px;cursor:pointer;transition:.2s}
.card:hover{border-color:var(--ac);transform:translateY(-2px)}
.card h3{font-size:15px;margin-bottom:6px}.card .meta{font-size:12px;color:var(--tx2)}
.status-pill{font-size:11px;padding:3px 10px;border-radius:12px;font-weight:600;text-transform:uppercase;letter-spacing:.3px}
.st-draft{background:#2d3436;color:var(--tx2)}.st-generated{background:#0a3d62;color:#74b9ff}
.st-rendering{background:#6c3483;color:#d2b4de}.st-rendered{background:#0e6655;color:#82e0aa}.st-published{background:#1e8449;color:#a9dfbf}
.tbl{width:100%;border-collapse:collapse}
.tbl th{text-align:left;font-size:12px;color:var(--tx2);padding:8px 12px;border-bottom:1px solid var(--bd);text-transform:uppercase;letter-spacing:.5px}
.tbl td{padding:10px 12px;border-bottom:1px solid var(--bd);font-size:13px;vertical-align:middle}
.tbl tr:hover td{background:var(--s2)}
.ar-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:10px}
.ar-btn{background:var(--s2);border:2px solid var(--bd);border-radius:6px;padding:6px 4px;color:var(--tx2);font-size:11px;cursor:pointer;text-align:center;transition:.2s}
.ar-btn:hover{border-color:var(--ac);color:var(--tx)}.ar-btn.active{border-color:var(--ac);background:rgba(108,92,231,.15);color:var(--ac2)}
.idea-card{background:var(--s2);border:1px solid var(--bd);border-radius:8px;padding:12px;margin-bottom:8px;cursor:pointer;transition:.2s}
.idea-card:hover{border-color:var(--ac)}
.idea-card h4{font-size:14px;margin-bottom:4px;color:var(--ac2)}.idea-card p{font-size:12px;color:var(--tx2)}
.seo-title-opt{background:var(--s2);border:1px solid var(--bd);border-radius:8px;padding:10px 14px;margin-bottom:6px;cursor:pointer;font-size:14px;transition:.2s}
.seo-title-opt:hover,.seo-title-opt.selected{border-color:var(--ac);background:rgba(108,92,231,.1)}
.tag-cloud{display:flex;flex-wrap:wrap;gap:6px}.tag-chip{background:var(--s2);border:1px solid var(--bd);border-radius:14px;padding:4px 12px;font-size:12px;color:var(--tx2)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(100px,1fr));gap:10px}
.stat{background:var(--s2);border-radius:8px;padding:10px;text-align:center}
.stat .v{font-size:22px;font-weight:700;color:var(--ac2)}.stat .l{font-size:10px;color:var(--tx2);text-transform:uppercase;letter-spacing:.5px;margin-top:2px}
.loading{display:inline-block;width:16px;height:16px;border:2px solid var(--bd);border-top-color:var(--ac);border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.toast{position:fixed;bottom:24px;right:24px;padding:14px 20px;border-radius:10px;font-size:14px;z-index:9999;max-width:500px;animation:slideIn .3s}
.toast-ok{background:#27ae60;color:#fff}.toast-err{background:#e74c3c;color:#fff}.toast-info{background:var(--s1);color:var(--tx);border:1px solid var(--bd)}
@keyframes slideIn{from{transform:translateY(20px);opacity:0}to{transform:translateY(0);opacity:1}}
.hidden{display:none!important}
.modal-bg{position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:100;display:flex;align-items:center;justify-content:center}
.modal{background:var(--s1);border:1px solid var(--bd);border-radius:14px;padding:24px;width:600px;max-width:90vw;max-height:80vh;overflow-y:auto}
.modal h2{font-size:18px;margin-bottom:16px}
.split{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:900px){.split{grid-template-columns:1fr}.row-3,.row-4{grid-template-columns:1fr 1fr}}
</style>
</head>
<body>

<!-- NAV -->
<div class="nav">
  <div class="logo">VideoMaker<span>AI Pipeline</span></div>
  <button class="tab-btn active" onclick="showTab('dashboard')">Dashboard</button>
  <button class="tab-btn" onclick="showTab('create')">Create Video</button>
  <button class="tab-btn" onclick="showTab('videos')">Video Library</button>
  <button class="tab-btn" onclick="showTab('settings')">Settings</button>
  <div class="nav-right">
    <span class="provider-badge" id="providerBadge">AI: —</span>
    <span title="ComfyUI status"><span class="comfy-dot off" id="comfyDot"></span> ComfyUI</span>
  </div>
</div>

<!-- ═══ DASHBOARD ═══ -->
<div class="page active" id="page-dashboard">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px">
    <h2>Channels</h2>
    <button class="btn btn-p" onclick="openChannelModal()">+ New Channel</button>
  </div>
  <div class="cards" id="channelCards"></div>
  <h2 style="margin:28px 0 16px">Recent Videos</h2>
  <div id="recentVideos"></div>
</div>

<!-- ═══ CREATE VIDEO ═══ -->
<div class="page" id="page-create">
  <div class="split">
    <!-- LEFT -->
    <div>
      <!-- Channel + Topic -->
      <div class="panel">
        <div class="panel-title">Channel & Topic</div>
        <div class="fg">
          <label>Channel</label>
          <select id="cChannel" onchange="onChannelChange()"><option value="">Select channel...</option></select>
        </div>
        <div class="fg">
          <label>Topic / Video Idea</label>
          <div style="display:flex;gap:8px">
            <input type="text" id="cTopic" placeholder="Enter topic or click AI Suggest...">
            <button class="btn btn-p btn-sm" onclick="aiSuggestTopics()" id="btnSuggest">AI Suggest</button>
          </div>
        </div>
        <div id="topicIdeas" class="hidden"></div>
      </div>

      <!-- Aspect Ratio -->
      <div class="panel">
        <div class="panel-title">Resolution</div>
        <div class="ar-grid" id="arGrid"></div>
        <div class="row row-2 hidden" id="customRes">
          <div class="fg"><label>Width</label><input type="number" id="cW" value="1024" min="64" max="2048" step="32"></div>
          <div class="fg"><label>Height</label><input type="number" id="cH" value="576" min="64" max="2048" step="32"></div>
        </div>
      </div>

      <!-- Gen Settings -->
      <div class="panel">
        <div class="panel-title">Generation Settings</div>
        <div class="row row-3">
          <div class="fg"><label>Sec / Clip</label><input type="number" id="cSpc" value="5" min="1" max="30" step="0.5"></div>
          <div class="fg"><label>FPS</label><input type="number" id="cFps" value="24" min="8" max="60"></div>
          <div class="fg"><label>Guide Strength</label><input type="number" id="cGs" value="0.99" min="0" max="1" step="0.01"></div>
        </div>
        <div class="fg"><label>Negative Prompt</label><input type="text" id="cNeg" value="blurry, low res, low quality, bad anatomy, worst quality, text, watermark, ugly, deformed, morphed, bad proportions, unnatural movement, jittery, still frame"></div>
        <div class="fg"><label>Global Prompt</label><input type="text" id="cGlobal" placeholder="Optional..."></div>
        <div class="row row-2">
          <div class="fg"><label>Output Dir</label><input type="text" id="cOutDir" value="D:/assets_genrated/ltx_output"></div>
          <div class="fg"><label>Filename Prefix</label><input type="text" id="cPrefix" value="%date:yyyy-MM-dd%/LTX-2_3"></div>
        </div>
      </div>

      <!-- SEO -->
      <div class="panel">
        <div class="panel-title">SEO Metadata <span class="badge badge-blue">AI</span></div>
        <button class="btn btn-p btn-sm" onclick="aiGenerateSeo()" id="btnSeo">Generate SEO</button>
        <div id="seoResults" class="hidden" style="margin-top:12px">
          <label>Pick a Title</label>
          <div id="seoTitles"></div>
          <div class="fg" style="margin-top:10px"><label>Description</label><textarea id="seoDesc" rows="6"></textarea></div>
          <div class="fg"><label>Tags</label><div class="tag-cloud" id="seoTags"></div></div>
        </div>
      </div>
    </div>

    <!-- RIGHT -->
    <div>
      <!-- Prompts -->
      <div class="panel" style="display:flex;flex-direction:column">
        <div class="panel-title">Clip Prompts <span class="badge badge-purple" id="clipBadge">0</span></div>
        <div class="btn-row" style="margin-top:0;margin-bottom:10px">
          <button class="btn btn-p btn-sm" onclick="aiGeneratePrompts()" id="btnGenPrompts">AI Generate</button>
          <button class="btn btn-s btn-sm" onclick="loadFairyTemplate()">Fairy Template</button>
          <button class="btn btn-s btn-sm" onclick="document.getElementById('cPrompts').value='';updateStats()">Clear</button>
        </div>
        <textarea id="cPrompts" rows="22" placeholder="Enter clip prompts separated by ---" oninput="updateStats()"></textarea>
      </div>

      <!-- Stats + Actions -->
      <div class="panel">
        <div class="stats" style="margin-bottom:14px">
          <div class="stat"><div class="v" id="stClips">0</div><div class="l">Clips</div></div>
          <div class="stat"><div class="v" id="stDur">0s</div><div class="l">Duration</div></div>
          <div class="stat"><div class="v" id="stFrames">0</div><div class="l">Frames</div></div>
          <div class="stat"><div class="v" id="stRes">—</div><div class="l">Resolution</div></div>
        </div>
        <div class="btn-row" style="justify-content:flex-end">
          <button class="btn btn-p btn-lg" onclick="generateWorkflow()">Generate Workflow</button>
          <button class="btn btn-ok btn-lg" onclick="generateAndRun()">Generate + Run ComfyUI</button>
        </div>
        <div id="genStatus" class="hidden" style="margin-top:12px;padding:12px;border-radius:8px;background:var(--s2);font-size:13px"></div>
      </div>
    </div>
  </div>
</div>

<!-- ═══ VIDEO LIBRARY ═══ -->
<div class="page" id="page-videos">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
    <h2>Video Library</h2>
    <div style="display:flex;gap:10px">
      <select id="vFilterCh" onchange="loadVideos()" style="width:180px"><option value="">All channels</option></select>
      <select id="vFilterSt" onchange="loadVideos()" style="width:140px">
        <option value="">All statuses</option>
        <option value="draft">Draft</option><option value="generated">Generated</option>
        <option value="rendering">Rendering</option><option value="rendered">Rendered</option>
        <option value="published">Published</option>
      </select>
    </div>
  </div>
  <table class="tbl">
    <thead><tr><th>Title / Topic</th><th>Channel</th><th>Clips</th><th>Duration</th><th>Status</th><th>Created</th><th>Actions</th></tr></thead>
    <tbody id="videoRows"></tbody>
  </table>
</div>

<!-- ═══ SETTINGS ═══ -->
<div class="page" id="page-settings">
  <h2 style="margin-bottom:20px">Settings</h2>
  <div class="split">
    <div class="panel">
      <div class="panel-title">AI Provider</div>
      <div class="fg"><label>Active Provider</label>
        <select id="sProvider" onchange="saveProviderSetting()"><option value="claude">Claude (Anthropic)</option><option value="gemini">Gemini (Google)</option></select>
      </div>
      <div class="fg"><label>Claude API Key</label>
        <div style="display:flex;gap:8px"><input type="password" id="sClaudeKey" placeholder="sk-ant-..."><button class="btn btn-s btn-sm" onclick="saveApiKey('claude')">Save</button></div>
      </div>
      <div class="fg"><label>Gemini API Key</label>
        <div style="display:flex;gap:8px"><input type="password" id="sGeminiKey" placeholder="AI..."><button class="btn btn-s btn-sm" onclick="saveApiKey('gemini')">Save</button></div>
      </div>
      <div id="settingsStatus" style="margin-top:8px;font-size:13px;color:var(--tx2)"></div>
    </div>
    <div class="panel">
      <div class="panel-title">Paths</div>
      <div class="fg"><label>ComfyUI Directory</label><input type="text" id="sComfyDir" value=""></div>
      <div class="fg"><label>Default Output Directory</label><input type="text" id="sOutDir" value=""></div>
      <div class="btn-row">
        <button class="btn btn-p btn-sm" onclick="savePaths()">Save Paths</button>
        <button class="btn btn-err btn-sm" onclick="killComfyUI()">Kill ComfyUI</button>
      </div>
    </div>
  </div>
</div>

<!-- ═══ CHANNEL MODAL ═══ -->
<div class="modal-bg hidden" id="channelModal" onclick="if(event.target===this)this.classList.add('hidden')">
  <div class="modal">
    <h2 id="chModalTitle">New Channel</h2>
    <input type="hidden" id="chId">
    <div class="row row-2">
      <div class="fg"><label>Name</label><input type="text" id="chName"></div>
      <div class="fg"><label>Niche</label><input type="text" id="chNiche"></div>
    </div>
    <div class="fg"><label>Visual Style Prompt</label><textarea id="chStyle" rows="3"></textarea></div>
    <div class="row row-2">
      <div class="fg"><label>Target Audience</label><input type="text" id="chAudience"></div>
      <div class="fg"><label>Voice Style</label><input type="text" id="chVoice"></div>
    </div>
    <div class="row row-2">
      <div class="fg"><label>Language</label><input type="text" id="chLang" value="English"></div>
      <div class="fg"><label>Made for Kids</label><select id="chKids"><option value="1">Yes</option><option value="0">No</option></select></div>
    </div>
    <div class="fg"><label>Keywords (comma-separated)</label><input type="text" id="chKeywords"></div>
    <div class="fg"><label>Default Negative Prompt</label><input type="text" id="chNegPrompt"></div>
    <div class="btn-row">
      <button class="btn btn-p" onclick="saveChannel()">Save Channel</button>
      <button class="btn btn-s" onclick="document.getElementById('channelModal').classList.add('hidden')">Cancel</button>
      <button class="btn btn-err btn-sm hidden" id="chDeleteBtn" onclick="deleteChannel()">Delete</button>
    </div>
  </div>
</div>

<!-- ═══ VIDEO DETAIL MODAL ═══ -->
<div class="modal-bg hidden" id="videoModal" onclick="if(event.target===this)this.classList.add('hidden')">
  <div class="modal" style="width:700px">
    <h2 id="vmTitle">Video Details</h2>
    <div id="vmContent"></div>
  </div>
</div>

<script>
// ═══ State ═══
let channels = [], currentAR = '16:9 (1024x576)', lastWorkflowPath = '', WORKFLOW_PATH = '';

// ═══ Tabs ═══
function showTab(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('page-'+name).classList.add('active');
  document.querySelectorAll('.tab-btn').forEach(b => { if(b.textContent.toLowerCase().includes(name.replace('dashboard','dash'))) b.classList.add('active'); });
  // find exact match
  document.querySelectorAll('.tab-btn').forEach(b => {
    const map = {dashboard:'Dashboard',create:'Create Video',videos:'Video Library',settings:'Settings'};
    if(b.textContent.trim()===map[name]) { b.classList.add('active'); }
  });
  if(name==='dashboard') { loadChannels(); loadRecentVideos(); }
  if(name==='create') loadChannelDropdown();
  if(name==='videos') { loadVideoFilters(); loadVideos(); }
  if(name==='settings') loadSettings();
}

// ═══ Toast ═══
function toast(msg, type='info') {
  const d = document.createElement('div');
  d.className = 'toast toast-'+type;
  d.textContent = msg;
  document.body.appendChild(d);
  setTimeout(() => d.remove(), 4000);
}

// ═══ API helpers ═══
async function api(url, opts={}) {
  if(opts.body && typeof opts.body==='object') {
    opts.headers = {'Content-Type':'application/json',...(opts.headers||{})};
    opts.body = JSON.stringify(opts.body);
  }
  const r = await fetch(url, opts);
  const j = await r.json();
  if(j.error) throw new Error(j.error);
  return j;
}

// ═══ Dashboard ═══
async function loadChannels() {
  channels = await api('/api/channels');
  const el = document.getElementById('channelCards');
  if(!channels.length) { el.innerHTML='<p style="color:var(--tx2)">No channels yet. Create one!</p>'; return; }
  el.innerHTML = channels.map(c => `
    <div class="card" onclick="editChannel(${c.id})">
      <h3>${esc(c.name)}</h3>
      <div class="meta">${esc(c.niche||'')}</div>
      <div class="meta" style="margin-top:6px">${c.video_count} videos &middot; ${c.published_count} published</div>
      <div class="meta">${c.for_kids?'Made for Kids':''} ${esc(c.language||'')}</div>
      <div class="btn-row"><button class="btn btn-p btn-sm" onclick="event.stopPropagation();startVideoForChannel(${c.id})">+ New Video</button></div>
    </div>
  `).join('');
}

async function loadRecentVideos() {
  try {
    const vids = await api('/api/videos');
    const el = document.getElementById('recentVideos');
    if(!vids.length) { el.innerHTML='<p style="color:var(--tx2)">No videos generated yet.</p>'; return; }
    el.innerHTML = '<table class="tbl"><thead><tr><th>Topic</th><th>Channel</th><th>Status</th><th>Created</th></tr></thead><tbody>' +
      vids.slice(0,10).map(v => `<tr style="cursor:pointer" onclick="openVideoDetail(${v.id})">
        <td>${esc(v.topic||v.title||'Untitled')}</td><td>${esc(v.channel_name||'')}</td>
        <td><span class="status-pill st-${v.status}">${v.status}</span></td>
        <td>${v.created_at?.slice(0,16)||''}</td></tr>`).join('') + '</tbody></table>';
  } catch(e) {}
}

function startVideoForChannel(cid) {
  showTab('create');
  setTimeout(() => { document.getElementById('cChannel').value = cid; onChannelChange(); }, 100);
}

// ═══ Channel Modal ═══
function openChannelModal(ch=null) {
  document.getElementById('chModalTitle').textContent = ch ? 'Edit Channel' : 'New Channel';
  document.getElementById('chId').value = ch?.id || '';
  document.getElementById('chName').value = ch?.name || '';
  document.getElementById('chNiche').value = ch?.niche || '';
  document.getElementById('chStyle').value = ch?.style_prompt || '';
  document.getElementById('chAudience').value = ch?.target_audience || '';
  document.getElementById('chVoice').value = ch?.voice_style || '';
  document.getElementById('chLang').value = ch?.language || 'English';
  document.getElementById('chKids').value = ch?.for_kids ? '1' : '0';
  document.getElementById('chKeywords').value = ch?.keywords || '';
  document.getElementById('chNegPrompt').value = ch?.negative_prompt || '';
  document.getElementById('chDeleteBtn').classList.toggle('hidden', !ch);
  document.getElementById('channelModal').classList.remove('hidden');
}

function editChannel(id) { const ch = channels.find(c=>c.id===id); if(ch) openChannelModal(ch); }

async function saveChannel() {
  const id = document.getElementById('chId').value;
  const data = {
    name: document.getElementById('chName').value,
    niche: document.getElementById('chNiche').value,
    style_prompt: document.getElementById('chStyle').value,
    target_audience: document.getElementById('chAudience').value,
    voice_style: document.getElementById('chVoice').value,
    language: document.getElementById('chLang').value,
    for_kids: document.getElementById('chKids').value === '1',
    keywords: document.getElementById('chKeywords').value,
    negative_prompt: document.getElementById('chNegPrompt').value,
  };
  try {
    if(id) await api(`/api/channels/${id}`, {method:'PUT', body:data});
    else await api('/api/channels', {method:'POST', body:data});
    document.getElementById('channelModal').classList.add('hidden');
    toast('Channel saved!', 'ok');
    loadChannels();
  } catch(e) { toast(e.message, 'err'); }
}

async function deleteChannel() {
  const id = document.getElementById('chId').value;
  if(!id || !confirm('Delete this channel and all its videos?')) return;
  await api(`/api/channels/${id}`, {method:'DELETE'});
  document.getElementById('channelModal').classList.add('hidden');
  toast('Channel deleted', 'ok');
  loadChannels();
}

// ═══ Create Video ═══
async function loadChannelDropdown() {
  if(!channels.length) channels = await api('/api/channels');
  const sel = document.getElementById('cChannel');
  const cur = sel.value;
  sel.innerHTML = '<option value="">Select channel...</option>' + channels.map(c => `<option value="${c.id}">${esc(c.name)}</option>`).join('');
  if(cur) sel.value = cur;
  buildARGrid();
}

function onChannelChange() {
  const cid = document.getElementById('cChannel').value;
  const ch = channels.find(c => c.id == cid);
  if(ch) {
    document.getElementById('cNeg').value = ch.negative_prompt || '';
    document.getElementById('cGlobal').value = ch.global_prompt || '';
  }
}

function buildARGrid() {
  const grid = document.getElementById('arGrid');
  grid.innerHTML = Object.entries(ASPECT_RATIOS).map(([name, [w,h]]) =>
    `<button class="ar-btn${name===currentAR?' active':''}" onclick="selectAR(this,'${name}',${w},${h})">${name}</button>`
  ).join('');
}
const ASPECT_RATIOS = {"16:9 (1024x576)":[1024,576],"16:9 (768x432)":[768,432],"9:16 (576x1024)":[576,1024],"1:1 (576x576)":[576,576],"1:1 (768x768)":[768,768],"4:3 (768x576)":[768,576],"21:9 (1024x432)":[1024,432]};

function selectAR(btn, name, w, h) {
  document.querySelectorAll('.ar-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  currentAR = name;
  document.getElementById('cW').value = w;
  document.getElementById('cH').value = h;
  updateStats();
}

function getRes() {
  return { w: parseInt(document.getElementById('cW').value)||1024, h: parseInt(document.getElementById('cH').value)||576 };
}

function countClips() {
  const t = document.getElementById('cPrompts').value.trim();
  if(!t) return 0;
  return t.includes('---') ? t.split(/\n?---+\n?/).filter(p=>p.trim()).length : t.split(/\n\s*\n/).filter(p=>p.trim()).length;
}

function updateStats() {
  const clips = countClips();
  const spc = parseFloat(document.getElementById('cSpc').value)||5;
  const fps = parseInt(document.getElementById('cFps').value)||24;
  const res = getRes();
  const dur = clips*spc;
  document.getElementById('stClips').textContent = clips;
  document.getElementById('stDur').textContent = dur>=60 ? (dur/60).toFixed(1)+'m' : dur+'s';
  document.getElementById('stFrames').textContent = clips*Math.round(spc*fps);
  document.getElementById('stRes').textContent = res.w+'x'+res.h;
  document.getElementById('clipBadge').textContent = clips;
}

// ═══ AI Functions ═══
async function aiSuggestTopics() {
  const cid = document.getElementById('cChannel').value;
  if(!cid) { toast('Select a channel first', 'err'); return; }
  const btn = document.getElementById('btnSuggest');
  btn.disabled = true; btn.innerHTML = '<span class="loading"></span>';
  try {
    const r = await api('/api/ai/suggest-topics', {method:'POST', body:{channel_id:parseInt(cid)}});
    const el = document.getElementById('topicIdeas');
    el.classList.remove('hidden');
    el.innerHTML = (r.ideas||[]).map(i => `
      <div class="idea-card" onclick="document.getElementById('cTopic').value='${esc(i.topic)}';document.getElementById('topicIdeas').classList.add('hidden')">
        <h4>${esc(i.topic)}</h4>
        <p>${esc(i.hook||'')} &mdash; ${esc(i.why||'')}</p>
      </div>
    `).join('');
  } catch(e) { toast(e.message, 'err'); }
  btn.disabled = false; btn.textContent = 'AI Suggest';
}

async function aiGeneratePrompts() {
  const cid = document.getElementById('cChannel').value;
  const topic = document.getElementById('cTopic').value;
  if(!cid) { toast('Select a channel first', 'err'); return; }
  if(!topic) { toast('Enter a topic first', 'err'); return; }
  const btn = document.getElementById('btnGenPrompts');
  btn.disabled = true; btn.innerHTML = '<span class="loading"></span> Generating...';
  try {
    const r = await api('/api/ai/generate-prompts', {method:'POST', body:{
      channel_id: parseInt(cid), topic,
      num_clips: parseInt(document.getElementById('cSpc').value) <= 3 ? 10 : 6,
      seconds_per_clip: parseFloat(document.getElementById('cSpc').value)||5,
    }});
    document.getElementById('cPrompts').value = (r.prompts||[]).join('\n---\n');
    if(r.topic) document.getElementById('cTopic').value = r.topic;
    updateStats();
    toast('Prompts generated!', 'ok');
  } catch(e) { toast(e.message, 'err'); }
  btn.disabled = false; btn.textContent = 'AI Generate';
}

async function aiGenerateSeo() {
  const cid = document.getElementById('cChannel').value;
  const prompts = document.getElementById('cPrompts').value;
  if(!cid) { toast('Select a channel first', 'err'); return; }
  if(!prompts.trim()) { toast('Generate prompts first', 'err'); return; }
  const btn = document.getElementById('btnSeo');
  btn.disabled = true; btn.innerHTML = '<span class="loading"></span> Generating...';
  try {
    const r = await api('/api/ai/generate-seo', {method:'POST', body:{channel_id:parseInt(cid), prompts_text:prompts}});
    const el = document.getElementById('seoResults');
    el.classList.remove('hidden');
    document.getElementById('seoTitles').innerHTML = (r.titles||[]).map((t,i) =>
      `<div class="seo-title-opt${i===0?' selected':''}" onclick="selectSeoTitle(this)">${esc(t)}</div>`
    ).join('');
    document.getElementById('seoDesc').value = r.description || '';
    document.getElementById('seoTags').innerHTML = (r.tags||[]).map(t => `<span class="tag-chip">${esc(t)}</span>`).join('');
    toast('SEO metadata generated!', 'ok');
  } catch(e) { toast(e.message, 'err'); }
  btn.disabled = false; btn.textContent = 'Generate SEO';
}

function selectSeoTitle(el) {
  document.querySelectorAll('.seo-title-opt').forEach(e=>e.classList.remove('selected'));
  el.classList.add('selected');
}

function loadFairyTemplate() {
  document.getElementById('cPrompts').value = `Cute magical fairy flying over glowing pink flower garden, sparkling particles, soft pastel colors, children's storybook illustration style, dreamy atmosphere
---
Slow cinematic drift through enchanted crystal forest, glowing mushrooms, fireflies, pastel purple and blue, soft volumetric light
---
Unicorn walking through rainbow flower field, butterflies, sparkles, pastel colors, gentle movement, fantasy world, children's book style
---
Magical castle floating in cotton candy clouds, rainbow waterfalls, tiny fairies dancing, warm golden sunlight, whimsical, dreamy
---
Gentle butterfly garden with oversized colorful flowers, dewdrops sparkling like diamonds, soft breeze, pastel sky at golden hour
---
Sparkling fairy dust trail swirling through moonlit enchanted garden, glowing flowers closing for night, peaceful blue and purple tones, calming bedtime atmosphere`;
  updateStats();
}

// ═══ Workflow Generation ═══
async function generateWorkflow(andRun=false) {
  const prompts = document.getElementById('cPrompts').value.trim();
  if(!prompts) { toast('Enter prompts first', 'err'); return; }
  const res = getRes();
  const selectedTitle = document.querySelector('.seo-title-opt.selected')?.textContent || '';
  const body = {
    prompts_text: prompts,
    seconds_per_clip: parseFloat(document.getElementById('cSpc').value),
    frame_rate: parseInt(document.getElementById('cFps').value),
    guide_strength: parseFloat(document.getElementById('cGs').value),
    width: res.w, height: res.h,
    negative_prompt: document.getElementById('cNeg').value,
    global_prompt: document.getElementById('cGlobal').value,
    filename_prefix: document.getElementById('cPrefix').value,
    output_dir: document.getElementById('cOutDir').value,
    workflow_path: WORKFLOW_PATH,
    channel_id: parseInt(document.getElementById('cChannel').value) || null,
    topic: document.getElementById('cTopic').value,
    title: selectedTitle,
  };
  const el = document.getElementById('genStatus');
  el.classList.remove('hidden');
  el.innerHTML = '<span class="loading"></span> Generating workflow...';
  try {
    const r = await api('/api/generate', {method:'POST', body});
    lastWorkflowPath = r.output_file;
    el.innerHTML = `<span style="color:var(--ok)">Workflow saved:</span> ${esc(r.output_file)}<br>
      ${r.clip_count} clips, ${r.total_seconds.toFixed(0)}s, ${r.resolution}
      <br><button class="btn btn-s btn-sm" style="margin-top:8px" onclick="copyPath('${r.output_file.replace(/\\/g,'\\\\')}')">Copy Path</button>`;
    toast('Workflow generated!', 'ok');
    if(andRun) runComfyUI(r.output_file);
  } catch(e) {
    el.innerHTML = `<span style="color:var(--err)">Error:</span> ${esc(e.message)}`;
    toast(e.message, 'err');
  }
}

async function generateAndRun() { await generateWorkflow(true); }

async function runComfyUI(wfPath) {
  const el = document.getElementById('genStatus');
  el.classList.remove('hidden');
  el.innerHTML += '<br><span class="loading"></span> Starting ComfyUI & queuing workflow...';
  try {
    const r = await api('/api/comfyui/run', {method:'POST', body:{workflow_path: wfPath||lastWorkflowPath}});
    if(r.queued) {
      el.innerHTML += `<br><span style="color:var(--ok)">Workflow queued and rendering!</span> Prompt ID: ${esc(r.prompt_id||'')}
        <br><button class="btn btn-s btn-sm" style="margin-top:6px" onclick="checkQueueStatus()">Check Progress</button>`;
      toast('Workflow queued in ComfyUI!', 'ok');
      checkComfyUI();
    } else {
      el.innerHTML += `<br><span style="color:var(--warn)">${esc(r.message)}</span>
        <br><button class="btn btn-s btn-sm" style="margin-top:6px" onclick="copyPath('${(wfPath||lastWorkflowPath).replace(/\\/g,'\\\\')}')">Copy Workflow Path</button>`;
      toast('ComfyUI running — load workflow manually', 'info');
    }
  } catch(e) { el.innerHTML += `<br><span style="color:var(--err)">${esc(e.message)}</span>`; }
}

async function checkQueueStatus() {
  try {
    const r = await api('/api/comfyui/queue-status');
    const el = document.getElementById('genStatus');
    if(r.total===0) {
      el.innerHTML += `<br><span style="color:var(--ok)">Queue empty — rendering complete!</span>`;
      toast('Rendering complete!', 'ok');
    } else {
      el.innerHTML += `<br>Queue: ${r.running} running, ${r.pending} pending`;
      toast(`Queue: ${r.running} running, ${r.pending} pending`, 'info');
    }
  } catch(e) { toast(e.message, 'err'); }
}

function copyPath(p) { navigator.clipboard.writeText(p); toast('Path copied!', 'ok'); }

// ═══ Video Library ═══
async function loadVideoFilters() {
  if(!channels.length) channels = await api('/api/channels');
  const sel = document.getElementById('vFilterCh');
  sel.innerHTML = '<option value="">All channels</option>' + channels.map(c => `<option value="${c.id}">${esc(c.name)}</option>`).join('');
}

async function loadVideos() {
  const cid = document.getElementById('vFilterCh').value;
  const st = document.getElementById('vFilterSt').value;
  let url = '/api/videos?';
  if(cid) url += `channel_id=${cid}&`;
  if(st) url += `status=${st}&`;
  const vids = await api(url);
  const el = document.getElementById('videoRows');
  if(!vids.length) { el.innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--tx2);padding:40px">No videos found</td></tr>'; return; }
  el.innerHTML = vids.map(v => `<tr>
    <td style="cursor:pointer" onclick="openVideoDetail(${v.id})">${esc(v.topic||v.title||'Untitled')}</td>
    <td>${esc(v.channel_name||'')}</td>
    <td>${v.clip_count||0}</td>
    <td>${v.total_seconds ? v.total_seconds.toFixed(0)+'s' : '—'}</td>
    <td><span class="status-pill st-${v.status}">${v.status}</span></td>
    <td>${v.created_at?.slice(0,16)||''}</td>
    <td>
      <div style="display:flex;gap:4px">
        ${v.status==='generated'?`<button class="btn btn-ok btn-sm" onclick="event.stopPropagation();markRendered(${v.id})">Rendered</button>`:''}
        ${v.status==='rendered'?`<button class="btn btn-ok btn-sm" onclick="event.stopPropagation();markPublished(${v.id})">Publish</button>`:''}
        ${v.workflow_path?`<button class="btn btn-s btn-sm" onclick="event.stopPropagation();runComfyUI('${v.workflow_path.replace(/\\/g,'\\\\')}')">Run</button>`:''}
        <button class="btn btn-err btn-sm" onclick="event.stopPropagation();deleteVideo(${v.id})">Del</button>
      </div>
    </td>
  </tr>`).join('');
}

async function markRendered(id) { await api(`/api/videos/${id}`,{method:'PUT',body:{status:'rendered'}}); loadVideos(); toast('Marked as rendered','ok'); }
async function markPublished(id) {
  const url = prompt('YouTube URL (optional):','');
  await api(`/api/videos/${id}`,{method:'PUT',body:{status:'published', yt_url:url||''}});
  loadVideos(); toast('Published!','ok');
}
async function deleteVideo(id) { if(!confirm('Delete this video?')) return; await api(`/api/videos/${id}`,{method:'DELETE'}); loadVideos(); }

async function openVideoDetail(id) {
  const v = await api(`/api/videos/${id}`);
  document.getElementById('vmTitle').textContent = v.topic || v.title || 'Video Details';
  let seo = {};
  try { seo = JSON.parse(v.seo_json||'{}'); } catch(e) {}
  document.getElementById('vmContent').innerHTML = `
    <div class="row row-2" style="margin-bottom:12px">
      <div><strong>Channel:</strong> ${esc(v.channel_name)}</div>
      <div><strong>Status:</strong> <span class="status-pill st-${v.status}">${v.status}</span></div>
    </div>
    <div class="row row-3" style="margin-bottom:12px">
      <div><strong>Clips:</strong> ${v.clip_count}</div>
      <div><strong>Duration:</strong> ${v.total_seconds?v.total_seconds.toFixed(0)+'s':'—'}</div>
      <div><strong>Resolution:</strong> ${v.resolution||'—'}</div>
    </div>
    ${v.title?`<div class="fg"><strong>Title:</strong> ${esc(v.title)}</div>`:''}
    ${v.yt_url?`<div class="fg"><strong>YouTube:</strong> <a href="${esc(v.yt_url)}" target="_blank" style="color:var(--ac2)">${esc(v.yt_url)}</a></div>`:''}
    ${v.workflow_path?`<div class="fg"><strong>Workflow:</strong> <code style="font-size:12px;color:var(--tx2)">${esc(v.workflow_path)}</code> <button class="btn btn-s btn-sm" onclick="copyPath('${v.workflow_path.replace(/\\/g,'\\\\')}')">Copy</button></div>`:''}
    <div class="fg"><strong>Prompts:</strong><pre style="background:var(--s2);padding:12px;border-radius:8px;font-size:12px;max-height:200px;overflow-y:auto;white-space:pre-wrap">${esc(v.prompts_text||'')}</pre></div>
    ${seo.description?`<div class="fg"><strong>Description:</strong><pre style="background:var(--s2);padding:12px;border-radius:8px;font-size:12px;max-height:150px;overflow-y:auto;white-space:pre-wrap">${esc(seo.description)}</pre></div>`:''}
    ${seo.tags?`<div class="fg"><strong>Tags:</strong><div class="tag-cloud">${seo.tags.map(t=>`<span class="tag-chip">${esc(t)}</span>`).join('')}</div></div>`:''}
    <div class="btn-row">
      <input type="text" id="vmYtUrl" value="${esc(v.yt_url||'')}" placeholder="YouTube URL" style="flex:1">
      <button class="btn btn-p btn-sm" onclick="updateVideoUrl(${v.id})">Save URL</button>
      ${v.status==='rendered'?`<button class="btn btn-s btn-sm" onclick="api('/api/videos/${v.id}/stitch',{method:'POST'}).then(r=>{toast('Stitched!','ok');loadVideos();openVideoDetail(${v.id});}).catch(e=>toast(e.message,'err'))">Stitch Chunks</button>`:''}
      ${v.status!=='published'?`<button class="btn btn-ok btn-sm" onclick="api('/api/videos/${v.id}',{method:'PUT',body:{status:'published',yt_url:document.getElementById('vmYtUrl').value}}).then(()=>{toast('Published!','ok');document.getElementById('videoModal').classList.add('hidden');loadVideos()})">Mark Published</button>`:''}
    </div>
  `;
  document.getElementById('videoModal').classList.remove('hidden');
}

async function updateVideoUrl(id) {
  await api(`/api/videos/${id}`,{method:'PUT',body:{yt_url:document.getElementById('vmYtUrl').value}});
  toast('URL saved','ok');
}

// ═══ Settings ═══
async function loadSettings() {
  const s = await api('/api/settings');
  document.getElementById('sProvider').value = s.ai_provider || 'claude';
  document.getElementById('sClaudeKey').value = s.claude_api_key_set ? '••••••••' : '';
  document.getElementById('sGeminiKey').value = s.gemini_api_key_set ? '••••••••' : '';
  document.getElementById('sComfyDir').value = s.comfyui_dir || '';
  document.getElementById('sOutDir').value = s.output_dir || '';
  document.getElementById('settingsStatus').innerHTML =
    `Claude: ${s.claude_api_key_set?'<span style="color:var(--ok)">Key set</span>':'<span style="color:var(--err)">Not set</span>'} &middot; ` +
    `Gemini: ${s.gemini_api_key_set?'<span style="color:var(--ok)">Key set</span>':'<span style="color:var(--err)">Not set</span>'} &middot; ` +
    `Available: ${s.available_providers.join(', ')||'none (install anthropic or google-generativeai)'}`;
  updateProviderBadge(s.ai_provider);
}

async function saveProviderSetting() {
  const p = document.getElementById('sProvider').value;
  await api('/api/settings', {method:'POST', body:{ai_provider:p}});
  updateProviderBadge(p);
  toast('Provider switched to '+p, 'ok');
}

async function saveApiKey(provider) {
  const key = document.getElementById(provider==='claude'?'sClaudeKey':'sGeminiKey').value;
  if(!key || key.includes('••')) { toast('Enter a valid key', 'err'); return; }
  await api('/api/settings', {method:'POST', body:{[provider+'_api_key']:key}});
  toast(provider+' key saved!', 'ok');
  loadSettings();
}

async function savePaths() {
  await api('/api/settings', {method:'POST', body:{
    comfyui_dir: document.getElementById('sComfyDir').value,
    output_dir: document.getElementById('sOutDir').value,
  }});
  toast('Paths saved!', 'ok');
}

function updateProviderBadge(p) {
  document.getElementById('providerBadge').textContent = 'AI: ' + (p||'—').charAt(0).toUpperCase() + (p||'').slice(1);
}

// ═══ ComfyUI Status ═══
async function checkComfyUI() {
  try {
    const r = await api('/api/comfyui/status');
    document.getElementById('comfyDot').className = 'comfy-dot ' + (r.running?'on':'off');
  } catch(e) { document.getElementById('comfyDot').className = 'comfy-dot off'; }
}


async function killComfyUI() {
  if(!confirm('Forcefully kill ComfyUI?')) return;
  try {
    const r = await api('/api/comfyui/kill', {method:'POST'});
    toast(r.message, 'ok');
    setTimeout(checkComfyUI, 1000);
  } catch(e) { toast(e.message, 'err'); }
}

// ═══ Utils ═══
function esc(s) { const d=document.createElement('div'); d.textContent=s||''; return d.innerHTML; }

// ═══ Init ═══
api('/api/config').then(c => { WORKFLOW_PATH = c.workflow_path; document.getElementById('cOutDir').value = c.output_dir || document.getElementById('cOutDir').value; }).catch(()=>{});
loadChannels();
loadRecentVideos();
checkComfyUI();
setInterval(checkComfyUI, 15000);
loadSettings().catch(()=>{});
document.getElementById('cSpc').addEventListener('input', updateStats);
document.getElementById('cFps').addEventListener('input', updateStats);
updateStats();
</script>
</body>
</html>'''
