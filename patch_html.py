import os

with open('video_maker.py', 'r', encoding='utf-8') as f:
    text = f.read()

settings_html = """      <div class="fg"><label>ComfyUI Directory</label><input type="text" id="sComfyDir" value=""></div>
      <div class="fg"><label>Default Output Directory</label><input type="text" id="sOutDir" value=""></div>
      <div class="btn-row">
        <button class="btn btn-p btn-sm" onclick="savePaths()">Save Paths</button>
        <button class="btn btn-err btn-sm" onclick="killComfyUI()">Kill ComfyUI</button>
      </div>"""
text = text.replace("""      <div class="fg"><label>ComfyUI Directory</label><input type="text" id="sComfyDir" value=""></div>
      <div class="fg"><label>Default Output Directory</label><input type="text" id="sOutDir" value=""></div>
      <button class="btn btn-p btn-sm" onclick="savePaths()">Save Paths</button>""", settings_html)

stitch_html = """    <div class="btn-row">
      <input type="text" id="vmYtUrl" value="${esc(v.yt_url||'')}" placeholder="YouTube URL" style="flex:1">
      <button class="btn btn-p btn-sm" onclick="updateVideoUrl(${v.id})">Save URL</button>
      ${v.status==='rendered'?`<button class="btn btn-s btn-sm" onclick="api('/api/videos/${v.id}/stitch',{method:'POST'}).then(r=>{toast('Stitched!','ok');loadVideos();openVideoDetail(${v.id});}).catch(e=>toast(e.message,'err'))">Stitch Chunks</button>`:''}
      ${v.status!=='published'?`<button class="btn btn-ok btn-sm" onclick="api('/api/videos/${v.id}',{method:'PUT',body:{status:'published',yt_url:document.getElementById('vmYtUrl').value}}).then(()=>{toast('Published!','ok');document.getElementById('videoModal').classList.add('hidden');loadVideos()})">Mark Published</button>`:''}
    </div>"""
text = text.replace("""    <div class="btn-row">
      <input type="text" id="vmYtUrl" value="${esc(v.yt_url||'')}" placeholder="YouTube URL" style="flex:1">
      <button class="btn btn-p btn-sm" onclick="updateVideoUrl(${v.id})">Save URL</button>
      ${v.status!=='published'?`<button class="btn btn-ok btn-sm" onclick="api('/api/videos/${v.id}',{method:'PUT',body:{status:'published',yt_url:document.getElementById('vmYtUrl').value}}).then(()=>{toast('Published!','ok');document.getElementById('videoModal').classList.add('hidden');loadVideos()})">Mark Published</button>`:''}
    </div>""", stitch_html)

kill_js = """
async function killComfyUI() {
  if(!confirm('Forcefully kill ComfyUI?')) return;
  try {
    const r = await api('/api/comfyui/kill', {method:'POST'});
    toast(r.message, 'ok');
    setTimeout(checkComfyUI, 1000);
  } catch(e) { toast(e.message, 'err'); }
}

// ═══ Utils ═══"""
text = text.replace('// ═══ Utils ═══', kill_js)

with open('video_maker.py', 'w', encoding='utf-8') as f:
    f.write(text)
