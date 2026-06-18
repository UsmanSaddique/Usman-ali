import re

html_path = r"c:\Users\PC\Desktop\VideoMaker\ai-director\frontend\index.html"

with open(html_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update the Scenes card header
scene_header_old = """      <!-- Scenes -->
      <div class="card">
        <h3>Scenes <span class="text-dim text-sm" id="scene-count">—</span></h3>
        <div class="scene-grid" id="scene-grid"></div>
      </div>"""

scene_header_new = """      <!-- Scenes -->
      <div class="card">
        <h3 class="flex-between">
          <span>Scenes <span class="text-dim text-sm" id="scene-count">—</span></span>
          <div class="btn-group">
            <button class="btn-outline btn-sm" onclick="selectAllScenes()">Select All</button>
            <button class="btn-outline btn-sm" onclick="deselectAllScenes()">None</button>
            <button class="btn-primary btn-sm hidden" id="btn-generate-selected" onclick="openGenerateModal()">Generate Selected</button>
          </div>
        </h3>
        <div class="scene-grid" id="scene-grid"></div>
      </div>"""

if scene_header_old in content:
    content = content.replace(scene_header_old, scene_header_new)

# 2. Add the Generate Modal
modal_old = """<!-- ═══ Scene Edit Modal ═══ -->"""
modal_new = """<!-- ═══ Generate Scenes Modal ═══ -->
<div class="modal-overlay hidden" id="generate-scenes-modal">
  <div class="modal">
    <h3>Generate Selected Scenes (<span id="gen-scenes-count">0</span>)</h3>
    <div class="form-grid">
      <div class="form-group full">
        <div class="section-label">Video Model</div>
        <div id="gen-model-grid" class="model-grid"></div>
        <input type="hidden" id="gen-video-model" value="">
      </div>
      <div class="form-group full">
        <div class="section-label">LoRAs (optional)</div>
        <div id="gen-loras-list" class="lora-grid"></div>
      </div>
    </div>
    <div class="btn-group mt-16">
      <button class="btn-generate" onclick="submitGenerateScenes()">
        <span class="btn-icon">▶</span> Start Generation
      </button>
      <button class="btn-outline" onclick="closeGenerateModal()">Cancel</button>
    </div>
  </div>
</div>

<!-- ═══ Scene Edit Modal ═══ -->"""

if modal_old in content:
    content = content.replace(modal_old, modal_new)

# 3. Update loadModelsAndLoras to render into the new modal instead of 'newProject'
js_models_old = """function loadModelsAndLoras() {
  api('/api/models').then(data => {
    _availableModels = data.video || [];
    buildModelGrid('new-model-grid', 'new-video-model', _availableModels, selectModel);
  });
  api('/api/loras').then(data => {
    _availableLoras = data || [];
    filterLoras();
  });
}"""

js_models_new = """function loadModelsAndLoras() {
  api('/api/models').then(data => {
    _availableModels = data.video || [];
    buildModelGrid('gen-model-grid', 'gen-video-model', _availableModels, selectModel);
  });
  api('/api/loras').then(data => {
    _availableLoras = data || [];
    filterLoras();
  });
}"""

if js_models_old in content:
    content = content.replace(js_models_old, js_models_new)

# 4. Update filterLoras to use gen-* inputs
js_filter_old = """function filterLoras() {
  const selectedModel = document.getElementById('new-video-model')?.value || '';
  const container = document.getElementById('new-loras-list');"""

js_filter_new = """function filterLoras() {
  const selectedModel = document.getElementById('gen-video-model')?.value || '';
  const container = document.getElementById('gen-loras-list');"""

if js_filter_old in content:
    content = content.replace(js_filter_old, js_filter_new)


# 5. Add UI logic for checkboxes, selections, and submitting
js_logic = """
// ── Scene Checkboxes & Generation Modal ────────────────────────────────
function selectAllScenes() {
  document.querySelectorAll('.scene-checkbox').forEach(cb => cb.checked = true);
  updateGenerateBtn();
}
function deselectAllScenes() {
  document.querySelectorAll('.scene-checkbox').forEach(cb => cb.checked = false);
  updateGenerateBtn();
}
function toggleSceneSelection(id) {
  const cb = document.querySelector(`.scene-checkbox[value="${id}"]`);
  if(cb) { cb.checked = !cb.checked; updateGenerateBtn(); }
}
function updateGenerateBtn() {
  const count = document.querySelectorAll('.scene-checkbox:checked').length;
  const btn = document.getElementById('btn-generate-selected');
  if (count > 0) {
    btn.classList.remove('hidden');
    btn.textContent = `Generate Selected (${count})`;
  } else {
    btn.classList.add('hidden');
  }
}
function openGenerateModal() {
  const count = document.querySelectorAll('.scene-checkbox:checked').length;
  document.getElementById('gen-scenes-count').textContent = count;
  loadModelsAndLoras();
  document.getElementById('generate-scenes-modal').classList.remove('hidden');
}
function closeGenerateModal() {
  document.getElementById('generate-scenes-modal').classList.add('hidden');
}
async function submitGenerateScenes() {
  const checkboxes = document.querySelectorAll('.scene-checkbox:checked');
  const sceneIds = Array.from(checkboxes).map(cb => cb.value);
  const videoModel = document.getElementById('gen-video-model').value;
  const selectedLoras = getSelectedLoras();
  
  if (sceneIds.length === 0) { alert('No scenes selected'); return; }
  if (!videoModel) { alert('Please select a video model'); return; }
  
  try {
    await api(`/api/projects/${currentProject.id}/generate-scenes`, 'POST', {
      scene_ids: sceneIds,
      video_model: videoModel,
      lora_ids: selectedLoras.ids,
      lora_weights: selectedLoras.weights
    });
    closeGenerateModal();
    deselectAllScenes();
    setTimeout(() => openProject(currentProject.id), 1000);
  } catch(e) {
    alert('Error: ' + e.message);
  }
}
"""

if "function selectAllScenes" not in content:
    content = content.replace("// ── Scene Management ───────────────────────────────────────────", js_logic + "\n// ── Scene Management ───────────────────────────────────────────")

# 6. Update renderProject to include checkboxes in scene cards
scene_card_regex = r"<div class=\"scene-card \$\{statusClass\}\" onclick=\"editScene\('\$\{s\.id\}', \$\{s\.scene_number\}\)\">.*?<div class=\"scene-num\">Scene \$\{s\.scene_number\} · \$\{s\.scene_type\}</div>"
scene_card_new = """<div class="scene-card ${statusClass}" onclick="toggleSceneSelection('${s.id}')">
      <div class="flex-between">
        <div class="scene-num" style="margin-bottom:0;">
          <input type="checkbox" class="scene-checkbox" value="${s.id}" onclick="event.stopPropagation(); updateGenerateBtn()">
          Scene ${s.scene_number} · ${s.scene_type}
        </div>
        <div class="btn-group">
          <button class="btn-outline btn-sm" onclick="event.stopPropagation(); editScene('${s.id}', ${s.scene_number})">Edit</button>
        </div>
      </div>"""
content = re.sub(scene_card_regex, scene_card_new, content, flags=re.DOTALL)

# 7. Remove Edit/Approve buttons from the bottom of the card since they moved or aren't strictly needed if we just use Edit.
# Or just let them be, but Edit is now at the top. Let's remove the bottom btn-group.
bottom_btns_regex = r"<div class=\"btn-group\">\s*<button class=\"btn-outline btn-sm\" onclick=\"event\.stopPropagation\(\); approveScene\('\$\{s\.id\}'\)\">✓ Approve</button>\s*<button class=\"btn-outline btn-sm\" onclick=\"event\.stopPropagation\(\); editScene\('\$\{s\.id\}', \$\{s\.scene_number\}\)\">Edit</button>\s*</div>"
content = re.sub(bottom_btns_regex, "", content)

# 8. Add regenerate capability to individual scene by hooking up the "Regenerate" button in the edit modal.
# The user asked: "i can also do resume / or start again thing etc please do that for me"
# I can change the "Regenerate" button inside editScene modal to just select that scene and open Generate modal.
regen_func_old = """async function regenerateScene() {
  if (!currentSceneId) return;
  try {
    await api(`/api/scenes/${currentSceneId}/regenerate`, 'POST');
    closeModal();
    setTimeout(() => openProject(currentProject.id), 1000);
  } catch(e) { alert('Error: ' + e.message); }
}"""

regen_func_new = """function regenerateScene() {
  if (!currentSceneId) return;
  closeModal();
  deselectAllScenes();
  const cb = document.querySelector(`.scene-checkbox[value="${currentSceneId}"]`);
  if(cb) cb.checked = true;
  updateGenerateBtn();
  openGenerateModal();
}"""

if regen_func_old in content:
    content = content.replace(regen_func_old, regen_func_new)


with open(html_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Patch applied.")
