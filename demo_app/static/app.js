const form = document.querySelector("#analyze-form");
const videoInput = document.querySelector("#video");
const preview = document.querySelector("#preview");
const emptyPreview = document.querySelector("#empty-preview");
const fileLabel = document.querySelector("#file-label");
const toast = document.querySelector("#toast");
const toastText = document.querySelector("#toast-text");
const health = document.querySelector("#health");

const resultGrid = document.querySelector("#result-grid");
const clipsPanel = document.querySelector("#clips-panel");
const claimsPanel = document.querySelector("#claims-panel");
const conceptsPanel = document.querySelector("#concepts-panel");
const recommendationsPanel = document.querySelector("#recommendations-panel");

videoInput.addEventListener("change", () => {
  const file = videoInput.files?.[0];
  if (!file) return;
  fileLabel.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MB`;
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  emptyPreview.hidden = true;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = videoInput.files?.[0];
  if (!file) {
    showToast("Chọn video trước nha", false);
    return;
  }

  const button = form.querySelector("button");
  button.disabled = true;
  showToast("Đang chạy student inference...", true);

  const body = new FormData(form);
  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      body,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(extractError(payload));
    }
    renderResult(payload);
    showToast("Done", false);
    setTimeout(() => (toast.hidden = true), 1400);
  } catch (error) {
    showToast(error.message || "Inference failed", false);
  } finally {
    button.disabled = false;
  }
});

fetch("/health")
  .then((res) => res.json())
  .then((data) => {
    health.textContent = data.checkpoint ? "Checkpoint loaded" : "Demo fallback";
  })
  .catch(() => {
    health.textContent = "Offline";
  });

function renderResult(data) {
  const score = Number(data.scores?.student_ecr || 0);
  const band = data.scores?.band?.label_vi || data.scores?.band?.label || "-";
  document.querySelector("#score").textContent = score.toFixed(3);
  document.querySelector("#band").textContent = band;
  document.querySelector("#meter-fill").style.width = `${Math.max(0, Math.min(100, score * 100))}%`;
  document.querySelector("#summary").textContent = data.nla_style_explanation?.summary || "";

  const faith = data.nla_style_explanation?.faithfulness || {};
  document.querySelector("#keep-only").textContent = fmt(fathValue(faith.keep_only_selected_ecr));
  document.querySelector("#remove").textContent = fmt(fathValue(faith.remove_selected_ecr));
  document.querySelector("#confidence").textContent = data.nla_style_explanation?.confidence || "-";

  const thumbs = data.assets?.top_clip_thumbnails || [];
  const clipRows = data.evidence?.top_clips || [];
  const clipList = document.querySelector("#clip-list");
  clipList.innerHTML = "";
  clipRows.forEach((clip, index) => {
    const thumb = thumbs[index];
    const el = document.createElement("article");
    el.className = "clip";
    el.innerHTML = `
      ${thumb?.url ? `<img src="${thumb.url}" alt="Top clip ${index + 1}" />` : ""}
      <div class="clip-body">
        <div class="clip-time">${clip.relative_time?.label || `clip ${clip.clip_index}`}</div>
        <div class="clip-label">${clip.semantic_label || "Student-selected temporal evidence"}</div>
        <div class="chip-row">
          <span class="chip">attn ${fmt(clip.student_attention)}</span>
          <span class="chip">Δ ${fmt(clip.contribution_to_score)}</span>
        </div>
      </div>
    `;
    clipList.appendChild(el);
  });

  const claims = document.querySelector("#claims");
  claims.innerHTML = "";
  (data.nla_style_explanation?.claims || []).forEach((claim) => {
    const li = document.createElement("li");
    li.textContent = claim;
    claims.appendChild(li);
  });

  const conceptRows = data.concept_bottleneck?.concepts || [];
  const concepts = document.querySelector("#concepts");
  concepts.innerHTML = "";
  conceptRows.forEach((concept) => {
    const el = document.createElement("article");
    el.className = "concept";
    const score = Math.max(0, Math.min(1, Number(concept.score || 0)));
    el.innerHTML = `
      <div class="concept-top">
        <span class="concept-name">${prettyConcept(concept.name)}</span>
        <span class="concept-score">${concept.label_vi || concept.label}</span>
      </div>
      <div class="concept-bar"><span style="width: ${score * 100}%"></span></div>
      <p>${concept.rationale || ""}</p>
    `;
    concepts.appendChild(el);
  });

  const recommendations = document.querySelector("#recommendations");
  recommendations.innerHTML = "";
  (data.recommendations || []).forEach((rec) => {
    const li = document.createElement("li");
    li.textContent = rec;
    recommendations.appendChild(li);
  });

  resultGrid.hidden = false;
  clipsPanel.hidden = clipRows.length === 0;
  claimsPanel.hidden = !data.nla_style_explanation?.claims?.length;
  conceptsPanel.hidden = conceptRows.length === 0;
  recommendationsPanel.hidden = !data.recommendations?.length;
}

function showToast(message, loading) {
  toast.hidden = false;
  toastText.textContent = message;
  toast.querySelector(".spinner").style.display = loading ? "block" : "none";
}

function fmt(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(3) : "-";
}

function fathValue(value) {
  return value === null || value === undefined ? NaN : value;
}

function extractError(payload) {
  if (typeof payload?.detail === "string") return payload.detail;
  if (payload?.detail?.message) {
    return `${payload.detail.message}: ${payload.detail.stderr || payload.detail.stdout || ""}`;
  }
  return "Inference failed";
}

function prettyConcept(name) {
  return String(name || "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}
