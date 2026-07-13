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
const postProductionGroup = document.querySelector("#post-production-group");
const contentGroup = document.querySelector("#content-group");
const postProductionRecommendations = document.querySelector("#post-production-recommendations");
const contentRecommendations = document.querySelector("#content-recommendations");
const autoEditPanel = document.querySelector("#auto-edit-panel");
const autoEditButton = document.querySelector("#auto-edit-button");
const autoEditPlan = document.querySelector("#auto-edit-plan");
const editedResultPanel = document.querySelector("#edited-result-panel");
const editedResult = document.querySelector("#edited-result");
const editedDelta = document.querySelector("#edited-delta");
const suggestedTitle = document.querySelector("#suggested-title");
const suggestedDescription = document.querySelector("#suggested-description");

let currentResult = null;

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
    editedResultPanel.hidden = true;
    editedResult.innerHTML = "";
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
    const model = data.model_ready ? "Checkpoint loaded" : "Checkpoint missing";
    const llm =
      data.llm_backend === "local" ? "Local LLM" : data.llm_backend === "api" ? "API LLM" : "Template";
    health.textContent = `${model} · ${llm}`;
  })
  .catch(() => {
    health.textContent = "Offline";
  });

autoEditButton.addEventListener("click", async () => {
  if (!currentResult?.ui?.run_id) {
    showToast("Chưa có video gốc để chỉnh", false);
    return;
  }
  autoEditButton.disabled = true;
  showToast("Đang chỉnh video và chấm lại...", true);
  try {
    const response = await fetch("/api/auto_edit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        run_id: currentResult.ui.run_id,
        topk: Number(document.querySelector("#topk").value || 5),
        title: suggestedTitle?.value || "",
        description: suggestedDescription?.value || "",
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(extractError(payload));
    }
    renderAutoEditResult(payload);
    showToast("Đã chỉnh và chấm lại xong", false);
    setTimeout(() => (toast.hidden = true), 1400);
  } catch (error) {
    showToast(error.message || "Auto edit failed", false);
  } finally {
    autoEditButton.disabled = false;
  }
});

function renderResult(data) {
  currentResult = data;
  const score = Number(data.scores?.student_ecr || 0);
  const band = data.scores?.band?.label_vi || data.scores?.band?.label || "-";
  document.querySelector("#score").textContent = score.toFixed(3);
  document.querySelector("#band").textContent = band;
  document.querySelector("#meter-fill").style.width = `${Math.max(0, Math.min(100, score * 100))}%`;
  document.querySelector("#summary").textContent = data.nla_style_explanation?.summary || "";

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
        <div class="clip-label">${clip.semantic_label || "Bằng chứng thời gian được mô hình chọn"}</div>
        <div class="chip-row">
          <span class="chip">chú ý ${fmt(clip.student_attention)}</span>
          <span class="chip">ảnh hưởng ${fmt(clip.contribution_to_score)}</span>
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

  const conceptRows = data.semantic_attributes?.attributes || data.concept_bottleneck?.concepts || [];
  const concepts = document.querySelector("#concepts");
  concepts.innerHTML = "";
  conceptRows.forEach((concept) => {
    const el = document.createElement("article");
    el.className = "concept";
    const score = Math.max(0, Math.min(1, Number(concept.score || 0)));
    const percent = formatPercent(score);
    el.innerHTML = `
      <div class="concept-top">
        <span class="concept-name">${prettyConcept(concept.name)}</span>
        <span class="concept-score">${concept.label_vi || concept.label} · ${percent}</span>
      </div>
      <div class="concept-bar"><span style="width: ${score * 100}%"></span></div>
      <p>${concept.rationale || ""}</p>
    `;
    concepts.appendChild(el);
  });

  renderOriginalRecommendations(data);
  renderMetadataSuggestion(data);

  autoEditPlan.innerHTML = "";
  const groups = getRecommendationGroups(data);
  const visualPost = groups.post.filter((item) => item.toLowerCase().startsWith("hậu kì"));
  const autoRows = visualPost.length
    ? visualPost
    : ["Auto edit sẽ chỉ áp dụng chỉnh sửa hình ảnh nhẹ nếu tìm thấy clip phù hợp."];
  const metadataChanges = data.metadata_suggestion?.changes || [];
  metadataChanges.forEach((item) => autoRows.push(`Metadata: ${item}`));
  autoRows.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    autoEditPlan.appendChild(li);
  });

  resultGrid.hidden = false;
  clipsPanel.hidden = clipRows.length === 0;
  claimsPanel.hidden = !data.nla_style_explanation?.claims?.length;
  conceptsPanel.hidden = conceptRows.length === 0;
  recommendationsPanel.hidden = !hasAnyRecommendations(data);
  autoEditPanel.hidden = !data.ui?.run_id;
}

function renderAutoEditResult(payload) {
  renderEditPlan(payload.edit_plan);
  const edited = payload.edited;
  const comparison = payload.comparison || {};
  const delta = Number(comparison.delta_ecr || 0);
  editedDelta.textContent = `Δ ${delta >= 0 ? "+" : ""}${delta.toFixed(3)}`;
  editedDelta.className = `status-pill ${delta >= 0 ? "delta-good" : "delta-bad"}`;
    editedResult.innerHTML = renderResultBlock(edited, comparison);
  editedResultPanel.hidden = false;
  editedResultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderMetadataSuggestion(data) {
  const suggestion = data?.metadata_suggestion || {};
  if (suggestedTitle) {
    suggestedTitle.value = suggestion.title || data?.input_context?.title || "";
  }
  if (suggestedDescription) {
    suggestedDescription.value = suggestion.description || data?.input_context?.description || "";
  }
}

function renderEditPlan(plan) {
  autoEditPlan.innerHTML = "";
  const edits = plan?.edits || [];
  if (!edits.length) {
    const li = document.createElement("li");
    li.textContent = "Không tìm thấy clip cần chỉnh khả thi; video được giữ nguyên để chấm lại.";
    autoEditPlan.appendChild(li);
    return;
  }
  edits.forEach((edit) => {
    const li = document.createElement("li");
    const ops = Object.entries(edit.operations || {})
      .map(([key, value]) => `${prettyOperation(key)} ${formatOperationValue(key, value)}`)
      .join(", ");
    li.textContent = `${edit.label}: ${edit.reasons?.join(" ")} (${ops})`;
    autoEditPlan.appendChild(li);
  });
}

function renderOriginalRecommendations(data) {
  const groups = getRecommendationGroups(data);
  renderRecommendationList(postProductionRecommendations, groups.post);
  renderRecommendationList(contentRecommendations, groups.content);
  postProductionGroup.hidden = groups.post.length === 0;
  contentGroup.hidden = groups.content.length === 0;
}

function renderRecommendationList(container, rows) {
  container.innerHTML = "";
  rows.forEach((rec) => {
    const li = document.createElement("li");
    li.textContent = rec;
    container.appendChild(li);
  });
}

function getRecommendationGroups(data) {
  const grouped = data?.recommendations_grouped || {};
  const post = Array.isArray(grouped.post_production) ? grouped.post_production : [];
  const content = Array.isArray(grouped.content_reshoot) ? grouped.content_reshoot : [];
  if (post.length || content.length) {
    return { post, content };
  }
  return {
    post: [],
    content: Array.isArray(data?.recommendations) ? data.recommendations : [],
  };
}

function hasAnyRecommendations(data) {
  const groups = getRecommendationGroups(data);
  return groups.post.length > 0 || groups.content.length > 0;
}

function renderResultBlock(data, comparison) {
  const score = Number(data?.scores?.student_ecr || 0);
  const original = Number(comparison?.original_ecr || 0);
  const delta = Number(comparison?.delta_ecr || score - original);
  const band = data?.scores?.band?.label_vi || data?.scores?.band?.label || "-";
  const claims = data?.nla_style_explanation?.claims || [];
  const concepts = data?.semantic_attributes?.attributes || data?.concept_bottleneck?.concepts || [];
  const clips = data?.evidence?.top_clips || [];
  const thumbs = data?.assets?.top_clip_thumbnails || [];
  const videoUrl = data?.ui?.video_url;
  return `
    ${videoUrl ? `<video class="edited-video" src="${escapeHtml(videoUrl)}" controls playsinline></video>` : ""}
    <section class="edited-score-card">
      <p class="eyebrow">Dự đoán ECR sau chỉnh</p>
      <div class="score-row">
        <strong>${fmt(score)}</strong>
        <span class="band-pill">${escapeHtml(band)}</span>
      </div>
      <div class="meter"><span style="width: ${Math.max(0, Math.min(100, score * 100))}%"></span></div>
      <p class="summary">${escapeHtml(data?.nla_style_explanation?.summary || "")}</p>
    </section>
    <div class="comparison-grid">
      <article class="compare-card">
        <span>Video gốc</span>
        <strong>${fmt(original)}</strong>
      </article>
      <article class="compare-card">
        <span>Video đã chỉnh</span>
        <strong>${fmt(score)}</strong>
      </article>
      <article class="compare-card ${delta >= 0 ? "positive" : "negative"}">
        <span>Thay đổi ECR</span>
        <strong>${delta >= 0 ? "+" : ""}${fmt(delta)}</strong>
      </article>
    </div>
    ${renderClipGrid(clips, thumbs)}
    ${renderList("Giải thích sau chỉnh", claims, "claims")}
    ${renderConceptGrid(concepts)}
    ${renderList(
      "Gợi ý còn lại sau chỉnh (cần thay đổi cảnh quay/dựng)",
      getRecommendationGroups(data).content,
      "recommendations content-production"
    )}
  `;
}

function renderClipGrid(clips, thumbs) {
  if (!clips.length) return "";
  return `
    <div class="edited-section">
      <p class="eyebrow">Đoạn nổi bật sau chỉnh</p>
      <div class="clip-list">
        ${clips
          .map((clip, index) => {
            const thumb = thumbs[index];
            return `
              <article class="clip">
                ${thumb?.url ? `<img src="${escapeHtml(thumb.url)}" alt="Edited top clip ${index + 1}" />` : ""}
                <div class="clip-body">
                  <div class="clip-time">${escapeHtml(clip.relative_time?.label || `clip ${clip.clip_index}`)}</div>
                  <div class="clip-label">${escapeHtml(clip.semantic_label || "Bằng chứng thời gian được mô hình chọn")}</div>
                  <div class="chip-row">
                    <span class="chip">chú ý ${fmt(clip.student_attention)}</span>
                    <span class="chip">ảnh hưởng ${fmt(clip.contribution_to_score)}</span>
                  </div>
                </div>
              </article>
            `;
          })
          .join("")}
      </div>
    </div>
  `;
}

function renderList(title, rows, className) {
  if (!rows.length) return "";
  return `
    <div class="edited-section">
      <p class="eyebrow">${escapeHtml(title)}</p>
      <ul class="${className}">
        ${rows.map((row) => `<li>${escapeHtml(row)}</li>`).join("")}
      </ul>
    </div>
  `;
}

function renderConceptGrid(concepts) {
  if (!concepts.length) return "";
  return `
    <div class="edited-section">
      <p class="eyebrow">Thuộc tính ngữ nghĩa sau chỉnh</p>
      <div class="concept-grid">
        ${concepts
          .map((concept) => {
            const score = Math.max(0, Math.min(1, Number(concept.score || 0)));
            return `
              <article class="concept">
                <div class="concept-top">
                  <span class="concept-name">${escapeHtml(prettyConcept(concept.name))}</span>
                  <span class="concept-score">${escapeHtml(concept.label_vi || concept.label || "")} · ${formatPercent(score)}</span>
                </div>
                <div class="concept-bar"><span style="width: ${score * 100}%"></span></div>
                <p>${escapeHtml(concept.rationale || "")}</p>
              </article>
            `;
          })
          .join("")}
      </div>
    </div>
  `;
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

function formatPercent(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${Math.round(Math.max(0, Math.min(1, number)) * 100)}%` : "-";
}

function extractError(payload) {
  if (typeof payload?.detail === "string") return payload.detail;
  if (payload?.detail?.message) {
    return `${payload.detail.message}: ${payload.detail.stderr || payload.detail.stdout || ""}`;
  }
  return "Inference failed";
}

function prettyConcept(name) {
  const labels = {
    hook_strength: "Độ mạnh phần mở đầu",
    motion_action: "Chuyển động/hành động",
    visual_clarity: "Độ rõ hình ảnh",
    lighting_quality: "Chất lượng ánh sáng",
    text_specificity: "Độ cụ thể văn bản",
    pacing_variety: "Độ đa dạng nhịp hình",
  };
  return labels[name] || String(name || "").replaceAll("_", " ");
}

function prettyOperation(name) {
  return {
    brightness: "tăng sáng",
    contrast: "contrast",
    sharpness: "sharpness",
    saturation: "saturation",
  }[name] || name;
}

function formatOperationValue(name, value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "";
  if (name === "brightness") return `${number >= 0 ? "+" : ""}${number.toFixed(1)}`;
  return `×${number.toFixed(2)}`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
