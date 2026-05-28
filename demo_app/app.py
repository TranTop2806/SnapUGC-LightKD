from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
RUN_DIR = Path(os.environ.get("SNAPUGC_DEMO_RUN_DIR", ROOT / "results/demo_runs"))
UPLOAD_DIR = RUN_DIR / "uploads"
OUTPUT_DIR = RUN_DIR / "outputs"
STATIC_DIR = APP_DIR / "static"

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

app = FastAPI(title="SnapUGC LightKD Demo", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "ok": True,
        "root": str(ROOT),
        "report_json": str(resolve_report_path()),
        "checkpoint": str(resolve_checkpoint_path(resolve_report_path())),
        "efficientnet_weights": str(resolve_efficientnet_path()),
        "llm_explainer": bool(os.environ.get("SNAPUGC_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")),
    }


@app.post("/api/analyze")
async def analyze(
    video: UploadFile = File(...),
    title: str = Form(""),
    description: str = Form(""),
    device: str = Form("cpu"),
    topk: int = Form(3),
) -> dict[str, object]:
    if not video.filename:
        raise HTTPException(status_code=400, detail="Missing uploaded video")
    suffix = Path(video.filename).suffix.lower() or ".mp4"
    if suffix not in {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}:
        raise HTTPException(status_code=400, detail=f"Unsupported video type: {suffix}")

    run_id = f"{int(time.time())}_{safe_stem(video.filename)}"
    upload_path = UPLOAD_DIR / f"{run_id}{suffix}"
    assets_dir = OUTPUT_DIR / run_id / "assets"
    out_json = OUTPUT_DIR / run_id / "result.json"
    assets_dir.mkdir(parents=True, exist_ok=True)

    with upload_path.open("wb") as f:
        shutil.copyfileobj(video.file, f)

    report_path = resolve_report_path()
    checkpoint_path = resolve_checkpoint_path(report_path)
    labels_path = resolve_labels_path()
    efficientnet_path = resolve_efficientnet_path()
    cmd = [
        sys.executable,
        str(ROOT / "scripts/infer_new_video_with_student_expl.py"),
        "--video",
        str(upload_path),
        "--title",
        title,
        "--description",
        description,
        "--device",
        device,
        "--topk",
        str(topk),
        "--out-json",
        str(out_json),
        "--assets-dir",
        str(assets_dir),
    ]
    if report_path:
        cmd.extend(["--report-json", str(report_path)])
    if checkpoint_path:
        cmd.extend(["--checkpoint", str(checkpoint_path)])
    if labels_path:
        cmd.extend(["--labels-csv", str(labels_path)])
    if efficientnet_path:
        cmd.extend(["--efficientnet-weights", str(efficientnet_path)])

    try:
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
            timeout=int(os.environ.get("SNAPUGC_DEMO_TIMEOUT", "240")),
        )
    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Inference failed",
                "stdout": exc.stdout[-4000:],
                "stderr": exc.stderr[-4000:],
            },
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail=f"Inference timed out: {exc}") from exc

    try:
        result = json.loads(out_json.read_text(encoding="utf-8"))
    except Exception:
        result = json.loads(completed.stdout)
    result["ui"] = {
        "run_id": run_id,
        "video_url": f"/uploads/{upload_path.name}",
    }
    result["assets"] = normalize_asset_urls(result.get("assets", {}), run_id)
    return result


@app.get("/uploads/{name}")
def uploaded_video(name: str) -> FileResponse:
    path = UPLOAD_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path)


def normalize_asset_urls(assets: dict, run_id: str) -> dict:
    out = dict(assets or {})
    thumbs = []
    for item in out.get("top_clip_thumbnails", []):
        path = Path(item.get("path", ""))
        thumbs.append(
            {
                **item,
                "url": f"/outputs/{run_id}/assets/{path.name}",
            }
        )
    out["top_clip_thumbnails"] = thumbs
    return out


def resolve_report_path() -> Path | None:
    raw = os.environ.get("SNAPUGC_REPORT_JSON")
    candidates = []
    if raw:
        candidates.append(Path(raw).expanduser())
    candidates.extend(
        [
            ROOT / "results/kd_tuning_official_5k/v05_small_cosine_rank/official_student_kd_report.json",
            Path.home()
            / "workspace/results/kd_tuning_official_5k/v05_small_cosine_rank/official_student_kd_report.json",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def resolve_checkpoint_path(report_path: Path | None) -> Path | None:
    raw = os.environ.get("SNAPUGC_STUDENT_CHECKPOINT")
    candidates = []
    if raw:
        candidates.append(Path(raw).expanduser())
    if report_path:
        candidates.append(report_path.parent / "student_kd_best.pth")
    for path in candidates:
        if path.exists():
            return path
    return None


def resolve_labels_path() -> Path | None:
    raw = os.environ.get("SNAPUGC_LABELS_CSV")
    candidates = []
    if raw:
        candidates.append(Path(raw).expanduser())
    candidates.extend(
        [
            ROOT / "data/official_5k_split/split_all_5000.csv",
            Path.home() / "workspace/SnapUGC-LightKD/data/official_5k_split/split_all_5000.csv",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def resolve_efficientnet_path() -> Path | None:
    raw = os.environ.get("SNAPUGC_EFFICIENTNET_WEIGHTS")
    candidates = []
    if raw:
        candidates.append(Path(raw).expanduser())
    candidates.extend(
        [
            ROOT / "checkpoints/efficientnet_v2_s_21k_ft1k-dbb43f38.pth",
            Path.home()
            / "workspace/snapugc-checkpoints/efficientnet_v2_s_21k_ft1k-dbb43f38.pth",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def safe_stem(name: str) -> str:
    stem = Path(name).stem.lower()
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)[:60]
