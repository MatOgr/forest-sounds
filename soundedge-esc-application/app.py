import atexit
import os
import tempfile
import uuid
from pathlib import Path

import torch
from classes import ESC50_CLASSES
from flask import Flask, jsonify, render_template_string, request, send_from_directory
from preprocessing import preprocess_audio

from utils import load_compressed_model, load_model, predict

BASE_DIR = Path(__file__).resolve().parent
ORIGINAL_MODEL_PATH = BASE_DIR / "weights" / "esc50_model.pth"
COMPRESSED_MODEL_PATH = BASE_DIR / "weights" / "esc50_model_compressed.pth"
STATS_PATH = BASE_DIR / "stats" / "esc50_mel_stats.json"
SAMPLES_DIR = BASE_DIR / "samples"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "soundedge_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".wav"}
LOW_CONFIDENCE_THRESHOLD = 0.6
MAX_UPLOAD_BYTES = 5 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

_gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model_cache = {"Original": None, "Compressed": None}


def _safe_label(raw_name: str) -> str:
    return raw_name.replace("_", " ").title()


def _sample_files() -> list[str]:
    if not SAMPLES_DIR.is_dir():
        return []
    return sorted([f.name for f in SAMPLES_DIR.iterdir() if f.suffix.lower() == ".wav"])


def _validate_wav_filename(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS


def _get_active_model(model_choice: str):
    if model_choice == "Compressed":
        if _model_cache["Compressed"] is None:
            _model_cache["Compressed"] = load_compressed_model(
                str(ORIGINAL_MODEL_PATH),
                str(COMPRESSED_MODEL_PATH),
                num_classes=len(ESC50_CLASSES),
            )
        return _model_cache["Compressed"], torch.device("cpu")

    if _model_cache["Original"] is None:
        _model_cache["Original"] = load_model(
            str(ORIGINAL_MODEL_PATH),
            _gpu_device,
            num_classes=len(ESC50_CLASSES),
        )
    return _model_cache["Original"], _gpu_device


def _prediction_payload(model_choice: str, source_path: Path):
    model, device = _get_active_model(model_choice)
    input_tensor = preprocess_audio(str(source_path), str(STATS_PATH))
    top_class, top_prob, all_probs = predict(model, input_tensor, device)

    return {
        "topClass": top_class,
        "topClassLabel": _safe_label(top_class),
        "topProbability": top_prob,
        "topProbabilityPct": round(top_prob * 100, 2),
        "lowConfidence": top_prob < LOW_CONFIDENCE_THRESHOLD,
        "top3": [
            {
                "className": p["class_name"],
                "classLabel": _safe_label(p["class_name"]),
                "probability": p["probability"],
                "probabilityPct": round(p["probability"] * 100, 2),
            }
            for p in all_probs[:3]
        ],
        "allProbs": [
            {
                "className": p["class_name"],
                "classLabel": _safe_label(p["class_name"]),
                "probability": p["probability"],
                "probabilityPct": round(p["probability"] * 100, 2),
            }
            for p in all_probs
        ],
    }


@app.get("/")
def index():
    classes = [_safe_label(c) for c in ESC50_CLASSES]
    samples = [{"name": s, "label": _safe_label(Path(s).stem)} for s in _sample_files()]
    return render_template_string(
        PAGE_TEMPLATE,
        class_badges=classes,
        samples=samples,
    )


@app.get("/samples/<path:filename>")
def serve_sample(filename: str):
    return send_from_directory(SAMPLES_DIR, filename)


@app.get("/uploads/<path:filename>")
def serve_uploaded_file(filename: str):
    return send_from_directory(UPLOAD_DIR, filename)


@app.post("/upload")
def upload_audio():
    uploaded = request.files.get("file")
    if uploaded is None:
        return jsonify(
            {"error": "Missing file field. Use multipart form key 'file'."},
        ), 400

    filename = uploaded.filename or ""
    if not filename:
        return jsonify({"error": "No file selected."}), 400
    if not _validate_wav_filename(filename):
        return jsonify({"error": "Only .wav files are supported."}), 400

    file_id = uuid.uuid4().hex
    stored_name = f"{file_id}.wav"
    destination = UPLOAD_DIR / stored_name
    uploaded.save(destination)

    return jsonify(
        {
            "fileId": file_id,
            "filename": filename,
            "audioUrl": f"/uploads/{stored_name}",
        },
    )


@app.post("/predict")
def predict_audio():
    payload = request.get_json(silent=True) or {}
    source_type = payload.get("source", "upload")
    model_choice = payload.get("model", "Original")
    if model_choice not in ("Original", "Compressed"):
        return jsonify({"error": "Invalid model. Use 'Original' or 'Compressed'."}), 400

    try:
        if source_type == "upload":
            file_id = payload.get("fileId", "")
            if not file_id:
                return jsonify({"error": "Missing fileId for uploaded source."}), 400
            source_path = UPLOAD_DIR / f"{file_id}.wav"
            if not source_path.exists():
                return jsonify({"error": "Uploaded file not found. Upload again."}), 404
        elif source_type == "sample":
            sample_name = payload.get("sampleName", "")
            if not sample_name:
                return jsonify({"error": "Missing sampleName for sample source."}), 400
            source_path = SAMPLES_DIR / sample_name
            if not source_path.exists() or source_path.suffix.lower() != ".wav":
                return jsonify({"error": "Sample not found."}), 404
        else:
            return jsonify({"error": "Invalid source. Use 'upload' or 'sample'."}), 400

        result = _prediction_payload(model_choice=model_choice, source_path=source_path)
        return jsonify(result)
    except Exception as exc:  # noqa – FIXME
        return jsonify({"error": f"Error during inference: {exc}"}), 500


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


def _cleanup_uploads():
    if not UPLOAD_DIR.exists():
        return
    for wav in UPLOAD_DIR.glob("*.wav"):
        try:
            wav.unlink()
        except OSError:
            pass


atexit.register(_cleanup_uploads)


PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>SoundEdge - Environmental Sound Classification</title>
    <style>
        :root {
            --bg: #f1f5f9;
            --text: #1f2937;
            --muted: #64748b;
            --line: #d8dee8;
            --chip-bg: #e9edf3;
            --chip-border: #d8e0ec;
            --panel: transparent;
            --blue: #0b75b8;
            --violet: #6d46e8;
            --danger-bg: #fff1f2;
            --danger: #dc2626;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: "Segoe UI", "Helvetica Neue", Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
        }
        .container {
            width: min(1020px, 92vw);
            margin: 1.4rem auto 3rem;
        }
        .hero {
            background: linear-gradient(135deg, #d8e7fb 0%, #dfe4f8 100%);
            border: 1px solid #b7d1f5;
            border-radius: 14px;
            padding: 2rem 1.4rem 1.6rem;
            text-align: center;
            margin-bottom: 1.9rem;
        }
        .hero h1 {
            margin: 0;
            font-size: clamp(2.2rem, 3.8vw, 3.6rem);
            font-weight: 800;
            background: linear-gradient(90deg, #1f82c4, #4f48d9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero p {
            margin: 1rem 0 0;
            color: #445d78;
            font-size: 2.1vmin;
            min-font-size: 1rem;
        }
        .hero-icon {
            font-size: clamp(2rem, 4vmin, 2.8rem);
            margin-right: 0.55rem;
            vertical-align: 0.28em;
        }
        .section {
            border-top: 1px solid var(--line);
            padding-top: 1.9rem;
            margin-top: 1.7rem;
        }
        .section:first-of-type {
            border-top: 0;
            padding-top: 0;
            margin-top: 0;
        }
        .panel {
            background: var(--panel);
            border: 0;
        }
        .section-title {
            font-size: 1.72rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--violet);
            font-weight: 800;
            margin: 0 0 1rem;
        }
        .badge-grid { display: flex; flex-wrap: wrap; gap: 0.46rem; }
        .badge {
            background: var(--chip-bg);
            border: 1px solid var(--chip-border);
            border-radius: 8px;
            padding: 0.32rem 0.74rem;
            font-size: 1rem;
            color: #425569;
        }
        .model-row {
            display: flex;
            gap: 1.2rem;
            align-items: center;
            font-size: 1.08rem;
        }
        .model-row label {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            cursor: pointer;
        }
        .tabs {
            display: flex;
            gap: 1rem;
            border-bottom: 2px solid var(--line);
            margin-bottom: 1.05rem;
        }
        .tab-btn {
            border: 0;
            background: transparent;
            color: #334155;
            font-size: 1.08rem;
            padding: 0.32rem 0.1rem 0.7rem;
            border-bottom: 3px solid transparent;
            cursor: pointer;
        }
        .tab-btn.active {
            color: #0f73c6;
            border-color: #0f73c6;
        }
        .tab-panel {
            display: none;
        }
        .tab-panel.active {
            display: block;
        }
        .guide-box {
            background: #eff8ff;
            border: 1px solid #b9def9;
            border-radius: 12px;
            padding: 0.95rem 1rem;
            color: #36516e;
            margin-bottom: 0.95rem;
            font-size: 0.95rem;
        }
        .guide-title {
            font-weight: 700;
            color: #0c5f9e;
            margin-bottom: 0.45rem;
        }
        .input-grid {
            display: flex;
            gap: 0.75rem;
            align-items: center;
            flex-wrap: wrap;
        }
        .upload-input {
            border: 1px solid #ccd7e4;
            border-radius: 9px;
            background: #f7fafc;
            padding: 0.42rem;
            min-width: 280px;
        }
        .sample-select {
            width: 100%;
            border-radius: 10px;
            border: 1px solid #d6deea;
            font-size: 1.2rem;
            color: #374b60;
            background: #eef2f7;
            padding: 0.58rem 0.8rem;
        }
        .audio-wrap {
            margin-top: 0.8rem;
            width: 100%;
            max-width: 100%;
            border-radius: 999px;
            background: #e8edf4;
            padding: 0.3rem 0.65rem;
        }
        audio {
            width: 100%;
            height: 38px;
        }
        .sample-btn {
            margin-top: 0.95rem;
            width: 100%;
            border: 1px solid #cfd9e5;
            background: #f2f5f8;
            color: #233548;
            border-radius: 11px;
            padding: 0.72rem 0.9rem;
            font-size: 2.1vmin;
            cursor: pointer;
        }
        .btn {
            border: 1px solid #0b77bb;
            background: #0b77bb;
            color: #fff;
            border-radius: 10px;
            padding: 0.57rem 1rem;
            font-weight: 600;
            cursor: pointer;
        }
        .btn.secondary {
            border-color: #6d46e8;
            background: #6d46e8;
        }
        .btn:disabled {
            opacity: 0.55;
            cursor: not-allowed;
        }
        .status {
            margin-top: 0.6rem;
            font-size: 0.96rem;
            color: var(--muted);
            min-height: 1.2rem;
        }
        .error {
            color: var(--danger);
            background: var(--danger-bg);
            border: 1px solid #fecdd3;
            border-radius: 10px;
            padding: 0.65rem;
            margin-top: 0.7rem;
        }
        #resultsPanel { display: none; }
        .result-card {
            background: linear-gradient(135deg, #dbeafe 0%, #ede9fe 100%);
            border: 1px solid #5f96ff;
            border-radius: 14px;
            padding: 1.9rem;
            text-align: center;
            margin-bottom: 1.2rem;
        }
        .result-label {
            font-size: 1.35rem;
            letter-spacing: 0.12em;
            color: #6a7c90;
            text-transform: uppercase;
        }
        .result-class {
            margin: 0.58rem 0 0;
            color: #0d6ba9;
            font-size: clamp(2.1rem, 4vmin, 3.4rem);
            font-weight: 800;
        }
        .result-prob {
            margin-top: 0.58rem;
            color: #6d46e8;
            font-size: clamp(1.35rem, 2.6vmin, 2rem);
            font-weight: 600;
        }
        .bar-row { margin-top: 0.9rem; }
        .bar-label {
            display: flex;
            justify-content: space-between;
            font-size: 1.1rem;
            margin-bottom: 0.25rem;
            color: #3a4b5c;
        }
        .bar-bg {
            height: 14px;
            border-radius: 999px;
            background: #cfd6df;
            overflow: hidden;
        }
        .bar-fill {
            height: 14px;
            border-radius: 999px;
            background: linear-gradient(90deg, #0284c7, #7c3aed);
        }
        .all-prob-list {
            margin-top: 0.75rem;
            max-height: 420px;
            overflow-y: auto;
            padding-right: 0.25rem;
        }
        .all-prob-row {
            margin-bottom: 0.48rem;
        }
        .all-prob-head {
            display: flex;
            justify-content: space-between;
            font-size: 0.98rem;
            color: #3f5061;
            margin-bottom: 0.16rem;
        }
        .all-prob-bg {
            height: 8px;
            border-radius: 999px;
            background: #d9e0e8;
            overflow: hidden;
        }
        .all-prob-fill {
            height: 8px;
            border-radius: 999px;
            background: #0b86c8;
        }
        @media (max-width: 760px) {
            .section-title { font-size: 1.14rem; }
            .badge { font-size: 0.85rem; }
            .hero p { font-size: 0.98rem; }
            .sample-select { font-size: 1.18rem; }
            .sample-btn { font-size: 0.95rem; }
            .upload-input { min-width: 100%; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="hero">
            <h1><span class="hero-icon">🔊</span>SoundEdge</h1>
            <p>Environmental Sound Classification - upload a short audio clip and let the model identify the sound.</p>
        </div>

        <div class="panel section">
            <div class="section-title">Supported Sound Classes</div>
            <div class="badge-grid">
                {% for cls in class_badges %}
                    <span class="badge">{{ cls }}</span>
                {% endfor %}
            </div>
        </div>

        <div class="panel section">
            <div class="section-title">Select Model</div>
            <div class="model-row">
                <label><input type="radio" name="model" value="Original" checked> Original</label>
                <label><input type="radio" name="model" value="Compressed"> Compressed</label>
            </div>
        </div>

        <div class="panel section">
            <div class="section-title">Choose Audio Input</div>
            <div class="tabs">
                <button id="tabUpload" class="tab-btn active" type="button">⬆️ Upload a File</button>
                <button id="tabSample" class="tab-btn" type="button">🎵 Try a Sample</button>
            </div>

            <div id="uploadPanel" class="tab-panel active">
                <div class="guide-box">
                    <div class="guide-title">Upload Guide</div>
                    <div>Format: WAV (.wav) only.</div>
                    <div>Duration: Around 5 seconds gives best results.</div>
                    <div>Size: Keep under 5 MB.</div>
                </div>
                <div class="input-grid">
                    <input id="uploadInput" class="upload-input" type="file" accept=".wav,audio/wav" />
                    <button id="uploadBtn" class="btn" type="button">Upload</button>
                    <button id="predictUploadedBtn" class="btn secondary" type="button" disabled>Classify Uploaded Audio</button>
                </div>
                <div class="audio-wrap" id="uploadedAudioWrap" style="display:none;">
                    <audio id="uploadedAudio" controls></audio>
                </div>
                <div id="uploadStatus" class="status"></div>
                <div id="uploadError"></div>
            </div>

            <div id="samplePanel" class="tab-panel">
                {% if samples %}
                    <select id="sampleSelect" class="sample-select">
                        {% for sample in samples %}
                            <option value="{{ sample.name }}">{{ sample.label }}</option>
                        {% endfor %}
                    </select>
                    <div class="audio-wrap">
                        <audio id="sampleAudio" controls></audio>
                    </div>
                    <button id="predictSampleBtn" class="sample-btn" type="button">Classify this sample ></button>
                {% else %}
                    <div class="status">No sample files found in the samples folder.</div>
                {% endif %}
                <div id="sampleStatus" class="status"></div>
            </div>
        </div>

        <div id="resultsPanel" class="panel section"></div>
    </div>

    <script>
        let uploadedFileId = null;
        const uploadInput = document.getElementById("uploadInput");
        const uploadBtn = document.getElementById("uploadBtn");
        const predictUploadedBtn = document.getElementById("predictUploadedBtn");
        const uploadStatus = document.getElementById("uploadStatus");
        const uploadError = document.getElementById("uploadError");
        const uploadedAudio = document.getElementById("uploadedAudio");
        const sampleSelect = document.getElementById("sampleSelect");
        const sampleAudio = document.getElementById("sampleAudio");
        const sampleStatus = document.getElementById("sampleStatus");
        const predictSampleBtn = document.getElementById("predictSampleBtn");
        const resultsPanel = document.getElementById("resultsPanel");
        const uploadedAudioWrap = document.getElementById("uploadedAudioWrap");
        const tabUpload = document.getElementById("tabUpload");
        const tabSample = document.getElementById("tabSample");
        const uploadPanel = document.getElementById("uploadPanel");
        const samplePanel = document.getElementById("samplePanel");

        function currentModel() {
            const selected = document.querySelector('input[name="model"]:checked');
            return selected ? selected.value : "Original";
        }

        function setError(target, message) {
            target.innerHTML = message ? `<div class="error">${message}</div>` : "";
        }

        function setTab(mode) {
            const isUpload = mode === "upload";
            tabUpload?.classList.toggle("active", isUpload);
            tabSample?.classList.toggle("active", !isUpload);
            uploadPanel?.classList.toggle("active", isUpload);
            samplePanel?.classList.toggle("active", !isUpload);
        }

        function renderResult(result) {
            resultsPanel.style.display = "block";
            if (result.lowConfidence) {
                resultsPanel.innerHTML = `
                    <div class="error" style="margin-top:0;">
                        Unable to confidently identify the sound. Please upload a clearer clip and try again.
                    </div>
                `;
                return;
            }

            let top3Html = "";
            result.top3.forEach(item => {
                top3Html += `
                    <div class="bar-row">
                        <div class="bar-label"><span>${item.classLabel}</span><span>${item.probabilityPct.toFixed(2)}%</span></div>
                        <div class="bar-bg"><div class="bar-fill" style="width:${item.probabilityPct}%;"></div></div>
                    </div>
                `;
            });

            let allHtml = "";
            result.allProbs.forEach(item => {
                allHtml += `
                    <div class="all-prob-row">
                        <div class="all-prob-head"><span>${item.classLabel}</span><span>${item.probabilityPct.toFixed(1)}%</span></div>
                        <div class="all-prob-bg"><div class="all-prob-fill" style="width:${item.probabilityPct}%;"></div></div>
                    </div>
                `;
            });

            resultsPanel.innerHTML = `
                <div class="result-card">
                    <div class="result-label">Predicted Sound</div>
                    <h2 class="result-class">${result.topClassLabel}</h2>
                    <div class="result-prob">Confidence ${result.topProbabilityPct.toFixed(1)}%</div>
                </div>
                <div class="section-title" style="margin-top:1rem;">Top 3 Predictions</div>
                ${top3Html}
                <div class="section-title" style="margin-top:1.2rem;">All Class Probabilities</div>
                <div class="all-prob-list">${allHtml}</div>
            `;
        }

        async function predict(payload, statusEl) {
            statusEl.textContent = "Analysing audio...";
            const resp = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ ...payload, model: currentModel() })
            });
            const data = await resp.json();
            if (!resp.ok) {
                throw new Error(data.error || "Prediction failed.");
            }
            statusEl.textContent = "Classification completed.";
            renderResult(data);
        }

        uploadInput?.addEventListener("change", () => {
            setError(uploadError, "");
            uploadStatus.textContent = "";
            uploadedFileId = null;
            predictUploadedBtn.disabled = true;

            const file = uploadInput.files && uploadInput.files[0];
            if (!file) {
                uploadedAudioWrap.style.display = "none";
                return;
            }
            const objectUrl = URL.createObjectURL(file);
            uploadedAudio.src = objectUrl;
            uploadedAudioWrap.style.display = "block";
        });

        uploadBtn?.addEventListener("click", async () => {
            setError(uploadError, "");
            uploadStatus.textContent = "";
            const file = uploadInput.files && uploadInput.files[0];
            if (!file) {
                setError(uploadError, "Select a WAV file before uploading.");
                return;
            }

            const formData = new FormData();
            formData.append("file", file);

            try {
                uploadStatus.textContent = "Uploading...";
                const resp = await fetch("/upload", { method: "POST", body: formData });
                const data = await resp.json();
                if (!resp.ok) {
                    throw new Error(data.error || "Upload failed.");
                }
                uploadedFileId = data.fileId;
                uploadedAudio.src = data.audioUrl;
                uploadedAudioWrap.style.display = "block";
                predictUploadedBtn.disabled = false;
                uploadStatus.textContent = `Uploaded: ${data.filename}`;
            } catch (err) {
                setError(uploadError, err.message || "Upload failed.");
                uploadStatus.textContent = "";
            }
        });

        predictUploadedBtn?.addEventListener("click", async () => {
            setError(uploadError, "");
            if (!uploadedFileId) {
                setError(uploadError, "Upload a file first.");
                return;
            }
            try {
                await predict({ source: "upload", fileId: uploadedFileId }, uploadStatus);
            } catch (err) {
                setError(uploadError, err.message || "Prediction failed.");
                uploadStatus.textContent = "";
            }
        });

        if (sampleSelect && sampleAudio) {
            function syncSampleAudio() {
                sampleAudio.src = `/samples/${encodeURIComponent(sampleSelect.value)}`;
            }
            sampleSelect.addEventListener("change", syncSampleAudio);
            syncSampleAudio();
        }

        tabUpload?.addEventListener("click", () => setTab("upload"));
        tabSample?.addEventListener("click", () => setTab("sample"));
        setTab("sample");

        predictSampleBtn?.addEventListener("click", async () => {
            sampleStatus.textContent = "";
            try {
                await predict({ source: "sample", sampleName: sampleSelect.value }, sampleStatus);
            } catch (err) {
                sampleStatus.textContent = err.message || "Prediction failed.";
            }
        });
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8501"))
    host = os.environ.get("HOST", "127.0.0.1")
    app.run(host=host, port=port, debug=False)
