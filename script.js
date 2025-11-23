// ===== ELEMENTS =====
const uploadArea = document.getElementById("upload-area");
const imageInput = document.getElementById("image-input");
const browseBtn = document.getElementById("browse-btn");
const previewImg = document.getElementById("preview-img");
const previewBox = document.getElementById("preview-box");
const analyzeBtn = document.getElementById("analyze-btn");

const loadingEl = document.getElementById("loading");
const resultEl = document.getElementById("result");
const resultTextEl = document.getElementById("result-text");
const resultScoreEl = document.getElementById("result-score");
const errorBox = document.getElementById("error-box");

let selectedFile = null;

// ===== HELPER FUNCTIONS =====
function resetResult() {
  resultEl.classList.add("hidden");
  errorBox.classList.add("hidden");
  errorBox.textContent = "";
}

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.classList.remove("hidden");
}

function showPreview(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewImg.style.display = "block";
    const placeholder = previewBox.querySelector(".preview-placeholder");
    if (placeholder) placeholder.style.display = "none";
  };
  reader.readAsDataURL(file);
}

// ===== DRAG & DROP =====
uploadArea.addEventListener("click", () => imageInput.click());
browseBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  imageInput.click();
});

uploadArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadArea.classList.add("dragover");
});

uploadArea.addEventListener("dragleave", () => {
  uploadArea.classList.remove("dragover");
});

uploadArea.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadArea.classList.remove("dragover");
  handleFile(e.dataTransfer.files[0]);
});

imageInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

// ===== FILE VALIDATION =====
function handleFile(file) {
  resetResult();

  if (!file) return;

  if (!file.type.startsWith("image/")) {
    showError("Please upload a valid image file.");
    return;
  }

  if (file.size > 5 * 1024 * 1024) {
    showError("Image is too large. Max 5 MB allowed.");
    return;
  }

  selectedFile = file;
  showPreview(file);
  analyzeBtn.disabled = false;
}

// ===== ANALYZE BUTTON =====
analyzeBtn.addEventListener("click", async () => {
  resetResult();

  if (!selectedFile) {
    showError("Please select an image.");
    return;
  }

  analyzeBtn.disabled = true;
  loadingEl.classList.remove("hidden");

  try {
    const formData = new FormData();
    formData.append("image", selectedFile);

    // 🔥 Correct backend API URL
    const response = await fetch("http://localhost:5000/detect", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) throw new Error("Server error");

    const data = await response.json();

    const label = data.label.toLowerCase();
    const confidence = data.confidence;

    resultTextEl.classList.remove("real", "deepfake");

    if (label === "real") {
      resultTextEl.textContent = "Real Image";
      resultTextEl.classList.add("real");
    } else {
      resultTextEl.textContent = "Deepfake Image";
      resultTextEl.classList.add("deepfake");
    }

    resultScoreEl.textContent = (confidence * 100).toFixed(1) + "%";
    resultEl.classList.remove("hidden");

  } catch (err) {
    showError(err.message);
  } finally {
    loadingEl.classList.add("hidden");
    analyzeBtn.disabled = !selectedFile;
  }
});
