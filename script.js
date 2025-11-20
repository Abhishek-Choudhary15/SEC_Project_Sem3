/* Professional frontend JS:
 - drag & drop + keyboard accessible file select
 - preview + file metadata
 - upload with progress *visual* (server doesn't stream progress)
 - shows loading/result/error states, download report
*/

const fileInput = document.getElementById('fileInput');
const dropZone = document.getElementById('dropZone');
const previewWrap = document.getElementById('previewWrap');
const previewImage = document.getElementById('previewImage');
const fileNameEl = document.getElementById('fileName');
const fileSizeEl = document.getElementById('fileSize');
const clearBtn = document.getElementById('clearBtn');
const analyzeBtn = document.getElementById('analyzeBtn');

const resultEmpty = document.getElementById('resultEmpty');
const resultLoading = document.getElementById('resultLoading');
const resultFinal = document.getElementById('resultFinal');
const resultError = document.getElementById('resultError');

const progBar = document.getElementById('progBar');
const labelText = document.getElementById('labelText');
const confidenceText = document.getElementById('confidenceText');
const badge = document.getElementById('badge');
const rawJson = document.getElementById('rawJson');
const downloadBtn = document.getElementById('downloadBtn');
const reAnalyzeBtn = document.getElementById('reAnalyzeBtn');
const retryBtn = document.getElementById('retryBtn');

let selectedFile = null;
let lastResult = null;

function bytesToSize(bytes){
  if (bytes === 0) return '0 B';
  const k = 1024, sizes = ['B','KB','MB','GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function resetUI(){
  previewWrap.classList.add('hidden');
  analyzeBtn.disabled = true;
  selectedFile = null;
  resultEmpty.classList.remove('hidden');
  resultLoading.classList.add('hidden');
  resultFinal.classList.add('hidden');
  resultError.classList.add('hidden');
  progBar.style.width = '0%';
  rawJson.textContent = '';
  lastResult = null;
}

function showPreview(file){
  const reader = new FileReader();
  reader.onload = e => {
    previewImage.src = e.target.result;
    previewWrap.classList.remove('hidden');
    fileNameEl.textContent = file.name;
    fileSizeEl.textContent = bytesToSize(file.size);
    analyzeBtn.disabled = false;
    resultEmpty.classList.add('hidden');
  };
  reader.readAsDataURL(file);
}

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fileInput.click();
  }
});

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('dragover');
});
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f) handleFile(f);
});

fileInput.addEventListener('change', e => {
  const f = e.target.files && e.target.files[0];
  if (f) handleFile(f);
});

function handleFile(file){
  if (!file.type.startsWith('image/')) {
    alert('Please select an image file (jpg, png).');
    return;
  }
  if (file.size > 8 * 1024 * 1024) {
    alert('Max file size is 8 MB.');
    return;
  }
  selectedFile = file;
  showPreview(file);
}

clearBtn.addEventListener('click', resetUI);

analyzeBtn.addEventListener('click', () => {
  if (!selectedFile) return;
  startAnalysis(selectedFile);
});

async function startAnalysis(file){
  // UI states
  resultEmpty.classList.add('hidden');
  resultError.classList.add('hidden');
  resultFinal.classList.add('hidden');
  resultLoading.classList.remove('hidden');
  progBar.style.width = '6%';

  try {
    // prepare form
    const form = new FormData();
    form.append('image', file, file.name);

    // send request
    const resp = await fetch('/upload', {
      method: 'POST',
      body: form
    });

    // visual progress simulation
    for (let p = 10; p <= 70; p += 8){
      progBar.style.width = p + '%';
      await new Promise(r => setTimeout(r, 120));
    }

    if (!resp.ok){
      const txt = await resp.text();
      throw new Error(txt || `Server returned ${resp.status}`);
    }
    const data = await resp.json();

    // finish progress
    progBar.style.width = '100%';

    showResult(data);
  } catch (err) {
    showError(err.message || 'Unknown error');
  } finally {
    // small delay for UX
    await new Promise(r => setTimeout(r, 300));
    resultLoading.classList.add('hidden');
  }
}

function showResult(data){
  lastResult = data;
  resultFinal.classList.remove('hidden');
  resultError.classList.add('hidden');

  const label = data.label || 'unknown';
  const conf = (typeof data.confidence === 'number') ? Math.round(data.confidence * 10000) / 100 : 'N/A';

  labelText.textContent = label.toUpperCase();
  confidenceText.textContent = `Confidence: ${conf === 'N/A' ? conf : conf + '%'}`;

  rawJson.textContent = JSON.stringify(data, null, 2);

  badge.className = 'badge';
  if (label === 'fake') {
    badge.textContent = 'FAKE';
    badge.classList.add('fake');
  } else if (label === 'real') {
    badge.textContent = 'REAL';
    badge.classList.add('real');
  } else {
    badge.textContent = 'UNKNOWN';
    badge.classList.add('unknown');
  }
}

function showError(msg){
  resultError.classList.remove('hidden');
  resultFinal.classList.add('hidden');
  const errorMsg = document.getElementById('errorMsg');
  errorMsg.textContent = msg;
}

downloadBtn.addEventListener('click', () => {
  if (!lastResult) return;
  const blob = new Blob([JSON.stringify(lastResult, null, 2)], {type:'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `analysis-${Date.now()}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
});

reAnalyzeBtn.addEventListener('click', resetUI);
retryBtn.addEventListener('click', () => {
  if (selectedFile) startAnalysis(selectedFile);
  else resetUI();
});

// init
resetUI();
