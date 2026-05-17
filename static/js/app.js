/**
 * Smart PDF Summarizer — Client-side logic
 */

// ── Theme ──────────────────────────────────────────────────────────────────
function initTheme() {
  const html = document.documentElement;
  const saved = localStorage.getItem('theme') || 'dark';
  html.setAttribute('data-theme', saved);

  const btn = document.querySelector('[data-theme-toggle]');
  if (!btn) return;
  btn.addEventListener('click', () => {
    const next = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
  });
}

// ── Toasts ─────────────────────────────────────────────────────────────────
function showToast(message, type = 'info', duration = 3500) {
  const container = document.getElementById('toast-container');
  if (!container) return;
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), duration);
}

// ── Auth ───────────────────────────────────────────────────────────────────
function initAuthForms() {
  document.querySelectorAll('[data-auth-form]').forEach(form => {
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const endpoint = form.dataset.authEndpoint;
      const payload = Object.fromEntries(new FormData(form));
      const btn = form.querySelector('button[type="submit"]');
      const orig = btn?.textContent;
      if (btn) { btn.disabled = true; btn.textContent = 'Please wait…'; }
      try {
        const res = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (data.success) {
          showToast(data.message, 'success');
          if (data.redirect) setTimeout(() => window.location.href = data.redirect, 600);
        } else {
          showToast(data.message || 'An error occurred.', 'error');
        }
      } catch (err) {
        showToast('Network error: ' + err.message, 'error');
      } finally {
        if (btn) { btn.disabled = false; btn.textContent = orig; }
      }
    });
  });
}

// ── Upload ─────────────────────────────────────────────────────────────────
function initUpload() {
  const dropzone = document.getElementById('dropzone');
  const form = document.querySelector('[data-upload-form]');
  const input = document.getElementById('pdf-input');
  const fileList = document.getElementById('file-list');
  if (!dropzone || !form) return;

  dropzone.addEventListener('click', () => input?.click());

  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(ev =>
    dropzone.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); })
  );
  dropzone.addEventListener('dragover', () => dropzone.classList.add('dragover'));
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
  dropzone.addEventListener('drop', (e) => {
    dropzone.classList.remove('dragover');
    input.files = e.dataTransfer?.files;
    renderFileList(input.files);
  });
  input?.addEventListener('change', e => renderFileList(e.target.files));

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!input?.files?.length) { showToast('Select at least one PDF.', 'warning'); return; }
    const fd = new FormData();
    Array.from(input.files).forEach(f => fd.append('files', f));
    const btn = document.getElementById('upload-btn');
    if (btn) { btn.disabled = true; btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Uploading…'; }
    try {
      const res = await fetch('/upload', { method: 'POST', body: fd });
      const data = await res.json();
      if (data.success) {
        showToast(data.message, 'success');
        input.value = '';
        if (fileList) fileList.innerHTML = '';
        setTimeout(() => window.location.href = '/dashboard', 1200);
      } else {
        showToast(data.message || 'Upload failed.', 'error');
      }
    } catch (err) {
      showToast('Upload error: ' + err.message, 'error');
    } finally {
      if (btn) { btn.disabled = false; btn.innerHTML = '<i class="fa-solid fa-upload"></i> Upload files'; }
    }
  });
}

function renderFileList(files) {
  const fileList = document.getElementById('file-list');
  if (!fileList) return;
  fileList.innerHTML = '';
  Array.from(files).forEach(f => {
    const el = document.createElement('div');
    el.className = 'file-item';
    el.innerHTML = `<strong><i class="fa-solid fa-file-pdf" style="color:var(--primary);margin-right:6px;"></i>${f.name}</strong><p>${(f.size/1024/1024).toFixed(2)} MB</p>`;
    fileList.appendChild(el);
  });
}

// ── Summary ────────────────────────────────────────────────────────────────
function initSummary() {
  const genBtn = document.querySelector('[data-generate-summary]');
  const delBtn = document.querySelector('[data-delete-summary]');
  const delPdfBtn = document.querySelector('[data-delete-pdf]');
  const lengthSel = document.querySelector('[data-summary-length]');
  const toggleBtn = document.getElementById('toggle-fullview');

  if (genBtn) genBtn.addEventListener('click', handleGenerateSummary);
  if (delBtn) delBtn.addEventListener('click', handleDeleteSummary);
  if (delPdfBtn) delPdfBtn.addEventListener('click', handleDeletePdf);

  if (toggleBtn) {
    const grid = document.getElementById('summary-grid');
    toggleBtn.addEventListener('click', () => {
      const full = grid?.classList.toggle('full');
      localStorage.setItem('summaryFull', full ? '1' : '0');
    });
    if (localStorage.getItem('summaryFull') === '1') grid?.classList.add('full');
  }

  initFlashcards();
}

async function handleGenerateSummary(e) {
  const pdfId = e.currentTarget.dataset.pdfId;
  const length = document.querySelector('[data-summary-length]')?.value || 'medium';
  const btn = e.currentTarget;
  const orig = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Generating…';
  try {
    const res = await fetch(`/api/summary/${pdfId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ length }),
    });
    const data = await res.json();
    if (data.success && data.summary) {
      displaySummary(data.summary);
      showToast('Summary generated!', 'success');
    } else {
      showToast(data.message || 'Generation failed.', 'error');
    }
  } catch (err) {
    showToast('Error: ' + err.message, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = orig;
  }
}

function displaySummary(summary) {
  const st = document.getElementById('summary-text');
  if (st) st.innerHTML = `<p>${(summary.summary_text || '').replace(/\n\n/g, '</p><p>').replace(/\n/g, '<br>')}</p>`;

  const kw = document.getElementById('keyword-list');
  if (kw && summary.keywords?.length) {
    kw.innerHTML = summary.keywords.map(k => `<span class="tag">${k}</span>`).join('');
  }

  const fc = document.getElementById('flashcard-list');
  if (fc && summary.flashcards?.length) {
    fc.innerHTML = summary.flashcards.map(c => `
      <button type="button" class="flashcard" data-flip-card>
        <span class="flashcard-front">${c.front}</span>
        <span class="flashcard-back">${c.back}</span>
      </button>`).join('');
    initFlashcards();
  }
}

async function handleDeleteSummary(e) {
  const pdfId = e.currentTarget.dataset.pdfId;
  if (!confirm('Delete this summary?')) return;
  const btn = e.currentTarget;
  const orig = btn.innerHTML;
  btn.disabled = true; btn.innerHTML = 'Deleting…';
  try {
    const res = await fetch(`/api/summary/${pdfId}`, { method: 'DELETE' });
    const data = await res.json();
    if (data.success) {
      showToast('Summary deleted.', 'success');
      document.getElementById('summary-text').innerHTML = '<p style="color:var(--muted);">No summary yet. Choose a length and click <strong>Generate</strong>.</p>';
      const kw = document.getElementById('keyword-list');
      if (kw) kw.innerHTML = '<p class="empty-state">Keywords appear after generation.</p>';
      const fc = document.getElementById('flashcard-list');
      if (fc) fc.innerHTML = '<p class="empty-state">Flashcards appear after generation.</p>';
    } else { showToast(data.message || 'Could not delete.', 'error'); }
  } catch (err) { showToast('Error: ' + err.message, 'error'); }
  finally { btn.disabled = false; btn.innerHTML = orig; }
}

async function handleDeletePdf(e) {
  const pdfId = e.currentTarget.dataset.pdfId;
  if (!confirm('Permanently delete this PDF, its summary and chat history? This cannot be undone.')) return;
  const btn = e.currentTarget;
  const orig = btn.innerHTML;
  btn.disabled = true; btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Deleting…';
  try {
    const res = await fetch(`/api/pdf/${pdfId}`, { method: 'DELETE' });
    const data = await res.json();
    if (data.success) {
      showToast('PDF deleted.', 'success');
      setTimeout(() => window.location.href = '/dashboard', 1000);
    } else {
      showToast(data.message || 'Could not delete.', 'error');
      btn.disabled = false; btn.innerHTML = orig;
    }
  } catch (err) {
    showToast('Error: ' + err.message, 'error');
    btn.disabled = false; btn.innerHTML = orig;
  }
}

// ── Flashcards ─────────────────────────────────────────────────────────────
function initFlashcards() {
  document.querySelectorAll('[data-flip-card]').forEach(card => {
    card.addEventListener('click', () => card.classList.toggle('flipped'));
  });
}

// ── Chat ───────────────────────────────────────────────────────────────────
function initChat() {
  const form = document.querySelector('[data-chat-form]');
  if (!form) return;
  form.addEventListener('submit', handleChatSubmit);
}

async function handleChatSubmit(e) {
  e.preventDefault();
  const root = document.querySelector('[data-chat-root]');
  const pdfId = root?.dataset.pdfId;
  const input = document.getElementById('chat-input');
  const question = input?.value?.trim();
  if (!question) return;

  const messages = document.getElementById('chat-messages');
  const sendBtn = document.getElementById('chat-send-btn');

  // User bubble
  appendBubble(messages, 'user', question);
  input.value = '';
  if (sendBtn) { sendBtn.disabled = true; sendBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>'; }

  // Typing indicator
  const typingId = 'typing-' + Date.now();
  appendTyping(messages, typingId);

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pdf_id: pdfId, question }),
    });
    const data = await res.json();
    document.getElementById(typingId)?.remove();

    if (data.success && data.response) {
      const { answer, confidence, low_confidence } = data.response;
      appendBotBubble(messages, answer, confidence, low_confidence);

    } else {
      appendBubble(messages, 'bot', data.message || 'Sorry, I could not find an answer.');
    }
  } catch (err) {
    document.getElementById(typingId)?.remove();
    appendBubble(messages, 'bot', 'Network error: ' + err.message);
  } finally {
    if (sendBtn) { sendBtn.disabled = false; sendBtn.innerHTML = '<i class="fa-solid fa-paper-plane"></i> Send'; }
  }
}

function appendBubble(container, type, text) {
  const div = document.createElement('div');
  div.className = `chat-bubble ${type}`;
  const ans = document.createElement('div');
  ans.className = 'chat-answer';
  ans.textContent = text;
  div.appendChild(ans);
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function appendBotBubble(container, text, confidence, lowConfidence) {
  const div = document.createElement('div');
  div.className = 'chat-bubble bot';

  if (lowConfidence) {
    div.style.borderColor = 'rgba(251,191,36,0.4)';
    div.style.background = 'rgba(251,191,36,0.06)';
  }

  const ans = document.createElement('div');
  ans.className = 'chat-answer';
  ans.style.whiteSpace = 'pre-wrap';
  ans.textContent = text;

  div.appendChild(ans);

  if (!lowConfidence) {
    const pct = Math.round((confidence || 0) * 100);
    const color = pct >= 65 ? 'var(--success)' : pct >= 35 ? 'var(--accent)' : 'var(--danger)';
    const meta = document.createElement('div');
    meta.className = 'chat-meta';
    meta.innerHTML = `
      <span style="color:${color};font-weight:600;">${pct}% match</span>
      <div class="confidence-bar">
        <div class="confidence-fill" style="width:${pct}%;background:${color};"></div>
      </div>`;
    div.appendChild(meta);
  }

  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function appendTyping(container, id) {
  const div = document.createElement('div');
  div.className = 'chat-bubble bot';
  div.id = id;
  div.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

// ── Boot ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initAuthForms();
  initUpload();
  initSummary();
  initChat();
});
