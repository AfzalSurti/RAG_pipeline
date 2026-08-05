const authView = document.getElementById("authView");
const appView = document.getElementById("appView");
const authForm = document.getElementById("authForm");
const authName = document.getElementById("authName");
const authEmail = document.getElementById("authEmail");
const authPassword = document.getElementById("authPassword");
const authError = document.getElementById("authError");
const authSubmit = document.getElementById("authSubmit");
const nameField = document.getElementById("nameField");
const userLabel = document.getElementById("userLabel");
const chatList = document.getElementById("chatList");
const chatWindow = document.getElementById("chatWindow");
const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const statusText = document.getElementById("statusText");
const statusDot = document.getElementById("statusDot");
const chatTitle = document.getElementById("chatTitle");
const filesStrip = document.getElementById("filesStrip");
const fileInput = document.getElementById("fileInput");
const newChatBtn = document.getElementById("newChatBtn");
const logoutBtn = document.getElementById("logoutBtn");
const typingTemplate = document.getElementById("typingTemplate");
const messageTemplate = document.getElementById("messageTemplate");

let authMode = "login";
let currentUser = null;
let currentChatId = null;
let chats = [];
let isLoading = false;

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatAnswer(text) {
  const normalized = String(text || "").replace(/\r\n/g, "\n");
  const lines = normalized.split("\n");
  let html = "";
  let inOrderedList = false;
  let inUnorderedList = false;

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) {
      if (inOrderedList) {
        html += "</ol>";
        inOrderedList = false;
      }
      if (inUnorderedList) {
        html += "</ul>";
        inUnorderedList = false;
      }
      html += '<div class="msg-gap"></div>';
      continue;
    }

    const numbered = line.match(/^(\d+)\.\s+(.*)$/);
    const bulleted = line.match(/^[-*]\s+(.*)$/);
    const safeLine = escapeHtml(numbered ? numbered[2] : bulleted ? bulleted[1] : line)
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\(([^,]+?),\s*([^\)]+?)\)/g, '<span class="citation">($1, $2)</span>');

    if (numbered) {
      if (inUnorderedList) {
        html += "</ul>";
        inUnorderedList = false;
      }
      if (!inOrderedList) {
        html += "<ol>";
        inOrderedList = true;
      }
      html += `<li>${safeLine}</li>`;
      continue;
    }

    if (bulleted) {
      if (inOrderedList) {
        html += "</ol>";
        inOrderedList = false;
      }
      if (!inUnorderedList) {
        html += "<ul>";
        inUnorderedList = true;
      }
      html += `<li>${safeLine}</li>`;
      continue;
    }

    if (inOrderedList) {
      html += "</ol>";
      inOrderedList = false;
    }
    if (inUnorderedList) {
      html += "</ul>";
      inUnorderedList = false;
    }
    html += `<p>${safeLine}</p>`;
  }

  if (inOrderedList) html += "</ol>";
  if (inUnorderedList) html += "</ul>";
  return html || "<p>No answer returned.</p>";
}

function setAuthMode(mode) {
  authMode = mode;
  document.querySelectorAll(".auth-tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.mode === mode);
  });
  nameField.classList.toggle("hidden", mode !== "signup");
  authSubmit.textContent = mode === "signup" ? "Create account" : "Log in";
  authPassword.autocomplete = mode === "signup" ? "new-password" : "current-password";
  authError.hidden = true;
}

function showAuth() {
  authView.classList.remove("hidden");
  appView.classList.add("hidden");
}

function showApp() {
  authView.classList.add("hidden");
  appView.classList.remove("hidden");
  userLabel.textContent = currentUser?.name || currentUser?.email || "Signed in";
}

function setStatus(type, text) {
  statusText.textContent = text;
  statusDot.classList.remove("ok", "warn", "err");
  statusDot.classList.add(type);
}

async function api(url, options = {}) {
  const res = await fetch(url, {
    credentials: "same-origin",
    ...options,
    headers: {
      ...(options.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...(options.headers || {}),
    },
  });
  let data = {};
  try {
    data = await res.json();
  } catch (_) {
    data = {};
  }
  if (!res.ok) {
    const err = new Error(data.error || `Request failed (${res.status})`);
    err.status = res.status;
    throw err;
  }
  return data;
}

function appendMessage(text, role = "bot", isHtml = false) {
  const node = messageTemplate.content.firstElementChild.cloneNode(true);
  const roleNode = node.querySelector(".msg-role");
  const messageNode = node.querySelector(".msg");
  node.classList.add(role);
  roleNode.textContent =
    role === "user" ? "You" : role === "error" ? "System" : role === "system" ? "ExamMind" : "ExamMind";
  if (isHtml) messageNode.innerHTML = text;
  else messageNode.textContent = text;
  chatWindow.appendChild(node);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function appendTyping() {
  const typingNode = typingTemplate.content.firstElementChild.cloneNode(true);
  chatWindow.appendChild(typingNode);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return typingNode;
}

function renderEmptyState() {
  chatWindow.innerHTML = `
    <div class="empty-state">
      <h3>Start with a paper or a chapter</h3>
      <p>Ask from the built-in exam corpus, or upload a PDF/TXT/CSV for this chat and ask questions from it.</p>
    </div>
  `;
}

function renderFiles(files = []) {
  if (!files.length) {
    filesStrip.hidden = true;
    filesStrip.innerHTML = "";
    return;
  }
  filesStrip.hidden = false;
  filesStrip.innerHTML = files
    .map((f) => `<span class="file-chip">${escapeHtml(f.filename)}${f.indexed ? " · indexed" : ""}</span>`)
    .join("");
}

function renderChatList() {
  if (!chats.length) {
    chatList.innerHTML = `<p class="muted small">No chats yet. Create one to begin.</p>`;
    return;
  }
  chatList.innerHTML = chats
    .map(
      (c) => `
      <button type="button" class="chat-item ${c.id === currentChatId ? "active" : ""}" data-id="${c.id}">
        <div class="title">${escapeHtml(c.title || "New chat")}</div>
        <div class="meta">${c.updated_at ? new Date(c.updated_at).toLocaleString() : ""}</div>
      </button>`
    )
    .join("");

  chatList.querySelectorAll(".chat-item").forEach((btn) => {
    btn.addEventListener("click", () => openChat(Number(btn.dataset.id)));
  });
}

async function checkHealth() {
  try {
    const res = await fetch("/api/health", { credentials: "same-origin" });
    const data = await res.json();
    if (res.ok && data.ready) {
      setStatus("ok", "Ready");
      return true;
    }
    if (data.error) setStatus("err", data.error);
    else if (data.loading) setStatus("warn", "Loading RAG…");
    else setStatus("warn", "Starting…");
    return false;
  } catch (_) {
    setStatus("err", "Backend offline");
    return false;
  }
}

async function refreshChats() {
  const data = await api("/api/chats");
  chats = data.chats || [];
  renderChatList();
}

async function createChat() {
  const data = await api("/api/chats", {
    method: "POST",
    body: JSON.stringify({ title: "New chat" }),
  });
  await refreshChats();
  await openChat(data.chat.id);
}

async function openChat(chatId) {
  const data = await api(`/api/chats/${chatId}`);
  currentChatId = data.chat.id;
  chatTitle.textContent = data.chat.title || "Chat";
  renderFiles(data.chat.files || []);
  renderChatList();

  chatWindow.innerHTML = "";
  const messages = data.messages || [];
  if (!messages.length) {
    renderEmptyState();
    return;
  }
  messages.forEach((m) => {
    if (m.role === "user") appendMessage(m.content, "user");
    else if (m.role === "system") appendMessage(formatAnswer(m.content), "system", true);
    else appendMessage(formatAnswer(m.content), "bot", true);
  });
}

document.querySelectorAll(".auth-tab").forEach((tab) => {
  tab.addEventListener("click", () => setAuthMode(tab.dataset.mode));
});

authForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  authError.hidden = true;
  authSubmit.disabled = true;
  try {
    const payload = {
      email: authEmail.value.trim(),
      password: authPassword.value,
    };
    if (authMode === "signup") payload.name = authName.value.trim();
    const endpoint = authMode === "signup" ? "/api/auth/signup" : "/api/auth/login";
    const data = await api(endpoint, { method: "POST", body: JSON.stringify(payload) });
    currentUser = data.user;
    showApp();
    await refreshChats();
    if (chats.length) await openChat(chats[0].id);
    else await createChat();
  } catch (err) {
    authError.textContent = err.message;
    authError.hidden = false;
  } finally {
    authSubmit.disabled = false;
  }
});

logoutBtn.addEventListener("click", async () => {
  await api("/api/auth/logout", { method: "POST", body: "{}" });
  currentUser = null;
  currentChatId = null;
  chats = [];
  showAuth();
});

newChatBtn.addEventListener("click", async () => {
  try {
    await createChat();
  } catch (err) {
    appendMessage(err.message, "error");
  }
});

fileInput.addEventListener("change", async () => {
  const file = fileInput.files?.[0];
  fileInput.value = "";
  if (!file) return;
  if (!currentChatId) {
    appendMessage("Create or open a chat before uploading.", "error");
    return;
  }

  const thinking = appendTyping();
  try {
    const form = new FormData();
    form.append("file", file);
    const data = await api(`/api/chats/${currentChatId}/upload`, {
      method: "POST",
      body: form,
      headers: {},
    });
    thinking.remove();
    appendMessage(formatAnswer(data.notice || `Uploaded ${file.name}`), "system", true);
    if (data.chat) {
      chatTitle.textContent = data.chat.title || chatTitle.textContent;
      renderFiles(data.chat.files || []);
    }
    await refreshChats();
  } catch (err) {
    thinking.remove();
    appendMessage(err.message || "Upload failed.", "error");
  }
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message || isLoading) return;
  if (!currentChatId) {
    appendMessage("Create a chat first.", "error");
    return;
  }

  if (chatWindow.querySelector(".empty-state")) chatWindow.innerHTML = "";
  appendMessage(message, "user");
  messageInput.value = "";
  sendButton.disabled = true;
  messageInput.disabled = true;
  isLoading = true;
  const thinking = appendTyping();

  try {
    const data = await api(`/api/chats/${currentChatId}/messages`, {
      method: "POST",
      body: JSON.stringify({ message, top_k: 6 }),
    });
    thinking.remove();
    appendMessage(formatAnswer(data.answer || "No answer returned."), "bot", true);
    await refreshChats();
    const active = chats.find((c) => c.id === currentChatId);
    if (active) chatTitle.textContent = active.title;
  } catch (err) {
    thinking.remove();
    appendMessage(err.message || "Request failed.", "error");
  } finally {
    isLoading = false;
    sendButton.disabled = false;
    messageInput.disabled = false;
    messageInput.focus();
  }
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    messageInput.value = chip.dataset.prompt || "";
    messageInput.focus();
  });
});

async function boot() {
  setAuthMode("login");
  checkHealth();
  setInterval(checkHealth, 10000);
  try {
    const me = await api("/api/auth/me");
    if (me.user) {
      currentUser = me.user;
      showApp();
      await refreshChats();
      if (chats.length) await openChat(chats[0].id);
      else await createChat();
    } else {
      showAuth();
    }
  } catch (_) {
    showAuth();
  }
}

boot();
