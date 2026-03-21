const chatWindow = document.getElementById("chatWindow");
const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const statusText = document.getElementById("statusText");
const typingTemplate = document.getElementById("typingTemplate");
const messageTemplate = document.getElementById("messageTemplate");
const chips = document.querySelectorAll(".chip");


function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}


function formatAnswer(text) {
  const normalized = (text || "").replace(/\r\n/g, "\n");
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
      html += "<div class=\"msg-gap\"></div>";
      continue;
    }

    const numbered = line.match(/^(\d+)\.\s+(.*)$/);
    const bulleted = line.match(/^[-*]\s+(.*)$/);
    const safeLine = escapeHtml(numbered ? numbered[2] : bulleted ? bulleted[1] : line)
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\(([^,]+?),\s*([^\)]+?)\)/g, "<span class=\"citation\">($1, $2)</span>");

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


function appendMessage(text, role = "bot", isHtml = false) {
  const node = messageTemplate.content.firstElementChild.cloneNode(true);
  const roleNode = node.querySelector(".msg-role");
  const messageNode = node.querySelector(".msg");

  node.classList.add(role);
  roleNode.textContent = role === "user" ? "You" : role === "error" ? "System" : "Copilot";

  if (isHtml) {
    messageNode.innerHTML = text;
  } else {
    messageNode.textContent = text;
  }

  chatWindow.appendChild(node);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}


function appendTyping() {
  const typingNode = typingTemplate.content.firstElementChild.cloneNode(true);
  chatWindow.appendChild(typingNode);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return typingNode;
}


function setStatus(type, text) {
  statusText.textContent = text;
  statusText.classList.remove("ok", "warn", "err");
  statusText.classList.add(type);
}


async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();

    if (res.ok && data.ready) {
      setStatus("ok", "Ready. Ask your question.");
      return true;
    }

    if (data.error) {
      setStatus("err", `Startup error: ${data.error}`);
    } else if (data.loading) {
      setStatus("warn", "Loading RAG engine. First run can take time.");
    } else {
      setStatus("warn", "Starting services...");
    }
    return false;
  } catch (err) {
    setStatus("err", "Backend not reachable.");
    return false;
  }
}

let isLoading = false;

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message || isLoading) return;

  appendMessage(message, "user");
  messageInput.value = "";
  sendButton.disabled = true;
  messageInput.disabled = true;
  isLoading = true;

  const thinkingNode = appendTyping();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, top_k: 6 }),
    });

    const data = await res.json();
    thinkingNode.remove();

    if (!res.ok) {
      appendMessage(data.error || "Request failed.", "error");
    } else {
      appendMessage(formatAnswer(data.answer || "No answer returned."), "bot", true);
    }
  } catch (err) {
    thinkingNode.remove();
    appendMessage("Network error. Please retry.", "error");
  } finally {
    isLoading = false;
    sendButton.disabled = false;
    messageInput.disabled = false;
    messageInput.focus();
  }
});

chips.forEach((chip) => {
  chip.addEventListener("click", () => {
    messageInput.value = chip.dataset.prompt || "";
    messageInput.focus();
  });
});

appendMessage(
  "Welcome. Ask exam questions and I will answer using your document context with citations.",
  "bot"
);
checkHealth();
setInterval(checkHealth, 10000);
