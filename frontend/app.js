const chatWindow = document.getElementById("chatWindow");
const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const statusText = document.getElementById("statusText");

function appendMessage(text, role = "bot") {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.ready) {
      statusText.textContent = "Pipeline ready. Ask your question.";
      return true;
    }
    statusText.textContent = data.error ? `Startup error: ${data.error}` : "Pipeline initializing...";
    return false;
  } catch (err) {
    statusText.textContent = "Backend not reachable.";
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
  isLoading = true;

  appendMessage("Thinking...", "bot");
  const thinkingNode = chatWindow.lastElementChild;

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
      appendMessage(data.answer || "No answer returned.", "bot");
    }
  } catch (err) {
    thinkingNode.remove();
    appendMessage("Network error. Please retry.", "error");
  } finally {
    isLoading = false;
    sendButton.disabled = false;
    messageInput.focus();
  }
});

appendMessage("Hello! Ask a question based on your loaded documents.", "bot");
checkHealth();
setInterval(checkHealth, 10000);
