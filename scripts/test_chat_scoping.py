"""Manual check: a chat with uploads must answer only from those uploads."""
import sys
import uuid

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://127.0.0.1:8000"
UPLOAD_PDF = r"D:\RAG_pipeline\data\pdf1\01_Introduction_to_Attention_Mechanisms.pdf"
FOREIGN_TERMS = ["CG_External", "TOC_External", "CO_External", "BWP_External", "EE_External"]


def main() -> int:
    session = requests.Session()
    email = f"scopetest_{uuid.uuid4().hex[:8]}@example.com"

    resp = session.post(
        f"{BASE}/api/auth/signup",
        json={"name": "Scope Test", "email": email, "password": "test1234"},
        timeout=60,
    )
    resp.raise_for_status()
    print("signup:", resp.json()["user"]["email"])

    chat_id = session.post(f"{BASE}/api/chats", json={"title": "New chat"}, timeout=60).json()["chat"]["id"]
    print("chat id:", chat_id)

    with open(UPLOAD_PDF, "rb") as fh:
        upload = session.post(
            f"{BASE}/api/chats/{chat_id}/upload",
            files={"file": (UPLOAD_PDF.split("\\")[-1], fh, "application/pdf")},
            timeout=600,
        )
    upload.raise_for_status()
    print("upload:", upload.json()["file"])

    answer = session.post(
        f"{BASE}/api/chats/{chat_id}/messages",
        json={"message": "what this pdf is about?", "top_k": 6},
        timeout=600,
    ).json().get("answer", "")

    print("\n--- ANSWER ---\n", answer[:1200])

    leaked = [term for term in FOREIGN_TERMS if term.lower() in answer.lower()]
    if leaked:
        print("\nFAIL: answer leaked other papers:", leaked)
        return 1

    print("\nPASS: answer stayed inside this chat's uploaded document.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
