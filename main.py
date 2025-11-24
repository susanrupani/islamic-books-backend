from http.server import BaseHTTPRequestHandler, HTTPServer
from openai import OpenAI
import json
import os
from typing import Any

# -------- Configuration --------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

# Your OpenAI vector store ID (already created on the OpenAI platform)
VECTOR_STORE_ID = "vs_69232ab76db88191872d480a707c4733"

# Default model if none is provided
DEFAULT_MODEL = "gpt-4.1-mini"

# OpenAI client
client: Any = OpenAI(api_key=OPENAI_API_KEY)


# -------- HTTP Handler --------

class Handler(BaseHTTPRequestHandler):

    def _set_headers(self, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/health":
            self._set_headers(200)
            self.wfile.write(json.dumps({"status": "ok"}).encode())
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Not found"}).encode())

    def do_POST(self) -> None:
        if self.path != "/vector-search":
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Not found"}).encode())
            return

        # Read body
        content_len = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_len).decode()

        # Parse JSON body
        try:
            data = json.loads(raw_body)
        except json.JSONDecodeError:
            self._set_headers(200)
            self.wfile.write(json.dumps({
                "answer": "",
                "model_used": "",
                "error": "Invalid JSON"
            }).encode())
            return

        question = (data.get("question") or "").strip()
        if not question:
            self._set_headers(200)
            self.wfile.write(json.dumps({
                "answer": "",
                "model_used": "",
                "error": "No question provided"
            }).encode())
            return

        model = (data.get("model") or "").strip() or DEFAULT_MODEL

        try:
            # Call OpenAI Responses API with file_search tool
            response = client.responses.create(
                model=model,
                input=question,
                tools=[{"type": "file_search"}],
                tool_resources={
                    "file_search": {"vector_store_ids": [VECTOR_STORE_ID]}
                },
            )

            # Preferred helper (new SDK)
            answer_text = getattr(response, "output_text", None)

            # Fallback: manual extraction
            if not answer_text:
                chunks = []
                for item in getattr(response, "output", []) or []:
                    for content in getattr(item, "content", []) or []:
                        if getattr(content, "type", "") == "output_text":
                            chunks.append(content.text.value)
                answer_text = "\n\n".join(chunks) if chunks else ""

            self._set_headers(200)
            self.wfile.write(json.dumps({
                "answer": answer_text,
                "model_used": model,
                "error": ""
            }).encode())

        except Exception as e:
            # Log error
            print("Error in /vector-search:", repr(e))

            # Always return 200 so GPT Actions do NOT fail
            self._set_headers(200)
            self.wfile.write(json.dumps({
                "answer": "",
                "model_used": model,
                "error": str(e)
            }).encode())


# -------- Server Runner --------

def run(server_class=HTTPServer, handler_class=Handler) -> None:
    # Render sets PORT in env; default to 8000 for local dev
    port = int(os.environ.get("PORT", "8000"))
    server_address = ("0.0.0.0", port)
    httpd = server_class(server_address, handler_class)
    print(f"Serving on port {port}...")
    httpd.serve_forever()


if __name__ == "__main__":
    run()
