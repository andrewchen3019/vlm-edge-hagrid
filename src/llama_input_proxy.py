import base64
import hashlib
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests


TARGET = "http://127.0.0.1:8081"
OUT_DIR = Path("results/proxy_received_images")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sha16(x):
    if isinstance(x, str):
        x = x.encode("utf-8")
    return hashlib.sha256(x).hexdigest()[:16]


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        print("\n" + "=" * 100)
        print("POST", self.path)
        print("request body bytes:", len(body))
        print("request body sha:", sha16(body))

        try:
            payload = json.loads(body)

            print("\nTOP-LEVEL KEYS:")
            print(list(payload.keys()))

            prompt = payload.get("prompt", "")
            print("\nPROMPT:")
            print("type:", type(prompt).__name__)
            if isinstance(prompt, str):
                print("contains <__media__>:", "<__media__>" in prompt)
                print("media marker count:", prompt.count("<__media__>"))
                print("prompt preview:", repr(prompt[:500]))
            else:
                print("prompt preview:", repr(str(prompt)[:500]))

            multimodal_data = payload.get("multimodal_data", [])
            print("\nMULTIMODAL_DATA:")
            print("type:", type(multimodal_data).__name__)
            print("num images:", len(multimodal_data) if isinstance(multimodal_data, list) else "not a list")

            if isinstance(multimodal_data, list):
                for idx, b64 in enumerate(multimodal_data):
                    print(f"\nimage {idx}:")
                    print("base64 len:", len(b64))
                    print("base64 sha:", sha16(b64))
                    print("base64 start:", b64[:80])
                    print("base64 middle:", b64[len(b64)//2:len(b64)//2 + 80])
                    print("base64 end:", b64[-80:])

                    try:
                        raw = base64.b64decode(b64)
                        raw_sha = sha16(raw)
                        print("decoded image bytes:", len(raw))
                        print("decoded image sha:", raw_sha)

                        out_path = OUT_DIR / f"request_{sha16(body)}_image{idx}_{raw_sha}.jpg"
                        out_path.write_bytes(raw)
                        print("saved decoded image:", out_path)

                    except Exception as e:
                        print("could not decode image:", e)

            print("\nFORWARDING TO:", TARGET + self.path)

        except Exception as e:
            print("Could not parse JSON:", e)
            print("Raw body preview:")
            print(body[:1000])

        # Forward exact request body to real llama-server
        try:
            r = requests.post(
                TARGET + self.path,
                data=body,
                headers={"Content-Type": self.headers.get("Content-Type", "application/json")},
                timeout=600,
            )

            print("real server status:", r.status_code)
            try:
                response_json = r.json()
                print("real server content:", repr(response_json.get("content", "")))
                print("tokens_evaluated:", response_json.get("tokens_evaluated"))
            except Exception:
                print("real server raw response:", r.text[:500])

            self.send_response(r.status_code)
            self.send_header("Content-Type", r.headers.get("Content-Type", "application/json"))
            self.end_headers()
            self.wfile.write(r.content)

        except Exception as e:
            print("proxy forwarding error:", e)
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(str(e).encode("utf-8"))


if __name__ == "__main__":
    print("Proxy listening on http://127.0.0.1:8080")
    print("Forwarding to real llama-server at", TARGET)
    print("Decoded images will be saved in", OUT_DIR)
    ThreadingHTTPServer(("127.0.0.1", 8080), Handler).serve_forever()
