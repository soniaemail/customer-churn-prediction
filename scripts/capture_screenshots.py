"""Capture README screenshots of the running app with headless Edge/Chrome.

Start the app (``streamlit run main.py``), then::

    python scripts/capture_screenshots.py

The script drives the browser through the Chrome DevTools Protocol, waits
for Streamlit to finish rendering and saves PNGs to ``assets/images/``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from tornado.websocket import websocket_connect

APP_URL = "http://localhost:8501"
DEBUG_PORT = 9333
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "assets" / "images"
PAGES = {
    "home": "",
    "dashboard": "Dashboard",
    "prediction": "Prediction",
    "model_comparison": "Model_Comparison",
    "explainability": "Explainability",
}
BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "google-chrome",
    "chromium",
    "microsoft-edge",
]
# Clicks "Predict Churn" so the prediction screenshot shows a result
CLICK_PREDICT = """
(() => { const b = [...document.querySelectorAll('button')].find(x => x.innerText.includes('Predict Churn'));
         if (b) b.click(); return !!b; })()
"""


def find_browser() -> str:
    for candidate in BROWSERS:
        if Path(candidate).exists() or shutil.which(candidate):
            return candidate
    raise FileNotFoundError("No Chromium-based browser found")


async def capture() -> None:
    profile = tempfile.mkdtemp(prefix="churn-shots-")
    browser = subprocess.Popen(
        [find_browser(), "--headless=new", "--disable-gpu", "--hide-scrollbars",
         f"--remote-debugging-port={DEBUG_PORT}", f"--user-data-dir={profile}",
         "--window-size=1440,1000", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(40):  # wait for the DevTools endpoint
            try:
                targets = json.load(urllib.request.urlopen(f"http://localhost:{DEBUG_PORT}/json"))
                break
            except OSError:
                time.sleep(0.25)
        else:
            raise RuntimeError("Browser did not start")

        page = next(t for t in targets if t["type"] == "page")
        socket = await websocket_connect(page["webSocketDebuggerUrl"], max_message_size=64 * 2**20)
        message_id = 0

        async def send(method: str, **params: object) -> dict:
            nonlocal message_id
            message_id += 1
            await socket.write_message(json.dumps({"id": message_id, "method": method, "params": params}))
            while True:
                reply = json.loads(await socket.read_message())
                if reply.get("id") == message_id:
                    return reply.get("result", {})

        await send("Emulation.setDeviceMetricsOverride", width=1440, height=1000,
                   deviceScaleFactor=1, mobile=False)
        for name, path in PAGES.items():
            await send("Page.navigate", url=f"{APP_URL}/{path}")
            await asyncio.sleep(12)
            if name == "prediction":
                await send("Runtime.evaluate", expression=CLICK_PREDICT)
                await asyncio.sleep(6)
                await send("Runtime.evaluate", expression=(
                    "document.querySelector('[data-testid=\"stMain\"]').scrollTo(0, 430)"))
                await asyncio.sleep(1)
            shot = await send("Page.captureScreenshot", format="png")
            target = OUTPUT_DIR / f"{name}.png"
            target.write_bytes(base64.b64decode(shot["data"]))
            print(f"saved {target.name} ({target.stat().st_size / 1024:.0f} kB)")
        socket.close()
    finally:
        browser.terminate()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    asyncio.run(capture())
