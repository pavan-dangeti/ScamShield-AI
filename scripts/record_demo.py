"""Record the README demo by driving the web UI in a real browser.

Start a server with an empty analysis log first, so the dashboard shows only
what the demo does:

    SCAMSHIELD_MODEL=tfidf-lr uvicorn src.main:app --port 8000
    uv run --no-project --with playwright --with imageio-ffmpeg \\
        python -m scripts.record_demo --url http://127.0.0.1:8000

Writes ``docs/demo/scamshield_demo.mp4`` (full quality),
``docs/demo/scamshield_demo.gif`` (for the README, kept under 10 MB) and the
screenshots in ``docs/screenshots/``, all from the same browser session.
Neither tool is a project dependency; ``uv run --with`` fetches them for the run.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import tempfile

GIF_LIMIT_MB = 10.0
ENGLISH_SCAM = (
    "URGENT: Your SBI account will be blocked in 30 minutes. "
    "Verify your KYC now at bit.ly/sbi-kyc-verify or lose access."
)


def drive(page, url: str, shots: str) -> None:
    def pause(seconds: float) -> None:
        page.wait_for_timeout(int(seconds * 1000))

    def shot(name: str) -> None:
        page.screenshot(path=os.path.join(shots, f"{name}.png"), full_page=True)

    def analyze(name: str) -> None:
        page.click("#analyze-submit-btn")
        page.wait_for_selector("#result-details:not(.hide)")
        pause(1.0)
        shot(name)
        page.locator("#explanation-box").scroll_into_view_if_needed()
        pause(4.0)
        page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
        pause(1.0)

    def example(label: str, name: str) -> None:
        page.get_by_role("button", name=label, exact=True).click()
        pause(1.2)
        analyze(name)

    page.goto(url)
    pause(2.0)
    shot("01_home")
    page.click("#message-input")
    page.keyboard.type(ENGLISH_SCAM, delay=28)
    pause(0.8)
    analyze("02_english_scam")
    example("Legit Bank SMS", "03_legitimate_alert")
    example("Tanglish Scam", "04_tanglish_scam")
    example("Hindi Scam", "05_hindi_scam")
    page.click("#tab-stats-btn")
    pause(5.0)
    shot("06_dashboard")


def encode(ffmpeg: str, webm: str, mp4: str, gif: str) -> None:
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", webm, "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "23", "-movflags", "+faststart", mp4],
        check=True,
    )
    # Two-pass palette keeps the dark UI legible; step down until the GIF fits.
    for fps, width in ((10, 960), (8, 860), (6, 760), (5, 640)):
        flt = f"fps={fps},scale={width}:-1:flags=lanczos"
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", mp4, "-filter_complex",
             f"{flt},split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];"
             "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle", gif],
            check=True,
        )
        size_mb = os.path.getsize(gif) / 1e6
        print(f"gif at {fps} fps, {width}px: {size_mb:.1f} MB")
        if size_mb <= GIF_LIMIT_MB:
            return
    raise SystemExit(f"could not get the GIF under {GIF_LIMIT_MB} MB")


def main() -> None:
    from imageio_ffmpeg import get_ffmpeg_exe
    from playwright.sync_api import sync_playwright

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--out-dir", default="docs/demo")
    parser.add_argument("--screenshot-dir", default="docs/screenshots")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as video_dir, sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        size = {"width": 1280, "height": 800}
        context = browser.new_context(viewport=size, record_video_dir=video_dir, record_video_size=size)
        os.makedirs(args.screenshot_dir, exist_ok=True)
        drive(context.new_page(), args.url, args.screenshot_dir)
        context.close()
        # A separate, unrecorded context, so the demo video holds only the UI walkthrough.
        api_context = browser.new_context(viewport=size)
        api_page = api_context.new_page()
        api_page.goto(args.url.rstrip("/") + "/docs")
        api_page.wait_for_selector(".opblock")
        api_page.screenshot(path=os.path.join(args.screenshot_dir, "07_api_reference.png"))
        api_context.close()
        browser.close()
        webm = glob.glob(os.path.join(video_dir, "*.webm"))[0]
        os.makedirs(args.out_dir, exist_ok=True)
        mp4 = os.path.join(args.out_dir, "scamshield_demo.mp4")
        gif = os.path.join(args.out_dir, "scamshield_demo.gif")
        encode(get_ffmpeg_exe(), webm, mp4, gif)
    print(f"wrote {mp4} ({os.path.getsize(mp4) / 1e6:.1f} MB) and {gif}")


if __name__ == "__main__":
    main()
