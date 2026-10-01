"""Record the README demo by driving the web UI in a real browser.

Start a server with an empty analysis log first, so the dashboard shows only
what the demo does:

    SCAMSHIELD_MODEL=tfidf-lr uvicorn src.main:app --port 8000
    uv run --no-project --with playwright --with imageio-ffmpeg \\
        python -m scripts.record_demo --url http://127.0.0.1:8000

Writes ``docs/demo/scamshield_demo.mp4`` (full quality, attached to releases)
and ``docs/demo/scamshield_demo.gif`` (for the README, kept under 10 MB).
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


def drive(page, url: str) -> None:
    def pause(seconds: float) -> None:
        page.wait_for_timeout(int(seconds * 1000))

    def analyze() -> None:
        page.click("#analyze-submit-btn")
        page.wait_for_selector("#result-details:not(.hide)")
        pause(1.0)
        page.locator("#explanation-box").scroll_into_view_if_needed()
        pause(4.0)
        page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
        pause(1.0)

    def example(label: str) -> None:
        page.get_by_role("button", name=label, exact=True).click()
        pause(1.2)
        analyze()

    page.goto(url)
    pause(2.0)
    page.click("#message-input")
    page.keyboard.type(ENGLISH_SCAM, delay=28)
    pause(0.8)
    analyze()
    example("Legit Bank SMS")
    example("Tanglish Scam")
    example("Hindi Scam")
    page.click("#tab-stats-btn")
    pause(5.0)


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
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as video_dir, sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        size = {"width": 1280, "height": 800}
        context = browser.new_context(viewport=size, record_video_dir=video_dir, record_video_size=size)
        drive(context.new_page(), args.url)
        context.close()
        browser.close()
        webm = glob.glob(os.path.join(video_dir, "*.webm"))[0]
        os.makedirs(args.out_dir, exist_ok=True)
        mp4 = os.path.join(args.out_dir, "scamshield_demo.mp4")
        gif = os.path.join(args.out_dir, "scamshield_demo.gif")
        encode(get_ffmpeg_exe(), webm, mp4, gif)
    print(f"wrote {mp4} ({os.path.getsize(mp4) / 1e6:.1f} MB) and {gif}")


if __name__ == "__main__":
    main()
