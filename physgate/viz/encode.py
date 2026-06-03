"""Encode numpy RGB frames into MP4/GIF via ffmpeg (rawvideo stdin pipe).

Pure logic: numpy + the system ffmpeg binary. No PIL, no Isaac — so the
encoder is unit-testable in the pure-logic venv and reusable by both the
Isaac camera recorder and the matplotlib trajectory animator.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

#: GIF palette size — 128 colors keeps README GIFs small with no visible loss
#: on synthetic scenes (flat-shaded boxes + robot).
_GIF_MAX_COLORS = 128


def _validate_frames(frames: list[np.ndarray]) -> tuple[int, int]:
    """Check the frame list is encodable; return (height, width)."""
    if not frames:
        raise ValueError("no frames to encode")
    first = frames[0]
    if first.ndim != 3 or first.shape[2] not in (3, 4):
        raise ValueError(f"frames must be (H, W, 3|4) uint8 arrays, got {first.shape}")
    if any(f.shape != first.shape for f in frames):
        raise ValueError("all frames must have the same shape")
    return int(first.shape[0]), int(first.shape[1])


def _to_rgb24_bytes(frames: list[np.ndarray]) -> bytes:
    """Concatenate frames as raw rgb24 bytes (alpha dropped if present)."""
    return b"".join(np.ascontiguousarray(f[:, :, :3], dtype=np.uint8).tobytes() for f in frames)


def _require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError("ffmpeg not found on PATH (required to encode recordings)")
    return path


def _run_ffmpeg(args: list[str], raw_frames: bytes, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [_require_ffmpeg(), "-y", *args, str(output)],
        input=raw_frames,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}): {result.stderr.decode(errors='replace')[-2000:]}"
        )


def _rawvideo_input_args(height: int, width: int, fps: int) -> list[str]:
    return [
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
    ]


def encode_frames_to_mp4(frames: list[np.ndarray], output: str | Path, fps: int = 20) -> Path:
    """Encode (H, W, 3|4) uint8 frames into an H.264 MP4.

    Odd frame dimensions are padded to even (H.264 requirement).
    """
    height, width = _validate_frames(frames)
    output = Path(output)
    _run_ffmpeg(
        [
            *_rawvideo_input_args(height, width, fps),
            # pad to even dimensions, keep pixel-perfect otherwise
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            "-movflags",
            "+faststart",
        ],
        _to_rgb24_bytes(frames),
        output,
    )
    return output


def encode_frames_to_gif(frames: list[np.ndarray], output: str | Path, fps: int = 12) -> Path:
    """Encode (H, W, 3|4) uint8 frames into a palette-optimized animated GIF.

    GIFs autoplay in GitHub READMEs (MP4s do not) — this is the format used for
    the embedded "watch it run" recordings.
    """
    height, width = _validate_frames(frames)
    output = Path(output)
    _run_ffmpeg(
        [
            *_rawvideo_input_args(height, width, fps),
            # two-pass palette in one filtergraph: generate then apply
            "-vf",
            (
                f"split[a][b];[a]palettegen=max_colors={_GIF_MAX_COLORS}[p];"
                "[b][p]paletteuse=dither=bayer:bayer_scale=3"
            ),
            "-loop",
            "0",
        ],
        _to_rgb24_bytes(frames),
        output,
    )
    return output
