"""Tests for physgate.viz.encode — numpy frames -> MP4/GIF via ffmpeg.

These run in the pure-logic venv (numpy + system ffmpeg only, no Isaac, no PIL).
The encoder is what turns Isaac Sim camera captures and trajectory animations
into the dynamic recordings embedded in the README.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


def _synthetic_frames(n: int = 12, h: int = 48, w: int = 64) -> list[np.ndarray]:
    """A white square sweeping across a dark background."""
    frames = []
    for i in range(n):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        x = int(i / n * (w - 8))
        frame[20:28, x : x + 8] = 255
        frames.append(frame)
    return frames


def _probe(path) -> dict:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)["streams"][0]


def test_encode_mp4_produces_playable_file_with_all_frames(tmp_path):
    from physgate.viz.encode import encode_frames_to_mp4

    out = tmp_path / "clip.mp4"
    encode_frames_to_mp4(_synthetic_frames(n=12), out, fps=10)

    assert out.exists() and out.stat().st_size > 0
    stream = _probe(out)
    assert int(stream["nb_read_frames"]) == 12
    assert stream["width"] == 64 and stream["height"] == 48


def test_encode_gif_produces_animated_file(tmp_path):
    from physgate.viz.encode import encode_frames_to_gif

    out = tmp_path / "clip.gif"
    encode_frames_to_gif(_synthetic_frames(n=12), out, fps=10)

    assert out.exists() and out.stat().st_size > 0
    stream = _probe(out)
    assert int(stream["nb_read_frames"]) == 12


def test_encode_pads_odd_dimensions(tmp_path):
    """H.264 requires even dimensions — odd-sized frames must still encode."""
    from physgate.viz.encode import encode_frames_to_mp4

    frames = [np.zeros((47, 63, 3), dtype=np.uint8) for _ in range(4)]
    out = tmp_path / "odd.mp4"
    encode_frames_to_mp4(frames, out, fps=5)

    stream = _probe(out)
    assert stream["width"] % 2 == 0 and stream["height"] % 2 == 0


def test_encode_rejects_empty_frame_list(tmp_path):
    from physgate.viz.encode import encode_frames_to_mp4

    with pytest.raises(ValueError, match="no frames"):
        encode_frames_to_mp4([], tmp_path / "empty.mp4", fps=10)


def test_encode_rejects_mismatched_frame_shapes(tmp_path):
    from physgate.viz.encode import encode_frames_to_gif

    frames = [
        np.zeros((48, 64, 3), dtype=np.uint8),
        np.zeros((32, 32, 3), dtype=np.uint8),
    ]
    with pytest.raises(ValueError, match="same shape"):
        encode_frames_to_gif(frames, tmp_path / "bad.gif", fps=10)


def test_encode_accepts_rgba_frames(tmp_path):
    """Isaac camera output is RGBA — the alpha channel must be dropped cleanly."""
    from physgate.viz.encode import encode_frames_to_mp4

    frames = [np.zeros((48, 64, 4), dtype=np.uint8) for _ in range(4)]
    out = tmp_path / "rgba.mp4"
    encode_frames_to_mp4(frames, out, fps=5)

    stream = _probe(out)
    assert int(stream["nb_read_frames"]) == 4
