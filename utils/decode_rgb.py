"""Decode HOI4D RGB videos (align_rgb/image.mp4) to JPEG frames.

Usage:
    python scripts/decode_rgb.py --root /path/to/HOI4D/HOI4D_release
    python scripts/decode_rgb.py --root /path/to/HOI4D/HOI4D_release --workers 8 --fps 15
    python scripts/decode_rgb.py --root /path/to/HOI4D/HOI4D_release --release release.txt

The script is idempotent: sequences that already contain .jpg frames are skipped.
Decoded frames are written in-place next to image.mp4 as 00000.jpg, 00001.jpg, ...

Environment setup:
    uvactivate hoi4d
    uv pip install imageio-ffmpeg tqdm
"""

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import imageio_ffmpeg
from tqdm import tqdm

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def find_rgb_videos(root: Path, release_txt: Path | None) -> list[Path]:
    """Return sorted list of align_rgb/image.mp4 paths to decode.

    Args:
        root: Root of HOI4D_release directory.
        release_txt: Optional path to release.txt for official sequence list.
                     If None or missing, auto-discovers all image.mp4 files.

    Returns:
        List of Path objects pointing to image.mp4 files.
    """
    if release_txt and release_txt.exists():
        with open(release_txt) as f:
            seqs = [s.strip() for s in f if s.strip()]
        videos = [root / seq / "align_rgb" / "image.mp4" for seq in seqs]
        missing = [v for v in videos if not v.exists()]
        if missing:
            print(f"[warn] {len(missing)} sequences from release.txt not found on disk — skipping.")
        videos = [v for v in videos if v.exists()]
    else:
        videos = sorted(root.rglob("align_rgb/image.mp4"))
    return videos


def decode_one(mp4: Path, fps: int) -> tuple[Path, str]:
    """Decode a single mp4 to JPEG frames at `fps` frames per second.

    Args:
        mp4: Path to image.mp4.
        fps: Target frames per second (default 15).

    Returns:
        Tuple of (mp4 path, status string: "ok" | "skipped" | error message).

    Example (toy):
        A 10-second clip at fps=15 produces 00000.jpg … 00149.jpg in the same folder.
    """
    out_dir = mp4.parent

    # Skip if frames already decoded
    if any(out_dir.glob("*.jpg")):
        return mp4, "skipped"

    cmd = [
        FFMPEG,
        "-i", str(mp4),
        "-f", "image2",
        "-start_number", "0",
        "-vf", f"fps=fps={fps}",
        "-qscale:v", "2",
        str(out_dir / "%05d.jpg"),
        "-loglevel", "error",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return mp4, f"ERROR: {result.stderr.strip()}"
    return mp4, "ok"


def decode_all(root: Path, release_txt: Path | None, fps: int, workers: int) -> None:
    """Decode all HOI4D RGB videos under root using parallel workers.

    Args:
        root: Root of HOI4D_release directory.
        release_txt: Optional path to release.txt.
        fps: Frames per second for decoding.
        workers: Number of parallel ffmpeg processes.
    """
    videos = find_rgb_videos(root, release_txt)
    print(f"Found {len(videos)} sequences to process (workers={workers}, fps={fps}).")

    ok, skipped, errors = 0, 0, []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(decode_one, v, fps): v for v in videos}
        with tqdm(total=len(videos), unit="seq") as pbar:
            for future in as_completed(futures):
                mp4, status = future.result()
                if status == "ok":
                    ok += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    errors.append((mp4, status))
                pbar.set_postfix(ok=ok, skip=skipped, err=len(errors))
                pbar.update(1)

    print(f"\nDone. decoded={ok}  skipped={skipped}  errors={len(errors)}")
    if errors:
        print("\nFailed sequences:")
        for mp4, msg in errors:
            print(f"  {mp4}: {msg}")


def main() -> None:
    """Parse args and run decoding."""
    parser = argparse.ArgumentParser(
        description="Decode HOI4D RGB videos to JPEG frames."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/Project_11_FunctionalGrasp/hoang.le/datasets/HOI4D/HOI4D_release"),
        help="Path to HOI4D_release directory (default: %(default)s).",
    )
    parser.add_argument(
        "--release",
        type=Path,
        default=None,
        help="Path to release.txt for official sequence list. "
             "If omitted, auto-discovers all image.mp4 files under --root.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Output frames per second (default: %(default)s).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel ffmpeg processes (default: %(default)s).",
    )
    args = parser.parse_args()

    if not args.root.exists():
        raise SystemExit(f"Root directory not found: {args.root}")

    decode_all(args.root, args.release, args.fps, args.workers)


if __name__ == "__main__":
    main()
