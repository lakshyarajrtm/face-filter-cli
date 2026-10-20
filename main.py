import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from deepface import DeepFace
from PIL import Image, ImageOps
from sklearn.metrics.pairwise import cosine_distances

try:
    from tqdm import tqdm
except ImportError:  # progress bar is optional
    def tqdm(it, **kwargs):
        return it

MODEL = "Facenet512"
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def list_images(root: Path, recursive: bool, exclude: Path | None = None) -> list[Path]:
    """Case-insensitive image listing, optionally recursive, skipping an excluded folder."""
    it = root.rglob("*") if recursive else root.glob("*")
    files = []
    for p in it:
        if not p.is_file() or p.suffix.lower() not in EXTENSIONS:
            continue
        if exclude is not None and exclude in p.resolve().parents:
            continue
        files.append(p)
    return sorted(files)


def load_image(path: Path) -> np.ndarray:
    """Load an image, apply EXIF rotation, and return a BGR array (DeepFace's convention)."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return np.array(im)[:, :, ::-1].copy()


def face_area(rep: dict) -> int:
    fa = rep.get("facial_area", {})
    return int(fa.get("w", 0)) * int(fa.get("h", 0))


def represent(path: Path, detector: str) -> list[dict]:
    """Return one representation per detected face. Raises if no face is found."""
    reps = DeepFace.represent(
        img_path=load_image(path),
        model_name=MODEL,
        detector_backend=detector,
        enforce_detection=True,
    )
    return [dict(r) for r in reps] if isinstance(reps, list) else []


def get_target_embedding(path: Path, detector: str) -> np.ndarray | None:
    """Embedding of the LARGEST face in the image (not just the first detected)."""
    try:
        reps = represent(path, detector)
    except Exception as e:
        print(f"[-] No usable face in target {path.name}: {e}")
        return None
    if not reps:
        return None
    main = max(reps, key=face_area)
    return np.asarray(main["embedding"], dtype=np.float32)


def load_target_embeddings(target_path: Path, detector: str) -> list[np.ndarray]:
    if target_path.is_file():
        files = [target_path]
    elif target_path.is_dir():
        files = list_images(target_path, recursive=False)
    else:
        raise ValueError(f"Target path {target_path} is invalid.")
    embs = [get_target_embedding(f, detector) for f in files]
    return [e for e in embs if e is not None]


def unique_destination(output: Path, src: Path) -> Path:
    dest = output / src.name
    n = 1
    while dest.exists():
        dest = output / f"{src.stem}_{n}{src.suffix}"
        n += 1
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy photos containing a target face into an output folder.")
    parser.add_argument("--targets", required=True, help="A single target photo or a folder of target photos.")
    parser.add_argument("--source", required=True, help="Directory of candidate images to scan.")
    parser.add_argument("--output", required=True, help="Destination folder for matched images.")
    parser.add_argument("--threshold", type=float, default=0.30,
                        help="Cosine distance threshold (default 0.30, DeepFace's Facenet512 default).")
    parser.add_argument("--detector", default="retinaface",
                        choices=["opencv", "ssd", "mtcnn", "retinaface", "yunet", "centerface"],
                        help="Face detector backend (opencv is fastest but least accurate).")
    parser.add_argument("--recursive", action="store_true", help="Scan source subfolders too.")
    parser.add_argument("--log", default="face_filter_errors.log", help="File to record images that failed.")
    args = parser.parse_args()

    target_path = Path(args.targets)
    source_path = Path(args.source)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("[+] Encoding target face(s)...")
    targets = load_target_embeddings(target_path, args.detector)
    if not targets:
        print("[-] No valid target faces found. Exiting.")
        sys.exit(1)
    target_matrix = np.vstack(targets)
    print(f"[+] Loaded {len(targets)} target embedding(s).")

    files = list_images(source_path, args.recursive, exclude=output_path.resolve())
    print(f"[+] Scanning {len(files)} candidate images...")

    matched = 0
    skipped_no_face = 0
    failures: list[str] = []

    for img_path in tqdm(files, unit="img"):
        try:
            reps = represent(img_path, args.detector)
        except ValueError:
            # DeepFace raises ValueError when no face is detected
            skipped_no_face += 1
            continue
        except Exception as e:
            failures.append(f"{img_path}: {e}")
            continue

        if not reps:
            skipped_no_face += 1
            continue

        cand_matrix = np.vstack([np.asarray(r["embedding"], dtype=np.float32) for r in reps])
        dists = cosine_distances(target_matrix, cand_matrix)  # (n_targets, n_faces)
        best_dist = float(dists.min())

        if best_dist <= args.threshold:
            dest = unique_destination(output_path, img_path)
            shutil.copy2(img_path, dest)
            matched += 1
            tqdm.write(f"[MATCH] {img_path.name} (best distance {best_dist:.4f}) -> {dest.name}") \
                if hasattr(tqdm, "write") else print(f"[MATCH] {img_path.name} ({best_dist:.4f})")

    if failures:
        Path(args.log).write_text("\n".join(failures), encoding="utf-8")

    print(f"\n[+] Done. Copied {matched} matching image(s) to {output_path}")
    print(f"    No face found in {skipped_no_face} image(s); {len(failures)} failed"
          + (f" (see {args.log})" if failures else ""))


if __name__ == "__main__":
    main()