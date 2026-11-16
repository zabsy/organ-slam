"""Render a synthetic colonoscopy sequence with ground-truth poses and depth."""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from organ_slam.dataset import write_camera  # noqa: E402
from organ_slam.synthetic import SceneParams, TubeScene, default_intrinsics  # noqa: E402

_scene = None


def _init(params):
    global _scene
    _scene = TubeScene(SceneParams(**params))


def _render(job):
    k, T_wc, K, size, out = job
    img, depth = _scene.render(K, T_wc, size, size, rng=np.random.default_rng(k))
    cv2.imwrite(str(out / "rgb" / f"{k:06d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    np.save(out / "depth" / f"{k:06d}.npy", depth)
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(exist_ok=True)
    params = SceneParams(seed=args.seed)
    scene = TubeScene(params)
    K = default_intrinsics(args.size, args.size)
    poses = scene.camera_path(args.frames)
    write_camera(out, K, args.size, args.size)
    np.savetxt(out / "poses.txt", np.array(poses).reshape(len(poses), 16))
    (out / "scene.json").write_text(json.dumps(scene.params_dict(), indent=2))

    jobs = [(k, T, K, args.size, out) for k, T in enumerate(poses)]
    with Pool(args.workers, initializer=_init, initargs=(scene.params_dict(),)) as pool:
        for k in pool.imap_unordered(_render, jobs):
            if k % 50 == 0:
                print(f"rendered frame {k}", flush=True)
    step = np.linalg.norm(np.diff(np.array(poses)[:, :3, 3], axis=0), axis=1)
    print(f"wrote {len(poses)} frames to {out}; path length {step.sum():.1f} mm, {step.mean():.2f} mm/frame")


if __name__ == "__main__":
    main()
