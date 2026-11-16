"""Run monocular SLAM on a sequence and report accuracy if ground truth exists."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from organ_slam.dataset import Sequence  # noqa: E402
from organ_slam.evaluate import evaluate_map, evaluate_trajectory  # noqa: E402
from organ_slam.slam import MonoSLAM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="data/synthetic")
    ap.add_argument("--out", default="out/synthetic")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--global-ba", action="store_true", help="final bundle adjustment over all keyframes")
    args = ap.parse_args()

    seq = Sequence(args.seq)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    slam = MonoSLAM(seq.K)
    n = len(seq) if args.max_frames is None else min(len(seq), args.max_frames)
    t0 = time.time()
    for i in range(n):
        slam.process(seq.image(i), i)
        if i % 50 == 0:
            print(f"frame {i}: {len(slam.keyframes)} keyframes, {int(slam.point_ok.sum())} points", flush=True)
    t_track = time.time() - t0
    if not slam.keyframes:
        sys.exit("SLAM never initialised")
    if args.global_ba:
        slam.global_ba()
    traj = slam.trajectory()
    pts = slam.map_points()
    print(f"tracked {len(traj)}/{n} frames, {len(slam.keyframes)} keyframes, {len(pts)} map points, "
          f"{n / t_track:.1f} fps, lost frames: {slam.stats.lost_frames}")

    ids = np.array(sorted(traj))
    np.savetxt(out / "trajectory.txt", np.hstack([ids[:, None], np.array([traj[i] for i in ids]).reshape(len(ids), 16)]))
    np.savetxt(out / "map_points.txt", pts)

    if seq.poses_wc is None:
        return
    res = evaluate_trajectory(traj, seq.poses_wc[:n])
    metrics = {k: v for k, v in res.items() if isinstance(v, (int, float))}
    scene = None
    if seq.scene_params is not None:
        from organ_slam.synthetic import SceneParams, TubeScene
        scene = TubeScene(SceneParams(**seq.scene_params))
        mres = evaluate_map(pts, res["sim3"], scene)
        metrics.update({k: v for k, v in mres.items() if isinstance(v, (int, float))})
        res["map_aligned"] = mres["aligned"]
    metrics["fps"] = n / t_track
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    np.savez(out / "eval.npz", frame_ids=res["frame_ids"], aligned=res["aligned"], gt=res["gt"], err=res["err"],
             map_aligned=res.get("map_aligned", np.zeros((0, 3))))


if __name__ == "__main__":
    main()
