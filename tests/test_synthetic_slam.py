"""End-to-end: render a short synthetic clip and check SLAM against ground truth."""
import cv2
import numpy as np
import pytest

from organ_slam.evaluate import evaluate_map, evaluate_trajectory
from organ_slam.geometry import inv_T, project
from organ_slam.slam import MonoSLAM
from organ_slam.synthetic import SceneParams, TubeScene, default_intrinsics

SIZE, N = 240, 36


@pytest.fixture(scope="module")
def clip():
    scene = TubeScene(SceneParams(seed=3))
    K = default_intrinsics(SIZE, SIZE)
    poses = scene.camera_path(300)[:N]
    frames = [scene.render(K, T, SIZE, SIZE, rng=np.random.default_rng(i)) for i, T in enumerate(poses)]
    return scene, K, np.array(poses), frames


def test_rendered_depth_matches_scene(clip):
    scene, K, poses, frames = clip
    _, depth = frames[0]
    v, u = np.mgrid[20:SIZE:40, 20:SIZE:40]
    u, v = u.ravel(), v.ravel()
    z = depth[v, u]
    keep = z > 0
    rays = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0], (v + 0.5 - K[1, 2]) / K[1, 1], np.ones(len(u))], 1)
    pts_w = (rays * z[:, None])[keep] @ poses[0][:3, :3].T + poses[0][:3, 3]
    assert np.abs(scene.sdf(pts_w)[0]).max() < 0.1            # back-projected depth lies on the wall
    uv, zc = project(K, inv_T(poses[0]), pts_w)
    assert np.allclose(zc, z[keep], atol=1e-3)


def test_slam_tracks_synthetic_clip(clip):
    scene, K, poses, frames = clip
    slam = MonoSLAM(K)
    for i, (rgb, _) in enumerate(frames):
        slam.process(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), i)
    assert len(slam.keyframes) >= 3
    slam.global_ba()
    traj = slam.trajectory()
    assert len(traj) >= N - 6
    res = evaluate_trajectory(traj, poses)
    assert res["ate_pct"] < 3.0
    assert res["rot_err_mean_deg"] < 2.0
    assert evaluate_map(slam.map_points(), res["sim3"], scene)["map_median"] < 1.5
