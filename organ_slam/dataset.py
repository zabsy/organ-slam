"""Sequence on disk:

    seq/
      camera.json        {"fx","fy","cx","cy","width","height"}
      rgb/000000.png ...
      depth/000000.npy   (optional, float32 z-depth)
      poses.txt          (optional, one row-major 4x4 camera-to-world per line)
      scene.json         (optional, synthetic scene parameters)
"""
import json
from pathlib import Path

import cv2
import numpy as np


class Sequence:
    def __init__(self, root):
        self.root = Path(root)
        cam = json.loads((self.root / "camera.json").read_text())
        self.width, self.height = cam["width"], cam["height"]
        self.K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1.0]])
        self.rgb_paths = sorted((self.root / "rgb").glob("*.png"))
        pose_file = self.root / "poses.txt"
        self.poses_wc = np.loadtxt(pose_file).reshape(-1, 4, 4) if pose_file.exists() else None
        scene_file = self.root / "scene.json"
        self.scene_params = json.loads(scene_file.read_text()) if scene_file.exists() else None

    def __len__(self):
        return len(self.rgb_paths)

    def image(self, i):
        """BGR uint8 image."""
        return cv2.imread(str(self.rgb_paths[i]), cv2.IMREAD_COLOR)

    def depth(self, i):
        p = self.root / "depth" / (self.rgb_paths[i].stem + ".npy")
        return np.load(p) if p.exists() else None


def write_camera(root, K, width, height):
    Path(root).mkdir(parents=True, exist_ok=True)
    cam = dict(fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]),
               width=int(width), height=int(height))
    (Path(root) / "camera.json").write_text(json.dumps(cam, indent=2))
