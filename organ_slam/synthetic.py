"""Synthetic colon-like scene: a curved, folded, textured tube rendered by
sphere tracing with a light at the camera (like an endoscope).

Units are millimetres. Camera frame: x right, y down, z forward.
"""
from dataclasses import dataclass, asdict

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from .geometry import make_T


@dataclass
class SceneParams:
    seed: int = 0
    length: float = 260.0       # centreline length
    radius: float = 15.0        # mean lumen radius
    fold_depth: float = 0.18    # haustral fold depth (fraction of radius)
    fold_period: float = 22.0
    fold_width: float = 2.5
    lobe: float = 0.06          # 3-lobed cross-section amplitude
    bend_x: float = 28.0        # centreline lateral excursions
    bend_y: float = 18.0


class TubeScene:
    DS = 0.25        # centreline sample spacing
    TEX_RES = 0.1    # texture resolution (mm per texel)

    def __init__(self, params: SceneParams = SceneParams()):
        self.p = params
        self._build_centerline()
        self._build_texture()

    # ---- geometry -------------------------------------------------------
    def _build_centerline(self):
        p = self.p
        u = np.linspace(0.0, 1.0, 20001)
        raw = np.stack([p.bend_x * np.sin(2.2 * np.pi * u),
                        p.bend_y * (1.0 - np.cos(1.4 * np.pi * u)),
                        p.length * 0.93 * u], axis=1)
        seg = np.linalg.norm(np.diff(raw, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        self.S = np.arange(0.0, arc[-1], self.DS)
        self.C = np.stack([np.interp(self.S, arc, raw[:, i]) for i in range(3)], axis=1)
        T = np.gradient(self.C, axis=0)
        self.T = T / np.linalg.norm(T, axis=1, keepdims=True)
        # Parallel-transport (twist-free) frame along the centreline.
        N = np.zeros_like(self.T)
        n = np.array([1.0, 0.0, 0.0])
        for i, t in enumerate(self.T):
            n = n - np.dot(n, t) * t
            n /= np.linalg.norm(n)
            N[i] = n
        self.N = N
        self.B = np.cross(self.T, self.N)
        self.tree = cKDTree(self.C)
        self.total_length = float(self.S[-1])

    def radius_at(self, s, theta):
        p = self.p
        phase = np.mod(s, p.fold_period) - 0.5 * p.fold_period
        fold = 1.0 - p.fold_depth * np.exp(-(phase / p.fold_width) ** 2)
        lobes = 1.0 + p.lobe * np.cos(3.0 * theta + 0.02 * s)
        return p.radius * fold * lobes

    def tube_coords(self, pts):
        """Return (s, theta, rho) tube coordinates of Nx3 points."""
        _, idx = self.tree.query(pts, workers=-1)
        v = pts - self.C[idx]
        ax = np.sum(v * self.T[idx], axis=1)
        rad = v - ax[:, None] * self.T[idx]
        rho = np.linalg.norm(rad, axis=1)
        theta = np.arctan2(np.sum(rad * self.B[idx], axis=1), np.sum(rad * self.N[idx], axis=1))
        return self.S[idx] + ax, theta, rho

    def sdf(self, pts):
        """Signed distance-like value: positive inside the lumen, 0 on the wall."""
        s, theta, rho = self.tube_coords(pts)
        return self.radius_at(s, theta) - rho, s, theta

    def frame_at(self, s):
        i = int(np.clip(round(s / self.DS), 0, len(self.S) - 1))
        return self.C[i], self.T[i], self.N[i], self.B[i]

    # ---- appearance -----------------------------------------------------
    def _build_texture(self):
        rng = np.random.default_rng(self.p.seed)
        ns = int(self.total_length / self.TEX_RES)
        nt = int(2 * np.pi * self.p.radius / self.TEX_RES)

        def noise(sigma):
            g = gaussian_filter(rng.standard_normal((ns, nt)), sigma, mode=("reflect", "wrap"))
            return g / g.std()

        base = 0.55 * noise(40) + 0.35 * noise(12) + 0.25 * noise(4) + 0.18 * noise(1.5)
        # Thin dark vessel-like curves drawn as random walks.
        vessels = np.zeros((ns, nt))
        for _ in range(int(ns * nt / 9000)):
            pos = np.array([rng.uniform(0, ns), rng.uniform(0, nt)])
            ang = rng.uniform(0, 2 * np.pi)
            for _ in range(int(rng.uniform(150, 700))):
                ang += rng.normal(0, 0.12)
                pos += [np.cos(ang), np.sin(ang)]
                if not 0 <= pos[0] < ns:
                    break
                vessels[int(pos[0]), int(pos[1]) % nt] = 1.0
        vessels = gaussian_filter(vessels, 1.0, mode=("reflect", "wrap"))
        vessels /= vessels.max()
        lum = np.clip(0.62 + 0.13 * base - 0.45 * vessels, 0.08, 1.0)
        tint = np.array([1.0, 0.62, 0.55])                      # pinkish mucosa (RGB)
        vessel_tint = np.array([0.0, 0.10, 0.06])
        self.tex = (lum[..., None] * tint - vessels[..., None] * vessel_tint).clip(0, 1).astype(np.float32)

    def albedo(self, s, theta):
        ns, nt, _ = self.tex.shape
        fs = np.clip(s / self.TEX_RES, 0, ns - 1.001)
        ft = np.mod(theta, 2 * np.pi) / (2 * np.pi) * nt
        i0, j0 = np.floor(fs).astype(int), np.floor(ft).astype(int)
        a, b = (fs - i0)[:, None], (ft - j0)[:, None]
        j0, j1 = j0 % nt, (j0 + 1) % nt
        return ((1 - a) * (1 - b) * self.tex[i0, j0] + (1 - a) * b * self.tex[i0, j1]
                + a * (1 - b) * self.tex[i0 + 1, j0] + a * b * self.tex[i0 + 1, j1])

    # ---- rendering ------------------------------------------------------
    def render(self, K, T_wc, width, height, max_dist=140.0, noise_std=0.008, rng=None):
        """Render one view. Returns (rgb uint8 HxWx3, depth float32 HxW in mm; 0 = no hit)."""
        u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
        dirs_c = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1).reshape(-1, 3)
        zscale = 1.0 / np.linalg.norm(dirs_c, axis=1)           # ray length -> z depth
        dirs = (dirs_c * zscale[:, None]) @ T_wc[:3, :3].T
        origin = T_wc[:3, 3]

        n = len(dirs)
        t = np.zeros(n)
        hit = np.zeros(n, bool)
        active = np.ones(n, bool)
        for _ in range(96):
            ids = np.flatnonzero(active)
            if len(ids) == 0:
                break
            d, s, _ = self.sdf(origin + t[ids, None] * dirs[ids])
            done = d < 0.02
            hit[ids[done]] = True
            t[ids] += np.where(done, 0.0, np.maximum(0.5 * d, 0.01))
            gone = (t[ids] > max_dist) | (s < 0.0) | (s > self.total_length)
            active[ids[done | gone]] = False

        rgb = np.zeros((n, 3), np.float32)
        depth = np.zeros(n, np.float32)
        ids = np.flatnonzero(hit)
        if len(ids):
            pts = origin + t[ids, None] * dirs[ids]
            _, s, theta = self.sdf(pts)
            eps = 0.05
            grad = np.stack([(self.sdf(pts + eps * e)[0] - self.sdf(pts - eps * e)[0]) for e in np.eye(3)], axis=1)
            normal = grad / (np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12)   # points into the lumen
            cosang = np.clip(-np.sum(normal * dirs[ids], axis=1), 0.0, 1.0)
            shade = cosang * (14.0 / np.maximum(t[ids], 1.0)) ** 2
            col = self.albedo(s, theta) * shade[:, None]
            rgb[ids] = np.clip(col, 0, 1) ** (1 / 2.2)
            depth[ids] = (t[ids] * zscale[ids]).astype(np.float32)
        if noise_std > 0:
            rng = rng or np.random.default_rng(0)
            rgb = rgb + rng.normal(0, noise_std, rgb.shape).astype(np.float32)
        img = (np.clip(rgb, 0, 1) * 255).astype(np.uint8).reshape(height, width, 3)
        return img, depth.reshape(height, width)

    # ---- camera path ----------------------------------------------------
    def camera_path(self, n_frames, s_start=12.0, s_end=None, lookahead=22.0):
        """Smooth hand-held-like path along the lumen. Returns list of T_wc (camera -> world)."""
        s_end = self.total_length - 70.0 if s_end is None else s_end
        poses = []
        for k, s in enumerate(np.linspace(s_start, s_end, n_frames)):
            c, t, nrm, b = self.frame_at(s)
            ph = 2 * np.pi * k / n_frames
            off = self.p.radius * (0.28 * np.sin(3.1 * ph) * nrm + 0.22 * np.cos(2.3 * ph + 0.7) * b)
            pos = c + off
            c2, _, n2, b2 = self.frame_at(s + lookahead)
            target = c2 + self.p.radius * (0.25 * np.sin(4.3 * ph + 1.0) * n2 + 0.25 * np.cos(3.7 * ph) * b2)
            z = target - pos
            z /= np.linalg.norm(z)
            roll = 0.35 * np.sin(1.7 * ph)
            up = np.cos(roll) * nrm + np.sin(roll) * b
            x = np.cross(up, z)
            x /= np.linalg.norm(x)
            y = np.cross(z, x)
            poses.append(make_T(np.stack([x, y, z], axis=1), pos))
        return poses

    def params_dict(self):
        return asdict(self.p)


def default_intrinsics(width=320, height=320, hfov_deg=100.0):
    f = 0.5 * width / np.tan(np.radians(hfov_deg) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])
