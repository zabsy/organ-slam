"""Monocular feature-based SLAM: two-view initialisation, PnP tracking,
keyframe triangulation and windowed bundle adjustment.

World scale is arbitrary (monocular); the first two keyframes are one unit apart.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from .ba import bundle_adjust
from .geometry import (epipolar_dist, fundamental_from_poses, inv_T, make_T, parallax_deg,
                       project, triangulate)


@dataclass
class Config:
    n_features: int = 2000
    ratio: float = 0.8              # Lowe ratio test
    init_min_shift_px: float = 12.0  # median feature shift needed to initialise
    init_min_points: int = 80
    min_parallax_deg: float = 1.0
    max_reproj_px: float = 2.0
    pnp_reproj_px: float = 3.0
    min_track_inliers: int = 25
    kf_inlier_ratio: float = 0.6    # new keyframe when tracking falls below this share
    kf_min_baseline: float = 0.03   # baseline / median depth
    kf_max_baseline: float = 0.12
    track_keyframes: int = 3        # recent keyframes matched against when tracking
    ba_window: int = 8
    max_depth_ratio: float = 8.0    # reject points far beyond the typical scene depth
    dark_thresh: int = 12           # ignore features in near-black regions


@dataclass
class KeyFrame:
    frame_id: int
    T_cw: np.ndarray
    kps: np.ndarray                 # (N,2) pixels
    desc: np.ndarray                # (N,128)
    mp_ids: np.ndarray              # (N,) map point id or -1
    n_tracked: int = 0


@dataclass
class FramePose:
    frame_id: int
    kf_id: int                      # reference keyframe
    T_c_kf: np.ndarray              # pose relative to it, so later BA moves the frame too


@dataclass
class Stats:
    lost_frames: list = field(default_factory=list)
    inliers: list = field(default_factory=list)


class MonoSLAM:
    def __init__(self, K, cfg: Config = Config()):
        self.K = np.asarray(K, float)
        self.cfg = cfg
        self.sift = cv2.SIFT_create(nfeatures=cfg.n_features, contrastThreshold=0.02)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.keyframes: list[KeyFrame] = []
        self.points = np.zeros((0, 3))
        self.point_ok = np.zeros(0, bool)
        self.frames: list[FramePose] = []
        self.stats = Stats()
        self._init_ref = None       # (frame_id, kps, desc) awaiting a second view
        self._pending = []          # frames seen between the reference and initialisation
        self._last_T_cw = None

    # ---- features -------------------------------------------------------
    def _extract(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        mask = (cv2.erode((gray > self.cfg.dark_thresh).astype(np.uint8), np.ones((5, 5), np.uint8)) * 255)
        kp, desc = self.sift.detectAndCompute(gray, mask)
        if desc is None:
            return np.zeros((0, 2)), np.zeros((0, 128), np.float32)
        pts = np.array([k.pt for k in kp], float)
        # SIFT reports one keypoint per dominant orientation; keep one per location.
        _, uniq = np.unique(np.round(pts, 2), axis=0, return_index=True)
        uniq.sort()
        desc = desc[uniq]
        desc = np.sqrt(desc / (desc.sum(1, keepdims=True) + 1e-9))      # RootSIFT
        return pts[uniq], desc.astype(np.float32)

    def _match(self, d1, d2):
        """Ratio test + mutual consistency. Returns (M,2) index pairs."""
        if len(d1) < 2 or len(d2) < 2:
            return np.zeros((0, 2), int)
        fwd = self.matcher.knnMatch(d1, d2, k=2)
        back = self.matcher.match(d2, d1)
        back_of = {m.queryIdx: m.trainIdx for m in back}
        out = [(a.queryIdx, a.trainIdx) for a, b in fwd
               if a.distance < self.cfg.ratio * b.distance and back_of.get(a.trainIdx) == a.queryIdx]
        return np.array(out, int).reshape(-1, 2)

    # ---- map bookkeeping ------------------------------------------------
    def _add_points(self, X):
        start = len(self.points)
        self.points = np.vstack([self.points, X])
        self.point_ok = np.concatenate([self.point_ok, np.ones(len(X), bool)])
        return np.arange(start, start + len(X))

    def _good_triangulation(self, T1, T2, x1, x2, X):
        uv1, z1 = project(self.K, T1, X)
        uv2, z2 = project(self.K, T2, X)
        c = self.cfg
        near = z2 < self.cfg.max_depth_ratio * np.median(z2[z2 > 0]) if np.any(z2 > 0) else z2 > 0
        return ((z1 > 0) & (z2 > 0) & near
                & (np.linalg.norm(uv1 - x1, axis=1) < c.max_reproj_px)
                & (np.linalg.norm(uv2 - x2, axis=1) < c.max_reproj_px)
                & (parallax_deg(T1, T2, X) > c.min_parallax_deg))

    # ---- initialisation -------------------------------------------------
    def _try_initialise(self, frame_id, kps, desc):
        c = self.cfg
        if self._init_ref is None:
            self._init_ref = (frame_id, kps, desc)
            self._pending = []
            return None
        ref_id, kps0, desc0 = self._init_ref
        m = self._match(desc0, desc)
        if len(m) < c.init_min_points:
            self._init_ref = (frame_id, kps, desc)
            self._pending = []
            return None
        self._pending.append((frame_id, kps, desc))
        x0, x1 = kps0[m[:, 0]], kps[m[:, 1]]
        if np.median(np.linalg.norm(x1 - x0, axis=1)) < c.init_min_shift_px:
            return None
        E, inl = cv2.findEssentialMat(x0, x1, self.K, cv2.RANSAC, 0.999, 1.0)
        if E is None or E.shape != (3, 3):
            return None
        _, R, t, inl = cv2.recoverPose(E, x0, x1, self.K, mask=inl)
        inl = inl.ravel() > 0
        T0, T1 = np.eye(4), make_T(R, t)
        X = triangulate(self.K, T0, T1, x0, x1)
        good = inl & self._good_triangulation(T0, T1, x0, x1, X)
        if good.sum() < c.init_min_points:
            return None

        ids = self._add_points(X[good])
        kf0 = KeyFrame(ref_id, T0, kps0, desc0, -np.ones(len(kps0), int))
        kf1 = KeyFrame(frame_id, T1, kps, desc, -np.ones(len(kps), int))
        kf0.mp_ids[m[good, 0]] = ids
        kf1.mp_ids[m[good, 1]] = ids
        kf0.n_tracked = kf1.n_tracked = int(good.sum())
        self.keyframes = [kf0, kf1]
        self._local_ba(fix_scale=True)
        self.frames.append(FramePose(ref_id, 0, np.eye(4)))
        for fid, k, d in self._pending[:-1]:            # back-fill frames skipped while waiting
            T, _ = self._solve_pose(k, self._associate(k, d, [1, 0]))
            if T is not None:
                self.frames.append(FramePose(fid, 0, T @ inv_T(self.keyframes[0].T_cw)))
        self.frames.append(FramePose(frame_id, 1, np.eye(4)))
        self._init_ref = None
        self._pending = []
        self._last_T_cw = self.keyframes[1].T_cw
        return self._last_T_cw

    # ---- tracking -------------------------------------------------------
    def _associate(self, kps, desc, kf_ids):
        """Match a frame against keyframes; return {frame kp index: map point id}."""
        assoc = {}
        for i in kf_ids:
            kf = self.keyframes[i]
            have = np.flatnonzero(kf.mp_ids >= 0)
            have = have[self.point_ok[kf.mp_ids[have]]]
            if len(have) < 8:
                continue
            for a, b in self._match(desc, kf.desc[have]):
                assoc.setdefault(int(a), int(kf.mp_ids[have[b]]))
        return assoc

    def _solve_pose(self, kps, assoc):
        c = self.cfg
        if len(assoc) < c.min_track_inliers:
            return None, None
        idx = np.array(list(assoc.keys()))
        pid = np.array(list(assoc.values()))
        ok, rvec, tvec, inl = cv2.solvePnPRansac(self.points[pid], kps[idx], self.K, None,
                                                 reprojectionError=c.pnp_reproj_px, iterationsCount=300,
                                                 confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < c.min_track_inliers:
            return None, None
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(self.points[pid[inl]], kps[idx[inl]], self.K, None, rvec, tvec)
        T = make_T(cv2.Rodrigues(rvec)[0], tvec)
        uv, z = project(self.K, T, self.points[pid])
        good = (z > 0) & (np.linalg.norm(uv - kps[idx], axis=1) < c.pnp_reproj_px)
        if good.sum() < c.min_track_inliers:
            return None, None
        return T, (idx[good], pid[good])

    def process(self, img, frame_id):
        """Track one frame. Returns T_cw, or None if not initialised / lost."""
        c = self.cfg
        kps, desc = self._extract(img)
        if not self.keyframes:
            return self._try_initialise(frame_id, kps, desc)

        n_kf = len(self.keyframes)
        recent = list(range(n_kf - 1, max(-1, n_kf - 1 - c.track_keyframes), -1))
        assoc = self._associate(kps, desc, recent)
        T, inl = self._solve_pose(kps, assoc)
        if T is None:                                   # relocalise against a wider set
            wider = list(range(n_kf - 1, max(-1, n_kf - 13), -1))
            T, inl = self._solve_pose(kps, self._associate(kps, desc, wider))
        if T is None:
            self.stats.lost_frames.append(frame_id)
            return None
        idx, pid = inl
        self.stats.inliers.append(len(idx))

        ref = self.keyframes[-1]
        depth = project(self.K, T, self.points[pid])[1]
        baseline = np.linalg.norm(inv_T(T)[:3, 3] - inv_T(ref.T_cw)[:3, 3]) / max(np.median(depth), 1e-9)
        weak = len(idx) < max(60, c.kf_inlier_ratio * ref.n_tracked)
        if baseline > c.kf_min_baseline and (weak or baseline > c.kf_max_baseline):
            T = self._insert_keyframe(frame_id, T, kps, desc, idx, pid)
            self.frames.append(FramePose(frame_id, len(self.keyframes) - 1, np.eye(4)))
        else:
            self.frames.append(FramePose(frame_id, n_kf - 1, T @ inv_T(ref.T_cw)))
        self._last_T_cw = T
        return T

    # ---- mapping --------------------------------------------------------
    def _insert_keyframe(self, frame_id, T, kps, desc, idx, pid):
        kf = KeyFrame(frame_id, T, kps, desc, -np.ones(len(kps), int))
        kf.mp_ids[idx] = pid
        new_id = len(self.keyframes)
        self.keyframes.append(kf)
        for j in range(new_id - 1, max(-1, new_id - 4), -1):      # triangulate against recent keyframes
            other = self.keyframes[j]
            free_new = np.flatnonzero(kf.mp_ids < 0)
            free_old = np.flatnonzero(other.mp_ids < 0)
            m = self._match(desc[free_new], other.desc[free_old])
            if len(m) == 0:
                continue
            a, b = free_new[m[:, 0]], free_old[m[:, 1]]
            x_new, x_old = kps[a], other.kps[b]
            F = fundamental_from_poses(self.K, other.T_cw, T)
            keep = epipolar_dist(F, x_old, x_new) < self.cfg.max_reproj_px
            a, b, x_new, x_old = a[keep], b[keep], x_new[keep], x_old[keep]
            if len(a) == 0:
                continue
            X = triangulate(self.K, other.T_cw, T, x_old, x_new)
            good = self._good_triangulation(other.T_cw, T, x_old, x_new, X)
            ids = self._add_points(X[good])
            kf.mp_ids[a[good]] = ids
            other.mp_ids[b[good]] = ids
        self._local_ba()
        kf.n_tracked = int(np.sum(kf.mp_ids >= 0))
        return kf.T_cw

    def _observations(self, kf_ids):
        cam, pt, obs, kp_index = [], [], [], []
        for local, i in enumerate(kf_ids):
            kf = self.keyframes[i]
            have = np.flatnonzero(kf.mp_ids >= 0)
            have = have[self.point_ok[kf.mp_ids[have]]]
            cam.append(np.full(len(have), local))
            pt.append(kf.mp_ids[have])
            obs.append(kf.kps[have])
            kp_index.append(have)
        return np.concatenate(cam), np.concatenate(pt), np.vstack(obs), np.concatenate(kp_index)

    def _run_ba(self, kf_ids, fixed, iters):
        cam, pt, obs, kp_index = self._observations(kf_ids)
        used, pt_local = np.unique(pt, return_inverse=True)
        poses = np.array([self.keyframes[i].T_cw for i in kf_ids])
        poses, X, err = bundle_adjust(self.K, poses, self.points[used], cam, pt_local, obs, fixed, max_iters=iters)
        for local, i in enumerate(kf_ids):
            self.keyframes[i].T_cw = poses[local]
        self.points[used] = X
        # Drop observations that still disagree, then points left with fewer than two views.
        bad = err > 2.0 * self.cfg.max_reproj_px
        z = np.einsum("nj,nj->n", poses[cam, 2, :3], X[pt_local]) + poses[cam, 2, 3]
        bad |= (z <= 0) | (z > self.cfg.max_depth_ratio * np.median(z))
        for local, k in zip(cam[bad], kp_index[bad]):
            self.keyframes[kf_ids[local]].mp_ids[k] = -1
        counts = np.zeros(len(self.points), int)
        for kf in self.keyframes:
            np.add.at(counts, kf.mp_ids[kf.mp_ids >= 0], 1)
        self.point_ok &= counts >= 2
        for kf in self.keyframes:
            dead = (kf.mp_ids >= 0) & ~self.point_ok[np.maximum(kf.mp_ids, 0)]
            kf.mp_ids[dead] = -1

    def _local_ba(self, fix_scale=False):
        n = len(self.keyframes)
        ids = list(range(max(0, n - self.cfg.ba_window), n))
        fixed = np.zeros(len(ids), bool)
        if fix_scale:
            # Two-view start: keep the first pose and let the second move; rescale afterwards.
            fixed[0] = True
            self._run_ba(ids, fixed, 15)
            c0 = inv_T(self.keyframes[0].T_cw)[:3, 3]
            c1 = inv_T(self.keyframes[1].T_cw)[:3, 3]
            s = 1.0 / max(np.linalg.norm(c1 - c0), 1e-9)
            self.points = (self.points - c0) * s + c0
            T1 = self.keyframes[1].T_cw
            self.keyframes[1].T_cw = make_T(T1[:3, :3], -T1[:3, :3] @ ((c1 - c0) * s + c0))
            return
        fixed[:min(2, len(ids) - 1)] = True
        self._run_ba(ids, fixed, 10)

    def global_ba(self, iters=25):
        ids = list(range(len(self.keyframes)))
        fixed = np.zeros(len(ids), bool)
        fixed[:2] = True
        self._run_ba(ids, fixed, iters)

    # ---- results --------------------------------------------------------
    def trajectory(self):
        """Dict frame_id -> T_wc (camera to world) for every tracked frame."""
        return {f.frame_id: inv_T(f.T_c_kf @ self.keyframes[f.kf_id].T_cw) for f in self.frames}

    def map_points(self):
        return self.points[self.point_ok]
