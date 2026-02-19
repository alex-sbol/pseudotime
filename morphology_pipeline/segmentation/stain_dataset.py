from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union
import math

from cp_measure.bulk import get_core_measurements
measurements = get_core_measurements()
sizeshape_fn = measurements["sizeshape"]

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
import cv2 as cv

# ----------------------------
# Constants & parsing
# ----------------------------

CHANNELS_CANON = ("background", "dapi", "yap", "actin")          # canonical names for user I/O

TOKEN_MAP = {
    # background synonyms
    "background": "background", "bg": "background", "back": "background",
    # stains
    "dapi": "dapi", "yap": "yap", "actin": "actin",
}

# Accept .tif or .tiff, case-insensitive. Optional "_<source>" at the end (ignored).
_FNAME_RE = re.compile(
    r"^obj(?P<obj>\d+)_stain(?P<chan>[A-Za-z0-9]+)(?:_[A-Za-z0-9]+)?\.(?:tif|tiff)$",
    re.IGNORECASE,
)

def _canon_channel(token: str) -> str:
    t = token.strip().lower()
    if t in {"backgroud", "backgroun", "backgrounds"}:
        t = "background"
    out = TOKEN_MAP.get(t)
    if out is None:
        raise ValueError(f"Unrecognized channel token '{token}'. Allowed: {tuple(TOKEN_MAP.keys())}")
    return out

# ----------------------------
# Normalization & composition
# ----------------------------

def _normalize_channelwise(arr: np.ndarray, ignore_zeros: bool = True) -> np.ndarray:
    """Per-channel scaling to [0,1] while keeping zeros at 0 (original behavior)."""
    x = np.asarray(arr, dtype=np.float32).copy()
    _, _, C = x.shape
    for i in range(C):
        ch = x[..., i]
        if ignore_zeros:
            mask = ch != 0
            minv = float(ch[mask].min()) if mask.any() else 0.0
        else:
            minv = float(ch.min())
        ch = np.maximum(ch - minv, 0.0)
        denom = float(ch.max()) + 1e-6
        x[..., i] = ch / denom
    return x

def _compose_topo_cells(arr01: np.ndarray, bg_norm: float = 0.4) -> np.ndarray:
    """Compose display RGB like the original project: BG*bg_norm + [YAP->R, Actin->G, DAPI->B]."""
    arr01 = np.asarray(arr01, dtype=np.float32)
    H, W, C = arr01.shape
    to_plot = np.zeros((H, W, 3), dtype=np.float32)
    if C == 4:
        to_plot += arr01[..., 0:1] * float(bg_norm)
        to_plot[..., 0] += arr01[..., 1]  # YAP -> R
        to_plot[..., 1] += arr01[..., 2]  # Actin -> G
        to_plot[..., 2] += arr01[..., 3]  # DAPI -> B
    elif C == 3:
        to_plot += arr01
    else:
        raise ValueError("Expected 3 or 4 channels for composition")
    return np.clip(np.round(to_plot * 255.0), 0, 255).astype(np.uint8)
# ----------------------------
# Helpers
# ----------------------------
def rotation_matrix(angle_deg: float) -> np.ndarray:
    """
    Build the 2x3 affine that matches skimage.transform.rotate(image, angle_deg, resize=False, center=None).
    Convention: points are (x, y) == (col, row). Positive angle = CCW.
    Uses center at ((W-1)/2, (H-1)/2), which is what skimage's warp math uses.
    """
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)

    # Forward mapping (raw -> rotated):
    # x' =  c*x - s*y + (1-c)*cx + s*cy
    # y' =  s*x + c*y + (1-c)*cy - s*cx
    
    R = np.array([[c, -s],
                  [s,  c]], dtype=float)
    return R


def apply_rotation(R: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    """pts_xy: (N,2) of (x,y). Returns (N,2)."""
    return pts_xy @ R

def flatten_dict(d: Dict) -> List:
    """turns a nested dict into a 1D dict by concatenating keys with '.'"""
    out = []
    for k, v in d.items():
            if isinstance(v, dict):
                sub = flatten_dict(v)
                for sk, sv in sub.items():
                    out.append((f"{k}.{sk}", sv))
            else:
                out.append((k, v))
    return dict(out)




# ----------------------------
# Dataset
# ----------------------------

@dataclass
class StainDataset:
    root: Path
    dataframe: pd.DataFrame = field(default_factory=pd.DataFrame)

    @classmethod
    def from_folder(
        cls,
        folder: Union[str, Path],
        *,
        min_area: int = 50,
        strict: bool = True,
    ) -> "StainDataset":

        folder = Path(folder).expanduser().resolve()
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")

        # 1. Scan folder and build object → channel mapping
        idx: Dict[int, Dict[str, Path]] = {}

        for f in sorted(folder.iterdir()):
            if not f.is_file():
                continue
            if f.suffix.lower() not in (".tif", ".tiff"):
                continue

            m = _FNAME_RE.match(f.name)
            if not m:
                continue

            oid = int(m.group("obj"))
            chan = _canon_channel(m.group("chan"))

            idx.setdefault(oid, {})
            if chan not in idx[oid]:
                idx[oid][chan] = f.resolve()

        if strict:
            for oid, chmap in idx.items():
                missing = [c for c in ("dapi", "yap", "actin") if c not in chmap]
                if missing:
                    raise ValueError(f"Object {oid} missing channels: {missing}")

        # 2. Run segmentation pipeline
        records = []

        for oid, chmap in sorted(idx.items()):

            seg_dict = cls.labels_from_chmap(
                oid,
                chmap,
                min_area=min_area
            )

            rows = cls.save_isolated_strains(
                oid,
                chmap,
                seg_dict
            )

            records.extend(rows)

        if not records:
            raise RuntimeError("No segmented objects found.")

        # 3. Build DataFrame
        df = pd.DataFrame(records)
        df = df.set_index("obj_id").sort_index()

        return cls(root=folder, dataframe=df)

    def  add_all_data(self) -> None:
        """Add all information about the cell to the """
        # centers = []
        # eccentricities = []
        properties_rows = []
        for obj_id in self.dataframe.index:
            row = {}
            try:
                dapi_img = self.get_channel(obj_id, "dapi", as_uint8=False)
                yapi_img = self.get_channel(obj_id, "yap", as_uint8=False)
                actin_img = self.get_channel(obj_id, "actin", as_uint8=False)
                labels_dapi = (dapi_img > 0).astype(np.uint8)
                labels_yapi = (yapi_img > 0).astype(np.uint8)
                labels_actin = (actin_img > 0).astype(np.uint8)
                for name, fn in measurements.items():
                    res_dapi = fn(labels_dapi, dapi_img)
                    res_yapi = fn(labels_yapi, yapi_img)
                    res_actin = fn(labels_actin, actin_img)
                    if isinstance(res_dapi, dict):
                        res_flat = flatten_dict(res_dapi)
                        for k, v in res_flat.items():
                            row[f"dapi.{name}.{k}"] = v[0]
                    else:
                        row[f"dapi.{name}"] = res_dapi[0]

                    if isinstance(res_yapi, dict):
                        res_flat = flatten_dict(res_yapi)
                        for k, v in res_flat.items():
                            row[f"yapi.{name}.{k}"] = v[0]
                    else:
                        row[f"yapi.{name}"] = res_yapi[0]

                    if isinstance(res_actin, dict):
                        res_flat = flatten_dict(res_actin)
                        for k, v in res_flat.items():
                            row[f"actin.{name}.{k}"] = v[0]
                    else:
                        row[f"actin.{name}"] = res_actin[0]

                properties_rows.append(row)
            except Exception as e:
                print(f"[WARN] Could not compute {name} for obj {obj_id}: {e}")
                # centers.append((np.nan, np.nan))
                # eccentricities.append(np.nan)
                properties_rows.append({})
        
        
        properties_df = pd.DataFrame(properties_rows)
        properties_df.index = self.dataframe.index
        self.dataframe = pd.concat([self.dataframe, properties_df], axis=1)

        # self.dataframe['center'] = centers
        # self.dataframe['eccentricity'] = eccentricities

    def add_corridor_ids(self, corridor_bboxes: List[Dict[str, int]]) -> None:
        """Add 'corridor_ids' column to the dataframe based on provided corridor bounding boxes.

        Parameters
        ----------
        corridor_bboxes : List[Dict[str, int]]
            A list of dictionaries, each representing a corridor bounding box with keys 'id', 'y0', 'y1', 'x0', 'x1'.
        """
        corridor_ids = []
        i = 0
        for obj_id in self.dataframe.index:
            centre = self.dataframe.at[obj_id, 'center']
            if np.isnan(centre[0]) or np.isnan(centre[1]):
                corridor_ids.append(None)
                continue
            cy = centre[1]
            i = 0
            while i < len(corridor_bboxes):
                bbox = corridor_bboxes[i]
                if bbox['y0'] <= cy <= bbox['y1']:
                    corridor_ids.append(bbox['id'])
                    break
                elif cy < bbox['y0']:
                    corridor_ids.append(None)
                    print(f"[WARN] Object {obj_id} center y={cy} proves objects are not ordered ")
                    break
                else:
                    i += 1
        self.dataframe['corridor_ids'] = corridor_ids
    # ------------- Introspection -------------

    def list_objects(self) -> List[int]:
        return sorted(self.index.keys())

    def available_channels(self, obj_id: int) -> List[str]:
        return sorted(self.index.get(obj_id, {}).keys())

    # ------------- Loading -------------

    def get_channel(self, obj_id: int, channel: str, *, as_uint8: bool = False) -> np.ndarray:
        c = _canon_channel(channel)
        p = self.dataframe.at[obj_id, f"path_{c}"]
        if p is None:
            raise KeyError(f"Channel '{c}' not found for obj {obj_id}. Available: {self.available_channels(obj_id)}")
        arr = tiff.imread(p)
        if as_uint8:
            a01 = _normalize_channelwise(np.stack([arr], axis=-1))[..., 0]
            return (np.clip(a01, 0, 1) * 255).astype(np.uint8)
        return arr

    def get_channels(self, obj_id: int, channels: Sequence[str]) -> Dict[str, np.ndarray]:
        return {c: self.get_channel(obj_id, c) for c in channels}

    def get_array_hwcn(self, obj_id: int) -> np.ndarray:
        """Return (H,W,4) ordered as [BG, YAP, Actin, DAPI] for composing."""
        bg   = self.get_channel(obj_id, "background")
        dapi = self.get_channel(obj_id, "dapi")
        yap  = self.get_channel(obj_id, "yap")
        act  = self.get_channel(obj_id, "actin")
        for ch in (dapi, yap, act):
            if ch.shape != bg.shape:
                raise ValueError(f"Shape mismatch for obj {obj_id}: BG{bg.shape} vs {ch.shape}")
        return np.stack([bg.astype(np.float32),
                         yap.astype(np.float32),
                         act.astype(np.float32),
                         dapi.astype(np.float32)], axis=-1)

    # ------------- Display -------------

    def display_object(
        self,
        obj_id: int,
        *,
        stains: Union[str, Sequence[str]] = "all",
        bg_norm: float = 0.4,
        channelwise_norm: bool = True,
        figsize: Tuple[int, int] = (6, 6),
        interpolation: str = "nearest",
    ) -> None:
        """Display one object using the original project's composition."""
        arr = self.get_array_hwcn(obj_id)  # (H,W,4) [BG,YAP,Actin,DAPI]
        arr01 = _normalize_channelwise(arr, ignore_zeros=True) if channelwise_norm else _normalize_channelwise(arr, ignore_zeros=False)

        # Build mask: always keep BG (index 0)
        mask = np.zeros_like(arr01); mask[..., 0] = 1.0
        if stains == "all":
            mask[..., 1:] = 1.0
        else:
            if isinstance(stains, str): stains = [stains]
            for st in stains:
                st = _canon_channel(st)
                if st == "yap":   mask[..., 1] = 1.0
                elif st == "actin": mask[..., 2] = 1.0
                elif st == "dapi":  mask[..., 3] = 1.0
                else:
                    raise ValueError("stains must be among {'yap','actin','dapi'} or 'all'")
        rgb = _compose_topo_cells(arr01 * mask, bg_norm=bg_norm)  # uint8

        plt.figure(figsize=figsize)
        plt.imshow(rgb, interpolation=interpolation)
        plt.title(f"obj {obj_id} | {'all' if stains=='all' else '+'.join(stains)}")
        plt.axis("off"); plt.show()

    def display_grid(
        self,
        obj_id: int,
        *,
        bg_norm: float = 0.4,
        channelwise_norm: bool = True,
        figsize: Tuple[int, int] = (15, 4),
        interpolation: str = "nearest",
    ) -> None:
        """Show tiles: BG-only, BG+YAP, BG+Actin, BG+DAPI, BG+ALL."""
        arr = self.get_array_hwcn(obj_id)
        arr01 = _normalize_channelwise(arr, ignore_zeros=True) if channelwise_norm else _normalize_channelwise(arr, ignore_zeros=False)

        masks = []
        names = ["background", "yap", "actin", "dapi", "all"]
        # BG only
        m = np.zeros_like(arr01); m[..., 0] = 1.0; masks.append(m)
        # BG + one stain each
        for k in (1, 2, 3):
            m = np.zeros_like(arr01); m[..., 0] = 1.0; m[..., k] = 1.0; masks.append(m)
        # BG + all
        m = np.zeros_like(arr01); m[..., 0] = 1.0; m[..., 1:] = 1.0; masks.append(m)

        imgs = [_compose_topo_cells(arr01 * m, bg_norm=bg_norm) for m in masks]

        fig, axes = plt.subplots(1, len(imgs), figsize=figsize)
        for ax, img, name in zip(axes, imgs, names):
            ax.imshow(img, interpolation=interpolation); ax.set_title(name); ax.axis("off")
        fig.suptitle(f"obj {obj_id}"); plt.show()

    def rotate_centers_like_skimage(self,
                                img_shape: Tuple[int, int],
                                angle_deg: float,
                                out_col: str = "center_rot"):
        """Add rotated centers column matching your deskew rotation."""
        H, W = img_shape
        M = rotation_matrix(W, H, angle_deg)
        cp_df = self.dataframe
        idxs, pts = [], []
        for idx, v in cp_df["center"].items():
            xy = None #TODO
            if xy is None:
                continue
            idxs.append(idx); pts.append(xy)
        if not pts:
            cp_df[out_col] = None
            return

        pts = np.asarray(pts, dtype=float)
        pts_rot = apply_rotation(M, pts)

        for i, idx in enumerate(idxs):
            cp_df.at[idx, out_col] = (float(pts_rot[i, 0]), float(pts_rot[i, 1]))

    @staticmethod
    def labels_from_chmap(oid, chmap, min_area=50):
        """
        Return one selected label for each channel: dapi, yap, actin.
        The selection is:
            - threshold channel
            - connected components
            - filter by min_area
            - pick the component whose centroid is closest to image center
        """

        # --- Load images (error if missing) -------------------------------------
        def load_gray(channel):
            path = chmap.get(channel)
            if path is None:
                raise ValueError(f"Object {oid} missing channel '{channel}'")
            img = tiff.imread(path)
            if img.ndim == 3:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            return img

        dapi_grey = load_gray("dapi")
        yap_grey = load_gray("yap")
        actin_grey = load_gray("actin")

        # All channels are assumed same shape; use DAPI for center reference
        H, W = dapi_grey.shape
        img_center = np.array([W / 2, H / 2])

        # --- Helper: label for image -------------------------------------
        def get_single_label(grey_img):
            # Threshold
            _, thresh = cv.threshold(grey_img, 10, 255, 0)

            # Connected components
            num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
                thresh, connectivity=8
            )

            # Filter by min area (skip background, index 0)
            valid = [
                i for i in range(1, num_labels)
                if stats[i, cv.CC_STAT_AREA] >= min_area
            ]
            if not valid:
                return None  # or raise

            # Compute distances to center = pick closest
            centers = np.array([centroids[i] for i in valid])  # [[cx, cy], ...]
            dists = np.linalg.norm(centers - img_center, axis=1)
            chosen_idx = valid[np.argmin(dists)]

            return chosen_idx, labels, stats, centroids

        # --- Run for each channel -----------------------------------------------
        dapi_label, dapi_labels_img, dapi_stats, dapi_centroids = get_single_label(dapi_grey)
        yap_label, yap_labels_img, yap_stats, yap_centroids = get_single_label(yap_grey)
        actin_label, actin_labels_img, actin_stats, actin_centroids = get_single_label(actin_grey)

        #

        # --- Return structured result -------------------------------------------
        return {
            "dapi": {
                "label": dapi_label,
                "labels_img": dapi_labels_img,
                "stats": dapi_stats,
                "centroids": dapi_centroids,
            },
            "yap": {
                "label": yap_label,
                "labels_img": yap_labels_img,
                "stats": yap_stats,
                "centroids": yap_centroids,
            },
            "actin": {
                "label": actin_label,
                "labels_img": actin_labels_img,
                "stats": actin_stats,
                "centroids": actin_centroids,
            },
        }

    @staticmethod
    def rows_from_labels(oid, chmap, seg_dict):
        # look at this thing and my pipeline and adapt this function,
        # such that no further refactoring would be needed during the run

        H, W = seg_dict[oid].shape

        rows = []

        for chan, info in seg_dict.items():
            label = info['label']
            stats = info['stats']
            centroids = info['centroids']

            if label is None:
                continue

            x = int(stats[label, cv.CC_STAT_LEFT])
            y = int(stats[label, cv.CC_STAT_TOP])
            w = int(stats[label, cv.CC_STAT_WIDTH])
            h = int(stats[label, cv.CC_STAT_HEIGHT])
            area = int(stats[label, cv.CC_STAT_AREA])

            cx, cy = centroids[label]  # float centroid

            row = {
                "obj_id": oid,
                "channel": chan,

                # geometric info
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": w,
                "bbox_h": h,
                "area": area,

                # centroid
                "centroid_x": float(cx),
                "centroid_y": float(cy),

                # source image
                "path": str(chmap.get(chan, "")),

                # shape metadata
                "height": H,
                "width": W,
            }

            rows.append(row)

        return rows

    @staticmethod
    def save_isolated_strains():

        return

    @staticmethod
    def labels_from_chmap(oid, chmap, min_area=50):
        """
        Select one connected component for each stain channel (dapi, yap, actin)
        based on:
            - thresholding
            - connected components
            - filtering by min area
            - choosing component closest to image center
        """

        def load_gray(channel):
            p = chmap.get(channel)
            if p is None:
                raise ValueError(f"Object {oid} missing channel '{channel}'")
            img = tiff.imread(p)
            return cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img

        dapi_grey = load_gray("dapi")
        yap_grey = load_gray("yap")
        actin_grey = load_gray("actin")

        H, W = dapi_grey.shape
        img_center = np.array([W / 2, H / 2])

        def get_single_label(grey_img):
            _, thresh = cv.threshold(grey_img, 10, 255, 0)

            num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
                thresh, connectivity=8
            )

            valid = [
                i for i in range(1, num_labels)
                if stats[i, cv.CC_STAT_AREA] >= min_area
            ]
            if not valid:
                return None, None, None, None

            centers = np.array([centroids[i] for i in valid])
            dists = np.linalg.norm(centers - img_center, axis=1)
            chosen_idx = valid[np.argmin(dists)]

            return chosen_idx, labels, stats, centroids

        dapi_label, dapi_labels_img, dapi_stats, dapi_centroids = get_single_label(dapi_grey)
        yap_label, yap_labels_img, yap_stats, yap_centroids = get_single_label(yap_grey)
        actin_label, actin_labels_img, actin_stats, actin_centroids = get_single_label(actin_grey)

        return {
            "dapi": {"label": dapi_label, "labels_img": dapi_labels_img, "stats": dapi_stats,
                     "centroids": dapi_centroids},
            "yap": {"label": yap_label, "labels_img": yap_labels_img, "stats": yap_stats, "centroids": yap_centroids},
            "actin": {"label": actin_label, "labels_img": actin_labels_img, "stats": actin_stats,
                      "centroids": actin_centroids},
        }

    @staticmethod
    def rows_from_labels(oid, chmap, seg_dict):
        rows = []

        # infer shape
        H = W = None
        for c in ("dapi", "yap", "actin"):
            p = chmap.get(c)
            if p:
                img = tiff.imread(p)
                H, W = img.shape[:2]
                break
        if H is None:
            raise ValueError(f"Cannot infer shape for object {oid}")

        for chan, info in seg_dict.items():
            label = info["label"]
            if label is None:
                continue

            stats = info["stats"]
            centroids = info["centroids"]

            x = int(stats[label, cv.CC_STAT_LEFT])
            y = int(stats[label, cv.CC_STAT_TOP])
            w = int(stats[label, cv.CC_STAT_WIDTH])
            h = int(stats[label, cv.CC_STAT_HEIGHT])
            area = int(stats[label, cv.CC_STAT_AREA])

            cx, cy = centroids[label]

            rows.append({
                "obj_id": oid,
                "channel": chan,
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": w,
                "bbox_h": h,
                "area": area,
                "centroid_x": float(cx),
                "centroid_y": float(cy),
                "path": str(chmap.get(chan, "")),
                "height": H,
                "width": W,
            })

        return rows

    @staticmethod
    def save_isolated_strains(oid, chmap, seg_dict, *, dilate_iter=3):
        rows = []

        # determine shape
        H = W = None
        for c in ("dapi", "yap", "actin"):
            p = chmap.get(c)
            if p:
                arr = tiff.imread(p)
                H, W = arr.shape[:2]
                break

        for chan, info in seg_dict.items():
            label = info["label"]
            if label is None:
                continue

            labels_img = info["labels_img"]
            stats = info["stats"]
            centroids = info["centroids"]

            new_oid = oid * 10000 + label
            in_path = chmap[chan]
            img = tiff.imread(in_path)

            mask = (labels_img == label).astype(np.uint8)

            if dilate_iter > 0:
                kernel = np.ones((3, 3), np.uint8)
                mask = cv.dilate(mask, kernel, iterations=dilate_iter)

            out_img = img * mask.astype(img.dtype)

            out_dir = Path(in_path).parent / f"isolated_{chan}"
            out_dir.mkdir(exist_ok=True)
            out_path = out_dir / f"obj{new_oid}_stain{chan.upper()}.tif"
            tiff.imwrite(out_path, out_img)

            x = int(stats[label, cv.CC_STAT_LEFT])
            y = int(stats[label, cv.CC_STAT_TOP])
            w = int(stats[label, cv.CC_STAT_WIDTH])
            h = int(stats[label, cv.CC_STAT_HEIGHT])
            area = int(stats[label, cv.CC_STAT_AREA])
            cx, cy = centroids[label]

            rows.append({
                "obj_id": new_oid,
                "channel": chan,
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": w,
                "bbox_h": h,
                "area": area,
                "centroid_x": float(cx),
                "centroid_y": float(cy),
                "path": str(out_path),
                "height": H,
                "width": W,
            })

        return rows

# Convenience
def build_dataframe(folder: Union[str, Path], *, strict: bool = True, drop_incomplete: bool = False) -> pd.DataFrame:
    return StainDataset.from_folder(folder, strict=strict).dataframe



