
"""
stain_dataset.py
=================
Dataset + plotting utilities adapted to the original project's conventions,
with a **simple folder scanner** as requested.

Folder expectation:
    The given folder contains TIFF files named like:
        obj####_stain<Channel>_<Source>.tif[f]
    Examples:
        obj0007_stainbackground_real.tiff
        obj0007_stainDAPI_fake.tif
    The "<Source>" suffix is optional and ignored for indexing.

Plotting behavior matches the original project's utils:
- Per-channel normalization to [0,1] while keeping zeros at 0.
- Composition is additive (no alpha):
    RGB = BG*bg_norm  +  [YAP -> R, Actin -> G, DAPI -> B]

API
---
ds = StainDataset.from_folder("/path/to/folder", strict=True, drop_incomplete=False)
ds.display_object(7, stains="all")
ds.display_grid(7)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

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
# Normalization & composition (matching original project)
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
# Dataset
# ----------------------------

@dataclass
class StainDataset:
    root: Path
    index: Dict[int, Dict[str, Path]] = field(default_factory=dict)  # obj_id -> {channel: path}
    dataframe: pd.DataFrame = field(default_factory=pd.DataFrame)

    @classmethod
    def from_folder(
        cls,
        folder: Union[str, Path],
        *,
        strict: bool = True,
        drop_incomplete: bool = False,
    ) -> "StainDataset":
        """Create dataset by scanning a folder.

        Parameters
        ----------
        folder : str | Path
            Directory with TIFF files.
        strict : bool
            If True (default), every object must have all 4 channels; otherwise raises.
            If False, missing channels are allowed but warned and left empty in DataFrame.
        drop_incomplete : bool
            Only used when strict=False. If True, rows missing any channel are dropped.

        Returns
        -------
        StainDataset
        """
        folder = Path(folder).expanduser().resolve()
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"Folder not found or not a directory: {folder}")

        idx: Dict[int, Dict[str, Path]] = {}
        duplicates: List[Tuple[int, str, Path, Path]] = []

        # Non-recursive scan; only files directly inside `folder`
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
            d = idx.setdefault(oid, {})
            if chan in d:
                duplicates.append((oid, chan, d[chan], f))
                # keep the first; ignore later duplicates to keep behavior simple
                continue
            d[chan] = f.resolve()

        if duplicates:
            print(f"[WARN] Found {len(duplicates)} duplicate files for the same obj/channel. "
                  "Keeping the first occurrence. Examples:")
            for k in duplicates[:5]:
                print("   obj", k[0], "chan", k[1], "kept:", k[2].name, "ignored:", k[3].name)

        records = []
        missing_total = 0
        for oid, chmap in sorted(idx.items()):
            missing = [c for c in CHANNELS_CANON if c not in chmap]
            if strict and missing:
                raise ValueError(f"Object {oid} missing channels: {missing}")
            if (not strict) and missing:
                missing_total += 1

            row: dict[str, int | str] = {"obj_id": oid}
            for c in CHANNELS_CANON:
                row[f"path_{c}"] = str(chmap.get(c, "")) if c in chmap else ""

            # Derive (H,W) from the first available channel in canonical order
            h = w = None
            for c in CHANNELS_CANON:
                p = chmap.get(c)
                if p:
                    try:
                        arr = tiff.imread(p)
                        if arr.ndim >= 2:
                            h, w = int(arr.shape[-2]), int(arr.shape[-1])
                            break
                    except Exception:
                        continue
            if h is None or w is None:
                raise ValueError(f"Could not read image shape for obj {oid} from any available channel.")
            row["height"], row["width"] = h, w
            row["channels"] = ",".join(sorted(chmap.keys()))
            records.append(row)

        df = pd.DataFrame(records).set_index("obj_id").sort_index()

        if (not strict) and drop_incomplete:
            need = [f"path_{c}" for c in CHANNELS_CANON]
            before = len(df)
            df = df[(df[need] != "").all(axis=1)]
            after = len(df)
            if before != after:
                print(f"[INFO] Dropped {before-after} incomplete objects (drop_incomplete=True).")

        if (not strict) and missing_total:
            print(f"[WARN] {missing_total} objects are missing one or more channels.")

        return cls(root=folder, index=idx, dataframe=df)

    # ------------- Introspection -------------

    def list_objects(self) -> List[int]:
        return sorted(self.index.keys())

    def available_channels(self, obj_id: int) -> List[str]:
        return sorted(self.index.get(obj_id, {}).keys())

    # ------------- Loading -------------

    def get_channel(self, obj_id: int, channel: str, *, as_uint8: bool = False) -> np.ndarray:
        c = _canon_channel(channel)
        p = self.index.get(obj_id, {}).get(c)
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

# Convenience
def build_dataframe(folder: Union[str, Path], *, strict: bool = True, drop_incomplete: bool = False) -> pd.DataFrame:
    return StainDataset.from_folder(folder, strict=strict, drop_incomplete=drop_incomplete).dataframe
