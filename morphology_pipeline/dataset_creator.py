from morphology_pipeline.segmentation.stain_dataset import (
    StainDataset,
    affine_like_skimage_no_resize,
    apply_affine_points,
)
from morphology_pipeline.pseudotime import window_segments, descriptor_from_segments, slide_windows_with_matching
from morphology_pipeline.corridor_mask import deskew_with_hull, detect_corridors_via_hull
from typing import List, Dict, Tuple
import numpy as np



def create_dataset(background, folder):
    SD = StainDataset.from_folder(folder)
    #TODO extend this function to add other CP measure metrics
    SD.add_center_eccentricity()
    # TODO make them parameters
    #This is pseudotome hyper params
    #I could also make pseudotime 
    m, L, stride, L_min = 2, 30, 1, 8

    rot_img, corr_mask, hull_mask, rot_deg = deskew_with_hull(background)

    corridors = detect_corridors_via_hull(
        corr_mask, hull_mask,
        row_cov_thresh_rel_hull=0.07,
        min_band_height=5,
        merge_gap_px=3
    )
    print(f"Detected {len(corridors)} corridors.")

    
    H0, W0 = background.shape[:2]
    M = affine_like_skimage_no_resize(W0, H0, rot_deg)

    #This could be incorporated as a segmentation StainDataset method
    if "center_rot" not in SD.dataframe.columns:
        SD.dataframe["center_rot"] = None

    
    #RECLAIM this code from AI
    # scaling to match background if dataset used different dimensions
    h_ref = float(SD.dataframe["height"].iloc[0]) if "height" in SD.dataframe.columns else H0
    w_ref = float(SD.dataframe["width"].iloc[0]) if "width" in SD.dataframe.columns else W0
    sx = W0 / w_ref if w_ref else 1.0
    sy = H0 / h_ref if h_ref else 1.0

    centers_rot_map: Dict[int, Tuple[float, float]] = {}

    for obj_id, val in SD.dataframe["center"].items():
        if val is None:
            continue

        
        #TODO This needs to be checked if the image is scaled correctly cuz the dapi image is not full size of the background,
        #but this looks like we loose original location of the dapi cell
        # val = (x, y)
        x_raw, y_raw = val
        x_arr = np.asarray(x_raw)
        y_arr = np.asarray(y_raw)
        if x_arr.size == 0 or y_arr.size == 0:
            continue

        # scale, preserve orientation
        y = float(y_arr.reshape(-1)[0]) * sy
        x = float(x_arr.reshape(-1)[0]) * sx

        pt_rot = apply_affine_points(
            M, np.asarray([[x, y]], dtype=float)
        )[0]

        x_rot, y_rot = float(pt_rot[0]), float(pt_rot[1])
        centers_rot_map[obj_id] = (x_rot, y_rot)
        SD.dataframe.at[obj_id, "center_rot"] = (x_rot, y_rot)


    

    print("hello")