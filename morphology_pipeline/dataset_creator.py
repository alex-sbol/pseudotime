from morphology_pipeline.segmentation.stain_dataset import (
    rotation_matrix,
    apply_rotation,
)
from morphology_pipeline.corridor_mask import deskew_with_hull, detect_corridors_via_hull
import numpy as np
from morphology_pipeline.pseudotime import process_corridors


def create_dataset(background, folder, SD):
    m, L, stride, L_min = 2, 30, 1, 8

    rot_img, corr_mask, hull_mask, rot_deg = deskew_with_hull(background)
    print(f"Deskewed by {rot_deg:.2f} degrees.")

    #list of bbox dicts {id, y0, y1, x0, x1}
    corridors = detect_corridors_via_hull(
        corr_mask, hull_mask,
        row_cov_thresh_rel_hull=0.07,
        min_band_height=5,
        merge_gap_px=3
    )

    print(f"Detected {len(corridors)} corridors.")

    try:
        pseudotime = process_corridors(corr_mask, corridors, m, min_period=5, max_period=30)
    except RuntimeError as e:
        print(f"Error in pseudotime calculation: {e}")
        pseudotime = {
        "period": 7,           # now: normalized cycle length
        "template": [1 ,2, 3, 6, 3, 2, 1],  # now: canonical intensity profile
        "corridors": [{
            "signal": [1, 2, 3, 4, 3, 2, 1] ,
            "line_id": [0, 1, 2, 3, 4, 5, 6] ,
            "confidence": [1, 1, 1, 1, 1, 1, 1] ,
            "peaks": [0, 3, 6],
        }] * len(corridors),
    }
        
    #return pseudotime, corridors, corr_mask

    
    
    H0, W0 = background.shape[:2]
    W1, H1 = rot_img.shape[:2] 
    CRX = (W1 -1) / 2
    CRY = (H1 -1) / 2
    R = rotation_matrix( rot_deg)



    # scaling to match background if dataset used different dimensions
    h_ref = float(SD.dataframe["height"].iloc[0]) if "height" in SD.dataframe.columns else H0
    w_ref = float(SD.dataframe["width"].iloc[0]) if "width" in SD.dataframe.columns else W0

    cols = [
    "sizeshape.Center_X",
    "sizeshape.Center_Y",
    "sizeshape.BoundingBoxMinimum_X",
    "sizeshape.BoundingBoxMaximum_X",
    "sizeshape.BoundingBoxMinimum_Y",
    "sizeshape.BoundingBoxMaximum_Y",
    ]

    for obj_id, x_raw, y_raw, bbox_min_x, bbox_max_x, bbox_min_y, bbox_max_y in SD.dataframe[cols].itertuples(index=True):
        

        if x_raw is None or y_raw is None:
            continue

        cx = (h_ref - 1) / 2
        cy = (w_ref - 1) / 2

        x = x_raw - cx
        y = y_raw - cy

        pt_rot_center = apply_rotation(R, np.array([[x, y]], dtype=float))[0]
        #print(pt_rot_center.shape, pt_rot_center)
        if pt_rot_center[0] is not None:
            x_rot_c, y_rot_c = float(pt_rot_center[0]), float(pt_rot_center[1])
        else:
            x_rot_c, y_rot_c = None, None

        corners = np.array([
        [bbox_min_x, bbox_min_y],
        [bbox_max_x, bbox_min_y],
        [bbox_max_x, bbox_max_y],
        [bbox_min_x, bbox_max_y],
        ], dtype=float)


        corners -= np.array([[cx, cy]], dtype=float)

        rot_corners = apply_rotation(R, corners)

        x_min_rot = rot_corners[:, 0].min()
        y_min_rot = rot_corners[:, 1].min()
        x_max_rot = rot_corners[:, 0].max()
        y_max_rot = rot_corners[:, 1].max()

        # if x_min_rot > x_max_rot:
        #     x_min_rot, x_max_rot = x_max_rot, x_min_rot
        # if y_min_rot > y_max_rot:
        #     y_min_rot, y_max_rot = y_max_rot, y_min_rot

        SD.dataframe.at[obj_id, "centered_center_x"] = x
        SD.dataframe.at[obj_id, "centered_center_y"] = y

        SD.dataframe.at[obj_id, "center_rot_x"] = x_rot_c
        SD.dataframe.at[obj_id, "center_rot_y"] = y_rot_c

        SD.dataframe.at[obj_id, "centered_bbox_min_x"] = bbox_min_x - cx
        SD.dataframe.at[obj_id, "centered_bbox_min_y"] = bbox_min_y - cy
        SD.dataframe.at[obj_id, "centered_bbox_max_x"] = bbox_max_x - cx
        SD.dataframe.at[obj_id, "centered_bbox_max_y"] = bbox_max_y - cy

        SD.dataframe.at[obj_id, "bbox_rot_min_x"] = x_min_rot
        SD.dataframe.at[obj_id, "bbox_rot_min_y"] = y_min_rot
        SD.dataframe.at[obj_id, "bbox_rot_max_x"] = x_max_rot
        SD.dataframe.at[obj_id, "bbox_rot_max_y"] = y_max_rot

        SD.dataframe.at[obj_id, "pseudotime"] = None
        SD.dataframe.at[obj_id, "pseudotime_widths"] = None

        for corridor_info in pseudotime["corridors"]:
            bbox = corridor_info["bbox"]

            if (bbox["x0"] <= x_rot_c + CRX <= bbox["x1"]) and (bbox["y0"] <= y_rot_c + CRY <= bbox["y1"]):

                lh = int((x_min_rot + CRX - bbox["x0"]) // m)
                rh = int((x_max_rot + CRX - bbox["x0"]) // m) + 1

                # if lh < 0 or rh < 0:
                #     continue

                line_ids = bbox["line_id"][lh:rh]
                widthss = bbox["width"][lh:rh]
                if len(line_ids) != len(widthss):
                    print("Length mismatch in pseudotime assignment! " + str(obj_id) + f" {len(line_ids)} vs {len(widthss)} " + str(lh) + " " + str(rh))
                if True:
                    print(len(line_ids), len(widthss))
                    SD.dataframe.at[obj_id, "pseudotime"] = line_ids
                    SD.dataframe.at[obj_id, "pseudotime_widths"] = widthss
                else:
                    SD.dataframe.at[obj_id, "pseudotime"] = None
                break

        #Now we need to assign pseudotime to each cell based on its rotated center location

        

    return pseudotime, corridors, corr_mask
