from stain_dataset import StainDataset
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import remove_small_holes
from scipy.ndimage import distance_transform_edt
import numpy as np
import matplotlib.pyplot as plt

ds = StainDataset.from_folder(r"C:\Users\sbsas\Documents\uni\Projects\Nikita PhD\Chip_1476_1_real", strict=False)  # strict=True if all 4 must exist
print(ds.dataframe.head())

def segment_cell(actin_img):
    # find the segmentation of the cell:
    actin_blurred = gaussian(actin_img, sigma=5)
    actin_mask = actin_img >= threshold_otsu(actin_blurred)
    actin_mask = remove_small_holes(actin_mask.numpy(), area_threshold=2500)
    return actin_mask

def find_cortical_mask(actin_img, cortical_width=3):
    actin_mask = segment_cell(actin_img)
    dist_to_bg = distance_transform_edt(actin_mask)
    cortical_and_outside_mask = dist_to_bg <= cortical_width  # type: ignore
    return cortical_and_outside_mask

def reduce_cortical_actin(img_no_bg, cortical_and_outside_mask, dim_factor=0.5):
    # actin channel has index 2. img and bg is channels-first
    assert img_no_bg.shape[0] == 3 and len(img_no_bg.shape) == 3
    img_no_bg = np.array(img_no_bg)
    dimmed_actin = img_no_bg[1] * dim_factor
    img_no_bg[1] = cortical_and_outside_mask * dimmed_actin + (1-cortical_and_outside_mask) * img_no_bg[1]
    return img_no_bg

# # One object, original look
ds.display_object(2, stains='all', bg_norm=0.5)