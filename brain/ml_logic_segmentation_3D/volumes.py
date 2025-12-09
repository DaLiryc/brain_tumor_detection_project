import nibabel as nib
import numpy as np
from pathlib import Path
from brain.params import *

def compute_tumor_volume_from_nii(seg_path, tumor_labels=None, in_ml=False):
    """
    Compute the tumor volume from a nii file.
    Applicable for 3D segmentation masks from BRATS 2023 dataset.
    -- Parameters --
    seg_path : path du fichier *-seg.nii.gz
    tumor_labels : liste des labels considérés comme tumeur.
                   - None -> tout label > 0
    in_ml : si True, renvoie le volume en mL, sinon en mm^3
    """
    seg_img = nib.load(seg_path)
    seg_data = seg_img.get_fdata()
    if tumor_labels is None:
        tumor_labels = np.unique(seg_data)
        tumor_labels = [l for l in tumor_labels if l > 0]
    else:
        tumor_labels = np.array(tumor_labels)
    
    # Calcul du volume pour chaque label de tumeur
    volumes = {}
    for label in tumor_labels:
        mask = (seg_data == label)
        volume = np.sum(mask) * np.prod(seg_img.header.get_zooms()[:3])
        if in_ml:
            volume *= 1e-3  # Conversion de mm^3 en mL
        volumes[label] = volume
        
    tumor_volume = sum(volumes.values())
    return tumor_volume


def compute_tumor_volume_from_mask(mask, voxel_spacing=(1.0, 1.0, 1.0), in_ml=False):
    """
    Compute the tumor volume from a mask.
    -- Parameters --
    mask : np.ndarray (H,W,D) or (H,W,D,1) with 0/1 or 0/label
    voxel_spacing : tuple of float, spacing between voxels in mm
    in_ml : if True, return the volume in mL, otherwise in mm^3
    """
    sx, sy, sz = voxel_spacing
    voxel_vol_mm3 = sx * sy * sz

    n_voxels = np.count_nonzero(mask)
    vol_mm3 = n_voxels * voxel_vol_mm3

    if in_ml:
        return vol_mm3 / 1000.0
    else:
        return vol_mm3
    
def volume_voxels(mask):
    """
    Compute the number of voxels in a mask.
    -- Parameters --
    mask : np.ndarray (H,W,D) or (H,W,D,1) with 0/1 or 0/label
    """
    if mask.ndim == 4:
        mask = mask[..., 0]
    return int(np.count_nonzero(mask))

def compute_volume_voxels(seg_path, pred_bin):
    """
    Compute the number of voxels in a segmentation mask and in the prediction.
    -- Parameters --
    mask : np.ndarray (H,W,D) or (H,W,D,1) with 0/1 or 0/label
    """
    gt_img = nib.load(str(seg_path))
    gt_seg = gt_img.get_fdata()          # (H,W,D)
    gt_tumor = gt_seg > 0                # ou np.isin(gt_seg, [1,2,4]) selon tes labels
    n_gt = volume_voxels(gt_tumor)
    
    n_pred = volume_voxels(pred_bin)

    print("GT   - voxels tumor :", n_gt)
    print("Pred - voxels tumor :", n_pred)
    return n_gt, n_pred