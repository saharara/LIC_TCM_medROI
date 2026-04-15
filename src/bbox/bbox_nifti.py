import os
import glob
import numpy as np
import nibabel as nib
import cv2
import pandas as pd
from scipy import ndimage
import time

# Parameters
SUBJECTS_ROOT = r"D:\UET\Tai_lieu\Nam_3\Ky_2\Chu_de_hien_dai_HTTT\MedROI\data"
NIFTI_PATTERN = "*.nii"
OUT_DIR       = "./bbox_outputs"
SAVE_CROPPED  = True

# Parameters - Sửa lại cho đúng thực tế của bạn
# SUBJECTS_ROOT = r"D:\UET\Tai_lieu\Nam_3\Ky_2\Chu_de_hien_dai_HTTT\MedROI\data"         # Trỏ đúng vào thư mục chứa ảnh
# NIFTI_PATTERN = "*.nii*"        # Nhận diện cả .nii và .nii.gz cho chắc chắn

MISS_THR  = 0.002
PADDING   = 3
EXTRA_PAD = 6

# Utility functions
def bbox_from_mask(mask_bool):
    if not np.any(mask_bool): return None
    ys, xs = np.where(mask_bool)
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())

def pad_box(box, H, W, pad):
    if box is None: return None
    y0,x0,y1,x1 = box
    return max(0,y0-pad), max(0,x0-pad), min(H-1,y1+pad), min(W-1,x1+pad)

def normalize_to_uint8(V):
    """Normalize volume to uint8 range considering original intensity distribution"""
    V_nz = V[V > 0]
    if V_nz.size == 0:
        return np.zeros_like(V, dtype=np.uint8)
    
    vmin = np.percentile(V_nz, 1)
    vmax = np.percentile(V_nz, 99)
    
    V_normalized = np.clip((V - vmin) / (vmax - vmin) * 255, 0, 255)
    return V_normalized.astype(np.uint8)

def compute_subject_boxes(nifti_path):
    """Compute 2D bounding boxes per X-axis slice and 3D bounding box for subject"""
    img = nib.load(nifti_path)
    V   = img.get_fdata()
    
    V8 = normalize_to_uint8(V)

    X, Y, Z = V8.shape
    nz = V8 > 0
    if not np.any(nz):
        return [], None, img

    t_global = float(V8[nz].mean())
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    per_slice = []
    for x in range(X):
        slc = V8[x, :, :]
        H, W = slc.shape
        ref_mask = (slc >= t_global).astype(np.uint8) * 255
        ref_mask = cv2.morphologyEx(ref_mask, cv2.MORPH_CLOSE, kernel)
        ref_bool = (ref_mask > 0)
        ref_pixels = int(ref_bool.sum())

        non_zero_pixels = slc[slc > 0]
        thr_slice = float(non_zero_pixels.mean()) if non_zero_pixels.size > 0 else 0.0

        box_a = bbox_from_mask(slc > thr_slice)
        box_a = pad_box(box_a, H, W, PADDING)

        if (box_a is None) or (ref_pixels == 0):
            miss_a = 0.0
        else:
            ya, xa, yb, xb = box_a
            bbA = np.zeros_like(ref_bool, bool)
            bbA[ya:yb+1, xa:xb+1] = True
            miss_a = (ref_bool & (~bbA)).sum() / ref_pixels

        final_box = box_a
        if (box_a is not None) and (miss_a > MISS_THR):
            final_box = pad_box(box_a, H, W, PADDING + EXTRA_PAD)

        if (final_box is not None) and (ref_pixels > 0):
            fy0, fx0, fy1, fx1 = final_box
            bbF = np.zeros_like(ref_bool, bool)
            bbF[fy0:fy1+1, fx0:fx1+1] = True
            miss_final = (ref_bool & (~bbF)).sum() / ref_pixels
        else:
            fy0=fx0=fy1=fx1=-1
            miss_final = 0.0

        per_slice.append({
            "slice_axis": "X",
            "slice_idx": x,
            "thr_slice": thr_slice,
            "thr_global": t_global,
            "y_min": fy0, "x_min": fx0, "y_max": fy1, "x_max": fx1,
            "miss_rate": miss_final
        })

    # 3D bounding box
    ref3d = (V8 >= t_global)
    ref3d = ndimage.binary_opening(ref3d, iterations=1)
    
    if not np.any(ref3d):
        bbox3d = None
    else:
        xs, ys, zs = np.where(ref3d)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        z0, z1 = int(zs.min()), int(zs.max())

        x0 = max(0, x0 - PADDING); y0 = max(0, y0 - PADDING); z0 = max(0, z0 - PADDING)
        x1 = min(X-1, x1 + PADDING); y1 = min(Y-1, y1 + PADDING); z1 = min(Z-1, z1 + PADDING)
        bbox3d = (x0, y0, z0, x1, y1, z1)

    return per_slice, bbox3d, img

def subject_id_from_path(nifti_path):
    """Extract subject ID from filename"""
    basename = os.path.basename(nifti_path)
    return basename.replace('.nii.gz', '').replace('.nii', '')

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if SAVE_CROPPED:
        os.makedirs(os.path.join(OUT_DIR, "cropped_volumes"), exist_ok=True)

    # = glob.glob(os.path.join(SUBJECTS_ROOT, NIFTI_PATTERN))
    nifti_list = glob.glob(os.path.join(SUBJECTS_ROOT, "**", NIFTI_PATTERN), recursive=True) + \
             glob.glob(os.path.join(SUBJECTS_ROOT, NIFTI_PATTERN))
    nifti_list = sorted(set(nifti_list))
    nifti_list.sort()

    print(f"Searching in: {os.path.abspath(SUBJECTS_ROOT)}")
    print(f"Found {len(nifti_list)} NIfTI files")

    rows_2d = []
    rows_3d = []
    timing_info = []
    success_count = 0

    for nifti_path in nifti_list:
        sid = subject_id_from_path(nifti_path)
        print(f"[{sid}] processing...")

        try:
            crop_start_time = time.time()
            
            per_slice, bbox3d, img = compute_subject_boxes(nifti_path)
            
            bbox_calc_time = time.time() - crop_start_time
            
            # Store 2D slice information
            for r in per_slice:
                r_out = {"subject_id": sid}
                r_out.update(r)
                rows_2d.append(r_out)

            # Store 3D bbox and save cropped volume
            if bbox3d is not None:
                x0,y0,z0,x1,y1,z1 = bbox3d
                X,Y,Z = img.shape[:3]
                
                rows_3d.append({
                    "subject_id": sid,
                    "x_min": x0, "y_min": y0, "z_min": z0,
                    "x_max": x1, "y_max": y1, "z_max": z1,
                    "dim_x": (x1-x0+1), "dim_y": (y1-y0+1), "dim_z": (z1-z0+1),
                    "orig_x": X, "orig_y": Y, "orig_z": Z
                })

                if SAVE_CROPPED:
                    save_start_time = time.time()
                    
                    V = img.get_fdata()
                    cropped = V[x0:x1+1, y0:y1+1, z0:z1+1]
                    
                    new_affine = img.affine.copy()
                    new_affine[:3, 3] = img.affine[:3, 3] + img.affine[:3, :3] @ np.array([x0, y0, z0])
                    
                    new_img = nib.Nifti1Image(cropped.astype(np.float32), new_affine)
                    out_path = os.path.join(OUT_DIR, "cropped_volumes", f"{sid}.nii.gz")
                    nib.save(new_img, out_path)
                    
                    save_time = time.time() - save_start_time
                    total_crop_time = time.time() - crop_start_time
                    
                    timing_info.append({
                        "subject_id": sid,
                        "filename": f"{sid}.nii.gz",
                        "bbox_calc_time": bbox_calc_time,
                        "file_save_time": save_time,
                        "total_cropping_time": total_crop_time,
                        "original_shape": f"{X}x{Y}x{Z}",
                        "cropped_shape": f"{x1-x0+1}x{y1-y0+1}x{z1-z0+1}"
                    })
                    
                    success_count += 1
                    print(f"[{sid}] Cropped in {total_crop_time:.4f}s "
                          f"(calc: {bbox_calc_time:.4f}s, save: {save_time:.4f}s)")
            else:
                print(f"[{sid}] No valid bbox found")

        except Exception as e:
            print(f"[{sid}] Error: {e}")

    # Save CSV outputs
    df2d = pd.DataFrame(rows_2d)
    df2d.to_csv(os.path.join(OUT_DIR, "per_slice_boxes.csv"), index=False)
    
    df3d = pd.DataFrame(rows_3d)
    df3d.to_csv(os.path.join(OUT_DIR, "bbox3d_by_subject.csv"), index=False)
    
    # Save timing statistics
    if timing_info:
        df_timing = pd.DataFrame(timing_info)
        timing_csv_path = os.path.join(OUT_DIR, "cropping_times.csv")
        df_timing.to_csv(timing_csv_path, index=False)
        
        avg_bbox_time = df_timing['bbox_calc_time'].mean()
        avg_save_time = df_timing['file_save_time'].mean()
        avg_total_time = df_timing['total_cropping_time'].mean()
        
        print("\n" + "="*50)
        print("Cropping Time Statistics:")
        print(f"  Average BBox calculation: {avg_bbox_time:.4f}s")
        print(f"  Average file save:        {avg_save_time:.4f}s")
        print(f"  Average total cropping:   {avg_total_time:.4f}s")
        print(f"  Total processing time:    {df_timing['total_cropping_time'].sum():.2f}s")
    
    print("\n" + "="*50)
    print(f"Total files processed: {len(nifti_list)}")
    print(f"Successfully cropped: {success_count}")
    print(f"Saved: {os.path.join(OUT_DIR, 'per_slice_boxes.csv')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'bbox3d_by_subject.csv')}")
    if timing_info:
        print(f"Saved: {timing_csv_path}")
    if SAVE_CROPPED:
        print(f"Cropped volumes: {os.path.join(OUT_DIR, 'cropped_volumes')}")

if __name__ == "__main__":
    main()