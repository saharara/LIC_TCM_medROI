import os
import sys
import numpy as np
import nibabel as nib
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import json
import argparse
from pathlib import Path
import logging
import time
import pandas as pd
import glob
import torch
import torch.nn.functional as F
from torchvision import transforms
from models import TCM
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========== ROI Metadata Management ==========

class ROIMetadata:
    """
    Metadata class for ROI-based compression.
    
    Stores bounding box coordinates, original volume shape, and affine matrix
    to enable reconstruction of ROI-compressed volumes.
    """
    
    BBOX_SIZE = 12      # 6 values × int16
    SHAPE_SIZE = 6      # 3 values × int16
    AFFINE_SIZE = 36    # 9 values (3×3) × float32
    TOTAL_SIZE = BBOX_SIZE + SHAPE_SIZE + AFFINE_SIZE  # 54 bytes
    
    def __init__(self, bbox=None, original_shape=None, affine_matrix=None):
        """
        Args:
            bbox: dict with keys {x_min, x_max, y_min, y_max, z_min, z_max}
            original_shape: tuple (width, height, depth)
            affine_matrix: 4×4 or 3×3 numpy array
        """
        self.bbox = bbox
        self.original_shape = original_shape
        self.affine_matrix = affine_matrix
    
    def to_bytes(self):
        """Serialize metadata to bytes"""
        data = b''
        
        # Bounding Box (12 bytes)
        if self.bbox:
            bbox_array = np.array([
                self.bbox['x_min'], self.bbox['x_max'],
                self.bbox['y_min'], self.bbox['y_max'],
                self.bbox['z_min'], self.bbox['z_max']
            ], dtype=np.int16)
            data += bbox_array.tobytes()
        else:
            data += np.zeros(6, dtype=np.int16).tobytes()
        
        # Original Shape (6 bytes)
        if self.original_shape:
            shape_array = np.array(self.original_shape, dtype=np.int16)
            data += shape_array.tobytes()
        else:
            data += np.zeros(3, dtype=np.int16).tobytes()
        
        # Affine Matrix (36 bytes, 3×3 only)
        if self.affine_matrix is not None:
            if self.affine_matrix.shape == (4, 4):
                affine_3x3 = self.affine_matrix[:3, :3].astype(np.float32)
            elif self.affine_matrix.shape == (3, 3):
                affine_3x3 = self.affine_matrix.astype(np.float32)
            else:
                affine_3x3 = np.eye(3, dtype=np.float32)
            data += affine_3x3.tobytes()
        else:
            data += np.eye(3, dtype=np.float32).tobytes()
        
        assert len(data) == self.TOTAL_SIZE
        return data
    
    @classmethod
    def from_bytes(cls, data):
        """Deserialize metadata from bytes"""
        assert len(data) == cls.TOTAL_SIZE
        
        offset = 0
        
        # Bounding Box
        bbox_array = np.frombuffer(data[offset:offset+cls.BBOX_SIZE], dtype=np.int16)
        bbox = {
            'x_min': int(bbox_array[0]), 'x_max': int(bbox_array[1]),
            'y_min': int(bbox_array[2]), 'y_max': int(bbox_array[3]),
            'z_min': int(bbox_array[4]), 'z_max': int(bbox_array[5])
        }
        offset += cls.BBOX_SIZE
        
        # Original Shape
        shape_array = np.frombuffer(data[offset:offset+cls.SHAPE_SIZE], dtype=np.int16)
        original_shape = tuple(int(x) for x in shape_array)
        offset += cls.SHAPE_SIZE
        
        # Affine Matrix
        affine_3x3 = np.frombuffer(data[offset:offset+cls.AFFINE_SIZE], 
                                   dtype=np.float32).reshape(3, 3)
        affine_4x4 = np.eye(4, dtype=np.float32)
        affine_4x4[:3, :3] = affine_3x3
        
        return cls(bbox=bbox, original_shape=original_shape, affine_matrix=affine_4x4)
    
    def save(self, filepath):
        """Save metadata to file"""
        with open(filepath, 'wb') as f:
            f.write(self.to_bytes())
    
    @classmethod
    def load(cls, filepath):
        """Load metadata from file"""
        with open(filepath, 'rb') as f:
            data = f.read()
        return cls.from_bytes(data)


# ========== Image Processing ==========

class BrainImageProcessor:
    """Brain image preprocessing and normalization"""
    
    def __init__(self, normalize_method='minmax'):
        self.normalize_method = normalize_method
        self.global_min = None
        self.global_max = None
    
    def load_nifti(self, nii_path):
        """Load NIfTI file"""
        try:
            img = nib.load(nii_path)
            data = img.get_fdata()
            logger.info(f"Loaded: {nii_path}, Shape: {data.shape}")
            return data, img.header, img.affine
        except Exception as e:
            logger.error(f"Error loading {nii_path}: {e}")
            return None, None, None
    
    def set_normalization_range(self, data):
        """Set global normalization range based on original volume"""
        self.global_min = np.min(data)
        self.global_max = np.max(data)
        logger.info(f"Normalization range: [{self.global_min:.2f}, {self.global_max:.2f}]")
    
    def normalize_image(self, image_data, use_global=False):
        """Normalize image to [0, 255] range"""
        if use_global and self.global_min is not None:
            min_val, max_val = self.global_min, self.global_max
        else:
            min_val, max_val = np.min(image_data), np.max(image_data)
        
        if max_val > min_val:
            return ((image_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        return np.zeros_like(image_data, dtype=np.uint8)
    
    def extract_slices(self, volume_data, axis=2, max_slices=None, use_global_norm=False):
        """Extract 2D slices from 3D volume"""
        slice_count = volume_data.shape[axis]
        
        if max_slices is None or max_slices >= slice_count:
            indices = range(slice_count)
        else:
            step = max(1, slice_count // max_slices)
            indices = range(slice_count // 4, 3 * slice_count // 4, step)[:max_slices]
        
        slices = []
        for i in indices:
            slice_data = np.take(volume_data, i, axis=axis)
            slices.append(self.normalize_image(slice_data, use_global=use_global_norm))
        
        logger.info(f"Extracted {len(slices)} slices from axis {axis}")
        return slices, list(indices)


# ========== TCM Compression ==========

class TCMCompressor:
    """TCM model compression wrapper"""
    
    def __init__(self, checkpoint_path, device='cpu', real_mode=True):
        self.device = device
        self.real_mode = real_mode
        self.padding_size = 128
        
        logger.info(f"Loading TCM model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["state_dict"]

        # Remove 'module.' prefix
        clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # Model parameters (from pretrained model specifications)
        self.N = 64
        self.M = 320
        logger.info(f"Model parameters: N={self.N}, M={self.M}")

        # Create model
        self.net = TCM(
            config=[2,2,2,2,2,2],
            head_dim=[8, 16, 32, 32, 16, 8],
            drop_path_rate=0.0,
            N=self.N,
            M=self.M,
        )

        self.net.load_state_dict(clean_state_dict)
        self.net = self.net.to(device)
        self.net.eval()

        if real_mode:
            logger.info("Updating entropy models for real compression...")
            self.net.update()
    
    def pad(self, x):
        """Pad image to multiple of padding_size"""
        p = self.padding_size
        h, w = x.size(2), x.size(3)
        new_h = (h + p - 1) // p * p
        new_w = (w + p - 1) // p * p
        padding_left = (new_w - w) // 2
        padding_right = new_w - w - padding_left
        padding_top = (new_h - h) // 2
        padding_bottom = new_h - h - padding_top
        x_padded = F.pad(x, (padding_left, padding_right, padding_top, padding_bottom),
                        mode="constant", value=0)
        return x_padded, (padding_left, padding_right, padding_top, padding_bottom)
    
    def crop(self, x, padding):
        """Remove padding"""
        return F.pad(x, (-padding[0], -padding[1], -padding[2], -padding[3]))
    
    def compress_image(self, image_array):
        """Compress single 2D image (grayscale → RGB)"""
        start_time = time.time()
        
        # Grayscale → RGB conversion
        if len(image_array.shape) == 2:
            rgb_array = np.stack([image_array] * 3, axis=-1)
        else:
            rgb_array = image_array
        
        # PIL → Tensor
        pil_img = Image.fromarray(rgb_array.astype(np.uint8))
        x = transforms.ToTensor()(pil_img).unsqueeze(0).to(self.device)
        
        # Padding
        x_padded, padding = self.pad(x)
        
        with torch.no_grad():
            if self.real_mode:
                # Real compression with entropy coding
                out_enc = self.net.compress(x_padded)
                comp_time = time.time() - start_time
                
                compressed_size = sum(len(s[0]) for s in out_enc["strings"]) 
                
                return {
                    'compressed_data': out_enc,
                    'padding': padding,
                    'compressed_size': compressed_size,
                    'original_shape': image_array.shape
                }, comp_time
            else:
                # Forward pass without entropy coding
                out_net = self.net.forward(x_padded)
                comp_time = time.time() - start_time
                
                size = out_net['x_hat'].size()
                num_pixels = size[0] * size[2] * size[3]
                bpp = sum(torch.log(likelihoods).sum() / (-np.log(2) * num_pixels)
                         for likelihoods in out_net['likelihoods'].values()).item()
                compressed_size = int(bpp * num_pixels / 8)
                
                return {
                    'compressed_data': out_net,
                    'padding': padding,
                    'compressed_size': compressed_size,
                    'original_shape': image_array.shape
                }, comp_time
    
    def decompress_image(self, compressed_data):
        """Decompress image"""
        start_time = time.time()
        
        with torch.no_grad():
            if self.real_mode:
                out_dec = self.net.decompress(
                    compressed_data['compressed_data']["strings"],
                    compressed_data['compressed_data']["shape"]
                )
                x_hat = self.crop(out_dec["x_hat"], compressed_data['padding'])
            else:
                out_net = compressed_data['compressed_data']
                out_net['x_hat'].clamp_(0, 1)
                x_hat = self.crop(out_net["x_hat"], compressed_data['padding'])
            
            decomp_time = time.time() - start_time
            
            # Tensor → numpy (first channel only)
            reconstructed = (x_hat[0, 0].cpu().numpy() * 255).astype(np.uint8)
            
            return reconstructed, decomp_time


# ========== Quality Evaluation ==========

class CompressionEvaluator:
    """Compression quality evaluator"""
    
    @staticmethod
    def calculate_metrics(original, reconstructed):
        """Calculate image quality metrics"""
        mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
        
        if mse == 0:
            psnr = 100.0
        else:
            psnr = 20 * np.log10(255 / np.sqrt(mse))
        
        ssim_val = ssim(original, reconstructed, data_range=255)
        
        return mse, psnr, ssim_val


# ========== Utility Functions ==========

def pad_or_crop_to_shape(img_u8, target_hw):
    """
    Center crop/pad image to target shape.
    
    Args:
        img_u8: (H, W) uint8 array
        target_hw: (target_h, target_w) tuple
        
    Returns:
        img_u8: resized array
    """
    h, w = img_u8.shape
    th, tw = target_hw

    # Center crop if larger
    if h > th:
        top = (h - th) // 2
        img_u8 = img_u8[top:top + th, :]
        h = th
    if w > tw:
        left = (w - tw) // 2
        img_u8 = img_u8[:, left:left + tw]
        w = tw

    # Zero pad if smaller
    pad_top = max(0, (th - h) // 2)
    pad_bottom = max(0, th - h - pad_top)
    pad_left = max(0, (tw - w) // 2)
    pad_right = max(0, tw - w - pad_left)

    if pad_top or pad_bottom or pad_left or pad_right:
        img_u8 = np.pad(img_u8, ((pad_top, pad_bottom), (pad_left, pad_right)),
                       mode="constant", constant_values=0)

    return img_u8[:th, :tw]


def load_bbox_info(bbox_csv_path):
    """Load 3D bounding box information from CSV"""
    if not os.path.exists(bbox_csv_path):
        logger.warning(f"bbox CSV not found: {bbox_csv_path}")
        return {}
    
    try:
        df = pd.read_csv(bbox_csv_path)
        bbox_dict = {}
        
        for _, row in df.iterrows():
            subject_id = row['subject_id']
            bbox_dict[subject_id] = {
                'x_min': int(row['x_min']),
                'x_max': int(row['x_max']),
                'y_min': int(row['y_min']),
                'y_max': int(row['y_max']),
                'z_min': int(row['z_min']),
                'z_max': int(row['z_max']),
                'original_shape': (int(row['orig_x']), int(row['orig_y']), int(row['orig_z']))
            }
        
        logger.info(f"Loaded bbox info for {len(bbox_dict)} subjects")
        return bbox_dict
        
    except Exception as e:
        logger.error(f"Failed to load bbox CSV: {e}")
        return {}


def find_file_pairs(cropped_dir, original_dir):
    """Match cropped files with original files"""
    cropped_files = sorted(glob.glob(os.path.join(cropped_dir, '*.nii*')))
    
    file_pairs = []
    for cropped_path in cropped_files:
        basename = os.path.basename(cropped_path)
        subject_id = basename.replace('.nii.gz', '').replace('.nii', '')
        subject_id = subject_id.replace('_cropped', '').replace('_roi', '')
        
        possible_extensions = ['.nii', '.nii.gz']
        original_path = None
        
        for ext in possible_extensions:
            candidate = os.path.join(original_dir, f"{subject_id}{ext}")
            if os.path.exists(candidate):
                original_path = candidate
                break
        
        if original_path is None:
            logger.warning(f"Original not found for: {basename}")
            file_pairs.append((cropped_path, None))
        else:
            file_pairs.append((cropped_path, original_path))
    
    return file_pairs


# ========== Main Experiment Class ==========

class TCMExperiment:
    """TCM compression experiment manager"""
    
    def __init__(self, checkpoint_path, device='cpu', real_mode=True):
        self.processor = BrainImageProcessor()
        self.compressor = TCMCompressor(checkpoint_path, device, real_mode)
        self.evaluator = CompressionEvaluator()
    
    def run_experiment(self, cropped_nifti_path, output_dir,
                      original_nifti_path=None,
                      cropping_time_info=None,
                      bbox_info=None,
                      axis=2,
                      target_slice=None,
                      pad_to_original=True):
        """
        Run compression experiment on NIfTI volume.
        
        Args:
            cropped_nifti_path: path to cropped NIfTI file
            output_dir: output directory
            original_nifti_path: path to original NIfTI file
            cropping_time_info: cropping time information
            bbox_info: bounding box information
            axis: slice axis (0=sagittal, 1=coronal, 2=axial)
            target_slice: specific slice to process (None for all)
            pad_to_original: pad reconstructed to original shape
            
        Returns:
            dict: experiment results
        """
        cropped_path = Path(cropped_nifti_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize variables
        affine_matrix = None
        original_file_size = None
        original_volume = None
        original_slices = None
        roi_metadata = None
        
        # Load original volume first (for normalization range)
        if original_nifti_path and os.path.exists(original_nifti_path):
            original_file_size = os.path.getsize(original_nifti_path)
            logger.info(f"Original file size: {original_file_size:,} bytes")
            
            original_volume, _, affine_matrix = self.processor.load_nifti(original_nifti_path)
            
            if original_volume is not None:
                self.processor.set_normalization_range(original_volume)
                
                original_slices, original_indices = self.processor.extract_slices(
                    original_volume, axis=axis, max_slices=None, use_global_norm=True
                )
                
                # Create ROI metadata (only for ROI compression)
                if bbox_info and affine_matrix is not None:
                    roi_metadata = ROIMetadata(
                        bbox=bbox_info,
                        original_shape=original_volume.shape,
                        affine_matrix=affine_matrix
                    )
                    logger.info(f"ROI Metadata created")
        
        # Load cropped ROI data
        cropped_volume, _, _ = self.processor.load_nifti(cropped_path)
        if cropped_volume is None:
            return None
        
        cropped_file_size = os.path.getsize(cropped_path)
        logger.info(f"Cropped file size: {cropped_file_size:,} bytes")
        
        use_global = (self.processor.global_min is not None)
        
        cropped_slices, crop_indices = self.processor.extract_slices(
            cropped_volume, axis=axis, max_slices=None, use_global_norm=use_global
        )
        
        # Calculate 2D bbox
        bbox_2d = None
        if bbox_info:
            if axis == 0:
                bbox_2d = (bbox_info['y_min'], bbox_info['y_max'],
                          bbox_info['z_min'], bbox_info['z_max'])
            elif axis == 1:
                bbox_2d = (bbox_info['x_min'], bbox_info['x_max'],
                          bbox_info['z_min'], bbox_info['z_max'])
            else:  # axis == 2
                bbox_2d = (bbox_info['x_min'], bbox_info['x_max'],
                          bbox_info['y_min'], bbox_info['y_max'])
        
        # Calculate slice index offset
        z_offset = 0
        if bbox_info and axis == 2:
            z_offset = bbox_info['z_min']
        elif bbox_info and axis == 0:
            z_offset = bbox_info['x_min']
        elif bbox_info and axis == 1:
            z_offset = bbox_info['y_min']
        
        # Handle cropping time
        if isinstance(cropping_time_info, dict):
            avg_cropping_time = cropping_time_info.get('cropping_time', 0)
        elif isinstance(cropping_time_info, (int, float)):
            avg_cropping_time = cropping_time_info
        else:
            avg_cropping_time = 0
        
        results = {
            'filename': cropped_path.name,
            'model': f'TCM_N{self.compressor.N}_M{self.compressor.M}',
            'original_nifti_path': str(original_nifti_path) if original_nifti_path else 'N/A',
            'cropped_shape': cropped_volume.shape,
            'slice_count': len(cropped_slices),
            'avg_cropping_time': avg_cropping_time,
            'bbox_info': bbox_info,
            'has_roi_metadata': roi_metadata is not None,
            'original_file_size': original_file_size,
            'cropped_file_size': cropped_file_size,
            'real_mode': self.compressor.real_mode
        }
        
        # Process slices
        compressed_slices_data = []
        slice_metrics = []
        total_comp_time = 0
        total_decomp_time = 0
        total_compressed_size = 0
        

        # Process each slice
        for i, cropped_slice in enumerate(cropped_slices):
            try:
                # Calculate original slice index
                if original_slices is not None:
                    if bbox_2d:
                        original_slice_idx = z_offset + i
                    else:
                        original_slice_idx = crop_indices[i] if i < len(crop_indices) else i
                else:
                    original_slice_idx = crop_indices[i] if i < len(crop_indices) else i

                # Filter for target slice if specified
                if target_slice is not None and original_slice_idx != target_slice:
                    continue

                if target_slice is not None:
                    logger.info(f"Processing target slice: {target_slice}")

                # Compression
                compressed, comp_time = self.compressor.compress_image(cropped_slice)
                compressed_slices_data.append(compressed)
                total_comp_time += comp_time
                total_compressed_size += compressed['compressed_size']

                # Decompression
                reconstructed, decomp_time = self.compressor.decompress_image(compressed)
                total_decomp_time += decomp_time

                # Quality evaluation
                mse, psnr, ssim_val = None, None, None
                restored_full = None
                original_full_slice = None

                if original_slices is not None and original_slice_idx < len(original_slices):
                    original_full_slice = original_slices[original_slice_idx]

                    if bbox_2d:
                        # ROI mode: create full-size restored slice
                        restored_full = np.zeros_like(original_full_slice, dtype=np.uint8)

                        x_min, x_max, y_min, y_max = bbox_2d
                        roi_h = x_max - x_min
                        roi_w = y_max - y_min

                        roi_patch = reconstructed.astype(np.uint8)
                        roi_patch = pad_or_crop_to_shape(roi_patch, (roi_h, roi_w))

                        restored_full[x_min:x_max, y_min:y_max] = roi_patch

                        mse, psnr, ssim_val = CompressionEvaluator.calculate_metrics(
                            original_full_slice, restored_full
                        )

                    else:
                        # Original-only mode
                        recon_u8 = reconstructed.astype(np.uint8)
                        if pad_to_original:
                            recon_u8 = pad_or_crop_to_shape(recon_u8, original_full_slice.shape)

                        mse, psnr, ssim_val = CompressionEvaluator.calculate_metrics(
                            original_full_slice, recon_u8
                        )
                        restored_full = recon_u8

                # Record metrics
                if mse is not None:
                    slice_metrics.append({
                        'slice_index': original_slice_idx,
                        'mse': float(mse),
                        'psnr': float(psnr),
                        'ssim': float(ssim_val),
                        'compressed_size': int(compressed['compressed_size']),
                        'success': True,
                    })
                    logger.info(
                        f"Slice {original_slice_idx}: "
                        f"PSNR={psnr:.2f}dB, SSIM={ssim_val:.4f}, "
                        f"Size={compressed['compressed_size']} bytes"
                    )
                else:
                    slice_metrics.append({
                        'slice_index': original_slice_idx,
                        'success': False,
                        'error': 'No original slice',
                    })

            except Exception as e:
                logger.warning(f"Slice {i} failed: {e}")
                slice_metrics.append({
                    'slice_index': crop_indices[i] if i < len(crop_indices) else i,
                    'success': False,
                    'error': str(e)
                })

        if len(slice_metrics) == 0 and target_slice is not None:
            logger.warning(f"Target slice {target_slice} was not processed")
            
        # Save ROI metadata if exists
        if roi_metadata is not None:
            metadata_path = output_dir / 'roi_metadata.bin'
            roi_metadata.save(metadata_path)
            logger.info(f"ROI Metadata saved: {metadata_path}")
        
        # Calculate volume metrics
        volume_metrics = {}
        
        metadata_size = ROIMetadata.TOTAL_SIZE if roi_metadata is not None else 0
        total_compressed_size_with_metadata = total_compressed_size + metadata_size
        
        if original_file_size and total_compressed_size > 0:
            volume_compression_ratio_file = original_file_size / total_compressed_size_with_metadata
            
            if original_slices:
                original_memory_size = sum(s.size * s.itemsize for s in original_slices)
                volume_compression_ratio_memory = original_memory_size / total_compressed_size_with_metadata
            else:
                volume_compression_ratio_memory = None
            
            # BPP calculation
            if bbox_info:
                if original_volume is not None:
                    total_pixels = (original_volume.shape[0] *
                                  original_volume.shape[1] *
                                  original_volume.shape[2])
                else:
                    total_pixels = bbox_info['original_shape'][0] * \
                                 bbox_info['original_shape'][1] * \
                                 bbox_info['original_shape'][2]
            else:
                total_pixels = (cropped_volume.shape[0] *
                              cropped_volume.shape[1] *
                              cropped_volume.shape[2])

            bpp = (total_compressed_size_with_metadata * 8) / total_pixels
            
            volume_metrics = {
                'total_compressed_size': total_compressed_size,
                'metadata_size': metadata_size,
                'total_compressed_size_with_metadata': total_compressed_size_with_metadata,
                'compression_ratio_file': volume_compression_ratio_file,
                'compression_ratio_memory': volume_compression_ratio_memory,
                'bpp': bpp,
                'total_compression_time': total_comp_time,
                'total_decompression_time': total_decomp_time,
                'avg_compression_time_per_slice': total_comp_time / len(cropped_slices) if cropped_slices else 0,
                'avg_decompression_time_per_slice': total_decomp_time / len(cropped_slices) if cropped_slices else 0
            }
            
            logger.info(f"\n[VOLUME METRICS]")
            logger.info(f"  Compression ratio: {volume_compression_ratio_file:.2f}:1")
            logger.info(f"  BPP: {bpp:.4f}")
            logger.info(f"  Metadata: {metadata_size} bytes")
        
        # Calculate average quality metrics
        successful = [m for m in slice_metrics if m.get('success', False)]
        if successful:
            quality_metrics = {
                'success_rate': len(successful) / len(cropped_slices),
                'avg_mse': np.mean([m['mse'] for m in successful]),
                'avg_psnr': np.mean([m['psnr'] for m in successful]),
                'avg_ssim': np.mean([m['ssim'] for m in successful])
            }
            
            logger.info(f"\n[QUALITY METRICS]")
            logger.info(f"  Avg PSNR: {quality_metrics['avg_psnr']:.2f} dB")
            logger.info(f"  Avg SSIM: {quality_metrics['avg_ssim']:.4f}")
        else:
            quality_metrics = {'success_rate': 0}
        
        results['slice_metrics'] = slice_metrics
        results['quality_metrics'] = quality_metrics
        results['volume_metrics'] = volume_metrics
        
        # Save results
        result_file = output_dir / f"{cropped_path.stem}_results.json"
        with open(result_file, 'w') as f:
            results_to_save = results.copy()
            results_to_save['slice_metrics'] = [
                {k: v for k, v in m.items() if k != 'compressed_data'}
                for m in slice_metrics
            ]
            json.dump(results_to_save, f, indent=2, default=str)
        
        logger.info(f"\nResults saved to {result_file}")
        return results
    
    def create_report(self, results, output_dir):
        """Generate CSV report"""
        output_dir = Path(output_dir)
        
        quality_metrics = results.get('quality_metrics', {})
        volume_metrics = results.get('volume_metrics', {})
        
        csv_data = [{
            'Model': results['model'],
            'Has_ROI_Metadata': results.get('has_roi_metadata', False),
            'Success_Rate': f"{quality_metrics.get('success_rate', 0):.2%}",
            'Avg_PSNR_dB': f"{quality_metrics.get('avg_psnr', 0):.2f}",
            'Avg_SSIM': f"{quality_metrics.get('avg_ssim', 0):.4f}",
            'Compression_Ratio_File': f"{volume_metrics.get('compression_ratio_file', 0):.2f}:1",
            'BPP': f"{volume_metrics.get('bpp', 0):.4f}",
            'Compressed_Size_KB': f"{volume_metrics.get('total_compressed_size', 0)/1024:.2f}",
            'Metadata_Size_B': volume_metrics.get('metadata_size', 0),
            'Total_Compressed_KB': f"{volume_metrics.get('total_compressed_size_with_metadata', 0)/1024:.2f}",
        }]
        
        df = pd.DataFrame(csv_data)
        csv_file = output_dir / f"{results['filename']}_summary.csv"
        df.to_csv(csv_file, index=False)
        
        logger.info(f"\nReport saved to {csv_file}")
        
        return df


# ========== Batch Processing Functions ==========

def batch_process(checkpoint_path, cropped_dir, original_dir, output_dir,
                 bbox_csv=None, max_files=None, device='cpu', real_mode=True,
                 target_slice=None):
    """
    Batch process ROI compressed files.
    
    Args:
        checkpoint_path: path to TCM checkpoint
        cropped_dir: directory with cropped ROI files
        original_dir: directory with original files
        output_dir: output directory
        bbox_csv: path to bbox CSV file
        max_files: maximum number of files to process
        device: 'cpu' or 'cpu'
        real_mode: use real compression with entropy coding
    """
    file_pairs = find_file_pairs(cropped_dir, original_dir)
    
    if not file_pairs:
        logger.error(f"No file pairs found")
        return
    
    if max_files:
        file_pairs = file_pairs[:max_files]
    
    bbox_dict = {}
    if bbox_csv:
        bbox_dict = load_bbox_info(bbox_csv)
    
    experiment = TCMExperiment(checkpoint_path, device, real_mode)
    results_list = []
    all_metrics = []
    
    for i, (cropped_path, original_path) in enumerate(file_pairs, 1):
        basename = os.path.basename(cropped_path)
        logger.info(f"\n[{i}/{len(file_pairs)}] Processing: {basename}")
        
        subject_id = basename.replace('.nii.gz', '').replace('.nii', '')
        subject_id = subject_id.replace('_cropped', '').replace('_roi', '')
        
        bbox_info = bbox_dict.get(subject_id)
        
        file_output_dir = Path(output_dir) / Path(cropped_path).stem
        
        try:
            results = experiment.run_experiment(
                cropped_nifti_path=cropped_path,
                output_dir=file_output_dir,
                original_nifti_path=original_path,
                bbox_info=bbox_info,
                target_slice=target_slice
            )
            
            if results:
                experiment.create_report(results, file_output_dir)
                
                quality = results.get('quality_metrics', {})
                volume = results.get('volume_metrics', {})
                
                if quality.get('success_rate', 0) > 0:
                    all_metrics.append({
                        'filename': basename,
                        'has_roi_metadata': results.get('has_roi_metadata', False),
                        'success_rate': quality.get('success_rate', 0),
                        'avg_psnr': quality.get('avg_psnr'),
                        'avg_ssim': quality.get('avg_ssim'),
                        'compression_ratio_file': volume.get('compression_ratio_file'),
                        'bpp': volume.get('bpp'),
                        'compressed_size': volume.get('total_compressed_size'),
                        'metadata_size': volume.get('metadata_size', 0),
                        'total_compressed_size': volume.get('total_compressed_size_with_metadata'),
                    })
                
                results_list.append({
                    'filename': basename,
                    'status': 'success',
                    'has_original': original_path is not None
                })
            else:
                results_list.append({'filename': basename, 'status': 'failed'})
                
        except Exception as e:
            results_list.append({
                'filename': basename,
                'status': 'error',
                'error': str(e)
            })
            logger.error(f"Error: {e}")
    
    # Calculate and save overall average
    if all_metrics:
        def safe_mean(key):
            values = [m[key] for m in all_metrics if m[key] is not None]
            return np.mean(values) if values else None
        
        overall_avg = {
            'Model': f'TCM_N{experiment.compressor.N}_M{experiment.compressor.M}',
            'Has ROI Metadata': any(m['has_roi_metadata'] for m in all_metrics),
            'Avg PSNR (dB)': f"{safe_mean('avg_psnr'):.2f}" if safe_mean('avg_psnr') else "N/A",
            'Avg SSIM': f"{safe_mean('avg_ssim'):.4f}" if safe_mean('avg_ssim') else "N/A",
            'Compression Ratio': f"{safe_mean('compression_ratio_file'):.2f}:1" if safe_mean('compression_ratio_file') else "N/A",
            'BPP': f"{safe_mean('bpp'):.4f}" if safe_mean('bpp') else "N/A",
            'Metadata Size (B)': f"{safe_mean('metadata_size'):.0f}" if safe_mean('metadata_size') else "N/A",
        }
        
        overall_df = pd.DataFrame([overall_avg])
        overall_file = Path(output_dir) / "overall_average.csv"
        overall_df.to_csv(overall_file, index=False)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"OVERALL AVERAGE (n={len(all_metrics)} subjects)")
        logger.info(f"{'='*80}")
        logger.info(f"\n{overall_df.to_string(index=False)}")
    
    successful = sum(1 for r in results_list if r['status'] == 'success')
    logger.info(f"\nBatch processing completed!")
    logger.info(f"Success: {successful}/{len(file_pairs)}")


def batch_process_original_only(checkpoint_path, original_dir, output_dir,
                                max_files=None, device='cpu', real_mode=True,
                                axis=2, target_slice=None):
    """Batch process original data without cropping"""
    original_files = sorted(glob.glob(os.path.join(original_dir, '*.nii*')))
    
    if not original_files:
        logger.error(f"No NIfTI files found in {original_dir}")
        return
    
    if max_files:
        original_files = original_files[:max_files]
    
    logger.info(f"Found {len(original_files)} files to process")
    
    experiment = TCMExperiment(checkpoint_path, device, real_mode)
    all_metrics = []
    
    for i, original_path in enumerate(original_files, 1):
        basename = os.path.basename(original_path)
        logger.info(f"\n[{i}/{len(original_files)}] Processing: {basename}")
        
        file_output_dir = Path(output_dir) / Path(original_path).stem
        
        try:
            results = experiment.run_experiment(
                cropped_nifti_path=original_path,
                output_dir=file_output_dir,
                original_nifti_path=original_path,
                cropping_time_info=0,
                axis=axis,
                target_slice=target_slice
            )
            
            if results:
                experiment.create_report(results, file_output_dir)
                
                quality = results.get('quality_metrics', {})
                volume = results.get('volume_metrics', {})
                
                if quality.get('success_rate', 0) > 0:
                    all_metrics.append({
                        'filename': basename,
                        'avg_psnr': quality.get('avg_psnr'),
                        'avg_ssim': quality.get('avg_ssim'),
                        'compression_ratio_file': volume.get('compression_ratio_file'),
                        'bpp': volume.get('bpp'),
                    })
                
        except Exception as e:
            logger.error(f"Error: {e}")
    
    if all_metrics:
        def safe_mean(key):
            values = [m[key] for m in all_metrics if m[key] is not None]
            return np.mean(values) if values else None
        
        overall_avg = {
            'Model': f'TCM_N{experiment.compressor.N}_M{experiment.compressor.M}',
            'Avg PSNR (dB)': f"{safe_mean('avg_psnr'):.2f}" if safe_mean('avg_psnr') else "N/A",
            'Avg SSIM': f"{safe_mean('avg_ssim'):.4f}" if safe_mean('avg_ssim') else "N/A",
            'Compression Ratio': f"{safe_mean('compression_ratio_file'):.2f}:1" if safe_mean('compression_ratio_file') else "N/A",
            'BPP': f"{safe_mean('bpp'):.4f}" if safe_mean('bpp') else "N/A",
        }
        
        overall_df = pd.DataFrame([overall_avg])
        overall_file = Path(output_dir) / "overall_average.csv"
        overall_df.to_csv(overall_file, index=False)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"OVERALL AVERAGE (n={len(all_metrics)} subjects)")
        logger.info(f"{'='*80}")
        logger.info(f"\n{overall_df.to_string(index=False)}")


# ========== Command Line Interface ==========

def main():
    parser = argparse.ArgumentParser(
        description='TCM Brain Image Compression Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ROI compression evaluation
  python eval.py \\
      --checkpoint ./model.pth.tar \\
      --cropped_dir ./cropped_volumes \\
      --original_dir ./original_nifti \\
      --bbox_csv ./bbox_info.csv \\
      --output ./results \\
      --real_mode

  # Original data evaluation
  python eval.py \\
      --checkpoint ./model.pth.tar \\
      --use_original_only \\
      --original_dir ./original_nifti \\
      --output ./results \\
      --real_mode
        """
    )
    
    # Required arguments
    parser.add_argument('--checkpoint', required=True,
                       help='TCM checkpoint path')
    parser.add_argument('--output', required=True,
                       help='Output directory')
    
    # Data paths
    parser.add_argument('--cropped_dir',
                       help='Directory with cropped ROI files')
    parser.add_argument('--original_dir',
                       help='Directory with original files')
    parser.add_argument('--bbox_csv',
                       help='CSV file with 3D bbox info')
    
    # Processing options
    parser.add_argument('--use_original_only', action='store_true',
                       help='Use original data without cropping')
    parser.add_argument('--max_files', type=int,
                       help='Maximum number of files to process')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cpu'],
                       help='Device to use')
    parser.add_argument('--real_mode', action='store_true',
                       help='Use real compression with entropy coding')
    parser.add_argument('--axis', type=int, default=2, choices=[0,1,2],
                       help='Slice axis: 0=sagittal, 1=coronal, 2=axial')
    parser.add_argument('--target_slice', type=int, default=None, help='Specific slice index')
    
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.use_original_only and args.original_dir:
        batch_process_original_only(
            checkpoint_path=args.checkpoint,
            original_dir=args.original_dir,
            output_dir=args.output,
            max_files=args.max_files,
            device=args.device,
            real_mode=args.real_mode,
            axis=args.axis,
            target_slice=args.target_slice
        )
    
    elif args.cropped_dir:
        if not args.original_dir:
            logger.error("--original_dir is required for ROI compression")
            sys.exit(1)
        
        batch_process(
            checkpoint_path=args.checkpoint,
            cropped_dir=args.cropped_dir,
            original_dir=args.original_dir,
            output_dir=args.output,
            bbox_csv=args.bbox_csv,
            max_files=args.max_files,
            device=args.device,
            real_mode=args.real_mode,
            target_slice=args.target_slice
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    logger.info("TCM Brain Image Compression Evaluation")
    logger.info("=" * 60)
    main()
