#!/usr/bin/env python3
"""
Pixel3DMM Batch Inference Script

Processes multiple images through the pixel3dmm pipeline and saves outputs in a structured format.

Usage:
    python inference_pixel3dmm.py --input_dir <path_to_images> --output_dir <path_to_output>

For each input image (e.g., person_001.png), creates:
    output_dir/person_001/
        ├── uv.png                    # UV texture map
        ├── normal.png                # Surface normal map
        ├── segmentation.png          # Face segmentation (colored)
        ├── camera_params.json        # Camera intrinsics and extrinsics
        ├── flame_parameters.json     # FLAME model parameters
        └── head_orientation.json     # Head pose (vectors + Euler angles)
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import trimesh
import pyvista as pv

# Add src to Python path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from pixel3dmm import env_paths
from pixel3dmm.utils.utils_3d import rotation_6d_to_matrix
from dreifus.matrix import Intrinsics, Pose, CameraCoordinateConvention, PoseType
from dreifus.pyvista import render_from_camera


def setup_output_dir(output_dir):
    """Create output directory if it doesn't exist."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_image_files(input_dir):
    """Get all PNG and JPG images from input directory."""
    input_dir = Path(input_dir).resolve()  # Convert to absolute path
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    image_files = []
    for ext in ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']:
        image_files.extend(input_dir.glob(ext))
    
    image_files = sorted(image_files)
    
    if not image_files:
        raise ValueError(f"No image files (PNG/JPG) found in: {input_dir}")
    
    return image_files


def run_preprocessing(image_path, output_dir):
    """Run preprocessing (face detection, cropping, MICA, segmentation)."""
    print(f"\n{'='*80}")
    print(f"STEP 1: Preprocessing {image_path.name}")
    print(f"{'='*80}")
    
    # Use absolute path for image
    image_path = Path(image_path).resolve()
    output_dir = Path(output_dir).resolve()
    
    cmd = [
        "python", "scripts/run_preprocessing.py",
        "--video_or_images_path", str(image_path),
        "--output_dir", str(output_dir)
    ]
    
    result = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Preprocessing failed for {image_path.name}")


def run_network_inference(vid_name, prediction_type, preprocessed_dir):
    """Run pixel3dmm network inference for normals or uv_map."""
    print(f"\n{'='*80}")
    print(f"STEP 2: Network Inference - {prediction_type}")
    print(f"{'='*80}")
    
    preprocessed_dir = Path(preprocessed_dir).resolve()
    
    cmd = [
        "python", "scripts/network_inference.py",
        f"model.prediction_type={prediction_type}",
        f"video_name={vid_name}",
        f"preprocessed_dir={str(preprocessed_dir)}"
    ]
    
    result = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Network inference ({prediction_type}) failed for {vid_name}")


def run_tracking(vid_name, iters, preprocessed_dir, tracking_output_dir):
    """Run pixel3dmm tracking/optimization."""
    print(f"\n{'='*80}")
    print(f"STEP 3: Tracking and Optimization")
    print(f"{'='*80}")
    
    preprocessed_dir = Path(preprocessed_dir).resolve()
    tracking_output_dir = Path(tracking_output_dir).resolve()
    
    cmd = [
        "python", "scripts/track.py",
        f"video_name={vid_name}",
        f"iters={iters}",
        f"preprocessed_dir={str(preprocessed_dir)}",
        f"tracking_output_dir={str(tracking_output_dir)}"
    ]
    
    result = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Tracking failed for {vid_name}")


def render_flame_segmentation(tracking_dir, preprocessed_dir, output_subdir):
    """
    Render FLAME mesh segmentation in the original (non-cropped) image space.
    
    Creates:
        - flame_segmentation.png: Binary mask of FLAME mesh
        - flame_overlay.png: FLAME mesh overlaid on original image
    """
    print("  - Rendering FLAME segmentation in original image space...")
    
    # Load the first mesh (frame 0)
    mesh_files = [f for f in os.listdir(tracking_dir / "mesh") if f.endswith('.ply') and 'canonical' not in f]
    if not mesh_files:
        print("    ✗ Warning: No mesh files found")
        return
    
    mesh_files.sort()
    mesh = trimesh.load(str(tracking_dir / "mesh" / mesh_files[0]), process=False)
    
    # Load checkpoint for camera and FLAME parameters
    ckpt_files = [f for f in os.listdir(tracking_dir / "checkpoint") if f.endswith('.frame')]
    ckpt_files.sort()
    ckpt = torch.load(str(tracking_dir / "checkpoint" / ckpt_files[0]), weights_only=False)
    
    # Load crop parameters to get original image size
    crop_params_file = preprocessed_dir / "crop_params" / "00000.npy"
    if not crop_params_file.exists():
        print("    ✗ Warning: Crop parameters not found")
        return
    
    crop_params = np.load(crop_params_file, allow_pickle=True).item()
    original_width, original_height = crop_params['original_size']
    crop_coords = crop_params['crop_coords']  # [x1, y1, x2, y2]
    
    # Load original RGB image
    rgb_path = preprocessed_dir / "rgb" / "00000.jpg"
    if not rgb_path.exists():
        print("    ✗ Warning: Original RGB image not found")
        return
    
    original_img = np.array(Image.open(rgb_path))
    
    # Transform mesh from FLAME space to world space
    head_rot = rotation_6d_to_matrix(torch.from_numpy(ckpt['flame']['R'])).numpy()[0]
    mesh.vertices = mesh.vertices @ head_rot.T + ckpt['flame']['t']
    
    # Build camera extrinsics (world to camera)
    extr_world_to_cam = np.eye(4)
    extr_world_to_cam[:3, :3] = ckpt['camera']['R_base_0'][0]
    extr_world_to_cam[:3, 3] = ckpt['camera']['t_base_0'][0]
    
    extr_world_to_cam = Pose(
        extr_world_to_cam,
        camera_coordinate_convention=CameraCoordinateConvention.OPEN_GL,
        pose_type=PoseType.WORLD_2_CAM
    )
    
    # Build camera intrinsics for the CROPPED image (512x512)
    intr_cropped = np.eye(3)
    intr_cropped[0, 0] = ckpt['camera']['fl'][0, 0] * 256  # focal length
    intr_cropped[1, 1] = ckpt['camera']['fl'][0, 0] * 256
    intr_cropped[:2, 2] = ckpt['camera']['pp'][0] * (256/2 + 0.5) + 256/2 + 0.5  # principal point
    
    # Scale intrinsics from 256x256 to 512x512 (cropped image size)
    scale_factor = 512.0 / 256.0
    intr_cropped_512 = intr_cropped * scale_factor
    intr_cropped_512[2, 2] = 1.0
    
    # Adjust intrinsics for the original image size
    # The crop operation changes the principal point
    x1, y1, x2, y2 = crop_coords
    crop_width = x2 - x1
    crop_height = y2 - y1
    
    # The cropped region was resized to 512x512, potentially with padding if not square
    # We need to account for this padding
    if crop_width != crop_height:
        # Padding was added to make it square
        max_dim = max(crop_width, crop_height)
        if crop_width < crop_height:
            # Padding on left/right
            x_padding = (max_dim - crop_width) // 2
            y_padding = 0
        else:
            # Padding on top/bottom
            x_padding = 0
            y_padding = (max_dim - crop_height) // 2
    else:
        x_padding = 0
        y_padding = 0
        max_dim = crop_width
    
    # Scale from 512x512 to the padded crop size
    scale_to_padded = max_dim / 512.0
    intr_full = intr_cropped_512.copy()
    intr_full[:2, :] *= scale_to_padded
    
    # Adjust principal point: account for padding removal, then add crop offset
    intr_full[0, 2] -= x_padding  # Remove x padding offset
    intr_full[1, 2] -= y_padding  # Remove y padding offset
    intr_full[0, 2] += x1  # Add crop x offset
    intr_full[1, 2] += y1  # Add crop y offset
    
    intr_full_obj = Intrinsics(intr_full)
    
    # Render FLAME mesh at original image resolution
    pll = pv.Plotter(off_screen=True, window_size=(original_width, original_height))
    pll.add_mesh(mesh, color='white')
    rendered = render_from_camera(pll, extr_world_to_cam, intr_full_obj)
    
    # Extract alpha channel as binary mask
    alpha = rendered[..., 3]
    flame_mask = (alpha > 0).astype(np.uint8) * 255
    
    # Save binary segmentation
    Image.fromarray(flame_mask).save(output_subdir / "flame_segmentation.png")
    print(f"    ✓ Saved: flame_segmentation.png")
    
    # Create overlay (50% original image + 50% rendered mesh)
    overlay = original_img.copy()
    mask_bool = alpha > 0
    if rendered.shape[:2] == original_img.shape[:2]:
        overlay[mask_bool] = (original_img[mask_bool] * 0.5 + rendered[mask_bool, :3] * 0.5).astype(np.uint8)
    
    Image.fromarray(overlay).save(output_subdir / "flame_overlay.png")
    print(f"    ✓ Saved: flame_overlay.png")


def extract_and_save_outputs(vid_name, output_subdir, preprocessed_dir, tracking_output_dir):
    """Extract relevant outputs and save in clean format."""
    print(f"\n{'='*80}")
    print(f"STEP 4: Extracting and organizing outputs")
    print(f"{'='*80}")
    
    output_subdir = Path(output_subdir)
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    preprocessed_dir = Path(preprocessed_dir)
    tracking_dirs = list(Path(tracking_output_dir).glob(f"{vid_name}_*"))
    
    if not tracking_dirs:
        raise RuntimeError(f"No tracking output found for {vid_name}")
    
    tracking_dir = tracking_dirs[0]  # Use first match
    
    # 1. Copy UV map
    print("  - Extracting UV map...")
    uv_source = preprocessed_dir / "p3dmm" / "uv_map" / "00000.png"
    if uv_source.exists():
        shutil.copy(uv_source, output_subdir / "uv.png")
        print(f"    ✓ Saved: uv.png")
    else:
        print(f"    ✗ Warning: UV map not found at {uv_source}")
    
    # 2. Copy normal map
    print("  - Extracting normal map...")
    normal_source = preprocessed_dir / "p3dmm" / "normals" / "00000.png"
    if normal_source.exists():
        shutil.copy(normal_source, output_subdir / "normal.png")
        print(f"    ✓ Saved: normal.png")
    else:
        print(f"    ✗ Warning: Normal map not found at {normal_source}")
    
    # 3. Copy segmentation files
    print("  - Extracting segmentations...")
    
    # Cropped grayscale segmentation
    seg_cropped_source = preprocessed_dir / "seg_og" / "00000.png"
    if seg_cropped_source.exists():
        shutil.copy(seg_cropped_source, output_subdir / "segmentation_cropped.png")
        print(f"    ✓ Saved: segmentation_cropped.png")
    else:
        print(f"    ✗ Warning: Cropped segmentation not found at {seg_cropped_source}")
    
    # Full image annotated colored segmentation
    seg_full_source = preprocessed_dir / "seg_full_annotations" / "color_00000.png"
    if seg_full_source.exists():
        shutil.copy(seg_full_source, output_subdir / "annotated_segmentation.png")
        print(f"    ✓ Saved: annotated_segmentation.png")
    else:
        print(f"    ✗ Warning: Full annotated segmentation not found at {seg_full_source}")
    
    # 4. Extract and save camera parameters
    print("  - Extracting camera parameters...")
    frame_file = tracking_dir / "checkpoint" / "00000.frame"
    if frame_file.exists():
        frame_data = torch.load(frame_file, weights_only=False)
        
        camera_params = {
            "focal_length": frame_data['camera']['fl'].tolist(),
            "principal_point": frame_data['camera']['pp'].tolist(),
            "rotation_base": {
                k.replace('R_base_', ''): v.tolist() 
                for k, v in frame_data['camera'].items() 
                if k.startswith('R_base_')
            },
            "translation_base": {
                k.replace('t_base_', ''): v.tolist() 
                for k, v in frame_data['camera'].items() 
                if k.startswith('t_base_')
            },
            "image_size": frame_data['img_size'].tolist()
        }
        
        with open(output_subdir / "camera_params.json", 'w') as f:
            json.dump(camera_params, f, indent=2)
        print(f"    ✓ Saved: camera_params.json")
    else:
        print(f"    ✗ Warning: Frame file not found at {frame_file}")
    
    # 5. Extract and save FLAME parameters
    print("  - Extracting FLAME parameters...")
    if frame_file.exists():
        flame_params = {
            "shape": frame_data['flame']['shape'].tolist(),
            "expression": frame_data['flame']['exp'].tolist(),
            "rotation_6d": frame_data['flame']['R'].tolist(),
            "rotation_matrix": frame_data['flame']['R_rotation_matrix'].tolist(),
            "translation": frame_data['flame']['t'].tolist(),
            "jaw": frame_data['flame']['jaw'].tolist(),
            "neck": frame_data['flame']['neck'].tolist(),
            "eyes": frame_data['flame']['eyes'].tolist(),
            "eyelids": frame_data['flame']['eyelids'].tolist(),
        }
        
        with open(output_subdir / "flame_parameters.json", 'w') as f:
            json.dump(flame_params, f, indent=2)
        print(f"    ✓ Saved: flame_parameters.json")
    
    # 6. Extract and save head orientation
    print("  - Extracting head orientation...")
    head_orient_file = tracking_dir / "checkpoint" / "00000_head_orientation.npy"
    if head_orient_file.exists():
        head_orient_data = np.load(head_orient_file, allow_pickle=True).item()
        
        # Convert numpy arrays to lists for JSON serialization
        head_orientation = {
            "forward_vector": head_orient_data['forward_vector'].tolist(),
            "up_vector": head_orient_data['up_vector'].tolist(),
            "right_vector": head_orient_data['right_vector'].tolist(),
            "euler_angles_xyz_radians": head_orient_data['euler_angles_xyz'].tolist(),
            "euler_angles_xyz_degrees": np.rad2deg(head_orient_data['euler_angles_xyz']).tolist()
        }
        
        with open(output_subdir / "head_orientation.json", 'w') as f:
            json.dump(head_orientation, f, indent=2)
        print(f"    ✓ Saved: head_orientation.json")
    else:
        print(f"    ✗ Warning: Head orientation file not found at {head_orient_file}")
    
    # 7. Extract and save crop box parameters
    print("  - Extracting crop box parameters...")
    crop_params_file = preprocessed_dir / "crop_params" / "00000.npy"
    if crop_params_file.exists():
        crop_params = np.load(crop_params_file, allow_pickle=True).item()
        
        # Save as JSON for easy reading
        with open(output_subdir / "crop_parameters.json", 'w') as f:
            json.dump(crop_params, f, indent=2)
        print(f"    ✓ Saved: crop_parameters.json")
    else:
        print(f"    ✗ Warning: Crop parameters not found at {crop_params_file}")
    
    # 8. Render FLAME segmentation in original image space
    try:
        render_flame_segmentation(tracking_dir, preprocessed_dir, output_subdir)
    except Exception as e:
        print(f"    ✗ Warning: Failed to render FLAME segmentation: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✓ All outputs saved to: {output_subdir}")
    print(f"  Output files:")
    print(f"    - uv.png (UV texture map)")
    print(f"    - normal.png (Normal map)")
    print(f"    - segmentation_cropped.png (Grayscale segmentation of cropped face)")
    print(f"    - annotated_segmentation.png (Colored segmentation of full image)")
    print(f"    - flame_segmentation.png (Binary FLAME mesh mask in original image space)")
    print(f"    - flame_overlay.png (FLAME mesh overlaid on original image)")
    print(f"    - camera_params.json (Camera parameters)")
    print(f"    - flame_parameters.json (FLAME model parameters)")
    print(f"    - head_orientation.json (Head orientation vectors and Euler angles)")
    print(f"    - crop_parameters.json (Face crop box parameters)")


def cleanup_intermediate_files(preprocessed_dir, tracking_output_dir, vid_name, keep_preprocessed=False):
    """Clean up intermediate files to save space."""
    preprocessed_dir = Path(preprocessed_dir)
    tracking_dirs = list(Path(tracking_output_dir).glob(f"{vid_name}_*"))
    
    if not keep_preprocessed:
        print(f"\n  - Cleaning up intermediate files for {vid_name}...")
        if preprocessed_dir.exists():
            shutil.rmtree(preprocessed_dir)
        for tracking_dir in tracking_dirs:
            if tracking_dir.exists():
                shutil.rmtree(tracking_dir)
        print(f"    ✓ Cleaned up")


def process_single_image(image_path, output_dir, iters=800, keep_intermediate=False):
    """Process a single image through the full pipeline."""
    vid_name = image_path.stem  # Filename without extension
    output_subdir = output_dir / vid_name
    
    # Create temporary directories for intermediate outputs
    import tempfile
    temp_base = Path(tempfile.mkdtemp(prefix=f"pixel3dmm_{vid_name}_"))
    preprocessed_dir = temp_base / "preprocessed"
    tracking_output_dir = temp_base / "tracking"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    tracking_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#'*80}")
    print(f"# Processing: {image_path.name}")
    print(f"# Output directory: {output_subdir}")
    print(f"# Temporary directory: {temp_base}")
    print(f"{'#'*80}")
    
    try:
        # Run pipeline steps with temporary directories
        run_preprocessing(image_path, preprocessed_dir)
        run_network_inference(vid_name, "normals", preprocessed_dir)
        run_network_inference(vid_name, "uv_map", preprocessed_dir)
        run_tracking(vid_name, iters, preprocessed_dir, tracking_output_dir)
        
        # Extract and organize outputs
        extract_and_save_outputs(vid_name, output_subdir, preprocessed_dir, tracking_output_dir)
        
        # Cleanup temporary files
        if not keep_intermediate:
            cleanup_intermediate_files(preprocessed_dir, tracking_output_dir, vid_name, keep_preprocessed=False)
            # Remove the temporary base directory
            if temp_base.exists():
                shutil.rmtree(temp_base)
        
        print(f"\n✓ Successfully processed: {image_path.name}")
        return True
        
    except Exception as e:
        print(f"\n✗ Error processing {image_path.name}: {str(e)}")
        import traceback
        traceback.print_exc()
        # Cleanup on error
        if temp_base.exists():
            shutil.rmtree(temp_base)
        return False


def inference_pixel3dmm(input_dir: str, output_dir: str, iters: int = 800, keep_intermediate: bool = False) -> int:
    """
    Run Pixel3DMM inference on a directory of images.
    
    Args:
        input_dir: Directory containing input images (PNG, JPG, JPEG)
        output_dir: Root directory for outputs (subdirs will be created per image)
        iters: Number of tracking iterations (default: 800)
        keep_intermediate: Keep intermediate preprocessing and tracking files
    
    Returns:
        0 if all images processed successfully, 1 if any failed
    """
    # Setup
    output_dir = setup_output_dir(output_dir)
    image_files = get_image_files(input_dir)

    # Process each image
    results = []
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing {image_path.name}...")
        success = process_single_image(
            image_path, 
            output_dir, 
            iters=iters,
            keep_intermediate=keep_intermediate
        )
        results.append((image_path.name, success))
    
   
    successful = sum(1 for _, success in results if success)
    failed = len(results) - successful
    
    print(f"Total images: {len(results)}")
    print(f"Successful:   {successful}")
    print(f"Failed:       {failed}")
    
    if failed > 0:
        print(f"\nFailed images:")
        for name, success in results:
            if not success:
                print(f"  - {name}")
    
    print(f"\nOutputs saved to: {output_dir}")    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Pixel3DMM inference on a directory of images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input images (PNG, JPG, JPEG)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root directory for outputs (subdirs will be created per image)"
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=800,
        help="Number of tracking iterations (default: 800)"
    )
    parser.add_argument(
        "--keep_intermediate",
        action="store_true",
        help="Keep intermediate preprocessing and tracking files"
    )
    
    args = parser.parse_args()

    inference_pixel3dmm(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        iters=args.iters,
        keep_intermediate=args.keep_intermediate
    )