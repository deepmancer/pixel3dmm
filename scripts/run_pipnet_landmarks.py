#!/usr/bin/env python3
"""
Extract PIPNet landmarks and MediaPipe landmarks from preprocessed cropped images.

This script:
1. Loads the PIPNet model
2. Extracts 98 facial landmarks using PIPNet
3. Automatically extracts 105 MediaPipe landmarks (via pipnet_utils.py)

Usage:
    python run_pipnet_landmarks.py --preprocessed_dir <path_to_preprocessed_data>
"""

import os
import sys
import importlib
import tyro
from pathlib import Path
import torchvision.transforms as transforms

from pixel3dmm import env_paths

# Add PIPNet to path
sys.path.append(f'{env_paths.CODE_BASE}/src/pixel3dmm/preprocessing/PIPNet/FaceBoxesV2/')

from pixel3dmm.preprocessing.pipnet_utils import demo_image


def main(preprocessed_dir: str):
    """
    Run PIPNet landmark detection on preprocessed (cropped) images.
    
    Args:
        preprocessed_dir: Path to the preprocessed data directory containing 'cropped/' folder
    """
    preprocessed_dir = Path(preprocessed_dir)
    
    if not preprocessed_dir.exists():
        raise ValueError(f"Preprocessed directory does not exist: {preprocessed_dir}")
    
    cropped_dir = preprocessed_dir / 'cropped'
    if not cropped_dir.exists():
        raise ValueError(f"Cropped images directory does not exist: {cropped_dir}")
    
    # Check if landmarks already exist
    pipnet_landmarks_dir = preprocessed_dir / 'PIPnet_landmarks'
    mediapipe_landmarks_dir = preprocessed_dir / 'mediapipe_landmarks'
    
    if pipnet_landmarks_dir.exists() and mediapipe_landmarks_dir.exists():
        pipnet_files = list(pipnet_landmarks_dir.glob('*.npy'))
        mediapipe_files = list(mediapipe_landmarks_dir.glob('*.npy'))
        cropped_files = list(cropped_dir.glob('*.jpg')) + list(cropped_dir.glob('*.png'))
        
        if len(pipnet_files) == len(cropped_files) and len(mediapipe_files) == len(cropped_files):
            print(f"\n{'='*80}")
            print("PIPNet and MediaPipe landmarks already exist and are complete")
            print(f"  PIPNet landmarks: {len(pipnet_files)} files")
            print(f"  MediaPipe landmarks: {len(mediapipe_files)} files")
            print("Skipping landmark extraction")
            print(f"{'='*80}\n")
            return
    
    # Load PIPNet configuration
    exp_path = 'experiments/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py'
    experiment_name = exp_path.split('/')[-1][:-3]
    data_name = exp_path.split('/')[-2]
    config_path = '.experiments.{}.{}'.format(data_name, experiment_name)
    
    my_config = importlib.import_module(config_path, package='pixel3dmm.preprocessing.PIPNet')
    Config = getattr(my_config, 'Config')
    cfg = Config()
    cfg.experiment_name = experiment_name
    cfg.data_name = data_name
    
    save_dir = os.path.join(
        f'{env_paths.CODE_BASE}/src/pixel3dmm/preprocessing/PIPNet/snapshots',
        cfg.data_name,
        cfg.experiment_name
    )
    
    # Setup preprocessing transforms
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize((cfg.input_size, cfg.input_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    print(f"\n{'='*80}")
    print(f"Running PIPNet landmark extraction")
    print(f"  Input directory: {cropped_dir}")
    print(f"  Output directory: {preprocessed_dir}")
    print(f"{'='*80}\n")
    
    # Run PIPNet landmark detection
    # This will also automatically extract MediaPipe landmarks
    try:
        demo_image(
            image_dir=str(cropped_dir),
            pid="dummy",  # Not used when processing directory
            save_dir=save_dir,
            preprocess=preprocess,
            cfg=cfg,
            input_size=cfg.input_size,
            net_stride=cfg.net_stride,
            num_nb=cfg.num_nb,
            use_gpu=cfg.use_gpu,
            start_frame=0,
            vertical_crop=False,
            static_crop=True,
            max_bbox=False,
            disable_cropping=True  # Images are already cropped
        )
        
        print(f"\n{'='*80}")
        print("✓ Landmark extraction completed successfully")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"✗ Error during landmark extraction: {e}")
        print(f"{'='*80}\n")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    tyro.cli(main)
