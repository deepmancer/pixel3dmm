import os
import tyro

from pixel3dmm import env_paths
from pixel3dmm.preprocessing.pipnet_utils import preprocess_with_mediapipe_cropping


def main(video_or_images_path: str, output_dir: str = None):

    if os.path.isdir(video_or_images_path):
        vid_name = video_or_images_path.split('/')[-1]
    else:
        vid_name = video_or_images_path.split('/')[-1][:-4]

    # Use provided output_dir or default to env_paths
    if output_dir is None:
        preprocess_dir = f'{env_paths.PREPROCESSED_DATA}/{vid_name}'
    else:
        preprocess_dir = output_dir
    
    # Run MediaPipe-based cropping (now integrated in pipnet_utils)
    print("\n" + "="*80)
    print("Running MediaPipe face detection and cropping...")
    print("="*80)
    preprocess_with_mediapipe_cropping(
        video_or_images_path=video_or_images_path,
        output_dir=preprocess_dir,
        target_size=512,
        start_frame=0
    )

    # Run PIPNet landmark detection on cropped images
    # This also extracts MediaPipe landmarks automatically
    print("\n" + "="*80)
    print("Running PIPNet landmark detection...")
    print("="*80)
    os.system(f'cd {env_paths.CODE_BASE}/scripts/ ; python run_pipnet_landmarks.py --preprocessed_dir {preprocess_dir}')

    # Run MICA for shape estimation
    os.system(f'cd {env_paths.CODE_BASE}/src/pixel3dmm/preprocessing/MICA ; python demo.py -video_name {vid_name} -a {preprocess_dir}/arcface/ -o {preprocess_dir}')

    # Run face segmentation
    os.system(f'cd {env_paths.CODE_BASE}/scripts/ ; python run_facer_segmentation.py --video_name {vid_name} --preprocessed_dir {preprocess_dir}')



if __name__ == '__main__':
    tyro.cli(main)