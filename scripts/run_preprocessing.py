import os
import tyro

from pixel3dmm import env_paths


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
    
    # Run cropping with output directory
    os.system(f'cd {env_paths.CODE_BASE}/scripts/ ; python run_cropping_mediapipe.py --video_or_images_path {video_or_images_path} --output_dir {preprocess_dir}')

    os.system(f'cd {env_paths.CODE_BASE}/src/pixel3dmm/preprocessing/MICA ; python demo.py -video_name {vid_name} -a {preprocess_dir}/arcface/ -o {preprocess_dir}')

    os.system(f'cd {env_paths.CODE_BASE}/scripts/ ; python run_facer_segmentation.py --video_name {vid_name} --preprocessed_dir {preprocess_dir}')



if __name__ == '__main__':
    tyro.cli(main)