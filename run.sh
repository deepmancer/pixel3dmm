export PYTHONPATH="/localhome/aha220/Hairdar/modules/pixel3dmm/src:$PYTHONPATH"
source ~/anaconda3/bin/activate
conda activate clip

PATH_TO_VIDEO="/localhome/aha220/Hairdar/modules/pixel3dmm/examplesss/hair_132.jpg"
base_name=$(basename $PATH_TO_VIDEO)
VID_NAME="${base_name%%.*}"

python scripts/run_preprocessing.py --video_or_images_path $PATH_TO_VIDEO
python scripts/network_inference.py model.prediction_type=normals video_name=$VID_NAME
python scripts/network_inference.py model.prediction_type=uv_map video_name=$VID_NAME
python scripts/track.py video_name=$VID_NAME iters=800
