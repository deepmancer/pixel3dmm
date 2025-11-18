import json
from pathlib import Path
from environs import Env


env = Env(expand_vars=True)
env_file_path = Path(f"{Path.home()}/.config/pixel3dmm/.env")
if env_file_path.exists():
    env.read_env(str(env_file_path), recurse=False)


with env.prefixed("PIXEL3DMM_"):
    CODE_BASE = "/localhome/aha220/Hairdar/modules/pixel3dmm/"
    PREPROCESSED_DATA = "/localhome/aha220/Hairdar/modules/pixel3dmm/preprocessed_data"
    TRACKING_OUTPUT = "/localhome/aha220/Hairdar/modules/pixel3dmm/tracking_output"



head_template = f'{CODE_BASE}/assets/head_template.obj'
head_template_color = f'{CODE_BASE}/assets/head_template_color.obj'
head_template_ply = f'{CODE_BASE}/assets/test_rigid.ply'
VALID_VERTICES_WIDE_REGION = f'{CODE_BASE}/assets/uv_valid_verty_noEyes_debug.npy'
VALID_VERTS_UV_MESH = f'{CODE_BASE}/assets/uv_valid_verty.npy'
VERTEX_WEIGHT_MASK = f'{CODE_BASE}/assets/flame_vertex_weights.npy'
MIRROR_INDEX = f'{CODE_BASE}/assets/flame_mirror_index.npy'
EYE_MASK = f'{CODE_BASE}/assets/uv_mask_eyes.png'
FLAME_UV_COORDS = f'{CODE_BASE}/assets/flame_uv_coords.npy'
VALID_VERTS_NARROW = f'{CODE_BASE}/assets/uv_valid_verty_noEyes.npy'
VALID_VERTS = f'{CODE_BASE}/assets/uv_valid_verty_noEyes_noEyeRegion_debug_wEars.npy'
FLAME_ASSETS = f'{CODE_BASE}/src/pixel3dmm/preprocessing/MICA/data/'

# FLAME model paths (from main Hairdar assets)
FLAME_MODEL_PATH = '/localhome/aha220/Hairdar/assets/body_models/base_models/flame/parametric_models'
FLAME_GENERIC_MODEL = f'{FLAME_MODEL_PATH}/generic_model.pkl'
FLAME_2020_PATH = f'{CODE_BASE}/src/pixel3dmm/preprocessing/MICA/data/FLAME2020'

# paths to pretrained pixel3dmm checkpoints
CKPT_UV_PRED = f'{CODE_BASE}/pretrained_weights/uv.ckpt'
CKPT_N_PRED = f'{CODE_BASE}/pretrained_weights/normals.ckpt'