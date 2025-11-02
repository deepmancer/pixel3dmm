import numpy as np
import torch
import os
import argparse


def load_frame_data(checkpoint_path):
    frame_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    return frame_data


def load_head_orientation(checkpoint_dir, frame_id):
    npy_path = os.path.join(checkpoint_dir, f'{frame_id:05d}_head_orientation.npy')
    if os.path.exists(npy_path):
        return np.load(npy_path, allow_pickle=True).item()
    
    frame_path = os.path.join(checkpoint_dir, f'{frame_id:05d}.frame')
    if os.path.exists(frame_path):
        frame_data = load_frame_data(frame_path)
        return frame_data['flame']['head_orientation']
    
    raise FileNotFoundError(f"Could not find head orientation data for frame {frame_id}")


def print_head_orientation(head_orient, frame_id=None):
    frame_str = f"Frame {frame_id}: " if frame_id is not None else ""
    
    print(f"\n{frame_str}Head Orientation")
    print("=" * 60)
    
    forward = head_orient['forward_vector'][0] if head_orient['forward_vector'].ndim > 1 else head_orient['forward_vector']
    print(f"Forward Vector (Z-axis): [{forward[0]:+.4f}, {forward[1]:+.4f}, {forward[2]:+.4f}]")
    
    up = head_orient['up_vector'][0] if head_orient['up_vector'].ndim > 1 else head_orient['up_vector']
    print(f"Up Vector (Y-axis):      [{up[0]:+.4f}, {up[1]:+.4f}, {up[2]:+.4f}]")
    
    right = head_orient['right_vector'][0] if head_orient['right_vector'].ndim > 1 else head_orient['right_vector']
    print(f"Right Vector (X-axis):   [{right[0]:+.4f}, {right[1]:+.4f}, {right[2]:+.4f}]")
    
    euler = head_orient['euler_angles_xyz'][0] if head_orient['euler_angles_xyz'].ndim > 1 else head_orient['euler_angles_xyz']
    pitch_deg = np.rad2deg(euler[0])
    yaw_deg = np.rad2deg(euler[1])
    roll_deg = np.rad2deg(euler[2])
    
    print(f"\nEuler Angles (XYZ convention):")
    print(f"  Pitch (X-axis rotation): {pitch_deg:+7.2f}° ({euler[0]:+.4f} rad)")
    print(f"  Yaw   (Y-axis rotation): {yaw_deg:+7.2f}° ({euler[1]:+.4f} rad)")
    print(f"  Roll  (Z-axis rotation): {roll_deg:+7.2f}° ({euler[2]:+.4f} rad)")
    
    print("=" * 60)


def load_all_camera_params(checkpoint_dir, frame_id):
    frame_path = os.path.join(checkpoint_dir, f'{frame_id:05d}.frame')
    frame_data = load_frame_data(frame_path)
    
    camera_params = frame_data.get('camera', {})
    
    print(f"\nFrame {frame_id}: Camera Parameters")
    print("=" * 60)
    
    if 'fl' in camera_params:
        fl = camera_params['fl']
        print(f"Focal Length: {fl[0][0]:.2f} pixels")
    
    if 'pp' in camera_params:
        pp = camera_params['pp']
        print(f"Principal Point: [{pp[0][0]:+.4f}, {pp[0][1]:+.4f}]")
    
    for key in camera_params.keys():
        if key.startswith('R_base_'):
            serial = key.replace('R_base_', '')
            R = camera_params[key]
            t = camera_params.get(f't_base_{serial}', None)
            print(f"\nCamera {serial}:")
            print(f"  Rotation shape: {R.shape}")
            print(f"  Translation: {t[0] if t is not None else 'N/A'}")
    
    print("=" * 60)
    
    return camera_params


def main():
    parser = argparse.ArgumentParser(description='Load and display head orientation from tracking results')
    parser.add_argument('checkpoint_dir', type=str, help='Path to checkpoint directory')
    parser.add_argument('--frame_id', type=int, default=0, help='Frame ID to load (default: 0)')
    parser.add_argument('--show_camera', action='store_true', help='Also show camera parameters')
    
    args = parser.parse_args()
    
    head_orient = load_head_orientation(args.checkpoint_dir, args.frame_id)
    print_head_orientation(head_orient, args.frame_id)
    
    if args.show_camera:
        load_all_camera_params(args.checkpoint_dir, args.frame_id)


if __name__ == '__main__':
    main()
