import os
import cv2
import numpy as np
import tyro
from pathlib import Path
from PIL import Image
import mediapipe as mp

from pixel3dmm import env_paths


def detect_face_mediapipe(image):
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    
    with mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    ) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.detections:
            return None
        
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = image.shape
        
        x = int(bboxC.xmin * iw)
        y = int(bboxC.ymin * ih)
        w = int(bboxC.width * iw)
        h = int(bboxC.height * ih)
        
        return (x, y, w, h)


def get_landmarks_mediapipe(image):
    mp_face_mesh = mp.solutions.face_mesh
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            return None
        
        landmarks = results.multi_face_landmarks[0]
        ih, iw, _ = image.shape
        
        landmark_points = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * iw)
            y = int(landmark.y * ih)
            landmark_points.append([x, y])
        
        return np.array(landmark_points)


def crop_face_from_bbox(image, bbox, scale=1.5):
    x, y, w, h = bbox
    
    center_x = x + w // 2
    center_y = y + h // 2
    
    size = max(w, h)
    size = int(size * scale)
    
    x1 = max(0, center_x - size // 2)
    y1 = max(0, center_y - size // 2)
    x2 = min(image.shape[1], center_x + size // 2)
    y2 = min(image.shape[0], center_y + size // 2)
    
    cropped = image[y1:y2, x1:x2]
    
    if cropped.shape[0] != cropped.shape[1]:
        max_dim = max(cropped.shape[0], cropped.shape[1])
        square = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        y_offset = (max_dim - cropped.shape[0]) // 2
        x_offset = (max_dim - cropped.shape[1]) // 2
        square[y_offset:y_offset+cropped.shape[0], x_offset:x_offset+cropped.shape[1]] = cropped
        cropped = square
    
    return cropped, (x1, y1, x2, y2)


def process_image_mediapipe(image_path, output_dir, target_size=512):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to load image: {image_path}")
        return False
    
    bbox = detect_face_mediapipe(image)
    if bbox is None:
        print(f"No face detected in: {image_path}")
        return False
    
    cropped, crop_coords = crop_face_from_bbox(image, bbox, scale=1.8)
    
    resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    
    output_path = output_dir / image_path.name
    cv2.imwrite(str(output_path), resized)
    
    landmarks = get_landmarks_mediapipe(resized)
    if landmarks is not None:
        landmarks_dir = output_dir.parent / 'landmarks'
        landmarks_dir.mkdir(exist_ok=True)
        np.save(str(landmarks_dir / f"{image_path.stem}.npy"), landmarks)
    
    return True


def main(
    video_or_images_path: str,
    start_frame: int = 0,
    target_size: int = 512
):
    if os.path.isdir(video_or_images_path):
        vid_name = os.path.basename(video_or_images_path.rstrip('/'))
        image_dir = Path(video_or_images_path)
        image_files = sorted([f for f in image_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    else:
        vid_name = Path(video_or_images_path).stem
        image_dir = Path(video_or_images_path).parent
        image_files = [Path(video_or_images_path)]
    
    output_base = Path(env_paths.PREPROCESSED_DATA) / vid_name
    output_base.mkdir(parents=True, exist_ok=True)
    
    cropped_dir = output_base / 'cropped'
    cropped_dir.mkdir(exist_ok=True)
    
    rgb_dir = output_base / 'rgb'
    rgb_dir.mkdir(exist_ok=True)
    
    arcface_dir = output_base / 'arcface'
    arcface_dir.mkdir(exist_ok=True)
    
    print(f"Processing {len(image_files)} images with MediaPipe face detection...")
    
    success_count = 0
    for i, image_file in enumerate(image_files[start_frame:], start=start_frame):
        print(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
        
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"  Failed to load image")
            continue
        
        # Save with numbered filenames (00000.jpg, 00001.jpg, etc.)
        frame_number = i - start_frame
        numbered_filename = f"{frame_number:05d}.jpg"
        
        cv2.imwrite(str(rgb_dir / numbered_filename), image)
        
        # Process and save cropped image with numbered filename
        bbox = detect_face_mediapipe(image)
        if bbox is None:
            print(f"  No face detected, skipping")
            continue
        
        cropped, crop_coords = crop_face_from_bbox(image, bbox, scale=1.8)
        resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
        
        cv2.imwrite(str(cropped_dir / numbered_filename), resized)
        cv2.imwrite(str(arcface_dir / numbered_filename), resized)
        
        # Save landmarks
        landmarks = get_landmarks_mediapipe(resized)
        if landmarks is not None:
            landmarks_dir = output_base / 'landmarks'
            landmarks_dir.mkdir(exist_ok=True)
            np.save(str(landmarks_dir / f"{frame_number:05d}.npy"), landmarks)
        
        success_count += 1
    
    print(f"\nSuccessfully processed {success_count}/{len(image_files)} images")
    print(f"Output directory: {output_base}")
    
    if success_count == 0:
        raise ValueError("No faces detected in any images. Please check your input images.")
    
    return True


if __name__ == '__main__':
    tyro.cli(main)
