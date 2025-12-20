from pathlib import Path
from typing import Union, List, Optional, Tuple
import numpy as np
from PIL import Image
import mediapipe as mp
import cv2


class FacialLandmarkDetector:
    
    # MediaPipe 478 to Dlib 68 correspondence mapping
    MP2DLIB_CORRESPONDENCE = [
        ## Face Contour
        [127],       # 1
        [234],       # 2
        [93],        # 3
        [132, 58],   # 4
        [58, 172],   # 5
        [136],       # 6
        [150],       # 7
        [176],       # 8
        [152],       # 9
        [400],       # 10
        [379],       # 11
        [365],       # 12
        [397, 288],  # 13
        [361],       # 14
        [323],       # 15
        [454],       # 16
        [356],       # 17
        
        ## Right Brow 
        [70],        # 18
        [63],        # 19
        [105],       # 20
        [66],        # 21
        [107],       # 22
        
        ## Left Brow
        [336],       # 23
        [296],       # 24
        [334],       # 25
        [293],       # 26
        [300],       # 27
        
        ## Nose
        [168, 6],    # 28
        [197, 195],  # 29
        [5],         # 30
        [4],         # 31
        [75],        # 32
        [97],        # 33
        [2],         # 34
        [326],       # 35
        [305],       # 36
        
        ## Right Eye
        [33],        # 37
        [160],       # 38
        [158],       # 39
        [133],       # 40
        [153],       # 41
        [144],       # 42
        
        ## Left Eye
        [362],       # 43
        [385],       # 44
        [387],       # 45
        [263],       # 46
        [373],       # 47
        [380],       # 48
        
        ## Upper Lip Contour Top
        [61],        # 49
        [39],        # 50
        [37],        # 51
        [0],         # 52
        [267],       # 53
        [269],       # 54
        [291],       # 55
        
        ## Lower Lip Contour Bottom
        [321],   # 56
        [314],   # 57
        [17],    # 58
        [84],    # 59
        [91],    # 60
        
        ## Upper Lip Contour Bottom
        [78],    # 61
        [82],    # 62
        [13],    # 63
        [312],   # 64
        [308],   # 65
        
        ## Lower Lip Contour Top
        [317],   # 66
        [14],    # 67
        [87],    # 68
    ]
    
    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        mediapipe_flame_embedding_path: str = "assets/body_models/landmarks/flame/mediapipe_landmark_embedding.npz"
    ):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mediapipe_flame_embedding_path = mediapipe_flame_embedding_path
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=self.static_image_mode,
            max_num_faces=self.max_num_faces,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=self.min_detection_confidence
        )
        
        # Store expected number of landmarks
        self.num_landmarks = 478 if refine_landmarks else 468
        
        # Load MediaPipe-FLAME landmark mapping
        self.mediapipe_flame_mapping = None
        self._load_mediapipe_flame_mapping()
        
        # Normalize MP2DLIB_CORRESPONDENCE for single-index entries
        self.mp2dlib_correspondence_normalized = []
        for indices in self.MP2DLIB_CORRESPONDENCE:
            if len(indices) == 1:
                self.mp2dlib_correspondence_normalized.append([indices[0], indices[0]])
            else:
                self.mp2dlib_correspondence_normalized.append(indices)
        
    def __del__(self):
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        if hasattr(self, 'face_detection'):
            self.face_detection.close()
    
    def _load_mediapipe_flame_mapping(self):
        try:
            mapping_path = Path(self.mediapipe_flame_embedding_path)
            if not mapping_path.exists():
                print(f"Warning: MediaPipe-FLAME mapping file not found at {mapping_path}")
                print("ldm105_flame will not be available in get_lmk_full()")
                return
            
            mapping_data = np.load(mapping_path)
            self.mediapipe_flame_mapping = {
                'landmark_indices': mapping_data['landmark_indices'],
            }
            print(f"Loaded MediaPipe-FLAME mapping: {len(self.mediapipe_flame_mapping['landmark_indices'])} landmarks")
            
        except Exception as e:
            print(f"Error loading MediaPipe-FLAME mapping: {e}")
            self.mediapipe_flame_mapping = None
    
    def convert_mediapipe_to_dlib68(self, lmks_mp: np.ndarray) -> np.ndarray:
        # Convert landmarks by averaging corresponding MediaPipe indices
        lmks_dlib = np.array([
            lmks_mp[indices].mean(axis=0) 
            for indices in self.mp2dlib_correspondence_normalized
        ])
        
        return lmks_dlib
    
    def get_mediapipe_flame_subset(self, lmks_mp: np.ndarray) -> Optional[np.ndarray]:
        if self.mediapipe_flame_mapping is None:
            return None
        
        landmark_indices = self.mediapipe_flame_mapping['landmark_indices']
        return lmks_mp[landmark_indices]
    
    def get_lmk_full(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        face_index: int = 0
    ) -> Optional[dict]:
        try:
            # Preprocess image
            image_array, height, width = self._preprocess_image(image)
            
            # Detect landmarks using MediaPipe
            results = self.face_mesh.process(image_array)
            
            # Check if any faces were detected
            if not results.multi_face_landmarks:
                print("Warning: No faces detected in the image")
                return None
            
            # Check if requested face index is valid
            num_faces = len(results.multi_face_landmarks)
            if face_index >= num_faces:
                print(
                    f"Warning: Requested face index {face_index} but only "
                    f"{num_faces} face(s) detected. Using face 0."
                )
                face_index = 0
            
            # Extract landmarks for the specified face
            face_landmarks = results.multi_face_landmarks[face_index]
            
            ldm478 = np.zeros((478, 2), dtype=np.float32)
            for idx, landmark in enumerate(face_landmarks.landmark):
                ldm478[idx, 0] = landmark.x * width
                ldm478[idx, 1] = landmark.y * height
            
            ldm468 = ldm478[:468].copy()
            ldm68 = self.convert_mediapipe_to_dlib68(ldm478)
            ldm105_flame = self.get_mediapipe_flame_subset(ldm478)
            
            return {
                'ldm478': ldm478,
                'ldm468': ldm468,
                'ldm105_flame': ldm105_flame,
                'ldm68': ldm68,
                'image_height': height,
                'image_width': width
            }
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None
        except Exception as e:
            print(f"Error during landmark detection: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _preprocess_image(
        self, 
        image: Union[str, Path, np.ndarray, Image.Image]
    ) -> Tuple[np.ndarray, int, int]:
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            pil_image = Image.open(image_path).convert('RGB')
            image_array = np.array(pil_image)
            
        elif isinstance(image, Image.Image):
            image_array = np.array(image.convert('RGB'))
            
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                image_array = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image_array = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 3:
                image_array = image.copy()
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
        else:
            raise ValueError(
                f"Unsupported image type: {type(image)}. "
                "Expected str, Path, np.ndarray, or PIL.Image.Image"
            )
        
        height, width = image_array.shape[:2]
        return image_array, height, width
    
    def get_lmk_478(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        face_index: int = 0
    ) -> Optional[np.ndarray]:
        try:
            # Preprocess image
            image_array, height, width = self._preprocess_image(image)
            
            # Detect landmarks using MediaPipe
            results = self.face_mesh.process(image_array)
            
            # Check if any faces were detected
            if not results.multi_face_landmarks:
                print("Warning: No faces detected in the image")
                return None
            
            # Check if requested face index is valid
            num_faces = len(results.multi_face_landmarks)
            if face_index >= num_faces:
                print(
                    f"Warning: Requested face index {face_index} but only "
                    f"{num_faces} face(s) detected. Using face 0."
                )
                face_index = 0
            
            # Extract landmarks for the specified face
            face_landmarks = results.multi_face_landmarks[face_index]
            
            landmarks_pixel = np.zeros((self.num_landmarks, 2), dtype=np.float32)
            
            for idx, landmark in enumerate(face_landmarks.landmark):
                landmarks_pixel[idx, 0] = landmark.x * width
                landmarks_pixel[idx, 1] = landmark.y * height
            
            return landmarks_pixel
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None
        except Exception as e:
            print(f"Error during landmark detection: {e}")
            return None
    
    def get_all_faces_lmk_478(
        self,
        image: Union[str, Path, np.ndarray, Image.Image]
    ) -> Optional[List[np.ndarray]]:
        try:
            # Preprocess image
            image_array, height, width = self._preprocess_image(image)
            
            # Detect landmarks using MediaPipe
            results = self.face_mesh.process(image_array)
            
            # Check if any faces were detected
            if not results.multi_face_landmarks:
                print("Warning: No faces detected in the image")
                return None
            
            # Extract landmarks for all detected faces
            all_landmarks = []
            for face_landmarks in results.multi_face_landmarks:
                landmarks_pixel = np.zeros((self.num_landmarks, 2), dtype=np.float32)
                
                for idx, landmark in enumerate(face_landmarks.landmark):
                    landmarks_pixel[idx, 0] = landmark.x * width
                    landmarks_pixel[idx, 1] = landmark.y * height
                
                all_landmarks.append(landmarks_pixel)
            
            return all_landmarks
            
        except Exception as e:
            print(f"Error during multi-face landmark detection: {e}")
            return None
    
    def get_lmk_478_with_confidence(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        face_index: int = 0
    ) -> Optional[Tuple[np.ndarray, List[float]]]:
        try:
            image_array, height, width = self._preprocess_image(image)
            results = self.face_mesh.process(image_array)
            
            if not results.multi_face_landmarks:
                return None
            
            num_faces = len(results.multi_face_landmarks)
            if face_index >= num_faces:
                face_index = 0
            
            face_landmarks = results.multi_face_landmarks[face_index]
            
            landmarks_pixel = np.zeros((self.num_landmarks, 2), dtype=np.float32)
            confidence_scores = []
            
            for idx, landmark in enumerate(face_landmarks.landmark):
                landmarks_pixel[idx, 0] = landmark.x * width
                landmarks_pixel[idx, 1] = landmark.y * height
                confidence_scores.append(landmark.visibility if hasattr(landmark, 'visibility') else 1.0)
            
            return landmarks_pixel, confidence_scores
            
        except Exception as e:
            print(f"Error during landmark detection with confidence: {e}")
            return None
    
    def get_lmk_478_3d(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        face_index: int = 0
    ) -> Optional[np.ndarray]:
        try:
            image_array, height, width = self._preprocess_image(image)
            results = self.face_mesh.process(image_array)
            
            if not results.multi_face_landmarks:
                return None
            
            num_faces = len(results.multi_face_landmarks)
            if face_index >= num_faces:
                face_index = 0
            
            face_landmarks = results.multi_face_landmarks[face_index]
            
            landmarks_3d = np.zeros((self.num_landmarks, 3), dtype=np.float32)
            
            for idx, landmark in enumerate(face_landmarks.landmark):
                landmarks_3d[idx, 0] = landmark.x * width
                landmarks_3d[idx, 1] = landmark.y * height
                landmarks_3d[idx, 2] = landmark.z * width
            
            return landmarks_3d
            
        except Exception as e:
            print(f"Error during 3D landmark detection: {e}")
            return None
    
    def batch_process(
        self,
        image_paths: List[Union[str, Path]]
    ) -> List[Optional[np.ndarray]]:
        results = []
        for image_path in image_paths:
            landmarks = self.get_lmk_478(image_path)
            results.append(landmarks)
        return results
    
    def get_landmark_subset(
        self,
        landmarks: np.ndarray,
        indices: List[int]
    ) -> np.ndarray:
        return landmarks[indices]
    
    @staticmethod
    def get_face_region_indices() -> dict:
        return {
            # Silhouette / Face oval
            'face_oval': [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
            ],
            'left_eye': [
                33, 7, 163, 144, 145, 153, 154, 155, 133,
                173, 157, 158, 159, 160, 161, 246
            ],
            'right_eye': [
                362, 382, 381, 380, 374, 373, 390, 249,
                263, 466, 388, 387, 386, 385, 384, 398
            ],
            'left_eyebrow': [
                46, 53, 52, 65, 55, 70, 63, 105, 66, 107
            ],
            'right_eyebrow': [
                276, 283, 282, 295, 285, 300, 293, 334, 296, 336
            ],
            'nose_bridge': [
                168, 6, 197, 195, 5, 4, 1, 19, 94, 2
            ],
            'nose_tip': [
                4, 5, 195, 197
            ],
            'lips_outer': [
                61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                291, 185, 40, 39, 37, 0, 267, 269, 270, 409
            ],
            'lips_inner': [
                78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
                308, 191, 80, 81, 82, 13, 312, 311, 310, 415
            ],
            'left_iris': [
                468, 469, 470, 471, 472
            ],
            'right_iris': [
                473, 474, 475, 476, 477
            ]
        }
    
    def get_face_bounding_box(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        face_index: int = 0,
        return_format: str = 'xyxy'
    ) -> Optional[Union[np.ndarray, dict]]:
        try:
            # Preprocess image
            image_array, height, width = self._preprocess_image(image)
            
            # Detect faces using MediaPipe Face Detection
            results = self.face_detection.process(image_array)
            
            # Check if any faces were detected
            if not results.detections:
                print("Warning: No faces detected in the image")
                return None
            
            # Check if requested face index is valid
            num_faces = len(results.detections)
            if face_index >= num_faces:
                print(
                    f"Warning: Requested face index {face_index} but only "
                    f"{num_faces} face(s) detected. Using face 0."
                )
                face_index = 0
            
            # Get the detection for the specified face
            detection = results.detections[face_index]
            
            # Extract bounding box in normalized coordinates
            bbox = detection.location_data.relative_bounding_box
            xmin_norm = bbox.xmin
            ymin_norm = bbox.ymin
            width_norm = bbox.width
            height_norm = bbox.height
            
            # Convert to pixel coordinates
            xmin_pixel = int(xmin_norm * width)
            ymin_pixel = int(ymin_norm * height)
            width_pixel = int(width_norm * width)
            height_pixel = int(height_norm * height)
            xmax_pixel = xmin_pixel + width_pixel
            ymax_pixel = ymin_pixel + height_pixel
            
            # Extract keypoints if available
            keypoints = {}
            if detection.location_data.relative_keypoints:
                keypoint_names = [
                    'right_eye', 'left_eye', 'nose_tip',
                    'mouth_center', 'right_ear_tragion', 'left_ear_tragion'
                ]
                for idx, kp in enumerate(detection.location_data.relative_keypoints):
                    if idx < len(keypoint_names):
                        keypoints[keypoint_names[idx]] = {
                            'x': int(kp.x * width),
                            'y': int(kp.y * height),
                            'x_normalized': kp.x,
                            'y_normalized': kp.y
                        }
            
            # Get detection confidence
            confidence = detection.score[0] if detection.score else 0.0
            
            # Return in requested format
            if return_format == 'xyxy':
                return np.array([xmin_pixel, ymin_pixel, xmax_pixel, ymax_pixel], dtype=np.int32)
            
            elif return_format == 'xywh':
                return np.array([xmin_pixel, ymin_pixel, width_pixel, height_pixel], dtype=np.int32)
            
            elif return_format == 'normalized':
                return {
                    'xmin': xmin_norm,
                    'ymin': ymin_norm,
                    'width': width_norm,
                    'height': height_norm
                }
            
            elif return_format == 'all':
                return {
                    'xyxy': np.array([xmin_pixel, ymin_pixel, xmax_pixel, ymax_pixel], dtype=np.int32),
                    'xywh': np.array([xmin_pixel, ymin_pixel, width_pixel, height_pixel], dtype=np.int32),
                    'normalized': {
                        'xmin': xmin_norm,
                        'ymin': ymin_norm,
                        'width': width_norm,
                        'height': height_norm
                    },
                    'confidence': confidence,
                    'keypoints': keypoints,
                    'image_width': width,
                    'image_height': height
                }
            
            else:
                raise ValueError(
                    f"Unsupported return_format: {return_format}. "
                    "Expected 'xyxy', 'xywh', 'normalized', or 'all'."
                )
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None
        except Exception as e:
            print(f"Error during face bounding box detection: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_all_faces_bounding_boxes(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        return_format: str = 'xyxy'
    ) -> Optional[List[Union[np.ndarray, dict]]]:
        try:
            # Preprocess image
            image_array, height, width = self._preprocess_image(image)
            
            # Detect faces using MediaPipe Face Detection
            results = self.face_detection.process(image_array)
            
            # Check if any faces were detected
            if not results.detections:
                print("Warning: No faces detected in the image")
                return None
            
            # Extract bounding boxes for all detected faces
            all_bboxes = []
            for face_idx, detection in enumerate(results.detections):
                # Extract bounding box in normalized coordinates
                bbox = detection.location_data.relative_bounding_box
                xmin_norm = bbox.xmin
                ymin_norm = bbox.ymin
                width_norm = bbox.width
                height_norm = bbox.height
                
                # Convert to pixel coordinates
                xmin_pixel = int(xmin_norm * width)
                ymin_pixel = int(ymin_norm * height)
                width_pixel = int(width_norm * width)
                height_pixel = int(height_norm * height)
                xmax_pixel = xmin_pixel + width_pixel
                ymax_pixel = ymin_pixel + height_pixel
                
                # Extract keypoints if available
                keypoints = {}
                if detection.location_data.relative_keypoints:
                    keypoint_names = [
                        'right_eye', 'left_eye', 'nose_tip',
                        'mouth_center', 'right_ear_tragion', 'left_ear_tragion'
                    ]
                    for idx, kp in enumerate(detection.location_data.relative_keypoints):
                        if idx < len(keypoint_names):
                            keypoints[keypoint_names[idx]] = {
                                'x': int(kp.x * width),
                                'y': int(kp.y * height),
                                'x_normalized': kp.x,
                                'y_normalized': kp.y
                            }
                
                # Get detection confidence
                confidence = detection.score[0] if detection.score else 0.0
                
                # Format based on return_format
                if return_format == 'xyxy':
                    all_bboxes.append(np.array([xmin_pixel, ymin_pixel, xmax_pixel, ymax_pixel], dtype=np.int32))
                
                elif return_format == 'xywh':
                    all_bboxes.append(np.array([xmin_pixel, ymin_pixel, width_pixel, height_pixel], dtype=np.int32))
                
                elif return_format == 'normalized':
                    all_bboxes.append({
                        'xmin': xmin_norm,
                        'ymin': ymin_norm,
                        'width': width_norm,
                        'height': height_norm
                    })
                
                elif return_format == 'all':
                    all_bboxes.append({
                        'xyxy': np.array([xmin_pixel, ymin_pixel, xmax_pixel, ymax_pixel], dtype=np.int32),
                        'xywh': np.array([xmin_pixel, ymin_pixel, width_pixel, height_pixel], dtype=np.int32),
                        'normalized': {
                            'xmin': xmin_norm,
                            'ymin': ymin_norm,
                            'width': width_norm,
                            'height': height_norm
                        },
                        'confidence': confidence,
                        'keypoints': keypoints,
                        'image_width': width,
                        'image_height': height
                    })
            
            return all_bboxes
            
        except Exception as e:
            print(f"Error during multi-face bounding box detection: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def visualize_landmarks(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        landmarks: Optional[np.ndarray] = None,
        show_indices: bool = False,
        point_size: int = 1,
        point_color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        try:
            # Preprocess image
            image_array, height, width = self._preprocess_image(image)
            
            # Get landmarks if not provided
            if landmarks is None:
                landmarks = self.get_lmk_478(image)
                if landmarks is None:
                    return image_array
            
            output_image = image_array.copy()
            
            for idx, (x, y) in enumerate(landmarks):
                cv2.circle(
                    output_image,
                    (int(x), int(y)),
                    point_size,
                    point_color,
                    -1
                )
                
                if show_indices:
                    cv2.putText(
                        output_image,
                        str(idx),
                        (int(x) + 2, int(y) - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.2,
                        (255, 255, 255),
                        1
                    )
            
            return output_image
            
        except Exception as e:
            print(f"Error during visualization: {e}")
            return image_array if 'image_array' in locals() else None
    
    def visualize_bounding_boxes(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        bboxes: Optional[List[Union[np.ndarray, dict]]] = None,
        bbox_color: Tuple[int, int, int] = (0, 255, 0),
        bbox_thickness: int = 2,
        show_confidence: bool = True,
        show_keypoints: bool = True
    ) -> np.ndarray:
        try:
            # Preprocess image
            image_array, height, width = self._preprocess_image(image)
            
            # Get bounding boxes if not provided
            if bboxes is None:
                bboxes = self.get_all_faces_bounding_boxes(image, return_format='all')
                if bboxes is None:
                    return image_array
            
            output_image = image_array.copy()
            
            for idx, bbox in enumerate(bboxes):
                if isinstance(bbox, dict) and 'xyxy' in bbox:
                    xyxy = bbox['xyxy']
                    confidence = bbox.get('confidence', None)
                    keypoints = bbox.get('keypoints', {})
                elif isinstance(bbox, np.ndarray):
                    if len(bbox) == 4:
                        xyxy = bbox
                        confidence = None
                        keypoints = {}
                    else:
                        continue
                else:
                    continue
                
                # Extract coordinates
                x_min, y_min, x_max, y_max = xyxy
                
                # Draw bounding box
                cv2.rectangle(
                    output_image,
                    (int(x_min), int(y_min)),
                    (int(x_max), int(y_max)),
                    bbox_color,
                    bbox_thickness
                )
                
                if show_confidence and confidence is not None:
                    text = f"Face {idx}: {confidence:.2f}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    
                    cv2.rectangle(
                        output_image,
                        (int(x_min), int(y_min) - text_size[1] - 10),
                        (int(x_min) + text_size[0] + 5, int(y_min)),
                        bbox_color,
                        -1
                    )
                    
                    cv2.putText(
                        output_image,
                        text,
                        (int(x_min) + 2, int(y_min) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )
                
                if show_keypoints and keypoints:
                    for kp_name, kp_data in keypoints.items():
                        kp_x = kp_data['x']
                        kp_y = kp_data['y']
                        cv2.circle(
                            output_image,
                            (int(kp_x), int(kp_y)),
                            3,
                            (255, 0, 0),
                            -1
                        )
            
            return output_image
            
        except Exception as e:
            print(f"Error during bounding box visualization: {e}")
            import traceback
            traceback.print_exc()
            return image_array if 'image_array' in locals() else None


if __name__ == "__main__":
    import sys
    
    # Initialize detector with refined landmarks (478 points)
    detector = FacialLandmarkDetector(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    
    # Example 1: Detect all landmark formats with get_lmk_full
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        print(f"Processing image: {image_path}")
        print("="*80)
        
        # Get all landmark formats in one call
        result = detector.get_lmk_full(image_path)
        
        if result is not None:
            print(f"\n✓ Successfully detected landmarks")
            print(f"\nImage dimensions:")
            print(f"  Width: {result['image_width']}")
            print(f"  Height: {result['image_height']}")
            
            print(f"\nLandmark formats:")
            print(f"  ldm478 (Full MediaPipe): {result['ldm478'].shape}")
            print(f"  ldm468 (MediaPipe without iris): {result['ldm468'].shape}")
            print(f"  ldm68 (Dlib format): {result['ldm68'].shape}")
            if result['ldm105_flame'] is not None:
                print(f"  ldm105_flame (FLAME subset): {result['ldm105_flame'].shape}")
            else:
                print(f"  ldm105_flame: Not available (mapping file not found)")
            
            print(f"\nSample landmarks (ldm478):")
            print(f"  Nose tip (index 4): {result['ldm478'][4]}")
            print(f"  Left iris center (index 468): {result['ldm478'][468]}")
            print(f"  Right iris center (index 473): {result['ldm478'][473]}")
            
            print(f"\nSample Dlib landmarks (ldm68):")
            print(f"  Nose tip (index 30): {result['ldm68'][30]}")
            print(f"  Left eye center (index 36): {result['ldm68'][36]}")
            print(f"  Right eye center (index 45): {result['ldm68'][45]}")
            
            # Get face region indices
            regions = detector.get_face_region_indices()
            print(f"\nAvailable regions: {list(regions.keys())}")
            
            # Extract nose landmarks
            nose_landmarks = detector.get_landmark_subset(result['ldm478'], regions['nose_tip'])
            print(f"\nNose tip landmarks (from ldm478):\n{nose_landmarks}")
            
            # Get 3D landmarks separately if needed
            landmarks_3d = detector.get_lmk_478_3d(image_path)
            if landmarks_3d is not None:
                print(f"\n✓ 3D landmarks shape: {landmarks_3d.shape}")
            
            # Visualize ldm478 (save to file)
            output_image = detector.visualize_landmarks(image_path, result['ldm478'])
            output_path = "landmarks_visualization_478.jpg"
            cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
            print(f"\n✓ Visualization (478 landmarks) saved to: {output_path}")
            
            # Visualize ldm68 (save to file)
            output_image_68 = detector.visualize_landmarks(image_path, result['ldm68'], point_size=2, point_color=(255, 0, 0))
            output_path_68 = "landmarks_visualization_68.jpg"
            cv2.imwrite(output_path_68, cv2.cvtColor(output_image_68, cv2.COLOR_RGB2BGR))
            print(f"✓ Visualization (68 Dlib landmarks) saved to: {output_path_68}")
            
            # Example 2: Detect face bounding box
            print(f"\n" + "="*80)
            print("Face Bounding Box Detection:")
            print("="*80)
            
            # Get bounding box in different formats
            bbox_xyxy = detector.get_face_bounding_box(image_path, return_format='xyxy')
            bbox_xywh = detector.get_face_bounding_box(image_path, return_format='xywh')
            bbox_all = detector.get_face_bounding_box(image_path, return_format='all')
            
            if bbox_xyxy is not None:
                print(f"\n✓ Successfully detected face bounding box")
                print(f"\nBounding box formats:")
                print(f"  xyxy: {bbox_xyxy}")
                print(f"  xywh: {bbox_xywh}")
                
                if bbox_all:
                    print(f"  confidence: {bbox_all['confidence']:.4f}")
                    print(f"\nKeypoints detected: {len(bbox_all['keypoints'])}")
                    for kp_name, kp_data in bbox_all['keypoints'].items():
                        print(f"    {kp_name}: ({kp_data['x']}, {kp_data['y']})")
                
                # Visualize bounding box
                output_bbox_image = detector.visualize_bounding_boxes(image_path)
                output_bbox_path = "bounding_box_visualization.jpg"
                cv2.imwrite(output_bbox_path, cv2.cvtColor(output_bbox_image, cv2.COLOR_RGB2BGR))
                print(f"\n✓ Bounding box visualization saved to: {output_bbox_path}")
            
        else:
            print("✗ Failed to detect landmarks")
    else:
        print("Usage: python facial_landmark_detector.py <image_path>")
        print("\nExample:")
        print("  python facial_landmark_detector.py face_image.jpg")
        print("\nFeatures:")
        print("  - Detects 478/468 MediaPipe facial landmarks")
        print("  - Converts to 68 Dlib landmark format")
        print("  - Extracts 105 FLAME-aligned landmarks")
        print("  - Detects face bounding boxes with confidence scores")
        print("  - Provides facial keypoints (eyes, nose, mouth, ears)")
