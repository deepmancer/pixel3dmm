from pathlib import Path
from typing import Union, List, Optional, Tuple
import numpy as np
from PIL import Image
import mediapipe as mp
import cv2


class FacialLandmarkDetector:
    """
    Attributes:
        static_image_mode (bool): Whether to treat input as static images vs video stream
        max_num_faces (int): Maximum number of faces to detect
        refine_landmarks (bool): Whether to refine landmarks around eyes and lips
        min_detection_confidence (float): Minimum confidence for face detection
        min_tracking_confidence (float): Minimum confidence for landmark tracking
    """
    
    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize the FacialLandmarkDetector with MediaPipe Face Mesh.
        
        Args:
            static_image_mode: If True, treats each image independently. 
                             If False, optimizes for video streams.
            max_num_faces: Maximum number of faces to detect in the image.
            refine_landmarks: If True, refines landmarks around eyes and lips 
                            for better accuracy (total 478 landmarks).
                            If False, uses 468 landmarks.
            min_detection_confidence: Minimum confidence value [0.0, 1.0] for 
                                    face detection to be considered successful.
            min_tracking_confidence: Minimum confidence value [0.0, 1.0] for 
                                   landmark tracking to be considered successful.
        """
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=self.static_image_mode,
            max_num_faces=self.max_num_faces,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
        # Store expected number of landmarks
        self.num_landmarks = 478 if refine_landmarks else 468
        
    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
    
    def _preprocess_image(
        self, 
        image: Union[str, Path, np.ndarray, Image.Image]
    ) -> Tuple[np.ndarray, int, int]:
        """
        Preprocess input image to RGB numpy array format.
        
        Args:
            image: Input image as file path, PIL Image, or numpy array
            
        Returns:
            Tuple of (RGB numpy array, height, width)
            
        Raises:
            ValueError: If image format is not supported
            FileNotFoundError: If image path does not exist
        """
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Read image using PIL for better format support
            pil_image = Image.open(image_path).convert('RGB')
            image_array = np.array(pil_image)
            
        elif isinstance(image, Image.Image):
            # Convert PIL Image to RGB numpy array
            image_array = np.array(image.convert('RGB'))
            
        elif isinstance(image, np.ndarray):
            # Ensure image is in RGB format
            if len(image.shape) == 2:
                # Grayscale to RGB
                image_array = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                # RGBA to RGB
                image_array = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 3:
                # Check if BGR (OpenCV format) and convert to RGB
                # MediaPipe expects RGB format
                # Assume it's BGR if it comes as numpy array from cv2.imread
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
        """
        Detect and return 478 facial landmarks in pixel space.
        
        This method detects facial landmarks using MediaPipe's Face Mesh model
        and returns them as 2D coordinates in pixel space, where x ranges from 
        0 to image width and y ranges from 0 to image height.
        
        MediaPipe's 478 landmarks include:
        - Face oval (0-16)
        - Left eyebrow (70-109)
        - Right eyebrow (336-345, etc.)
        - Left eye (362-373, etc.)
        - Right eye (33-133, etc.)
        - Nose bridge and tip (1-9, etc.)
        - Mouth outer (61, 185, 40, 39, etc.)
        - Mouth inner (78, 95, 88, etc.)
        - And many more detailed facial features
        
        Args:
            image: Input image as file path (str/Path), PIL Image, or numpy array
            face_index: Index of the face to extract landmarks from (default: 0)
                       Useful when multiple faces are detected
        
        Returns:
            numpy array of shape (478, 2) containing [x, y] pixel coordinates,
            or None if no face is detected or detection fails
            
        Example:
            >>> detector = FacialLandmarkDetector()
            >>> landmarks = detector.get_lmk_478("path/to/face.jpg")
            >>> if landmarks is not None:
            >>>     print(f"Detected {len(landmarks)} landmarks")
            >>>     print(f"Nose tip: {landmarks[4]}")  # Index 4 is nose tip
        """
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
            
            # Convert normalized landmarks to pixel coordinates
            landmarks_pixel = np.zeros((self.num_landmarks, 2), dtype=np.float32)
            
            for idx, landmark in enumerate(face_landmarks.landmark):
                # MediaPipe returns normalized coordinates [0, 1]
                # Convert to pixel space
                landmarks_pixel[idx, 0] = landmark.x * width   # x coordinate
                landmarks_pixel[idx, 1] = landmark.y * height  # y coordinate
            
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
        """
        Detect and return 478 landmarks for all detected faces.
        
        Args:
            image: Input image as file path, PIL Image, or numpy array
            
        Returns:
            List of numpy arrays, each of shape (478, 2) for each detected face,
            or None if no faces are detected
            
        Example:
            >>> detector = FacialLandmarkDetector(max_num_faces=5)
            >>> all_landmarks = detector.get_all_faces_lmk_478("group_photo.jpg")
            >>> if all_landmarks:
            >>>     print(f"Detected {len(all_landmarks)} faces")
        """
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
        """
        Detect landmarks with visibility/presence confidence scores.
        
        MediaPipe provides visibility scores for each landmark, indicating
        the likelihood that the landmark is visible in the image.
        
        Args:
            image: Input image as file path, PIL Image, or numpy array
            face_index: Index of the face to extract landmarks from
            
        Returns:
            Tuple of (landmarks array of shape (478, 2), list of confidence scores),
            or None if detection fails
        """
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
                
                # Store visibility score (0-1, higher is more visible)
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
        """
        Detect and return 478 facial landmarks in 3D space.
        
        MediaPipe also provides depth (z) coordinate, which represents
        the depth relative to the face center. Larger values mean the 
        landmark is closer to the camera.
        
        Args:
            image: Input image as file path, PIL Image, or numpy array
            face_index: Index of the face to extract landmarks from
            
        Returns:
            numpy array of shape (478, 3) containing [x, y, z] coordinates,
            where x and y are in pixel space and z is relative depth,
            or None if detection fails
        """
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
                landmarks_3d[idx, 2] = landmark.z * width  # Scale z by width for consistency
            
            return landmarks_3d
            
        except Exception as e:
            print(f"Error during 3D landmark detection: {e}")
            return None
    
    def batch_process(
        self,
        image_paths: List[Union[str, Path]]
    ) -> List[Optional[np.ndarray]]:
        """
        Process multiple images and return landmarks for each.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of landmark arrays (or None for failed detections)
        """
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
        """
        Extract a subset of landmarks by their indices.
        
        Useful for extracting specific facial features like eyes, nose, or mouth.
        
        Args:
            landmarks: Full landmark array of shape (478, 2) or (478, 3)
            indices: List of landmark indices to extract
            
        Returns:
            Subset of landmarks
            
        Example:
            >>> # Get landmarks for the nose tip and surrounding area
            >>> nose_indices = [1, 2, 3, 4, 5, 6, 49, 129, 203]
            >>> nose_landmarks = detector.get_landmark_subset(landmarks, nose_indices)
        """
        return landmarks[indices]
    
    @staticmethod
    def get_face_region_indices() -> dict:
        """
        Get predefined indices for different facial regions.
        
        Returns:
            Dictionary mapping region names to landmark index lists
            
        Reference:
            MediaPipe Face Mesh landmark topology:
            https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
        """
        return {
            # Silhouette / Face oval
            'face_oval': [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
            ],
            
            # Left eye
            'left_eye': [
                33, 7, 163, 144, 145, 153, 154, 155, 133,
                173, 157, 158, 159, 160, 161, 246
            ],
            
            # Right eye
            'right_eye': [
                362, 382, 381, 380, 374, 373, 390, 249,
                263, 466, 388, 387, 386, 385, 384, 398
            ],
            
            # Left eyebrow
            'left_eyebrow': [
                46, 53, 52, 65, 55, 70, 63, 105, 66, 107
            ],
            
            # Right eyebrow
            'right_eyebrow': [
                276, 283, 282, 295, 285, 300, 293, 334, 296, 336
            ],
            
            # Nose bridge
            'nose_bridge': [
                168, 6, 197, 195, 5, 4, 1, 19, 94, 2
            ],
            
            # Nose tip
            'nose_tip': [
                4, 5, 195, 197
            ],
            
            # Lips outer
            'lips_outer': [
                61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                291, 185, 40, 39, 37, 0, 267, 269, 270, 409
            ],
            
            # Lips inner
            'lips_inner': [
                78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
                308, 191, 80, 81, 82, 13, 312, 311, 310, 415
            ],
            
            # Left iris (requires refine_landmarks=True)
            'left_iris': [
                468, 469, 470, 471, 472
            ],
            
            # Right iris (requires refine_landmarks=True)
            'right_iris': [
                473, 474, 475, 476, 477
            ]
        }
    
    def visualize_landmarks(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        landmarks: Optional[np.ndarray] = None,
        show_indices: bool = False,
        point_size: int = 1,
        point_color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """
        Visualize detected landmarks on the image.
        
        Args:
            image: Input image
            landmarks: Precomputed landmarks (if None, will detect them)
            show_indices: Whether to show landmark indices as text
            point_size: Size of the landmark points
            point_color: RGB color tuple for landmark points
            
        Returns:
            Image with landmarks drawn on it
        """
        try:
            # Preprocess image
            image_array, height, width = self._preprocess_image(image)
            
            # Get landmarks if not provided
            if landmarks is None:
                landmarks = self.get_lmk_478(image)
                if landmarks is None:
                    return image_array
            
            # Create a copy for drawing
            output_image = image_array.copy()
            
            # Draw landmarks
            for idx, (x, y) in enumerate(landmarks):
                # Draw point
                cv2.circle(
                    output_image,
                    (int(x), int(y)),
                    point_size,
                    point_color,
                    -1
                )
                
                # Optionally draw index
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


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage demonstrating the FacialLandmarkDetector capabilities.
    """
    import sys
    
    # Initialize detector with refined landmarks (478 points)
    detector = FacialLandmarkDetector(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    
    # Example 1: Detect landmarks from file path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        print(f"Processing image: {image_path}")
        
        # Get 478 landmarks
        landmarks = detector.get_lmk_478(image_path)
        
        if landmarks is not None:
            print(f"✓ Successfully detected {len(landmarks)} landmarks")
            print(f"  Landmark shape: {landmarks.shape}")
            print(f"  Data type: {landmarks.dtype}")
            print(f"\nSample landmarks:")
            print(f"  Nose tip (index 4): {landmarks[4]}")
            print(f"  Left eye center (index 468): {landmarks[468]}")
            print(f"  Right eye center (index 473): {landmarks[473]}")
            
            # Get face region indices
            regions = detector.get_face_region_indices()
            print(f"\nAvailable regions: {list(regions.keys())}")
            
            # Extract nose landmarks
            nose_landmarks = detector.get_landmark_subset(landmarks, regions['nose_tip'])
            print(f"\nNose tip landmarks:\n{nose_landmarks}")
            
            # Get 3D landmarks
            landmarks_3d = detector.get_lmk_478_3d(image_path)
            if landmarks_3d is not None:
                print(f"\n✓ 3D landmarks shape: {landmarks_3d.shape}")
            
            # Visualize (save to file)
            output_image = detector.visualize_landmarks(image_path, landmarks)
            output_path = "landmarks_visualization.jpg"
            cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
            print(f"\n✓ Visualization saved to: {output_path}")
        else:
            print("✗ Failed to detect landmarks")
    else:
        print("Usage: python facial_landmark_detector.py <image_path>")
        print("\nExample:")
        print("  python facial_landmark_detector.py face_image.jpg")
