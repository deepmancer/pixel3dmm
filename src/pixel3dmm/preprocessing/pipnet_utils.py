import importlib
import os
import sys
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
import mediapipe as mp
from pathlib import Path


from pixel3dmm.preprocessing.PIPNet.FaceBoxesV2.faceboxes_detector import *
from pixel3dmm.preprocessing.PIPNet.lib.networks import *
from pixel3dmm.preprocessing.PIPNet.lib.functions import *
from pixel3dmm.preprocessing.PIPNet.lib.mobilenetv3 import mobilenetv3_large
from pixel3dmm import env_paths

# Add utils directory to path for FacialLandmarkDetector
from pixel3dmm.preprocessing.facial_landmark_detector import FacialLandmarkDetector


def detect_face_mediapipe(image):
    """
    Detect face in image using MediaPipe face detection.
    
    Args:
        image: BGR image from cv2.imread
        
    Returns:
        Tuple of (x, y, w, h) bounding box or None if no face detected
    """
    mp_face_detection = mp.solutions.face_detection
    
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


def crop_face_from_bbox_mediapipe(image, bbox, scale=1.8):
    """
    Crop face from image using bounding box with scaling.
    
    Args:
        image: Input image
        bbox: Tuple of (x, y, w, h) bounding box
        scale: Scale factor for cropping (default 1.8)
        
    Returns:
        Tuple of (cropped_image, (x1, y1, x2, y2) crop coordinates)
    """
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
    
    # Make square if not already
    if cropped.shape[0] != cropped.shape[1]:
        max_dim = max(cropped.shape[0], cropped.shape[1])
        square = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        y_offset = (max_dim - cropped.shape[0]) // 2
        x_offset = (max_dim - cropped.shape[1]) // 2
        square[y_offset:y_offset+cropped.shape[0], x_offset:x_offset+cropped.shape[1]] = cropped
        cropped = square
    
    return cropped, (x1, y1, x2, y2)


def preprocess_with_mediapipe_cropping(
    video_or_images_path: str,
    output_dir: str,
    target_size: int = 512,
    start_frame: int = 0
):
    """
    Preprocess images with MediaPipe face detection and cropping.
    
    This function:
    1. Detects faces using MediaPipe
    2. Crops and resizes to target_size x target_size
    3. Saves to rgb/, cropped/, and arcface/ directories
    4. Saves crop parameters
    
    Args:
        video_or_images_path: Path to input images or directory
        output_dir: Output directory for preprocessed data
        target_size: Target size for cropped images (default 512)
        start_frame: Starting frame index (default 0)
        
    Returns:
        True if successful
    """
    if os.path.isdir(video_or_images_path):
        vid_name = os.path.basename(video_or_images_path.rstrip('/'))
        image_dir = Path(video_or_images_path)
        image_files = sorted([f for f in image_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    else:
        vid_name = Path(video_or_images_path).stem
        image_dir = Path(video_or_images_path).parent
        image_files = [Path(video_or_images_path)]
    
    output_base = Path(output_dir)
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
        
        cropped, crop_coords = crop_face_from_bbox_mediapipe(image, bbox, scale=1.8)
        resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
        
        cv2.imwrite(str(cropped_dir / numbered_filename), resized)
        cv2.imwrite(str(arcface_dir / numbered_filename), resized)
        
        # Save crop box parameters
        crop_params_dir = output_base / 'crop_params'
        crop_params_dir.mkdir(exist_ok=True)
        crop_params = {
            'original_bbox': list(bbox),  # [x, y, w, h]
            'crop_coords': list(crop_coords),  # [x1, y1, x2, y2]
            'original_size': [image.shape[1], image.shape[0]],  # [width, height]
            'cropped_size': [resized.shape[1], resized.shape[0]],  # [width, height]
            'scale_factor': 1.8
        }
        np.save(str(crop_params_dir / f"{frame_number:05d}.npy"), crop_params)
        
        success_count += 1
    
    print(f"\nSuccessfully processed {success_count}/{len(image_files)} images")
    print(f"Output directory: {output_base}")
    
    if success_count == 0:
        raise ValueError("No faces detected in any images. Please check your input images.")
    
    return True


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError( "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError( "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y

def get_cstm_crop(image, detections, detections_max, max_bbox : bool = False):
    #Image.fromarray(image).show()
    image_width = image.shape[1]
    image_height = image.shape[0]

    det_box_scale = 1.42 #2.0#1.42
    if detections[4]*1.42 * detections[5]*1.42 < detections_max[4] * 1.1 * detections_max[5] * 1.1:
        detections = detections_max
        det_box_scale = 1.1

    det_xmin = detections[2]
    det_ymin = detections[3]
    det_width = detections[4]
    det_height = detections[5]
    if det_width > det_height:
        det_ymin -= (det_width - det_height)//2
        det_height = det_width
    if det_width < det_height:
        det_xmin -= (det_height - det_width)//2
        det_width = det_height

    det_xmax = det_xmin + det_width - 1
    det_ymax = det_ymin + det_height - 1


    det_xmin -= int(det_width * (det_box_scale - 1) / 2)
    det_ymin -= int(det_height * (det_box_scale - 1) / 2)
    det_xmax += int(det_width * (det_box_scale - 1) / 2)
    det_ymax += int(det_height * (det_box_scale - 1) / 2)
    if det_xmin < 0 or det_ymin < 0:
        min_overflow = min(det_xmin, det_ymin)
        det_xmin += -min_overflow
        det_ymin += -min_overflow
    if det_xmax > image_width -1 or det_ymax > image_height - 1:
        max_overflow = max(det_xmax - image_width -1, det_ymax - image_height-1)
        det_xmax -= max_overflow
        det_ymax -= max_overflow

    det_width = det_xmax - det_xmin + 1
    det_height = det_ymax - det_ymin + 1
    det_crop = image[det_ymin:det_ymax, det_xmin:det_xmax, :]
    return det_crop, det_ymin, det_ymax, det_xmin, det_xmax
    #Image.fromarray(det_crop).show()
    #exit()


def demo_image(image_dir, pid, save_dir, preprocess, cfg, input_size, net_stride, num_nb, use_gpu, flip=False, start_frame=0,
               vertical_crop : bool = False,
               static_crop : bool = False,
               max_bbox : bool = False,
               disable_cropping : bool = False,
               ):

    if cfg.use_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    detector = FaceBoxesDetector('FaceBoxes', f'{env_paths.CODE_BASE}/src/pixel3dmm/preprocessing/PIPNet/FaceBoxesV2/weights/FaceBoxesV2.pth', use_gpu, device)
    my_thresh = 0.6
    det_box_scale = 1.2
    meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface(
        os.path.join(f'{env_paths.CODE_BASE}/src/pixel3dmm/preprocessing/', 'PIPNet', 'data', cfg.data_name, 'meanface.txt'), cfg.num_nb)

    if cfg.backbone == 'resnet18':
        resnet18 = models.resnet18(pretrained=cfg.pretrained)
        net = Pip_resnet18(resnet18, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size,
                           net_stride=cfg.net_stride)
    elif cfg.backbone == 'resnet50':
        resnet50 = models.resnet50(pretrained=cfg.pretrained)
        net = Pip_resnet50(resnet50, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size,
                           net_stride=cfg.net_stride)
    elif cfg.backbone == 'resnet101':
        resnet101 = models.resnet101(pretrained=cfg.pretrained)
        net = Pip_resnet101(resnet101, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size,
                            net_stride=cfg.net_stride)
    elif cfg.backbone == 'mobilenet_v2':
        mbnet = models.mobilenet_v2(pretrained=cfg.pretrained)
        net = Pip_mbnetv2(mbnet, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
    elif cfg.backbone == 'mobilenet_v3':
        mbnet = mobilenetv3_large()
        if cfg.pretrained:
            mbnet.load_state_dict(torch.load('lib/mobilenetv3-large-1cd25616.pth'))
        net = Pip_mbnetv3(mbnet, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
    else:
        print('No such backbone!')
        exit(0)


    net = net.to(device)

    weight_file = os.path.join(save_dir, 'epoch%d.pth' % (cfg.num_epochs - 1))
    state_dict = torch.load(weight_file, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()

    if start_frame > 0:
        files = [f for f in os.listdir(f'{image_dir}/') if f.endswith('.jpg') or f.endswith('.png') and (((int(f.split('_')[-1].split('.')[0])-start_frame) % 3 )== 0)]
    else:
        files = [f for f in os.listdir(f'{image_dir}/') if f.endswith('.jpg') or f.endswith('.png')]
    files.sort()

    if not vertical_crop:
        all_detections = []
        all_images = []
        #all_normals = []
        succ_files = []
        for file_name in files:
                image = cv2.imread(f'{image_dir}/{file_name}')
                #normals = cv2.imread(f'{image_dir}/../normals/{file_name[:-4]}.png')

                if len(image.shape) < 3 or image.shape[-1] != 3:
                    continue

                image_height, image_width, _ = image.shape



                detections, _ = detector.detect(image, my_thresh, 1)
                dets_filtered = [det for det in detections if det[0] == 'face']
                dets_filtered.sort(key=lambda x: -1 * x[1])
                detections = dets_filtered
                if detections[0][1] < 0.75:
                    raise ValueError("Found face with too low detections confidence as max confidence")
                all_detections.append(detections[0])
                all_images.append(image)
                #all_normals.append(normals)
                succ_files.append(file_name)

        assert static_crop, 'Other options currently not supported anymore'
        if static_crop:
            #if max_bbox:
            det1_max = np.min(np.array([x[2] for x in all_detections]), axis=0)
            det2_max = np.min(np.array([x[3] for x in all_detections]), axis=0)
            det3_max = np.max(np.array([x[4]+x[2]-det1_max for x in all_detections]), axis=0)
            det4_max = np.max(np.array([x[5]+x[3]-det2_max for x in all_detections]), axis=0)
            det1 = np.mean(np.array([x[2] for x in all_detections]), axis=0)
            det2 = np.mean(np.array([x[3] for x in all_detections]), axis=0)
            det3 = np.mean(np.array([x[4] for x in all_detections]), axis=0)
            det4 = np.mean(np.array([x[5] for x in all_detections]), axis=0)

            det_smoothed = np.stack([det1, det2, det3, det4], axis=0).astype(np.int32)
            det_smoothed_max = np.stack([det1_max, det2_max, det3_max, det4_max], axis=0).astype(np.int32)
            all_detections_smoothed = []  # = [[x[0], x[1], x_smoothed[0], x_smoothed[1], x_smoothed[2], x_smoothed[3]] for x, x_smoothed in zip()]
            all_detections_max_smoothed = []  # = [[x[0], x[1], x_smoothed[0], x_smoothed[1], x_smoothed[2], x_smoothed[3]] for x, x_smoothed in zip()]
            for i, det in enumerate(all_detections):
                all_detections_smoothed.append(
                    [det[0], det[1], det_smoothed[0], det_smoothed[1], det_smoothed[2], det_smoothed[3]])
                all_detections_max_smoothed.append(
                    [det[0], det[1], det_smoothed_max[0], det_smoothed_max[1], det_smoothed_max[2], det_smoothed_max[3]])
            all_detections = all_detections_smoothed
            all_detections_max = all_detections_max_smoothed
        else:
            if len(all_detections) > 11:
                WINDOW_LENGTH = 11
                det1 = smooth(np.array([x[2] for x in all_detections]), window_len=WINDOW_LENGTH)
                det2 = smooth(np.array([x[3] for x in all_detections]), window_len=WINDOW_LENGTH)
                det3 = smooth(np.array([x[4] for x in all_detections]), window_len=WINDOW_LENGTH)
                det4 = smooth(np.array([x[5] for x in all_detections]), window_len=WINDOW_LENGTH)
                det_smoothed = np.stack([det1, det2,det3,det4], axis=1).astype(np.int32)
                all_detections_smoothed = [] #= [[x[0], x[1], x_smoothed[0], x_smoothed[1], x_smoothed[2], x_smoothed[3]] for x, x_smoothed in zip()]
                for i, det in enumerate(all_detections):
                    all_detections_smoothed.append([det[0], det[1], det_smoothed[i, 0], det_smoothed[i, 1], det_smoothed[i, 2], det_smoothed[i, 3]])
                all_detections = all_detections_smoothed
        # TODO: smooth detections!!!
        for file_name, detection, detection_max, image in zip(succ_files, all_detections, all_detections_max, all_images):

                        if not disable_cropping:
                            img_crop, det_ymin, det_ymax, det_xmin, det_xmax = get_cstm_crop(image, detection, detection_max, max_bbox=max_bbox)
                            #n_crop = get_cstm_crop(normals, detection)
                            image = img_crop
                            
                            # store cropping information:
                            if not os.path.exists(f'{image_dir}/../crop_ymin_ymax_xmin_xmax.npy'):
                                np.save(f'{image_dir}/../crop_ymin_ymax_xmin_xmax.npy', np.array([det_ymin, det_ymax, det_xmin, det_xmax]))
                        
                        # save cropped image (or original if disable_cropping=True)
                        os.makedirs(f'{image_dir}/../cropped/', exist_ok=True)
                        #os.makedirs(f'{image_dir}/../cropped_normals/', exist_ok=True)
                        cv2.imwrite(f'{image_dir}/../cropped/{file_name}', cv2.resize(image, (512, 512)))
                        #cv2.imwrite(f'{image_dir}/../cropped_normals/{file_name[:-4]}.png', cv2.resize(n_crop, (512, 512)))
    else:
        for file_name in files:
            image = cv2.imread(f'{image_dir}/{file_name}')
            if image.shape[0] != image.shape[1]:
                image = image[220:-220, 640:-640, :]
            os.makedirs(f'{image_dir}/../cropped/', exist_ok=True)
            cv2.imwrite(f'{image_dir}/../cropped/{file_name}', cv2.resize(image, (512, 512)))


    # run landmark detection
    lms = []
    image_dir = f'{image_dir}/../cropped/'
    for file_name in files:
                image = cv2.imread(f'{image_dir}/{file_name}')

                if len(image.shape) < 3 or image.shape[-1] != 3:
                    continue
                if flip:
                    image = cv2.transpose(image)

                image_height, image_width, _ = image.shape
                detections, _ = detector.detect(image, my_thresh, 1)
                pred_export = None
                dets_filtered = [det for det in detections if det[0] == 'face']
                dets_filtered.sort(key=lambda x: -1 * x[1])
                detections = dets_filtered


                print(detections)
                for i in range(min(1, len(detections))):
                    if detections[i][1] < 0.99:
                        continue
                    det_xmin = detections[i][2]
                    det_ymin = detections[i][3]
                    det_width = detections[i][4]
                    det_height = detections[i][5]
                    det_xmax = det_xmin + det_width - 1
                    det_ymax = det_ymin + det_height - 1


                    det_xmin -= int(det_width * (det_box_scale - 1) / 2)
                    # remove a part of top area for alignment, see paper for details
                    det_ymin += int(det_height * (det_box_scale - 1) / 2)
                    det_xmax += int(det_width * (det_box_scale - 1) / 2)
                    det_ymax += int(det_height * (det_box_scale - 1) / 2)
                    det_xmin = max(det_xmin, 0)
                    det_ymin = max(det_ymin, 0)
                    det_xmax = min(det_xmax, image_width - 1)
                    det_ymax = min(det_ymax, image_height - 1)
                    det_width = det_xmax - det_xmin + 1
                    det_height = det_ymax - det_ymin + 1
                    cv2.rectangle(image, (det_xmin, det_ymin), (det_xmax, det_ymax), (0, 0, 255), 2)
                    det_crop = image[det_ymin:det_ymax, det_xmin:det_xmax, :]
                    #np.save(f'{CROP_DIR}/{pid[:-4]}.npy', np.array([det_ymin, det_ymax, det_xmin, det_xmax]))
                    det_crop = cv2.resize(det_crop, (input_size, input_size))
                    inputs = Image.fromarray(det_crop[:, :, ::-1].astype('uint8'), 'RGB')
                    #inputs.show()
                    inputs = preprocess(inputs).unsqueeze(0)
                    inputs = inputs.to(device)
                    lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(net,
                                                                                                             inputs,
                                                                                                             preprocess,
                                                                                                             input_size,
                                                                                                             net_stride,
                                                                                                             num_nb)
                    lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
                    tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
                    tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
                    tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1, 1)
                    tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1, 1)
                    lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
                    lms_pred = lms_pred.cpu().numpy()
                    lms_pred_merge = lms_pred_merge.cpu().numpy()
                    pred_export = np.zeros([cfg.num_lms, 2])
                    for i in range(cfg.num_lms):
                        x_pred = lms_pred_merge[i * 2] * det_width
                        y_pred = lms_pred_merge[i * 2 + 1] * det_height
                        pred_export[i, 0] = (x_pred + det_xmin) / image_width
                        pred_export[i, 1] = (y_pred + det_ymin) / image_height
                        cv2.circle(image, (int(x_pred) + det_xmin, int(y_pred) + det_ymin), 1, (0, 0, 255), 2)
                        if i == 76:
                            cv2.circle(image, (int(x_pred) + det_xmin, int(y_pred) + det_ymin), 1, (255, 0, 0), 2)

                if pred_export is not None:
                    print('exporting stuff to ' + image_dir)
                    landmakr_dir =  f'{image_dir}/../PIPnet_landmarks/'
                    os.makedirs(landmakr_dir, exist_ok=True)
                    np.save(landmakr_dir + f'/{file_name[:-4]}.npy', pred_export)
                    lms.append(pred_export)
                    exp_dir = image_dir + '/../PIPnet_annotated_images/'
                    os.makedirs(exp_dir, exist_ok=True)
                    cv2.imwrite(exp_dir + f'/{file_name}', image)

                # cv2.imshow('1', image)
                # cv2.waitKey(0)

    lms = np.stack(lms, axis=0)
    os.makedirs(f'{image_dir}/../pipnet', exist_ok=True)
    np.save(f'{image_dir}/../pipnet/test.npy', lms)
    
    # Extract MediaPipe landmarks automatically
    print("\n" + "="*60)
    print("Extracting MediaPipe landmarks...")
    print("="*60)
    extract_mediapipe_landmarks_for_preprocessing(image_dir)


def extract_mediapipe_landmarks_for_preprocessing(image_dir):
    """
    Extract MediaPipe landmarks automatically during preprocessing.
    This function is called automatically after PIPNet landmark extraction.
    
    Args:
        image_dir: Path to the cropped images directory
    """
    from pathlib import Path
    from tqdm import tqdm
    import re
    
    # Get parent directory (preprocessed data folder)
    data_path = Path(image_dir).parent
    output_path = data_path / "mediapipe_landmarks"
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if already processed
    cropped_path = data_path / "cropped"
    if not cropped_path.exists():
        print(f"Warning: Cropped images folder not found: {cropped_path}")
        return
    
    image_files = sorted([f for f in cropped_path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    if len(image_files) == 0:
        print(f"Warning: No image files found in {cropped_path}")
        return

    print(f"Found {len(image_files)} images to process")
    
    # Load landmark indices mapping if available
    mapping_file = Path(env_paths.ASSETS) / "body_models/landmarks/flame/mediapipe_landmark_embedding.npz"
    landmark_indices = None
    
    if mapping_file.exists():
        try:
            mapping = np.load(str(mapping_file))
            landmark_indices = mapping['landmark_indices']
            print(f"Loaded landmark mapping: will extract {len(landmark_indices)} landmarks from 478")
        except Exception as e:
            print(f"Warning: Failed to load mapping file: {e}")
            print("Will extract all 478 landmarks")
    else:
        print(f"Warning: Mapping file not found at {mapping_file}")
        print("Will extract all 478 landmarks")
    
    # Initialize MediaPipe detector
    try:
        detector = FacialLandmarkDetector(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    except Exception as e:
        print(f"Error initializing MediaPipe detector: {e}")
        print("Skipping MediaPipe landmark extraction")
        return
    
    # Process each image
    successful = 0
    failed = 0
    failed_files = []
    
    for image_file in tqdm(image_files, desc="Extracting MediaPipe landmarks"):
        try:
            # Extract frame number from filename (handle different naming patterns)
            try:
                # Try to extract integer from filename
                frame_num = int(image_file.stem)
            except ValueError:
                # If filename is not pure integer, try to extract number
                match = re.search(r'(\d+)', image_file.stem)
                if match:
                    frame_num = int(match.group(1))
                else:
                    tqdm.write(f"Warning: Cannot extract frame number from {image_file.name}")
                    failed += 1
                    failed_files.append(image_file.name)
                    continue
            
            # Check if already processed
            output_file = output_path / f"mediapipe_lmk_{frame_num:05d}.npy"
            if output_file.exists():
                # Verify the file is valid
                try:
                    test_load = np.load(str(output_file))
                    expected_shape = (len(landmark_indices), 2) if landmark_indices is not None else (478, 2)
                    if test_load.shape == expected_shape:
                        successful += 1
                        continue
                    else:
                        tqdm.write(f"Warning: Existing file has wrong shape {test_load.shape}, re-extracting...")
                except Exception:
                    tqdm.write(f"Warning: Existing file is corrupted, re-extracting...")
            
            # Read image with cv2 and convert BGR to RGB
            image_bgr = cv2.imread(str(image_file))
            if image_bgr is None:
                failed += 1
                failed_files.append(image_file.name)
                tqdm.write(f"Error: Failed to read image {image_file.name}")
                continue
            
            # Get image dimensions for normalization
            image_height, image_width = image_bgr.shape[:2]
            
            # Convert BGR to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            # Extract 478 landmarks using RGB image (returns pixel coordinates)
            landmarks_478 = detector.get_lmk_478(image_rgb)
            
            if landmarks_478 is None:
                failed += 1
                failed_files.append(image_file.name)
                tqdm.write(f"Warning: Failed to detect face in {image_file.name} (frame {frame_num})")
                continue
            
            # Verify landmark shape
            if landmarks_478.shape != (478, 2):
                failed += 1
                failed_files.append(image_file.name)
                tqdm.write(f"Error: Unexpected landmark shape {landmarks_478.shape} for {image_file.name}")
                continue
            
            # Normalize landmarks to [0, 1] range (same format as PIPNet landmarks)
            # MediaPipe returns pixel coordinates, we need to normalize by image dimensions
            landmarks_478_normalized = landmarks_478.copy()
            landmarks_478_normalized[:, 0] = landmarks_478[:, 0] / image_width
            landmarks_478_normalized[:, 1] = landmarks_478[:, 1] / image_height
            
            # Subset to specified landmarks if mapping provided
            if landmark_indices is not None:
                landmarks = landmarks_478_normalized[landmark_indices]
            else:
                landmarks = landmarks_478_normalized
            
            # Verify final shape
            expected_shape = (len(landmark_indices), 2) if landmark_indices is not None else (478, 2)
            if landmarks.shape != expected_shape:
                failed += 1
                failed_files.append(image_file.name)
                tqdm.write(f"Error: Final landmark shape {landmarks.shape} != expected {expected_shape}")
                continue
            
            # Save normalized landmarks (same format as PIPNet: [0, 1] range)
            np.save(str(output_file), landmarks)
            successful += 1
            
        except Exception as e:
            failed += 1
            failed_files.append(image_file.name)
            tqdm.write(f"Error processing {image_file.name}: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("MediaPipe Landmark Extraction Complete")
    print("="*60)
    print(f"Successfully processed: {successful} / {len(image_files)}")
    print(f"Failed: {failed}")
    if failed > 0:
        print(f"\nFailed files:")
        for failed_file in failed_files[:10]:  # Show first 10
            print(f"  - {failed_file}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")
    if landmark_indices is not None:
        print(f"\nLandmark shape: ({len(landmark_indices)}, 2)")
    else:
        print(f"\nLandmark shape: (478, 2)")
    print(f"Format: Normalized coordinates [0, 1] (same as PIPNet)")
    print(f"Output saved to: {output_path}")
    print("="*60 + "\n")
    
    # Raise error if all extractions failed
    if successful == 0 and len(image_files) > 0:
        raise RuntimeError(
            "Failed to extract MediaPipe landmarks for any images. "
            "This will cause tracking to fail if use_mediapipe_landmarks=True. "
            "Please check the image quality and MediaPipe installation."
        )



