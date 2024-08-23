import glob
import os
import time
import tensorflow as tf
import numpy as np
import psutil    
import GPUtil
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as viz_utils
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import warnings

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, 
                        message="You are using `torch.load` with `weights_only=False`")

DEBUG = False

IMAGE_DIR = './images/test/'
IMAGE_PATHS = glob.glob(os.path.join(IMAGE_DIR, '*'))

# Load model
PATH_TO_SAVED_MODEL = './exported-models/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8' + "/saved_model"
SAM_CHECKPOINT = 'sam_vit_h_4b8939.pth'
MODEL_TYPE = 'vit_h'
DEVICE = 'gpu' if torch.cuda.is_available() else 'cpu'

# Load TensorFlow model
print('Loading TensorFlow model...', end='')
start_time = time.time()
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
end_time = time.time()
elapsed_time = end_time - start_time
print(f'Done! Took {elapsed_time:.2f} seconds')

# Free up memory if needed
torch.cuda.empty_cache()

# Load SAM model
print('Loading SAM model...', end='')
start_time = time.time()
sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(DEVICE)
predictor = SamPredictor(sam)
end_time = time.time()
elapsed_time = end_time - start_time
print(f'Done! Took {elapsed_time:.2f} seconds')

# Labels
PATH_TO_LABELS = './annotations/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS, use_display_name=True)

def log_hardware_usage():

    ram_usage = psutil.virtual_memory()
    cpu_usage = f"CPU RAM Free: {ram_usage.free / (1024 ** 3):.2f}GB | Used: {ram_usage.used / (1024 ** 3):.2f}GB | Util {ram_usage.percent:3.0f}% | Total {ram_usage.total / (1024 ** 3):.2f}GB"
    
    gpu = GPUtil.getGPUs()[0]
    gpu_usage = f"GPU RAM Free: {gpu.memoryFree / 1024:.2f}GB | Used: {gpu.memoryUsed / 1024:.2f}GB | Util {gpu.memoryUtil * 100:3.0f}% | Total {gpu.memoryTotal / 1024:.2f}GB"
    
    return cpu_usage, gpu_usage


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array."""
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)

def show_mask(mask):
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return mask_image

def show_points(coords, labels, ax, marker_size = 375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


for idx, image_path in enumerate(IMAGE_PATHS):
    print('Running inference for {}... '.format(image_path))

    start = time.time()
    image_np = load_image_into_numpy_array(image_path)
    if image_path.endswith('.jpg'):
        image_np = cv2.rotate(image_np, cv2.ROTATE_90_COUNTERCLOCKWISE)

    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    image_np_with_detections = image_np.copy()

    detection_boxes = tf.squeeze(detections['detection_boxes'], [0])
    detection_masks = tf.squeeze(detections['detection_masks'], [0])

    real_num_detection = tf.cast(detections['num_detections'][0], tf.int32)
    detection_boxes = tf.slice(detection_boxes, [0, 0],
                               [real_num_detection, -1])
    detection_masks = tf.slice(detection_masks, [0, 0, 0],
                               [real_num_detection, -1, -1])
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
        detection_masks, detection_boxes, image_np.shape[0], image_np.shape[1])
    detection_masks_reframed = tf.cast(
        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
    
    detections['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)

    detections['num_detections'] = int(detections['num_detections'][0])
    detections['detection_classes'] = detections['detection_classes'][0].numpy().astype(np.uint8)
    detections['detection_boxes'] = detections['detection_boxes'][0].numpy()
    detections['detection_scores'] = detections['detection_scores'][0].numpy()
    detections['detection_masks'] = detections['detection_masks'][0].numpy()

    image = Image.fromarray(np.uint8(image_np_with_detections))

    width, height = image.size
    (ymin, xmin, ymax, xmax) = detections['detection_boxes'][0]
    x_min, x_max, y_min, y_max = xmin * width, xmax * width, ymin * height, ymax * height # Boundig Box Coordinate
    
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    point = np.array([[center_x, y_max]])
    label = np.array([1])

    if DEBUG:
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            instance_masks=detections.get('detection_masks'),
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30)
        
    print(f"Bounding Box Coordinate: ({x_min}, {y_min}, {x_max}, {y_max})")

    predictor.set_image(image_np)
    masks, scores, logits = predictor.predict(
        point_coords = point,
        point_labels = label,
        multimask_output = True
    )
    max_idx = np.argmax(scores) 

    end = time.time()

    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    show_mask(masks[max_idx], plt.gca())
    show_points(point, label, plt.gca())
    plt.axis('off')
    plt.show()
    
    if DEBUG:
        output_path = f'images/test_annotated/{idx}.png'
        cv2.imwrite(output_path, image_np_with_detections)  
        print(f"Save as {output_path}")

    cpu_usage, gpu_usage= log_hardware_usage()
    print(cpu_usage)
    print(gpu_usage)
    print(f"Took {end - start} secs to annotate")
    print('-' * 50, end='\n\n')

print('Done!')
