import torchvision
import cv2
import time
import torch
import numpy as np
from functools import partial
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from retina_model.config import NUM_CLASSES, DEVICE

import logging

logger = logging.getLogger(__name__)
logger.setLevel(0)



def create_model(num_classes=91):
    """
    Creates a RetinaNet-ResNet50-FPN v2 model pre-trained on COCO.
    Replaces the classification head for the required number of classes.
    """
    model = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1)
    num_anchors = model.head.classification_head.num_anchors

    # Replace the classification head
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256, num_anchors=num_anchors, num_classes=num_classes, norm_layer=partial(torch.nn.GroupNorm, 32)
    )
    return model


class DetectionModel:
    def __init__(self):
        logger.info("Creating model start...")

        try:
            self.detect_model = create_model(num_classes=NUM_CLASSES)
            checkpoint = torch.load("models/best_retina_model.pth", map_location=DEVICE)
            self.detect_model.load_state_dict(checkpoint["model_state_dict"])
            self.detect_model.to(DEVICE).eval()
        except Exception as ex:
            logger.critical(f"Error creating model. {ex}")

        logger.info("Creating model success!")

    def detect(self, orig_image, resize_dim=None, threshold=0.25):
        """
        Runs inference on a single image (OpenCV BGR or NumPy array).
        - resize_dim: if not None, we resize to (resize_dim, resize_dim)
        - threshold: detection confidence threshold
        Returns: processed image with bounding boxes drawn.
        """

        logger.info("Detection persons start...")

        image = orig_image.copy()
        # Optionally resize for inference.
        if resize_dim is not None:
            image = cv2.resize(image, (resize_dim, resize_dim))

        # Convert BGR to RGB, normalize [0..1]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # Move channels to front (C,H,W)
        image_tensor = torch.tensor(image_rgb.transpose(2, 0, 1), dtype=torch.float).unsqueeze(0)
        start_time = time.time()
        # Inference
        with torch.no_grad():
            outputs = self.detect_model(image_tensor)
        end_time = time.time()
        # Get the current fps.
        fps = 1 / (end_time - start_time)
        fps_text = f"FPS: {fps:.2f}"
        # Move outputs to CPU numpy
        outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]
        boxes = outputs[0]["boxes"].numpy()
        scores = outputs[0]["scores"].numpy()
        labels = outputs[0]["labels"].numpy().astype(int)

        # Filter out boxes with low confidence
        valid_idx = np.where(scores >= threshold)[0]
        boxes = boxes[valid_idx].astype(int)
        labels = labels[valid_idx]

        h_orig, w_orig = orig_image.shape[:2]

        # If we resized for inference, rescale boxes back to orig_image size
        if resize_dim is not None:
            h_orig, w_orig = orig_image.shape[:2]
            h_new, w_new = resize_dim, resize_dim
            # scale boxes
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] / w_new) * w_orig
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] / h_new) * h_orig

        # Draw bounding boxes
        for box, label_idx in zip(boxes, labels):
            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 3)

        logger.info("Detection persons success!")
        return orig_image

if __name__ == "__main__":
    model = create_model(num_classes=NUM_CLASSES)
    print(model)
    # Total parameters:
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    # Trainable parameters:
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
