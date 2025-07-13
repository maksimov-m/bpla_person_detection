from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

args = dict(model="yolo11n.pt", data="coco8.yaml", epochs=3, )

trainer = DetectionTrainer(overrides=args)
