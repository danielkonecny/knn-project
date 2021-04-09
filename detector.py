# KNN Project - Traffic Sign Detector
# Authors: Daniel Konecny (xkonec75), Jan Pavlus (xpavlu10), David Sedlak (xsedla1d)


import os, json, random
import numpy as np
import cv2
import torch, torchvision

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from classes import grouped_classes_dict

setup_logger()


def define_model():
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = "cpu"
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

    cfg.DATASETS.TRAIN = ("traffic_signs_train",)
    cfg.DATASETS.TEST = ("traffic_signs_val",)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.0125  # pick a good LR
    cfg.SOLVER.MAX_ITER = 1000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.TEST.EVAL_PERIOD = 50
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(grouped_classes_dict.keys()) # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    return cfg


# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

def load_dataset(path):
    json_file = path+"/dataset.json"
    with open(json_file) as f:
        dataset_dicts = json.load(f)
    for i in dataset_dicts:
        filename = i["file_name"]
        i["file_name"] = path+"/"+filename
        for j in i["annotations"]:
            j["bbox_mode"] = BoxMode.XYWH_ABS
            j["category_id"] = int(j["category_id"])
    return dataset_dicts


def fetch_classes(path):
    annotations_folder = path+"/mtsd_v2_fully_annotated_annotation/mtsd_v2_fully_annotated/"
    _, _, annots = next(os.walk(annotations_folder + "/annotations"))
    classes = set()
    for annot_file in annots:
        with open(annotations_folder + "/annotations/" + annot_file) as f:
            a = json.load(f)
            for obj in a['objects']:
                classes.add(obj['label'])

    return classes


def load_mapillary_dataset(path, split="train"):
    annotations_folder = path+"/mtsd_v2_fully_annotated_annotation/mtsd_v2_fully_annotated/"
    train_images_folder = path+"/mtsd_v2_fully_annotated_images." + split + "/images/"
    images = set([os.path.splitext(f)[0] for f in os.listdir(train_images_folder) if os.path.isfile(os.path.join(train_images_folder, f))])

    with open(annotations_folder + "/splits/" + split + ".txt") as f:
        train_data = [i.rstrip("\n ") for i in f.readlines()]

    dataset_dicts = []
    for filename in train_data:
        if filename in images:
            try:
                with open(annotations_folder + "/annotations/" + filename + ".json") as f:
                    annotation = json.load(f)

                    annotations = []
                    for obj in annotation['objects']:
                        annot = {
                            'bbox' : [obj['bbox']['xmin'], obj['bbox']['ymin'], obj['bbox']['xmax'], obj['bbox']['ymax']],
                            'bbox_mode' : BoxMode.XYXY_ABS,
                            'category_id': grouped_classes_dict[obj['label'].split('--')[0]]
                        }
                        annotations.append(annot)
                    image_info = {
                        'image_id': filename,
                        'file_name': os.path.join(train_images_folder, filename + ".jpg"),
                        'height': annotation['height'],
                        'width': annotation['width'],
                        'annotations': annotations
                    }
                    dataset_dicts.append(image_info)
            except FileNotFoundError:
                print(f"Annotation file for {filename} not found.")

    return dataset_dicts


def preview_dataset():
    dataset_dicts = load_mapillary_dataset("mapillary", "train")
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=board_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow('', out.get_image()[:, :, ::-1])
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        cv2.waitKey(1)


def train(cfg):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    return trainer


def predict(cfg):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold

    predictor = DefaultPredictor(cfg)

    dataset_dicts = load_mapillary_dataset("mapillary", "val")
    for d in random.sample(dataset_dicts, 3):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                       metadata=board_metadata,
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE   # remove the colors of unsegmented pixels
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        print(outputs["instances"].pred_classes)
        print(outputs["instances"].pred_boxes)
        cv2.imshow('', out.get_image()[:, :, ::-1])
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        cv2.waitKey(1)


def evaluate(cfg, trainer):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold

    evaluator = COCOEvaluator("traffic_signs_val", distributed=False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "traffic_signs_val")
    print(inference_on_dataset(trainer.model, val_loader, evaluator))
    # another equivalent way to evaluate the model is to use `trainer.test`


if __name__ == '__main__':
    for d in ["train", "val"]:
        DatasetCatalog.register("traffic_signs_" + d, lambda d=d: load_mapillary_dataset("mapillary/", d))
        MetadataCatalog.get("traffic_signs_" + d).set(thing_classes=list(grouped_classes_dict.keys()))
    board_metadata = MetadataCatalog.get("traffic_signs_train")
    preview_dataset()

    training_model = define_model()
    trainer = train(training_model)

    predict(training_model)

    evaluate(training_model, trainer)
