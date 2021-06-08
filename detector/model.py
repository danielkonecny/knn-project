# KNN Project - Traffic Sign Detector
# Authors: Daniel Konecny (xkonec75), Jan Pavlus (xpavlu10), David Sedlak (xsedla1d)

import os
import random
import cv2
import datetime

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode

from detector.classes import grouped_classes_dict
from detector.dataset import load_mapillary_dataset


def define_model(train_dts, val_dts, device, model, lr, iterations, batch_size):
    cfg = get_cfg()
    cfg.OUTPUT_DIR = "output-training-" + datetime.datetime.now().strftime("%d-%m-%Y-(%H:%M:%S)")
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.MODEL.DEVICE = device
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)

    cfg.DATASETS.TRAIN = (train_dts,)
    cfg.DATASETS.TEST = (val_dts,)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.MAX_ITER = iterations
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.TEST.EVAL_PERIOD = 50
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(grouped_classes_dict.keys())

    return cfg


def train(cfg):
    os.makedirs(cfg.OUTPUT_DIR)

    # log model setup
    with open(cfg.OUTPUT_DIR + "/model-setup.log", "w") as f:
        f.write(str(cfg))

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    return trainer


def predict(cfg, path, dataset):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold

    predictor = DefaultPredictor(cfg)

    dataset_dicts = load_mapillary_dataset(path, "val")
    for d in random.sample(dataset_dicts, 3):
        im = cv2.imread(d["file_name"])
        # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        outputs = predictor(im)
        v = Visualizer(
            im[:, :, ::-1],
            metadata=MetadataCatalog.get(dataset),
            scale=0.5,
            instance_mode=ColorMode.IMAGE  # remove the colors of unsegmented pixels
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        print(outputs["instances"].pred_classes)
        print(outputs["instances"].pred_boxes)
        cv2.imshow('', out.get_image()[:, :, ::-1])
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        cv2.waitKey(1)


def evaluate(cfg, trainer, threshold=0.7, weights=None):
    if not weights == "unchanged":
        if weights:
            cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, weights)
        else:
            cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

    evaluator = COCOEvaluator(
        "traffic_signs_val",
        distributed=False,
        output_dir=cfg.OUTPUT_DIR
    )
    val_loader = build_detection_test_loader(cfg, "traffic_signs_val")
    print(inference_on_dataset(trainer.model, val_loader, evaluator))


def load_model(weights, base_model, device, threshold=0.6):
    cfg = get_cfg()
    cfg.OUTPUT_DIR = "output-" + datetime.datetime.now().strftime("%d-%m-%Y-(%H:%M:%S)")
    cfg.merge_from_file(model_zoo.get_config_file(base_model))
    cfg.MODEL.DEVICE = device
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(grouped_classes_dict.keys())

    return cfg
