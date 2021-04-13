# KNN Project - Traffic Sign Detector
# Authors: Daniel Konecny (xkonec75), Jan Pavlus (xpavlu10), David Sedlak (xsedla1d)

import os
import random
import cv2

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode

from classes import grouped_classes_dict
from dataset import load_mapillary_dataset


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


def train(cfg):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    return trainer


def predict(cfg, dataset):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold

    predictor = DefaultPredictor(cfg)

    dataset_dicts = load_mapillary_dataset("mapillary", "val")
    for d in random.sample(dataset_dicts, 3):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                       metadata=MetadataCatalog.get(dataset),
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
