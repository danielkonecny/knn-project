# KNN Project - Traffic Sign Detector
# Authors: Daniel Konecny (xkonec75), Jan Pavlus (xpavlu10), David Sedlak (xsedla1d)

from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog, DatasetCatalog

from classes import grouped_classes_dict
from dataset import preview_dataset, load_mapillary_dataset
from model import define_model, train, predict, evaluate


if __name__ == '__main__':
    setup_logger()
    for d in ["train", "val"]:
        DatasetCatalog.register("traffic_signs_" + d, lambda d=d: load_mapillary_dataset("mapillary/", d))
        MetadataCatalog.get("traffic_signs_" + d).set(thing_classes=list(grouped_classes_dict.keys()))
    board_metadata = MetadataCatalog.get("traffic_signs_train")
    preview_dataset("traffic_signs_train")

    training_model = define_model()
    trainer = train(training_model)

    predict(training_model, "traffic_signs_train")

    evaluate(training_model, trainer)
