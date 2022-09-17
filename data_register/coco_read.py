import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json


def register_coco_data():
    CLASS_NAMES =['closed_eye','closed_mouth', 'open_eye', 'open_mouth']

    # 数据集路径
    DATASET_ROOT = './datasets/coco'
    ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')

    TRAIN_PATH = os.path.join(os.path.dirname(DATASET_ROOT), 'train', "images")
    VAL_PATH = os.path.join(os.path.dirname(DATASET_ROOT), 'val', "images")

    TRAIN_JSON = os.path.join(ANN_ROOT, 'instances_train.json')
    #VAL_JSON = os.path.join(ANN_ROOT, 'val.json')
    VAL_JSON = os.path.join(ANN_ROOT, 'instances_val.json')

    # 声明数据集的子集
    PREDEFINED_SPLITS_DATASET = {
        "coco_my_train": (TRAIN_PATH, TRAIN_JSON),
        "coco_my_val": (VAL_PATH, VAL_JSON),
    }
    #训练集
    DatasetCatalog.register("coco_my_train", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH))
    MetadataCatalog.get("coco_my_train").set(thing_classes=CLASS_NAMES,  # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
                                                    evaluator_type='coco', # 指定评估方式
                                                    json_file=TRAIN_JSON,
                                                    image_root=TRAIN_PATH)

    #DatasetCatalog.register("coco_my_val", lambda: load_coco_json(VAL_JSON, VAL_PATH, "coco_2017_val"))
    #验证/测试集
    DatasetCatalog.register("coco_my_val", lambda: load_coco_json(VAL_JSON, VAL_PATH))
    MetadataCatalog.get("coco_my_val").set(thing_classes=CLASS_NAMES, # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
                                                evaluator_type='coco', # 指定评估方式
                                                json_file=VAL_JSON,
                                                image_root=VAL_PATH)
    return CLASS_NAMES