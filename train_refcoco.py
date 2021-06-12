from refer import REFER 
from pprint import pprint

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random

from detectron2.structures import BoxMode
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

def get_refer_dicts():
    refer = REFER(dataset='refcoco', data_root='./data', splitBy='google')
    ref_ids = refer.getRefIds(split='test')
    refer_train_list = []

    for ref_id in ref_ids:
        ref = refer.loadRefs(ref_id)[0]
        if len(ref['sentences']) < 2:
            continue

        # pprint(ref)
        # print('The label is %s.'.format(refer.Cats[ref['category_id']]))
        # print('bbox: ')
        # print(refer.getRefBox(ref_id))
        ref['bbox'] = refer.getRefBox(ref_id)
        ref['bbox_mode'] = BoxMode.XYXY_ABS

        # plt.figure()
        # refer.showMask(ref)
        # plt.show()
        ref['height'] = refer.loadImgs(ref['image_id'])[0]['height']
        ref['width'] = refer.loadImgs(ref['image_id'])[0]['width']

        ref['file_name'] = "data/images/mscoco/images/train2014/" + ref['file_name']
        refer_train_list.append(ref)
    return refer_train_list

def get_refer_classes():
    refer = REFER(dataset='refcoco', data_root='./data', splitBy='google')
    return refer.Cats

d = "train"
DatasetCatalog.register("refer_" + d, lambda d=d: get_refer_dicts())
MetadataCatalog.get("refer_" + d).set(thing_classes=get_refer_classes())
refer_metadata = MetadataCatalog.get("refer_train")

from detectron2.engine import DefaultTrainer

NUM_CLASSES = len(get_refer_classes())
OUTPUT_DIR = "fine_tuned_model"

cfg = get_cfg()
cfg.merge_from_file("/projectnb/statnlp/gik/py-bottom-up-attention/configs/VG-Detection/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml")
cfg.DATASETS.TRAIN = ("refer_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "https://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr.pkl"
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES

os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()