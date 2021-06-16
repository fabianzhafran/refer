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
from detectron2.engine import DefaultTrainer

def get_refer_dicts():
    refer = REFER(dataset='refcoco', data_root='./data', splitBy='google')
    ref_ids = refer.getRefIds(split='train')
    refer_train_list = []

    for ref_id in ref_ids:
        new_ref = {}
        ref = refer.loadRefs(ref_id)[0]
        if len(ref['sentences']) < 2:
            continue

        # pprint(ref)
        # print('The label is %s.'.format(refer.Cats[ref['category_id']]))
        # print('bbox: ')
        # print(refer.getRefBox(ref_id))
        new_ref['annotations'] = refer.loadAnns(ref['ann_id'])
        for i in range(len(new_ref['annotations'])):
            new_ref['annotations'][i]["bbox_mode"] = BoxMode.XYXY_ABS


        # plt.figure()
        # refer.showMask(ref)
        # plt.show()
        new_ref['height'] = refer.loadImgs(ref['image_id'])[0]['height']
        new_ref['width'] = refer.loadImgs(ref['image_id'])[0]['width']

        new_ref['file_name'] = "data/images/mscoco/images/train2014/" + refer.loadImgs(ref['image_id'])[0]['file_name']
        new_ref['image_id'] = ref['image_id']
        refer_train_list.append(new_ref)
    return refer_train_list

def get_refer_classes():
    refer = REFER(dataset='refcoco', data_root='./data', splitBy='google')
    
    lastIdx = 1
    for key, value in refer.Cats.items():
        lastIdx = max(lastIdx, int(key))
    list_classes = ['None' for i in range(lastIdx+1)]
    for key, value in refer.Cats.items():
        list_classes[int(key)] = value
    return list_classes

d = "train"
DatasetCatalog.register("refer_" + d, lambda d=d: get_refer_dicts())
MetadataCatalog.get("refer_" + d).set(thing_classes=get_refer_classes())
# refer_metadata = MetadataCatalog.get("refer_train")

NUM_CLASSES = len(get_refer_classes())
OUTPUT_DIR = "fine_tuned_model"

cfg = get_cfg()
cfg.merge_from_file("/projectnb/statnlp/gik/py-bottom-up-attention/configs/VG-Detection/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml")
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_C4_3x.yaml"))
cfg.DATASETS.TRAIN = ("refer_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "https://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr.pkl"
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025 
cfg.SOLVER.MAX_ITER = 5000    
# cfg.SOLVER.STEPS = []        
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 8
cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES

os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
