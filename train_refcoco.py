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
from detectron2.engine import DefaultPredictor

from IPython.display import clear_output, Image, display
import PIL.Image
import io

def get_refer_dicts(part_split):
    refer = REFER(dataset='refcoco', data_root='./data', splitBy='google')
    ref_ids = refer.getRefIds(split=part_split)
    refer_train_list = []
    
    list_data_image = {}

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
            new_ref['annotations'][i]["bbox_mode"] = BoxMode.XYWH_ABS


        # plt.figure()
        # refer.showMask(ref)
        # plt.show()
        new_ref['height'] = refer.loadImgs(ref['image_id'])[0]['height']
        new_ref['width'] = refer.loadImgs(ref['image_id'])[0]['width']

        new_ref['file_name'] = "data/images/mscoco/images/train2014/" + refer.loadImgs(ref['image_id'])[0]['file_name']
        new_ref['image_id'] = ref['image_id']
        refer_train_list.append(new_ref)
        
        if ref['image_id'] in list_data_image:
            list_data_image[ref['image_id']].append(new_ref)
        else:
            list_data_image[ref['image_id']] = [new_ref]
        
        
    final_refer_train_list = []
    for image_id in list_data_image:
        final_ref = {}
        final_ref['image_id'] = image_id
        final_ref['file_name'] = list_data_image[image_id][0]['file_name']
        final_ref['height'] = list_data_image[image_id][0]['height']
        final_ref['width'] = list_data_image[image_id][0]['width']
        final_ref['annotations'] = []
        for obj_ref in list_data_image[image_id]:
            final_ref['annotations'] += obj_ref['annotations']
            
        final_refer_train_list.append(final_ref)
        
#     print(final_refer_train_list[:5])
    print(len(final_refer_train_list))
    return final_refer_train_list

def get_refer_classes():
    refer = REFER(dataset='refcoco', data_root='./data', splitBy='google')
    
    lastIdx = 1
    for key, value in refer.Cats.items():
        lastIdx = max(lastIdx, int(key))
    list_classes = ['None' for i in range(lastIdx+1)]
    for key, value in refer.Cats.items():
        list_classes[int(key)] = value
    return list_classes

def showarray(a, fn, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = io.BytesIO()
    PIL.Image.fromarray(a).save(fn, fmt)

d = "train"
DatasetCatalog.register("refer_" + d, lambda d=d: get_refer_dicts(d))
MetadataCatalog.get("refer_" + d).set(thing_classes=get_refer_classes())
refer_metadata = MetadataCatalog.get("refer_" + d)

d = "val"
DatasetCatalog.register("refer_" + d, lambda d=d: get_refer_dicts(d))
MetadataCatalog.get("refer_" + d).set(thing_classes=get_refer_classes())


# for debug (show some images with bbox)

refer_dataset = DatasetCatalog.get("refer_train")
for i in range(5):
    sample = refer_dataset[i]
    img = cv2.imread(sample["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("refer_train"), scale=0.5)
    vis = visualizer.draw_dataset_dict(sample)
    showarray(vis.get_image()[:, :, ::-1], "output/" + "sampleObj_%i.jpg"%i)


NUM_CLASSES = len(get_refer_classes())
OUTPUT_DIR = "fine_tuned_model"

cfg = get_cfg()
cfg.merge_from_file("/projectnb/statnlp/gik/py-bottom-up-attention/configs/VG-Detection/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml")
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_C4_3x.yaml"))
cfg.DATASETS.TRAIN = ("refer_train",)
cfg.DATASETS.TEST = ("refer_test",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "https://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr.pkl"
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025 
cfg.SOLVER.MAX_ITER = 30000    
# cfg.SOLVER.STEPS = []        
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 8
cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES

os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

# test on train
cfg.MODEL.WEIGHTS = "/projectnb/statnlp/gik/refer/output/model_final.pth"
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
predictor = DefaultPredictor(cfg)
for i in range(5):
    sample = refer_dataset[i]
    im = cv2.imread(sample["file_name"])
    outputs_obj_only = predictor(im)
    print(outputs_obj_only)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get("vg"), scale=1.2)
    out = v.draw_instance_predictions(outputs_obj_only["instances"].to("cpu"))
    showarray(out.get_image()[:, :, ::-1], "output/"+"samplePredictedObj_%i.jpg"%i)
