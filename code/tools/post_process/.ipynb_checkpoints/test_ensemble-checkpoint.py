import time, argparse
from tqdm import tqdm
import os, cv2
import json
import mmcv
import glob 
import torch
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmcv.runner import wrap_fp16_model, load_checkpoint
from mmcv.cnn import fuse_conv_bn
from mmcv import Config
from mmcv.parallel import MMDataParallel
from ensemble_boxes import *

def parse_args():
    parser = argparse.ArgumentParser(description='json2submit_nms')
    parser.add_argument('--jsonfile', default='bbox-val.json', help='submit_file_name', type=str)
    args = parser.parse_args()
    return args

underwater_classes = ['Crack', 'Manhole', 'Net', 'Pothole', 'Patch-Crack', 'Patch-Net',
                      'Patch-Pothole', 'other']

def post_predictions(predictions, img_shape):
    bboxes_list, scores_list, labels_list = [], [], []
    for i, bboxes in enumerate(predictions):
        if len(bboxes) > 0:
            detect_label = i
            for bbox in bboxes:
                xmin, ymin, xmax, ymax, score = bbox.tolist()

                xmin /= img_shape[1]
                ymin /= img_shape[0]
                xmax /= img_shape[1]
                ymax /= img_shape[0]
                bboxes_list.append([xmin, ymin, xmax, ymax])
                scores_list.append(score)
                labels_list.append(detect_label)

    return bboxes_list, scores_list, labels_list
    
def main():
    args = parse_args()
    config_file1 = './ensemble_configs/cascade_rcnn_r50_rfp_carafe_sac.py'  # detectors_r50
    checkpoint_file1 = './ensemble_configs/cas-0bd921e5.pth'  
    config_file2 = './ensemble_configs/cascade_rcnn_s101_dcn_fpn.py'  # s101
    checkpoint_file2 = './ensemble_configs/s101_20-bd7b757b.pth'
    config_file3 = './ensemble_configs/cascade_rcnn_r2_101_dcn_fpn.py'  # r2_101
    checkpoint_file3 = './ensemble_configs/r2_101_20-487bd3ea.pth'
    config_file4 = './ensemble_configs/cascade_rcnn_r101_dcn_fpn.py'  # r101
    checkpoint_file4 = './ensemble_configs/r101_20-db83ab64.pth'
    config_file5 = './ensemble_configs/cascade_rcnn_x101_32x4d_dcn_fpn.py'  # x101_32x4d
    checkpoint_file5 = './ensemble_configs/x101_32x4d_20-f11fb360.pth'
    config_file6 = './ensemble_configs/cascade_rcnn_swin_small_fpn.py'  # swin_small
    checkpoint_file6 = './ensemble_configs/swin_small_e20-0df8a664.pth'
    
    device = 'cuda:0'
    cfg1 = Config.fromfile(config_file1)
    cfg2 = Config.fromfile(config_file2)
    cfg3 = Config.fromfile(config_file3)
    cfg4 = Config.fromfile(config_file4)
    cfg5 = Config.fromfile(config_file5)
    cfg6 = Config.fromfile(config_file6)
    
    # build model
    # model1
    model1 = build_detector(cfg1.model, test_cfg=cfg1.get('test_cfg'))
    load_checkpoint(model1, checkpoint_file1, map_location=device)
    # model2
    model2 = build_detector(cfg2.model, test_cfg=cfg2.get('test_cfg'))
    load_checkpoint(model2, checkpoint_file2, map_location=device)
    # model3
    model3 = build_detector(cfg3.model, test_cfg=cfg3.get('test_cfg'))
    load_checkpoint(model3, checkpoint_file3, map_location=device)
    # model4
    model4 = build_detector(cfg4.model, test_cfg=cfg4.get('test_cfg'))
    load_checkpoint(model4, checkpoint_file4, map_location=device)
    # model5
    model5 = build_detector(cfg5.model, test_cfg=cfg5.get('test_cfg'))
    load_checkpoint(model5, checkpoint_file5, map_location=device)
    # model6
    model6 = build_detector(cfg6.model, test_cfg=cfg6.get('test_cfg'))
    load_checkpoint(model6, checkpoint_file6, map_location=device)
    
    test_json_raw = json.load(open(cfg1.data.test.ann_file))
    imgid2name = {}
    for imageinfo in test_json_raw['images']:
        imgid = imageinfo['id']
        imgid2name[imageinfo['file_name']] = imgid
    wrap_fp16_model(model1)  # 采用fp16加速预测
    wrap_fp16_model(model2)
    wrap_fp16_model(model3)
    wrap_fp16_model(model4)
    wrap_fp16_model(model5)
    wrap_fp16_model(model6)
    
    # build the dataloader
    samples_per_gpu = cfg1.data.test.pop('samples_per_gpu', 1)  # aug_test不支持batch_size>1
    dataset = build_dataset(cfg1.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=4,
        dist=False,
        shuffle=False)
    model1 = MMDataParallel(model1, device_ids=[0])  # 为啥加？(不加就错了)
    model2 = MMDataParallel(model2, device_ids=[0])
    model3 = MMDataParallel(model3, device_ids=[0])
    model4 = MMDataParallel(model4, device_ids=[0])
    model5 = MMDataParallel(model5, device_ids=[0])
    model6 = MMDataParallel(model6, device_ids=[0])
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    model6.eval()
    
    json_results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result1 = model1(return_loss=False, rescale=True, **data)
            result2 = model2(return_loss=False, rescale=True, **data)
            result3 = model3(return_loss=False, rescale=True, **data)
            result4 = model4(return_loss=False, rescale=True, **data)
            result5 = model5(return_loss=False, rescale=True, **data)
            result6 = model6(return_loss=False, rescale=True, **data)
        batch_size = len(result1)
        assert len(result1) == len(result2)
        
        result1 = result1[0]  # 每次只输入一张
        result2 = result2[0]
        result3 = result3[0]
        result4 = result4[0]
        result5 = result5[0]
        result6 = result6[0]
        img_metas = data['img_metas'][0].data[0]
        img_shape = img_metas[0]['ori_shape']
        bboxes, scores, labels = post_predictions(result1, img_shape)
        e_bboxes, e_scores, e_labels = post_predictions(result2, img_shape)
        e_bboxes3, e_scores3, e_labels3 = post_predictions(result3, img_shape)
        e_bboxes4, e_scores4, e_labels4 = post_predictions(result4, img_shape)
        e_bboxes5, e_scores5, e_labels5 = post_predictions(result5, img_shape)
        e_bboxes6, e_scores6, e_labels6 = post_predictions(result6, img_shape)
        bboxes_list = [bboxes, e_bboxes, e_bboxes3, e_bboxes4, e_bboxes5, e_bboxes6]
        scores_list = [scores, e_scores, e_scores3, e_scores4, e_scores5, e_scores6]
        labels_list = [labels, e_labels, e_labels3, e_labels4, e_labels5, e_labels6]
        bboxes, scores, labels = weighted_boxes_fusion(
            bboxes_list,
            scores_list,
            labels_list,
            weights=[1.5, 1, 1, 1, 1, 1],
            iou_thr=0.6,
            skip_box_thr=0.0001,
            conf_type='avg')
#         basename = img_metas[0]['ori_filename']
#         image = cv2.imread(os.path.join(cfg1.data.test.img_prefix, basename))
        for (box, score, label) in zip(bboxes, scores, labels):
            xmin, ymin, xmax, ymax = box.tolist()
            xmin, ymin, xmax, ymax = round(
                float(xmin) * img_shape[1],
                2), round(float(ymin) * img_shape[0],
                          2), round(float(xmax) * img_shape[1],
                                    2), round(float(ymax) * img_shape[0], 2)
            data = dict()
            data['image_id'] = imgid2name[img_metas[0]['ori_filename']]
            data['bbox'] = [xmin, ymin, xmax-xmin, ymax-ymin]
            data['score'] = float(score)
            data['category_id'] = label+1
            json_results.append(data)
#             if score >= 0.1:
#                 cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 0), 2)
#                 cv2.putText(image, underwater_classes[int(label)] + ' ' + str(round(score, 5)),
#                         (int(xmin), int(ymin - 2)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), thickness=2
#                         )
#         cv2.imwrite(os.path.join('val_img', basename), image)
        for _ in range(batch_size):
            prog_bar.update()
    mmcv.dump(json_results, args.jsonfile)
            

if __name__ == "__main__":
    main()