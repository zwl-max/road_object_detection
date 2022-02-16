### [比赛地址](https://www.sodic.com.cn/competitions/900013)
### 2021.7.17 models/dense_heads/anchor_head.py中有个bug，已改正

### B榜结果：0.16970481815， 排名第五
## 环境信息
- sys.platform: linux(Ubuntu 7.5.0-3ubuntu1~18.04) 
- Python: 3.7.4 (default, Aug 13 2019, 20:35:49) [GCC 7.3.0] 
- GPU 0: GeForce RTX 2080 Ti 
- PyTorch: 1.6.0+cu101 
- TorchVision: 0.7.0+cu101 
- CUDA  10.1 
- CuDNN 7.6.3 
- OpenCV: 4.5.2 
- MMCV: 1.2.4 
- MMDetection: 2.11.0+41bb93f

## 最终的配置文件
- [detectors_r50](code/configs/cascade_rcnn_r50_rfp_carafe_sac.py)
- [s101](code/configs/cascade_rcnn_s101_dcn_fpn.py)
- [r2_101](code/configs/cascade_rcnn_r2_101_dcn_fpn.py)
- [r101](code/configs/cascade_rcnn_r101_dcn_fpn.py)
- [x101_32x4d](code/configs/cascade_rcnn_x101_32x4d_dcn_fpn.py)
- [swin_small](code/configs/cascade_rcnn_swin_small_fpn.py)
- [swin_base](code/configs/cascade_rcnn_swin_base_fpn.py)

## 测试
- `sh main.sh `
- 得到B榜的测试结果 prediction_result/result.json

## 训练
- `sh train.sh`

## 解决方案（实验迭代过程）：
### 基于[mmdetection](https://github.com/open-mmlab/mmdetection) 进行算法迭代
注意：测试采用的是多尺度测试(TTA) \
img_scale=[(4096, 600), (4096, 800), (4096, 1000)] + hflip(水平翻转)

1. baseline: cascade_rcnn_r50_rfp_sac + ms(600-1000) + nms(0.5) + epoch:12
    - 其中， ms指的是多尺度训练， nms(0.5): 使用nms, iou_thre等于0.5
- A榜结果：0.26490951

2. 基于baseline
   - 由于道路有些病害的框非常大，占据图像的大部分，而模型的感受野不足以覆盖这些特大框，因此，为了提高对特大框的检测能力，
     在每个roi上加入全局上下文信息，即gc(global context)。 [代码部分](code/mmdet/models/roi_heads/roi_extractors/single_level_roi_extractor.py)
- A榜结果：0.26810608   提升0.3%

3. 基于2
   - 由于训练数据集的有些框标注并不准确，因此为了提高检测框的泛化能力，对训练集的标注框进行扰动，扰动范围为
   [0.9, 1.1], 即bboxjitter, [代码部分](code/mmdet/datasets/pipelines/transforms.py)
- A榜结果：0.26970710918  提升0.16%     

4. 基于3
   - 为了提高模型的召回率， 用soft_nms替换nms
- A榜结果：0.27433556953  提升0.5%

5. 基于4
   - 由于FPN中上采样采用的是最近邻插值算法，在上采样的过程中，会导致信息的丢失。为了减少上采样过程中信息的丢失，因此采用
   [CARAFE](https://arxiv.org/abs/1905.02188) (可学习的轻量级上采样算法)。
- A榜结果：0.27521643  提升0.1%

6. 基于5
   - 由于从loss曲线上看，模型并没有收敛，且为了使模型更充分的利用训练数据，因此增加训练轮数， epoch:20。
- A榜结果：0.28022861478  提升0.5%

其他尝试，但不work:
- rpn部分采用atss
- anchor增加一个16尺寸
- cas iou调整为[0.55, 0.65, 0.75]
- autoaugmentv2 , v3
- color distort
- mixup
- 类别平衡采样
- fpn替换为acfpn
- box head的回归loss替换为giou loss, 替换为iou loss
- swa + epoch:12
- GeneralizedAttention(0010) 
- gcb_c5_r4(global context block)
- 增加训练尺寸，[600-1100]
   
为了提高模型的鲁棒性，尝试其他模型，便于模型融合
- a. cascade_rcnn_s101_dcn_fpn + gc(全局上下文信息) + bboxjitter(0.9,1.1) + ms(600-1000) + casiou(0.55-0.75) + softnms(0.5) + fp16 + bs2 + Adam(3e-5) + grad_clip(35) + e20
   - A榜结果：0.27352517935
   
- b. cascade_rcnn_r2_101_dcn_fpn + gc + bboxjitter(0.9,1.1) + ms(600-1000) + casiou(0.55-0.75) + softnms(0.5) + fp16 + bs2 + Adam(3e-5) + grad_clip(35) + e20
   - A榜结果：0.26708985523
   
- c. 配置同a,b, 只是换了backbone, 以下同理。  r101
   - A榜结果：0.26568430115
   
- d. x101_32x4d
   — A榜结果：0.25717720485
  
- e. swin_small
   - A榜结果：0.26715268562
   
- f. swin_base
   - A榜结果：0.26849687241

注意：e,f两种模型是在V100,显存32G上训练的， batch_size设置的是4。

#### 至此： 单模最好结果是detectors_r50: A榜：0.28022861478

### 模型融合
- a.
   - detectors_r50 
   - s101
   - 融合方式：先nms, 再融合，[WBF](https://github.com/ZFTurbo/Weighted-Boxes-Fusion), weights=[1.5,1], iou_thr=0.6, conf_type:avg
   - A榜结果：0.29156969777  提升1.1%
   
- b. 
   - detectors_r50
   - s101 
   - r2_101
   - 融合方式：先nms, 再融合，WBF, weights=[1.5,1,1], iou_thr=0.6, conf_type:avg
   - A榜结果：0.29706135891  提升0.6%
   
- c.
   - detectors_r50
   - s101 
   - r2_101
   - r101  
   - 融合方式：先nms, 再融合，WBF, weights=[1.5,1,1,1], iou_thr=0.6, conf_type:avg
   - A榜结果：0.29870779698  提升0.17%
   
- d.
   - detectors_r50
   - s101 
   - r2_101
   - r101  
   - x101_32x4d  
   - 融合方式：先nms, 再融合，WBF, weights=[1.5,1,1,1,1], iou_thr=0.6, conf_type:avg
   - A榜结果：0.30030742718  提升0.16%
   
- e.
   - detectors_r50
   - s101 
   - r2_101
   - r101  
   - x101_32x4d  
   - swin_small  
   - 融合方式：先nms, 再融合，WBF, weights=[1.5,1,1,1,1,1], iou_thr=0.6, conf_type:avg
   - A榜结果：0.30313734501  提升0.28%
   
- f.
   - detectors_r50
   - s101 
   - r2_101
   - r101  
   - x101_32x4d  
   - swin_small  
   - 融合方式：先融合，再soft_nms
   - A榜结果：0.30403569492  提升0.1%
   
- g.
   - detectors_r50
   - s101 
   - r2_101
   - r101  
   - x101_32x4d  
   - swin_small  
   - swin_base  
   - 融合方式：先融合，再soft_nms
   - A榜结果：0.3041911832 
   - B榜结果：0.16970481815
   
### [coco模型权重下载链接](code/download_weight.sh)

## 代码目录
```
/data 
├── raw_data            (数据集)
|   |—— train     (训练集目录)    
|   |—— test_A    (测试集_A目录）
|   |—— test_B    (测试集_B目录）
├── user_data           (用户中间数据目录)
├── prediction_result   (预测结果输出文件夹)
├── code                (代码文件夹）
├── main.sh             (预测脚本)
|—— train.sh            (训练脚本）
└── README.md
