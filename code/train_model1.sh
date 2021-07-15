python tools/train.py configs/cascade_rcnn_r50_rfp_carafe_sac.py --no-validate
sleep 1

echo "训练完成"
mv /data/user_data/cascade_rcnn_r50_rfp_carafe_sac_gc_box_e20/epoch_20.pth /data/code/ensemble_configs/detectors_r50_e20-0bd921e5.pth
