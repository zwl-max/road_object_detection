python tools/train.py configs/cascade_rcnn_s101_dcn_fpn.py --no-validate
sleep 1

echo "训练完成"
mv /data/user_data/cascade_rcnn_s101_dcn_fpn_gc_box_casiou0.55-0.75_e20/epoch_20.pth /data/code/ensemble_configs/s101_20-bd7b757b.pth
