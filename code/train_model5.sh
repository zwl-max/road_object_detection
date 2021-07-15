python tools/train.py configs/cascade_rcnn_x101_32x4d_dcn_fpn.py --no-validate
sleep 1

echo "训练完成"
mv /data/user_data/cascade_rcnn_x101_32x4d_dcn_fpn_casiou0.55-0.75_gc_box_e20/epoch_20.pth /data/code/ensemble_configs/x101_32x4d_20-f11fb360.pth
