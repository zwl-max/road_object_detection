python tools/train.py configs/cascade_rcnn_r2_101_dcn_fpn.py --no-validate
sleep 1

echo "训练完成"
mv /data/user_data/cascade_rcnn_r2_101_dcn_fpn_casiou0.55-0.75-gc_box_e20/epoch_20.pth /data/code/ensemble_configs/r2_101_20-487bd3ea.pth
