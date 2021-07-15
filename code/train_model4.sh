python tools/train.py configs/cascade_rcnn_r101_dcn_fpn.py --no-validate
sleep 1

echo "训练完成"
mv /data/user_data/cascade_rcnn_r101_dcn_fpn_casiou0.55-0.75_gc_box_e20/epoch_20.pth /data/code/ensemble_configs/r101_20-db83ab64.pth
