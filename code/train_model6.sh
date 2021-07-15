python tools/train.py configs/cascade_rcnn_swin_small_fpn.py --no-validate
sleep 1

echo "训练完成"
mv /data/user_data/cascade_rcnn_swin_small_fpn_casiou0.55-0.75_gc_box_e20/epoch_20.pth /data/code/ensemble_configs/swin_small_e20-0df8a664.pth
