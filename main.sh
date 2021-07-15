echo "生成testB.json"
cd /data/code/
python tools/data_process/generate_test_json.py
sleep 1

echo "开始预测"
python tools/predict_ensemble.py
echo "预测完成！"

cp /data/prediction_result/result.segm.json /data/prediction_result/result.json