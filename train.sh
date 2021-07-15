#echo "开始下载coco模型权重"
#cd /data/code
#sh download_weight.sh
#sleep 1
#echo "coco权重下载完成"

echo "开始训练第一个模型"
cd /data/code
sh train_model1.sh
sleep 1

echo "开始训练第二个模型"
cd /data/code
sh train_model2.sh
sleep 1

echo "开始训练第三个模型"
cd /data/code
sh train_model3.sh
sleep 1

echo "开始训练第四个模型"
cd /data/code
sh train_model4.sh
sleep 1

echo "开始训练第五个模型"
cd /data/code
sh train_model5.sh
sleep 1

echo "开始训练第六个模型"
cd /data/code
sh train_model6.sh
sleep 1

echo "开始训练第七个模型"
cd /data/code
sh train_model7.sh
