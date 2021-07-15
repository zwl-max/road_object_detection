# FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.4-cuda10.1-py3
ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip install mmcv-full==1.2.4 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html

## 把当前文件夹里的文件构建到镜像的/data下
ADD . /data

# Install MMDetection
RUN conda clean --all
WORKDIR /data
ENV FORCE_CUDA="1"
RUN pip install -r /data/code/requirements/build.txt
RUN pip install --no-cache-dir -e /data/code

# 镜像启动入口
CMD /bin/sh /data/run.sh
