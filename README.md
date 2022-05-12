# Requirements

整个项目采用Python 3.8 和 PyTorch 1.7.1构建。可使用下面的命令安装所有依赖包:

```
pip install -r requirements.txt
```
其中Apex的安装方法如下
```
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

# How to run
## baseline
```
python main.py dataset --gpu 0 --epochs 200 --lr 0.01 --augment none
```
## mixup
```
python main.py dataset --gpu 0 --epochs 200 --lr 0.01 --augment mixup
```
## cutout
```
python main.py dataset --gpu 0 --epochs 200 --lr 0.01 --augment cutout
```
## cutmix
```
python main.py dataset --gpu 0 --epochs 200 --lr 0.01 --augment cutmix
```
# Visualization
运行代码在visual.ipynb中

# Result & Tensorboard
模型运行结果会自动储存在results文件夹之下，运行tensorboard时需进入到对应的目标文件之下，以cutmix运行结果为例，其他运行结果只需将cutmix改成mixup等即可。
```
cd results/cutmix
tensorboard --logdir logs
```

# Reference
https://github.com/Lornatang/AlexNet-PyTorch.git



