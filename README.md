## 目录
- [MIMO-UNet描述](#1)
- [模型简介](#2)
- [数据集](#3)
- [环境要求](#4)
- [脚本说明](#5)
  - [脚本及样例代码](#6)
  - [脚本参数](#7)
    - [训练](#8)
    - [评估](#9)
  - [训练过程](#10)
    - [训练](#11)
  - [评估过程](#12)
    - [评估](#13)
- [性能](#14)
  - [训练性能](#15)
  - [评估性能](#16)
- [ModelZoo主页](#17)


 <p id="1"></p>

## MIMO-UNet描述

粗到细策略已被广泛用于单图像去模糊网络的架构设计。传统方法通常将子网络与多尺度输入图像堆叠在一起，并逐渐提高从底层子网到顶部子网的图像清晰度，不可避免地会产生高计算成本。为了实现快速准确的去模糊网络设计，我们重新审视了从粗到细的策略，并提出了一个多输入多输出U-net（MIMO-UNet）。首先，MIMO-UNet的单编码器拍摄多尺度输入图像，以减轻训练的难度。其次，MIMO-UNet的单个解码器输出多个不同尺度的去模糊图像，以使用单个U形网络模拟多级联U-net。最后，引入非对称特征融合，以有效的方式合并多尺度特征。在GoPro和RealBlur数据集上的广泛实验表明，所提出的网络在准确性和计算复杂性方面都优于最先进的方法。

参考论文/Paper: [Rethinking Coarse-to-Fine Approach in Single Image Deblurring](https://arxiv.org/abs/2108.05054)

参考代码/Code：[Code](https://github.com/chosj95/MIMO-UNet.)

---
<p id="2"></p>

## 模型简介

MIMO-UNet架构基于多输入多输出的单个U-Net，而非堆叠多个子网络来实现高效的由粗到细的图像去模糊。MIMO-UNet的编码器和解码器由三个编码器块（EB）和解码器块（DB）组成，它们使用卷积层从不同阶段中提取特征。此外，网络还引入了一种非对称地特征融合的方法来有效地融合多尺度特征以进行动态图像去模糊。

---
<p id="3"></p>

## 数据集
- deblur 数据集下载地址：[GOPRO_Large](https://drive.google.com/file/d/1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK/view?usp=sharing)
- GOPRO_Large数据集被广泛用于动态场景去模糊处理，其中：
  - 数据集大小: ~6.2G
  - 训练集: 3.9G, 其中由2103对图像组成
  - 测试集: 2.3G, 其中由1111对图像组成
  - 数据格式：图像
  - 数据将通过 src/data_augment.py 和 src/data_load.py 进行处理
- 数据集原始格式：
```
.
 ├─ GOPRO_Large
    ├─ train
    │  ├─GOPR0xxx_xx_xx
    │  │  ├─ blur   
    │  │  │  ├─ xxxx.png
    │  │  │  ├─ ......
    │  │  ├─ blur_gamma  
    │  │  │  ├─ xxxx.png
    │  │  │  ├─ ......
    │  │  ├─ sharp
    │  │  │  ├─ xxxx.png
    │  │  │  ├─ ......
    │  │  ├─ frames X offset X.txt
    │  ├─   ......
    ├─ test    
    │  ├─ ...... (same as train)

```
- 下载数据集后，通过运行命令```python src/preprocessing.py --root_src 解压的GOPRO_Large路径```来预处理数据集。
- 预处理数据集后，数据文件夹应如下所示：
```
.
 ├─ GOPRO
    ├─ train
    │  ├─ blur   
    │  │  ├─ 1.png
    │  │  ├─ ......
    │  │  ├─ 2103.png
    │  ├─ sharp
    │  │  ├─ 1.png
    │  │  ├─ ......
    │  │  ├─ 2103.png
    ├─ test    
    │  ├─ blur   
    │  │  ├─ 1.png
    │  │  ├─ ......
    │  │  ├─ 1111.png
    │  ├─ sharp
    │  │  ├─ 1.png
    │  │  ├─ ......
    │  │  ├─ 1111.png

```
---
<p id="4"></p>

## 环境要求
- 硬件（Ascend）
  - 使用昇腾处理器（Ascend）来搭建硬件环境。
- 框架
  - [MindSpore](#https://www.mindspore.cn/install)
- 有关详细信息，请参阅以下资源：
  - [MindSpore Tutorial](#https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
  - [MindSpore Python API](#https://www.mindspore.cn/docs/zh-CN/master/index.html)
  
---
<p id="5"></p>

## 脚本说明

<p id="6"></p>

### 脚本及样例代码
```
.
 ├─ MIMO-UNet
    ├─ config
    │  ├─ train.yaml             # 用于在NPU上训练的参数配置
    │  ├─ test.yaml              # 用于在NPU上测试的参数配置
    ├─ src  
    │  ├─  MIMOUNet.py           # MIMO-UNet网络架构
    │  ├─  ObsAndEnv.py          # Obs数据传输
    │  ├─  __init__.py           # 初始化文件
    │  ├─  callback.py           # 自定义回调函数
    │  ├─  config.py             # yaml文件解析
    │  ├─  data_augment.py       # 数据增强
    │  ├─  data_loder.py         # 数据加载
    │  ├─  init_weights.py       # 权重初始化
    │  ├─  layers.py             # 模型层
    │  ├─  loss_total.py         # 损失函数  
    │  ├─  preprocessing.py      # 数据集预处理
    │  ├─  trainers.py           #训练器
    ├─ eval.py                   # 测试网络
    ├─ train.py                  # 训练网络
    ├─ README.md                 # MIMO-UNet文件说明


```
<p id="7"></p>

### 脚本参数

<p id="8"></p>

#### 训练
```
"MODEL_NAME":MIMO-UNetPlus          # 设置残差块的个数
"NUMEPOCH": 3000,                   # 训练轮次数
"learning_rate": 1e-4 ,             # 初始学习率
"decay_rate": 0.5,                  # 学习率衰减率
"decay_epoch": 500,                 # 学习率每经过多少轮次进行衰减
"weight_decay": 0,                  # 权重衰减
"beta1": 0.9,                       # 第一个动量的衰减因子
"beta2": 0.999,                     # 无穷大范数的衰减因子
"eps": 1e-8,                        # eps是一个较小的值，确保不会遇到被零除的误差
"path": ./dataset/GOPRO             # 数据集路径
"mode": train                       # 模式，默认为训练
"column_names": ["input", "label"]  # 图片类别名
"shuffle": true                     # 是否将训练模型的数据集进行打乱
"num_workers":16,                   # 工作进程数
"batch_size": 4,                    # 输入张量的批次大小
"save_checkpoint_steps":526,        # 每隔 526 步保存
"keep_checkpoint_max":200,          # 最多保存多少个ckpt
"print_freq": 526                   # 打印频率
```
<p id="9"></p>

#### 评估
```
"PRETRAINED": "./mimo-unetPlus.ckpt"            # 预训练权重路径
"SAVE_IMG": false                           # 是否保存去模糊后得到的图片
"SAVED_PATH": ./restored                    # 图片保存路径
"path": ./dataset/GOPRO                     # 数据集路径
"mode": test                                # 模式，默认为测试
"column_names": ["input", "label"]          # 图片类别名
"shuffle": false                            # 是否将测试模型的数据集进行打乱
"num_workers":1,                            # 工作进程数
"batch_size": 1                             # 输入张量的批次大小

```
---
<p id="10"></p>

### 训练过程

<p id="11"></p>

#### 训练
- Ascend处理器环境训练MIMO-UNet，根据数据集是否预处理可分为以下两种情况：
  - 若未预处理数据集，则运行下列程序:
```python train.py --root_src 解压的 GOPRO_Large 数据集路径```
  - 若已经预处理数据集，当预处理后得到的数据集结构如下时：
    ```
    .
    ├─ GOPRO
        ├─ train
        │  ├─ blur   
        │  │  ├─ 1.png
        │  │  ├─ ......
        │  │  ├─ 2103.png
        │  ├─ sharp
        │  │  ├─ 1.png
        │  │  ├─ ......
        │  │  ├─ 2103.png
        ├─ test    
        │  ├─ blur   
        │  │  ├─ 1.png
        │  │  ├─ ......
        │  │  ├─ 1111.png
        │  ├─ sharp
        │  │  ├─ 1.png
        │  │  ├─ ......
        │  │  ├─ 1111.png

    ```

    运行下列程序来训练MIMO-UNet:

    ```python train.py```

- 训练结果
```
Epoch:[  0/3000], step:[  526/  526], loss:[0.101/0.103], time:476945.132 ms, lr:0.000100000
Epoch time: 1035279.755 ms, per step time: 1968.212 ms, avg loss: 0.103
Epoch:[  1/3000], step:[  526/  526], loss:[0.089/0.096], time:137.083 ms, lr:0.000100000
Epoch time: 74712.362 ms, per step time: 142.039 ms, avg loss: 0.096
Epoch:[  2/3000], step:[  526/  526], loss:[0.139/0.094], time:141.004 ms, lr:0.000100000
Epoch time: 75961.232 ms, per step time: 144.413 ms, avg loss: 0.094
Epoch:[  3/3000], step:[  526/  526], loss:[0.094/0.095], time:195.251 ms, lr:0.000100000
Epoch time: 73837.098 ms, per step time: 140.375 ms, avg loss: 0.095
Epoch:[  4/3000], step:[  526/  526], loss:[0.117/0.093], time:136.945 ms, lr:0.000100000
Epoch time: 75528.418 ms, per step time: 143.590 ms, avg loss: 0.093
Epoch:[  5/3000], step:[  526/  526], loss:[0.060/0.092], time:121.432 ms, lr:0.000100000
Epoch time: 75405.121 ms, per step time: 143.356 ms, avg loss: 0.092
...
Epoch:[2994/3000], step:[  526/  526], loss:[0.029/0.038], time:128.205 ms, lr:0.000003125
Epoch time: 74811.487 ms, per step time: 142.227 ms, avg loss: 0.038
Epoch:[2995/3000], step:[  526/  526], loss:[0.032/0.038], time:125.587 ms, lr:0.000003125
Epoch time: 73381.279 ms, per step time: 139.508 ms, avg loss: 0.038
Epoch:[2996/3000], step:[  526/  526], loss:[0.042/0.038], time:127.690 ms, lr:0.000003125
Epoch time: 73241.937 ms, per step time: 139.243 ms, avg loss: 0.038
Epoch:[2997/3000], step:[  526/  526], loss:[0.050/0.038], time:119.026 ms, lr:0.000003125
Epoch time: 74174.587 ms, per step time: 141.016 ms, avg loss: 0.038
Epoch:[2998/3000], step:[  526/  526], loss:[0.043/0.038], time:139.545 ms, lr:0.000003125
Epoch time: 75206.023 ms, per step time: 142.977 ms, avg loss: 0.038
Epoch:[2999/3000], step:[  526/  526], loss:[0.031/0.038], time:129.575 ms, lr:0.000003125
Epoch time: 74682.153 ms, per step time: 141.981 ms, avg loss: 0.038

```
<p id="12"></p>

### 评估过程

<p id="13"></p>

##### 评估
- 修改```config/test.yaml```中的.ckpt文件的路径，确保能正确读取权重文件
- 若未预处理数据集需提前预处理
- 评估MIMO-UNET，请执行如下命令：

  ```python eval.py```

---
<p id="14"></p>

## 模型性能
在MIMO-UNet原文中，作者提供了4个实验结果，分别为只使用了多尺度内容损失函数的MIMO-UNet，其PSNR精度为31.46；使用多尺度内容损失函数和多尺度频域重建损失函数的MIMO-UNet，其PSNR精度为31.73；为进一步验证模型的最佳性能，作者将MIMO-UNet中编码、解码器内残差块的个数从8个改为20个，得到MIMO-UNet+，其PSNR精度为32.45；在MIMO-UNet+的基础上，作者又在测试时增加了几何自集成策略，得到MIMO-UNet++，其PSNR精度为32.68。

由于 MSFR（多尺度频域重建）损失函数需 FFT2d 算子，Mindspore 1.5.1中未找到，因此在此自验报告中，复现模型仅对比使用了多尺度内容损失函数时的精度。我们采用 MindSpore 框架复现的 MIMO-UNet 模型的自验精度如下表所示：



|   Method    |  MIMO-UNet（论文精度）     | MIMO-UNet（自验精度） |
| :---------: |:------------------:|:---------------:|
|  PSNR (dB)  | 32.68 |   32.44（无MSFR）   |
| Runtime (s) |       0.04        |    0.4   |


<p id="15"></p>

### 训练性能

| 参数                         | MIMO-UNet (Ascend)                                                                                          |
|----------------------------|-------------------------------------------------------------------------------------------------------------|
| 模型版本                       | MIMO-UNet   |
| 资源                         | 	Ascend: 1 * Ascend-910(32GB)                                                                               |
| 上传日期                       | 2022-09-19                                                                                                  |
| MindSpore版本          | 1.5.1                                                                                                       |
| 数据集                        | GOPRO_Large                                                                                                 |
| 训练超参数 | batch_size=4, epoch=3000,  beta1=0.9, beta2=0.999, lr=0.0001,decay_rate=0.5,decay_epoch=500, num_workers=16 |
| 优化器 | Adam                                                                                                        |
| 损失函数 | 多尺度内容损失函数                                                                                                   |
| 速度 | 约74秒/epoch                                                                                               |
| 时间 | 60 时 59 分 30 秒                                                                                                   |
<p id="16"></p>

### 评估性能

| 参数          | MIMO-UNet (Ascend)            |
|-------------|-------------------------------|
| 模型版本        | MIMO-UNet                     |
| 资源          | 	Ascend: 1 * Ascend-910(32GB) |
| 上传日期        | 2022-09-22                    |
| MindSpore版本 | 1.5.1                         |
| 数据集         | GOPRO_Large                   |
| 评估超参数       | batch_size=1, num_workers=1   |
| 输出格式        | 去模糊后得到的图片                     |
| PSNR        | 32.44dB                       |


---
<p id="17"></p>

## ModelZoo主页
请浏览官网[主页](https://gitee.com/mindspore/models) 。





---









