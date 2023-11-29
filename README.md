# Digital-Image-Processing-Project
数字图像处理 project

### 文件结构

1. `./dataset/` 数据集，**需要在本地放置数据集**
2. `./network/` 网络模型
3. `./results/` 训练后生成，保存训练结果的`.csv`文件，**不会上传至github，需要另外保存**
4. `./checkpoints/` 训练后生成，保存训练时的模型参数，为`.pth`文件，**不会上传至github，需要另外保存**
5. `data.py` 数据读取和处理
6. `transform.py` 数据预处理
7. `utils.py` 包含了一些功能型函数
8. `train.py` 训练模型
9. `train.bat` *windows* 运行脚本
10. `train.sh` *linux* 运行脚本

### 注意事项

1. 运行前，先将数据集放在`./dataset`文件夹下，如`./dataset/1-Hypertensive Classification`，以及`./dataset/2-Hypertensive Retinopathy Classification`。
2. 最终保存下来的`.pth`模型权重文件位于`./checkpoints`文件夹中。`./checkpoints`文件夹不会被上传至github，需要另外保存。

### 记录

- *v0.1*: 建立项目
- *v0.2*: 完成数据读取，数据预处理，训练功能
  - *v0.2.1*: 完善代码并添加`utils.py`文件。
- *v0.3*: 添加`DenseNet`和`ResNet`网络代码，添加`is_search`命令行参数，完善代码。同时删除了github上的results文件夹，改为本地保存。