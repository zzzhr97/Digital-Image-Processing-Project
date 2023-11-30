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
3. 结果数据保存在`./results/`中，该文件夹不会被上传至github，需要另外保存。
4. 每次训练完毕，都会在主文件夹中生成`results.png`用于可视化训练过程。

### 记录

- *v0.1*: 建立项目
- *v0.2*: 完成数据读取，数据预处理，训练功能
  - *v0.2.1*: 完善代码并添加`utils.py`文件。
- *v0.3*: 添加`DenseNet`和`ResNet`网络代码，添加`is_search`命令行参数，完善代码。同时删除了github上的results文件夹，改为本地保存。
  - *v0.3.1*: 添加`transform_method_origin`和`transform_method_epoch`命令行参数，用于在训练时引入数据增强功能。修复一些小bug，并将最终结果保存成图片文件`results.png`，方便查看训练结果。此外，在训练时，不仅会打印`score`和`loss`，还会打印训练和验证集的`TP`，`TN`，`FP`，`FN`。
  - *v0.3.2*: 添加`out_dim`命令行参数，用于控制输出大小，为1输出大小为`[batch_size, 1]`，为2输出大小为`[batch_size, 2]`。如果为1，使用神经网络输出值经过`sigmoid`所得作为正样本概率；如果为2，使用神经网络输出的两个值分别作为负样本、正样本概率。