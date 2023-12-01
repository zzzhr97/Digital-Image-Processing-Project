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
5. 神经网络调用时，要求接收一个参数`num_classes`，用于调整最后输出的大小。

### 命令行参数
- *base parameters*:
  - `task`: 任务数字，`1`或`2`
  - `seed`: 随机种子编号。在超参数不变的情况下，seed设置相同，那么结果就会一模一样，如果seed设置不同，结果就会不同
  - `device`: 设备，`cuda`或`cpu`
  - `data_dir`: 数据集存放路径
- *optimization parameters*:
  - `n_valid`: 验证集样本个数
  - `transform_method_origin`: 读取数据时所用的预处理方法编号
  - `transform_method_epoch`: 训练时对训练数据进行的随机性预处理方法编号
  - `batch_size`: 每个轮次训练时的批大小
  - `n_epochs`: 轮次次数
  - `lr`: 学习率
  - `lr_decay_epochs`: 学习率发生改变的对应轮次
  - `lr_decay_values`: 学习率发生改变的目标数值
  - `weight_decay`: adam优化器的正则化参数，用于调整网络的敏感程度，越大，敏感程度越小
  - `is_shuffle`: 是否在读取数据、划分训练集/验证集时进行随机打乱
  - `optimizer`: 优化器，默认有两种选择,adam与sgd
- *model parameters*:
  - `out_dim`: 网络最后一层输出的形状，`1`或`2`
  - `threshold`: `out_dim`为`1`时，划分正负样本的边界，默认为0.5
  - `model`: 模型名称
  - `is_search`: `out_dim`为`1`时，是否在最佳模型权重上搜索`threshold`
- *logging parameters*:
  - `ckpt_dir`: 保存模型权重的路径
  - `result_dir`: 保存训练结果+命令行参数设置
  - `ckpt_every`: 每多少个epoch保存一次模型权重
  - `eval_every`: 每多少个epoch进行一次训练集和验证集loss和score的评估，打印并保存到results中
  - `print_every`: 每个epoch中，每多少个batch进行一次训练集上loss的打印

### 记录

- *v0.1*: 建立项目
- *v0.2*: 完成数据读取，数据预处理，训练功能
  - *v0.2.1*: 完善代码并添加`utils.py`文件。
- *v0.3*: 添加`DenseNet`和`ResNet`网络代码，添加`is_search`命令行参数，完善代码。同时删除了github上的results文件夹，改为本地保存。
  - *v0.3.1*: 添加`transform_method_origin`和`transform_method_epoch`命令行参数，用于在训练时引入数据增强功能。修复一些小bug，并将最终结果保存成图片文件`results.png`，方便查看训练结果。此外，在训练时，不仅会打印`score`和`loss`，还会打印训练和验证集的`TP`，`TN`，`FP`，`FN`。
  - *v0.3.2*: 添加`out_dim`命令行参数，用于控制输出大小，为`1`输出大小为`[batch_size, 1]`，为`2`输出大小为`[batch_size, 2]`。如果为`1`，使用神经网络输出值经过`sigmoid`所得作为正样本概率；如果为`2`，使用神经网络输出的两个值分别作为负样本、正样本概率。
  - *v0.3.3*: 在`README.md`中添加对命令行参数的详细中文解释，修改了一些命令行参数的数据类型