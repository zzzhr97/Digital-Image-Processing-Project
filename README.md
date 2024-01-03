# Digital-Image-Processing-Project
**数字图像处理 project：高血压视网膜病变图像的分类**

> *详细的项目记录、文件功能描述、命令行参数等内容请见 [RECORD](RECORD.md)*

## 项目介绍

## 环境
详细环境见requirements.txt。主要环境为
- `cuda11.8`
- `python3.7`
- `pytorch2.1.0`

## 项目运行

### 训练模型
运行以下命令，训练模型。运行脚本中的参数需要根据情况修改。
```shell
# Linux
bash scripts/train.sh

# Windows
scripts/train.bat
```

### 可视化
运行以下命令生成样例图片。运行脚本中的参数需要根据情况修改。
```shell
# Linux
bash scripts/visual.sh

# Windows
scripts/visual.bat
```

运行以下命令根据`.csv`结果文件获取可视化数据。
```shell
# Linux
bash scripts/show_result.sh

# Windows
scripts/show_result.bat
```

## 项目结果
