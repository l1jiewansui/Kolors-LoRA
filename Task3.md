# Task3

## 1.AI辅助理解代码

### 第一个单元格：安装和导入依赖

```
python复制代码!pip install simple-aesthetics-predictor
!pip install transformers modelscope

import torch
import os
from PIL import Image
from transformers import CLIPProcessor
from aesthetics_predictor import AestheticsPredictorV2Linear
from modelscope import snapshot_download
```

**解析**：

- `!pip install ...`：这是在 Jupyter Notebook 中运行的命令行命令，用于安装 Python 包。这里安装了 `simple-aesthetics-predictor`、`transformers` 和 `modelscope`，它们分别用于美学预测、处理 CLIP 模型以及从 ModelScope 下载模型。
- `import ...`：导入所需的 Python 库，包括处理张量的 `torch`、处理图像的 `PIL`、用于路径操作的 `os`、CLIP 处理器、以及美学评分模型。

### 第二个单元格：下载模型并加载

```
python复制代码# 下载并加载模型
model_id = snapshot_download('AI-ModelScope/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE', cache_dir="models/")
predictor = AestheticsPredictorV2Linear.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

# 选择设备
device = "cuda" if torch.cuda.is_available() else "cpu"
predictor = predictor.to(device)
```

**解析**：

- `snapshot_download`：从 ModelScope 下载指定 ID 的模型，并将其缓存到本地。
- `AestheticsPredictorV2Linear.from_pretrained(model_id)`：加载下载好的美学预测模型。
- `CLIPProcessor.from_pretrained(model_id)`：加载 CLIP 模型的处理器，用于预处理输入图像。
- `device = "cuda" if torch.cuda.is_available() else "cpu"`：检查是否有可用的 GPU，如果有则使用 GPU，否则使用 CPU。
- `predictor.to(device)`：将模型加载到选定的设备上（GPU 或 CPU）。

### 第三个单元格：定义美学评分函数

```
python复制代码def get_aesthetics_score(image):
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = predictor(**inputs)
    prediction = outputs.logits
    return prediction.tolist()[0][0]
```

**解析**：

- `get_aesthetics_score(image)`：这是一个函数，用于计算单张图像的美学分数。
- `processor(images=image, return_tensors="pt")`：使用 CLIP 处理器将图像转换为模型输入格式（张量）。
- `inputs = {k: v.to(device) for k, v in inputs.items()}`：将输入数据移动到选定的设备（GPU 或 CPU）。
- `with torch.no_grad():`：在评估模型时，不需要计算梯度，可以使用 `torch.no_grad()` 来禁用梯度计算，从而节省内存和加快计算速度。
- `outputs = predictor(**inputs)`：使用模型进行推理，获取美学分数。
- `prediction.tolist()[0][0]`：将模型的输出（张量）转换为 Python 列表，并提取美学分数的值。

### 第四个单元格：定义数据集评估函数

```
python复制代码def evaluate(folder):
    scores = []
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        if os.path.isfile(file_path):
            try:
                image = Image.open(file_path).convert("RGB")
                score = get_aesthetics_score(image)
                scores.append(score)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
    if len(scores) == 0:
        return 0
    else:
        return sum(scores) / len(scores)
```

**解析**：

- `evaluate(folder)`：这是一个函数，用于评估指定文件夹中所有图像的平均美学分数。
- `os.listdir(folder)`：列出文件夹中的所有文件。
- `os.path.join(folder, file_name)`：构建每个文件的完整路径。
- `Image.open(file_path).convert("RGB")`：打开图像并将其转换为 RGB 格式（有些图像可能是其他格式，如 RGBA，需要转换）。
- `get_aesthetics_score(image)`：计算单张图像的美学分数。
- `scores.append(score)`：将分数添加到列表中。
- `except Exception as e`：捕获并打印处理图像时可能出现的错误。
- `return sum(scores) / len(scores)`：计算并返回所有图像的平均美学分数。如果没有有效图像，则返回 0。

### 第五个单元格：运行评估并打印结果

```
python复制代码# 评估指定文件夹中的图像
score = evaluate("./images")
print(f"Aesthetic Score: {score}")
```

**解析**：

- 通过 `evaluate("./images")` 调用之前定义的评估函数，对指定文件夹（`./images`）中的图像进行评估，并计算平均美学分数。
- `print(f"Aesthetic Score: {score}")`：打印计算出来的美学分数。

实现效果：![image](https://github.com/user-attachments/assets/f5dc5775-73fa-46ef-b7ad-d5a1ab6da54c)

## 2.参数和数据集对lora质量的影响

本次任务我选用了两个数据集，一个是群像数据集（pixiv爬取的关键词为genshin impact的图像集），一个是单人数据集（pixiv爬取的关键词为ganyu的图像集）

示例：
![103033648_p0](https://github.com/user-attachments/assets/fd8e5858-e4d1-4e0b-aa82-ffad44357e08)

![87229432_p0](https://github.com/user-attachments/assets/5c8c5e99-834c-42cd-b31c-0c72cacb3b83)

标签采用pixiv自带的标签，数据集增强可以看https://www.modevol.com/episode/clib4cinm9s3j01mqapijdfq6

但经过验证，效果并没有想象中那么好，主要是模型无法准确识别关键词（如ganyu）

### 数据集风格

1.由于是关键词筛选，无法做到数据集风格统一，群像数据集训练较为困难。

2.单人数据集可以较为粗略地模仿角色的着装风格，如蓝色头发、犄角、黑色连裤袜等。

### 训练迭代次数

epoch=5较为适合，可以增强模型稳定性

### lora rank和lora alpha

lora alpha = 4的默认配置是最稳定的。 lora rank在训练时可以通过数据集类型选取

### 提示词

提示词非常关键，可以直接决定图像质量好坏，一般需要按照 角色，视角，镜头，姿势，头发，面部，动作，身体，服装，场景，时间，天气，事物 的顺序安排

## 测试结果展示和讨论

正向提示词：二次元，genshin，genshin impact，ganyu，蓝发少女，白色背景

反向提示词：模糊, 低分辨率, 变形, 扭曲, 噪点, 过曝, 欠曝, 丑陋, 不自然, 不完整, 错误颜色, 不对称, 低质量, 混乱, 杂乱, 错误比例, 伪影, 重影, 过度锐化

genshin-impact-lora

![image](https://github.com/user-attachments/assets/f1fbdfd2-6a43-46e4-874f-7c7614245225)

ganyu-lora

![image](https://github.com/user-attachments/assets/1b72dc83-b849-4827-8eff-5157605476eb)

我们可以看到 ganyu-lora 可以更加准确地展示出角色的正确服饰细节，证明单人数据集的训练是有效的。

但事实上我们生成的图片离真正的角色拟合仍然存在差距(上衣)，需要进一步处理数据集以完善。。。


