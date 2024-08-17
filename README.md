# 实战演练一一基于话剧的连环画制作 #
## 1.Chatgpt理解代码 ##
这个项目主要是围绕通过 `Data-Juicer` 和 `DiffSynth-Studio` 来处理数据和训练 LoRA 模型，然后生成图像的流程。下面我将逐步解释每个部分的作用和过程。

### 1. 安装依赖库

```
python复制代码!pip install simple-aesthetics-predictor
!pip install -v -e data-juicer
!pip uninstall pytorch-lightning -y
!pip install peft lightning pandas torchvision
!pip install -e DiffSynth-Studio
```

**作用**：

- **simple-aesthetics-predictor**: 这是一个用于图像美学评分的库。
- **data-juicer**: 这是一个用于数据处理的工具，可以帮助你进行数据的提取、转换和加载。
- **pytorch-lightning**: PyTorch Lightning 是一个用于简化 PyTorch 训练的库，安装了 peft、lightning、pandas 和 torchvision 这些常用库。首先卸载旧版本的 `pytorch-lightning` 以避免潜在的兼容性问题。
- **DiffSynth-Studio**: 这是一个用于高效微调大模型的工具，主要用于 LoRA 的训练和生成任务。

### 2. 下载和处理数据集

```
python复制代码from modelscope.msdatasets import MsDataset

ds = MsDataset.load(
    'AI-ModelScope/lowres_anime',
    subset_name='default',
    split='train',
    cache_dir="/mnt/workspace/kolors/data"
)
```

**作用**：

- 通过 `ModelScope` 下载名为 `lowres_anime` 的数据集，并将其缓存到指定的目录 `/mnt/workspace/kolors/data` 中。这个数据集将用于训练 LoRA 模型。

### 3. 数据预处理

```
python复制代码import json, os
from data_juicer.utils.mm_utils import SpecialTokens
from tqdm import tqdm

os.makedirs("./data/lora_dataset/train", exist_ok=True)
os.makedirs("./data/data-juicer/input", exist_ok=True)

with open("./data/data-juicer/input/metadata.jsonl", "w") as f:
    for data_id, data in enumerate(tqdm(ds)):
        image = data["image"].convert("RGB")
        image.save(f"/mnt/workspace/kolors/data/lora_dataset/train/{data_id}.jpg")
        metadata = {"text": "二次元", "image": [f"/mnt/workspace/kolors/data/lora_dataset/train/{data_id}.jpg"]}
        f.write(json.dumps(metadata))
        f.write("\n")
```

**作用**：

- 将下载的数据集进行处理，生成 RGB 格式的图片，并将这些图片保存到指定目录中。
- 生成一个包含图片路径和描述（`text`）的元数据文件 `metadata.jsonl`，以供后续的处理和过滤。

### 4. 配置和执行 Data-Juicer 数据过滤

```
python复制代码data_juicer_config = """
# global parameters
project_name: 'data-process'
dataset_path: './data/data-juicer/input/metadata.jsonl'
np: 4

text_keys: 'text'
image_key: 'image'
image_special_token: '<__dj__image>'

export_path: './data/data-juicer/output/result.jsonl'

# process schedule
process:
    - image_shape_filter:
        min_width: 1024
        min_height: 1024
        any_or_all: any
    - image_aspect_ratio_filter:
        min_ratio: 0.5
        max_ratio: 2.0
        any_or_all: any
"""

with open("data/data-juicer/data_juicer_config.yaml", "w") as file:
    file.write(data_juicer_config.strip())

!dj-process --config data/data-juicer/data_juicer_config.yaml
```

**作用**：

- 这里定义了 Data-Juicer 的配置文件，主要是用于过滤掉不符合要求的图片。比如尺寸不符合要求的图片会被过滤掉。
- 过滤后的数据会保存在 `result.jsonl` 文件中。

### 5. 生成经过筛选的训练数据集

```
python复制代码import pandas as pd
import os, json
from PIL import Image
from tqdm import tqdm

texts, file_names = [], []
os.makedirs("./data/lora_dataset_processed/train", exist_ok=True)

with open("./data/data-juicer/output/result.jsonl", "r") as file:
    for data_id, data in enumerate(tqdm(file.readlines())):
        data = json.loads(data)
        text = data["text"]
        texts.append(text)
        image = Image.open(data["image"][0])
        image_path = f"./data/lora_dataset_processed/train/{data_id}.jpg"
        image.save(image_path)
        file_names.append(f"{data_id}.jpg")

data_frame = pd.DataFrame()
data_frame["file_name"] = file_names
data_frame["text"] = texts
data_frame.to_csv("./data/lora_dataset_processed/train/metadata.csv", index=False, encoding="utf-8-sig")
data_frame
```

**作用**：

- 读取过滤后的数据，生成最终的训练数据集，并将其保存到 `./data/lora_dataset_processed/train` 目录中。同时生成一个 `metadata.csv` 文件，保存图片名称和描述信息。

### 6. 下载和训练 LoRA 模型

```
python复制代码from diffsynth import download_models

download_models(["Kolors", "SDXL-vae-fp16-fix"])
```

**作用**：

- 下载用于训练的基础模型（如 Kolors 和 SDXL-vae-fp16-fix）。

### 7. 执行 LoRA 训练

```
python复制代码import os
cmd = """
python DiffSynth-Studio/examples/train/kolors/train_kolors_lora.py \
  --pretrained_unet_path models/kolors/Kolors/unet/diffusion_pytorch_model.safetensors \
  --pretrained_text_encoder_path models/kolors/Kolors/text_encoder \
  --pretrained_fp16_vae_path models/sdxl-vae-fp16-fix/diffusion_pytorch_model.safetensors \
  --lora_rank 16 \
  --lora_alpha 4.0 \
  --dataset_path data/lora_dataset_processed \
  --output_path ./models \
  --max_epochs 1 \
  --center_crop \
  --use_gradient_checkpointing \
  --precision "16-mixed"
""".strip()

os.system(cmd)
```

**作用**：

- 运行训练脚本，对预训练的 UNet 和文本编码器模型进行 LoRA 微调，训练结果会保存到指定的 `./models` 目录中。

### 8. 加载微调后的模型并生成图像

```
python复制代码from diffsynth import ModelManager, SDXLImagePipeline
from peft import LoraConfig, inject_adapter_in_model
import torch

def load_lora(model, lora_rank, lora_alpha, lora_path):
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out"],
    )
    model = inject_adapter_in_model(lora_config, model)
    state_dict = torch.load(lora_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    return model

model_manager = ModelManager(
    torch_dtype=torch.float16,
    device="cuda",
    file_path_list=[
        "models/kolors/Kolors/text_encoder",
        "models/kolors/Kolors/unet/diffusion_pytorch_model.safetensors",
        "models/kolors/Kolors/vae/diffusion_pytorch_model.safetensors"
    ]
)

pipe = SDXLImagePipeline.from_model_manager(model_manager)
pipe.unet = load_lora(
    pipe.unet,
    lora_rank=16,
    lora_alpha=2.0,
    lora_path="models/lightning_logs/version_0/checkpoints/epoch=0-step=500.ckpt"
)

torch.manual_seed(0)
image = pipe(
    prompt="二次元，一个紫色短发小女孩，在家中沙发上坐着，双手托着腮，很无聊，全身，粉色连衣裙",
    negative_prompt="丑陋、变形、嘈杂、模糊、低对比度",
    cfg_scale=4,
    num_inference_steps=50,
    height=1024, width=1024,
)
image.save("1.jpg")
```

**作用**：

- 加载微调后的 LoRA 模型，并生成一张指定的图片。生成的图片保存在当前工作目录下，文件名为 `1.jpg`。

### 9. 图像拼接

```
python复制代码import numpy as np
from PIL import Image

images = [np.array(Image.open(f"{i}.jpg")) for i in range(1, 9)]
image = np.concatenate([
    np.concatenate(images[0:2], axis=1),
    np.concatenate(images[2:4], axis=1),
    np.concatenate(images[4:6], axis=1),
    np.concatenate(images[6:8], axis=1),
], axis=0)
image = Image.fromarray(image).resize((1024, 2048))
image
```

**作用**：

- 将生成的图像拼接在一起，生成一个大的图像，以便查看多个图像的拼接效果。

### 总结

这个项目的主要流程是：

1. 安装所需的依赖库。
2. 从数据集中下载和处理数据。
3. 使用 Data-Juicer 进行数据的过滤和处理。
4. 生成符合要求的训练数据。
5. 下载和训练 LoRA 模型。
6. 使用微调后的模型生成图像。
7. 最后，将生成的图像拼接在一起进行展示。

## 2.通过爬虫获取二次元风格图片和描述 ##

**Pixiv**

- Pixiv 是一个著名的二次元插画和漫画分享平台，拥有大量高质量的二次元风格图像。可以通过Pixiv API或者使用爬虫工具获取图片和相关描述。
- Pixiv API 相关链接：[Pixiv API](https://github.com/upbit/pixivpy)
- 示例爬虫项目：[Pixiv Crawler](https://github.com/KyuubiRan/pixiv_crawler)

**Sankaku Complex**

- Sankaku Complex 是一个涵盖多种二次元风格图片的社交平台，可以通过爬虫工具获取带有标签和描述的二次元图片。
- Sankaku Complex Crawler 示例项目：[Sankaku Complex Crawler](https://github.com/RaenonX/SankakuCacher)

**Danbooru**

- Danbooru 也可以通过爬虫工具获取图片及其对应的标签，然后将标签转换为文本描述，形成文生图数据集。
- Danbooru Crawler 示例项目：[Danbooru Crawler](https://github.com/KyuubiRan/danbooru-crawler)

最终采用的项目：https://github.com/IrisRainbowNeko/pixiv_AI_crawler

爬取命令：python AIcrawler.py --ckpt 模型权重 --n_images 总图像个数 [--keyword 关键字] 

示例图片：![121124788_p0](https://github.com/user-attachments/assets/c117ddce-f7aa-4d56-8c39-52b6577997f1)

## 3.自动生成文本描述（暂未实现） ##

使用预训练的自然语言处理模型（如OpenAI的CLIP或Hugging Face上的BERT模型），您可以将图片标签转换为自然语言描述。这些描述可以与图片配对，形成文生图数据集。

**示例代码**：

```
python复制代码from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("path_to_your_anime_image.jpg")
inputs = processor(text=["a photo of an anime character"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)
print(probs)
```

通过这些方法，可以获得或构建一个高质量的二次元风格文生图数据集，用于模型训练和微调。

## 4.更新baseline代码 ##

### 修改数据集

注意绝对路径
```
import os
import json
from tqdm import tqdm

# 定义图片所在的目录
image_directory = "./data/dataset1/train"

# 确保保存 metadata.jsonl 的目录存在
os.makedirs("./data/data-juicer/input", exist_ok=True)

# 创建 metadata.jsonl 文件
with open("./data/data-juicer/input/metadata.jsonl", "w") as f:
    for image_filename in tqdm(os.listdir(image_directory)):
        if image_filename.endswith(('.jpg', '.png', '.jpeg')):  # 确保只处理图片文件
            # 获取图片的完整路径
            image_path = os.path.abspath(os.path.join(image_directory, image_filename))
            
            # 创建 metadata 条目
            metadata = {"text": "二次元", "image": [image_path]}
            f.write(json.dumps(metadata))
            f.write("\n")
```

### 修改图片格式

```
import pandas as pd
import os, json
from PIL import Image
from tqdm import tqdm

texts, file_names = [], []
os.makedirs("./data/lora_dataset_processed/train", exist_ok=True)

with open("./data/data-juicer/output/result.jsonl", "r") as file:
    for data_id, data in enumerate(tqdm(file.readlines())):
        data = json.loads(data)
        text = data["text"]
        texts.append(text)
        
        # 打开图像并检查其模式
        image = Image.open(data["image"][0])
        if image.mode == 'RGBA':
            # 将图像从 RGBA 转换为 RGB
            image = image.convert('RGB')
        
        image_path = f"./data/lora_dataset_processed/train/{data_id}.jpg"
        image.save(image_path, 'JPEG')  # 明确指定保存为 JPEG 格式
        file_names.append(f"{data_id}.jpg")

data_frame = pd.DataFrame()
data_frame["file_name"] = file_names
data_frame["text"] = texts
data_frame.to_csv("./data/lora_dataset_processed/train/metadata.csv", index=False, encoding="utf-8-sig")

print(data_frame)
```

### 修改lora版本

```
lora_path="models/lightning_logs/version_1/checkpoints/epoch=0-step=500.ckpt"
```

图片对比

原lora

![11](https://github.com/user-attachments/assets/392170df-91bb-41db-b3ef-2889eafaaf52)

新lora

![1](https://github.com/user-attachments/assets/ea1a4079-f373-4804-a65a-556321c0095b)

可以看到差别不大，说明lora本身的影响并不大，当lora-alpha跳到4后，原lora：

![111](https://github.com/user-attachments/assets/8e83f247-e664-4bd2-bfc5-9a3fca7b6ba9)

可以显著看出简笔画的风格。

另一个对比：二次元，一个银色长发的少女，穿着黑色比基尼，站在海滩上，手中拿着帽子，神情严肃，全身，背景是蓝色的大海和天空

原lora

![21](https://github.com/user-attachments/assets/801b75e1-804d-40e3-a923-c43b7a2bc38d)

新lora

![31](https://github.com/user-attachments/assets/6bda8f9a-12f0-49d3-a194-68796ed21d0c)

## Task3

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

