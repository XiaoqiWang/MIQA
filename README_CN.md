<div align="center">
  <img src="./assets/logo1.png" alt="IQA Logo" width="1000"/>
  
  <h3><strong>面向机器的图像质量评估：新范式、大规模数据库与模型
</strong></h3> 

  [![Database](https://img.shields.io/badge/数据库-可访问-green?style=flat-square)](https://github.com/XiaoqiWang/MIQD-2.5M)
  [![Paper](https://img.shields.io/badge/arXiv-论文-red?style=flat-square)](https://arxiv.org/abs/2508.19850)
  [![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow?style=flat-square)](https://huggingface.co/)
  [![Colab](https://img.shields.io/badge/Colab-教程-blue?style=flat-square)](link-to-colab)
  [![GitHub Stars](https://img.shields.io/github/stars/XiaoqiWang/MIQA?style=social)](https://github.com/XiaoqiWang/MIQA)
  
[中文](README_CN.md) | [English](README.md) | [Colab](colab-link) | [博客](blog-link)
</div>
<div style="font-size: 13px;">
🎯 项目概览

- 🤖 **以机器为中心**: 绕过人类的主观感知，完全从深度学习模型的视角来评估图像质量。
- 📈 **由任务驱动**: 直接衡量图像降质（如模糊、噪声、压缩失真）对下游视觉任务性能的真实影响。
- 💡 **全新范式**: MIQA 为优化面向机器决策的图像处理流程提供了一个全新的、更有效的视角。
</div>

---

## 🔥 **最新动态**
> 待办事项:
> 
> **[2025-XX-XX]** 📊 [数据集发布与下载说明](link-to-dataset)
>
> **[2025-XX-XX]** 📖 [可交互的 Colab 教程已上线](colab-link)
>
> **[2025-XX-XX]** 🤗 [模型已上传至 HuggingFace Hub](huggingface-link)


## 🔬 研究背景
- **背景**: 机器视觉系统在理想环境下表现出色，但在真实世界的部署中，图像失真会严重降低其性能。传统的图像质量评估优先考虑人类的感知保真度，这与机器的“敏感点”存在根本性的错位。

- **核心优势**: 以机器为中心的评估框架，能够监控从图像采集、传输到处理的全过程质量，确保机器视觉系统在复杂条件下的性能可靠性，并为自动化视觉系统的优化提供关键支持。


## ✨ MIQA 是否有效？
<div align="center">
  <img src="./assets/cls_ratio.png" alt="分类任务性能" width="30%"/>
  <img src="./assets/det_ratio_ap75.png" alt="检测任务性能" width="30%"/>
  <img src="./assets/ins_ratio_ap75.png" alt="实例分割任务性能" width="30%"/>
  <p><em>使用 MIQA 分数过滤低质量图像后，各项任务的性能均得到提升</em></p>
</div>

<details>
<summary> 🗝️ 核心结论</summary>

我们的研究结果清晰地证明了 MIQA 在三个代表性的计算机视觉任务（图像分类、目标检测和实例分割）上的有效性。

该框架能够持续准确地识别出那些会降低模型性能的图像。通过滤除这些有害样本，MIQA 直接提升了下游任务的最终成果，并证明了“以机器为中心”方法的普适价值。这使得质量评估从一个被动的度量标准，转变为一个主动的性能保障工具，保护下游模型免受真实世界中不可预测的图像质量影响，确保其在关键时刻的稳健表现。
</details>

---
## 🛠️ 安装指南

#### 第 1 步: 安装核心依赖

首先，您需要安装两个必要的库：**mmcv** 和 **mmsegmentation**。
<details>
<summary> 安装 mmcv 和 mmsegmentation</summary>

*   若要安装最新版本的 **mmsegmentation**，请遵循其官方安装指南：
    [MMsegmentation 安装指南](https://mmsegmentation.readthedocs.io/en/main/get_started.html)

*   您也可以根据自己的 CUDA 和 PyTorch 版本，安装特定版本的 **mmsegmentation**。版本兼容性详情请参考：
    [MMCV 安装指南](https://mmcv.readthedocs.io/zh_CN/latest/get_started/installation.html)

</details>

#### 第 2 步: 处理 CUDA 版本兼容性问题

如果您的 CUDA 版本较高（例如 12.7 或更高），可能会遇到与 **mmcv** 的版本不匹配问题。此时，您需要安装一个兼容的 **mmcv** 版本。
<details>
<summary> 安装一个兼容的 mmcv 版本</summary>

例如，您可以通过以下命令卸载现有版本并安装一个兼容的新版本：

```bash
pip uninstall mmcv mmcv-full -y
mim install "mmcv>=2.0.0rc4,<2.2.0"  # 注意：此处版本号仅为示例，请根据您的 CUDA 和 PyTorch 环境选择最合适的版本。
```
</details>

#### 第 3 步: 安装其他所需库

```bash
pip install -r requirements.txt
```

---

## 📦 模型权重与性能
 
| 方法 | 图像分类 (SRCC/PLCC) | 目标检测 (SRCC/PLCC) | 实例分割 (SRCC/PLCC) |下载 |
|:----------------| :---: | :---: | :---: |:---: |
| ResNet-18 | `0.5131 / 0.5427` | `0.7541 / 0.7734` | `0.7582 / 0.7790` |[**下载**](YOUR_MODEL_LINK) |
| ResNet-50 | `0.5581 / 0.5797` | `0.7743 / 0.7925` | `0.7729 / 0.7933` |[**下载**](YOUR_MODEL_LINK) |
| EfficientNet-b1 | `0.5901 / 0.6130` | `0.7766 / 0.7950` | `0.7808 / 0.7999` |[**下载**](YOUR_MODEL_LINK) |
| EfficientNet-b5 | `0.6330 / 0.6440` | `0.7866 / 0.8041` | `0.7899 / 0.8074` |[**下载**](YOUR_MODEL_LINK) |
| ViT-small | `0.5998 / 0.6161` | `0.7992 / 0.8142` | `0.7968 / 0.8139` |[**下载**](YOUR_MODEL_LINK) |
| **RA-MIQA** | **`0.7003 / 0.6989`** | **`0.8125 / 0.8264`** | **`0.8188 / 0.8340`** |[**下载**](YOUR_MODEL_LINK) |

模型会在首次使用时自动下载：

```python
from machine_iqa import MIQAModel

# 通过指定方法名称来初始化 MIQA 模型
model = MIQAModel('ra-miqa')  # 模型将被自动下载
```

## 🚀 快速上手

### 评估单张图像

```python
from machine_iqa import MIQAModel
import cv2

# 加载模型
model = MIQAModel('ra-miqa')

# 读取并评估图像
image = cv2.imread('path/to/image.jpg')
quality_score = model.assess(image)
print(f"图像质量得分: {quality_score:.3f}")
```

<details>
<summary>📸 示例结果</summary>

| 图像 | 质量分数 | 预测结果 |
|-------|---------------|------------|
| ![Demo 1](assets/demo1.jpg) | 0.892 | 高质量 |
| ![Demo 2](assets/demo2.jpg) | 0.634 | 中等质量 |
| ![Demo 3](assets/demo3.jpg) | 0.298 | 低质量 |

</details>

### 批量处理图像

```python
from machine_iqa import batch_assess
import glob

# 处理整个目录的图像
image_paths = glob.glob('dataset/*.jpg')
scores = batch_assess(image_paths, model_name='ra-miqa')

for path, score in zip(image_paths, scores):
    print(f"{path}: {score:.3f}")
```

### 评估视频质量

```python
from machine_iqa import VideoMIQA

# 初始化视频评估器
video_iqa = VideoMIQA('ra-miqa')

# 处理视频文件
results = video_iqa.assess_video('path/to/video.mp4')
print(f"视频帧的质量分布: {results['mean_score']:.3f}")
```

<details>
<summary>🎥 视频评估示例</summary>

视频处理结果示例：
- **逐帧分析**: 获取每一帧的质量分数
- **时序一致性**: 观察质量随时间的变化
- **关键洞察**: 识别视频中质量下降的关键节点

</details>

### 实时摄像头评估

```python
from machine_iqa import RealTimeIQA
import cv2

# 初始化实时评估器 (建议使用轻量级模型以保证速度)
rt_iqa = RealTimeIQA('ra-miqa')  

# 启动摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 评估当前帧的质量
    score = rt_iqa.assess_frame(frame)
    
    # 在画面上显示结果
    cv2.putText(frame, f'Quality: {score:.3f}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Real-time IQA', frame)
    
    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 🏃 训练与评估

### 模型训练

```bash
# 基础训练
python train.py --config configs/iqa_base.yaml --data_path /path/to/dataset

# 使用自定义参数进行高级训练
python train.py \
    --config configs/iqa_large.yaml \
    --data_path /path/to/dataset \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --epochs 100 \
    --gpu_ids 0,1
```

### 在标准基准上进行评估

```bash
# 在测试集上评估
python evaluate.py --model_path checkpoints/best_model.pth --test_data /path/to/test

# 跨数据集评估
python evaluate.py --model_path checkpoints/best_model.pth --datasets miqa_cls
```

## 📈 基准测试

<details>
<summary>表 1: 基于综合性能的基准测试</summary>

<table>
<thead>
<tr>
<th rowspan="2" style="text-align: center;">类别</th>
<th rowspan="2" style="text-align: left;">方法</th>
<th colspan="4" style="text-align: center;">图像分类</th>
<th colspan="4" style="text-align: center;">目标检测</th>
<th colspan="4" style="text-align: center;">实例分割</th>
</tr>
<tr>
<th style="text-align: center;">SRCC ↑</th>
<th style="text-align: center;">PLCC ↑</th>
<th style="text-align: center;">KRCC ↑</th>
<th style="text-align: center;">RMSE ↓</th>
<th style="text-align: center;">SRCC ↑</th>
<th style="text-align: center;">PLCC ↑</th>
<th style="text-align: center;">KRCC ↑</th>
<th style="text-align: center;">RMSE ↓</th>
<th style="text-align: center;">SRCC ↑</th>
<th style="text-align: center;">PLCC ↑</th>
<th style="text-align: center;">KRCC ↑</th>
<th style="text-align: center;">RMSE ↓</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="7" style="text-align: center; vertical-align: middle;"><strong>基于人类视觉 (HVS-based)</strong></td>
<td style="text-align: left;">PSNR</td>
<td style="text-align: center;">0.2388</td>
<td style="text-align: center;">0.2292</td>
<td style="text-align: center;">0.1661</td>
<td style="text-align: center;">0.2928</td>
<td style="text-align: center;">0.3176</td>
<td style="text-align: center;">0.3456</td>
<td style="text-align: center;">0.2148</td>
<td style="text-align: center;">0.2660</td>
<td style="text-align: center;">0.3242</td>
<td style="text-align: center;">0.3530</td>
<td style="text-align: center;">0.2196</td>
<td style="text-align: center;">0.2553</td>
</tr>
<tr>
<td style="text-align: left;">SSIM</td>
<td style="text-align: center;">0.3027</td>
<td style="text-align: center;">0.2956</td>
<td style="text-align: center;">0.2119</td>
<td style="text-align: center;">0.2874</td>
<td style="text-align: center;">0.4390</td>
<td style="text-align: center;">0.4505</td>
<td style="text-align: center;">0.3011</td>
<td style="text-align: center;">0.2531</td>
<td style="text-align: center;">0.4391</td>
<td style="text-align: center;">0.4512</td>
<td style="text-align: center;">0.3011</td>
<td style="text-align: center;">0.2435</td>
</tr>
<tr>
<td style="text-align: left;">VSI</td>
<td style="text-align: center;">0.3592</td>
<td style="text-align: center;">0.3520</td>
<td style="text-align: center;">0.2520</td>
<td style="text-align: center;">0.2816</td>
<td style="text-align: center;">0.4874</td>
<td style="text-align: center;">0.4940</td>
<td style="text-align: center;">0.3355</td>
<td style="text-align: center;">0.2465</td>
<td style="text-align: center;">0.4919</td>
<td style="text-align: center;">0.4985</td>
<td style="text-align: center;">0.3392</td>
<td style="text-align: center;">0.2365</td>
</tr>
<tr>
<td style="text-align: left;">LPIPS</td>
<td style="text-align: center;">0.3214</td>
<td style="text-align: center;">0.3280</td>
<td style="text-align: center;">0.2258</td>
<td style="text-align: center;">0.2842</td>
<td style="text-align: center;">0.5264</td>
<td style="text-align: center;"><strong>0.5376</strong></td>
<td style="text-align: center;"><strong>0.3697</strong></td>
<td style="text-align: center;"><strong>0.2390</strong></td>
<td style="text-align: center;">0.5342</td>
<td style="text-align: center;"><strong>0.5453</strong></td>
<td style="text-align: center;"><strong>0.3754</strong></td>
<td style="text-align: center;"><strong>0.2287</strong></td>
</tr>
<tr>
<td style="text-align: left;">DISTS</td>
<td style="text-align: center;"><strong>0.3878</strong></td>
<td style="text-align: center;"><strong>0.3804</strong></td>
<td style="text-align: center;"><strong>0.2724</strong></td>
<td style="text-align: center;"><strong>0.2782</strong></td>
<td style="text-align: center;"><strong>0.5266</strong></td>
<td style="text-align: center;">0.5352</td>
<td style="text-align: center;">0.3659</td>
<td style="text-align: center;">0.2395</td>
<td style="text-align: center;"><strong>0.5363</strong></td>
<td style="text-align: center;">0.5450</td>
<td style="text-align: center;">0.3738</td>
<td style="text-align: center;">0.2288</td>
</tr>
<tr>
<td style="text-align: left;">HyperIQA</td>
<td style="text-align: center;">0.2496</td>
<td style="text-align: center;">0.2279</td>
<td style="text-align: center;">0.1741</td>
<td style="text-align: center;">0.2929</td>
<td style="text-align: center;">0.4462</td>
<td style="text-align: center;">0.4463</td>
<td style="text-align: center;">0.3031</td>
<td style="text-align: center;">0.2537</td>
<td style="text-align: center;">0.4456</td>
<td style="text-align: center;">0.4518</td>
<td style="text-align: center;">0.3031</td>
<td style="text-align: center;">0.2434</td>
</tr>
<tr>
<td style="text-align: left;">MANIQA</td>
<td style="text-align: center;">0.3403</td>
<td style="text-align: center;">0.3255</td>
<td style="text-align: center;">0.2387</td>
<td style="text-align: center;">0.2844</td>
<td style="text-align: center;">0.4574</td>
<td style="text-align: center;">0.4617</td>
<td style="text-align: center;">0.3124</td>
<td style="text-align: center;">0.2515</td>
<td style="text-align: center;">0.4636</td>
<td style="text-align: center;">0.4680</td>
<td style="text-align: center;">0.3176</td>
<td style="text-align: center;">0.2411</td>
</tr>
<tr>

</tr>
<tr>
<td rowspan="6" style="text-align: center; vertical-align: middle;"><strong>基于机器 (Machine-based)</strong></td>
<td style="text-align: left;">ResNet-18</td>
<td style="text-align: center;">0.5131</td>
<td style="text-align: center;">0.5427</td>
<td style="text-align: center;">0.3715</td>
<td style="text-align: center;">0.2527</td>
<td style="text-align: center;">0.7541</td>
<td style="text-align: center;">0.7734</td>
<td style="text-align: center;">0.5625</td>
<td style="text-align: center;">0.1797</td>
<td style="text-align: center;">0.7582</td>
<td style="text-align: center;">0.7790</td>
<td style="text-align: center;">0.5674</td>
<td style="text-align: center;">0.1711</td>
</tr>
<tr>
<td style="text-align: left;">ResNet-50</td>
<td style="text-align: center;">0.5581</td>
<td style="text-align: center;">0.5797</td>
<td style="text-align: center;">0.4062</td>
<td style="text-align: center;">0.2451</td>
<td style="text-align: center;">0.7743</td>
<td style="text-align: center;">0.7925</td>
<td style="text-align: center;">0.5824</td>
<td style="text-align: center;">0.1729</td>
<td style="text-align: center;">0.7729</td>
<td style="text-align: center;">0.7933</td>
<td style="text-align: center;">0.5826</td>
<td style="text-align: center;">0.1661</td>
</tr>
<tr>
<td style="text-align: left;">EfficientNet-b1</td>
<td style="text-align: center;">0.5901</td>
<td style="text-align: center;">0.6130</td>
<td style="text-align: center;">0.4320</td>
<td style="text-align: center;">0.2377</td>
<td style="text-align: center;">0.7766</td>
<td style="text-align: center;">0.7950</td>
<td style="text-align: center;">0.5859</td>
<td style="text-align: center;">0.1720</td>
<td style="text-align: center;">0.7808</td>
<td style="text-align: center;">0.7999</td>
<td style="text-align: center;">0.5918</td>
<td style="text-align: center;">0.1637</td>
</tr>
<tr>
<td style="text-align: left;">EfficientNet-b5</td>
<td style="text-align: center;">0.6330</td>
<td style="text-align: center;">0.6440</td>
<td style="text-align: center;">0.4680</td>
<td style="text-align: center;">0.2301</td>
<td style="text-align: center;">0.7866</td>
<td style="text-align: center;">0.8041</td>
<td style="text-align: center;">0.5971</td>
<td style="text-align: center;">0.1685</td>
<td style="text-align: center;">0.7899</td>
<td style="text-align: center;">0.8074</td>
<td style="text-align: center;">0.6013</td>
<td style="text-align: center;">0.1610</td>
</tr>
<tr>
<td style="text-align: left;">ViT-small</td>
<td style="text-align: center;">0.5998</td>
<td style="text-align: center;">0.6161</td>
<td style="text-align: center;">0.4407</td>
<td style="text-align: center;">0.2370</td>
<td style="text-align: center;">0.7992</td>
<td style="text-align: center;">0.8142</td>
<td style="text-align: center;">0.6099</td>
<td style="text-align: center;">0.1646</td>
<td style="text-align: center;">0.7968</td>
<td style="text-align: center;">0.8139</td>
<td style="text-align: center;">0.6083</td>
<td style="text-align: center;">0.1585</td>
</tr>
<tr style="background-color: #f0f8ff;">
<td style="text-align: left;"><strong>RA-MIQA (Ours)</strong></td>
<td style="text-align: center;"><strong>0.7003</strong></td>
<td style="text-align: center;"><strong>0.6989</strong></td>
<td style="text-align: center;"><strong>0.5255</strong></td>
<td style="text-align: center;"><strong>0.2152</strong></td>
<td style="text-align: center;"><strong>0.8125</strong></td>
<td style="text-align: center;"><strong>0.8264</strong></td>
<td style="text-align: center;"><strong>0.6263</strong></td>
<td style="text-align: center;"><strong>0.1596</strong></td>
<td style="text-align: center;"><strong>0.8188</strong></td>
<td style="text-align: center;"><strong>0.8340</strong></td>
<td style="text-align: center;"><strong>0.6333</strong></td>
<td style="text-align: center;"><strong>0.1505</strong></td>
</tr>
</tbody>
</table>

</details>

<details>
<summary>表 2: 一致性与准确性得分基准测试</summary>
<table>
<thead>
<tr>
<th rowspan="3" style="text-align: center; vertical-align: middle;">方法</th>
<th colspan="6" style="text-align: center;">图像分类</th>
<th colspan="6" style="text-align: center;">目标检测</th>
<th colspan="6" style="text-align: center;">实例分割</th>
</tr>
<tr>
<th colspan="3" style="text-align: center;">准确性得分</th>
<th colspan="3" style="text-align: center;">一致性得分</th>
<th colspan="3" style="text-align: center;">准确性得分</th>
<th colspan="3" style="text-align: center;">一致性得分</th>
<th colspan="3" style="text-align: center;">准确性得分</th>
<th colspan="3" style="text-align: center;">一致性得分</th>
</tr>
<tr>
<th style="text-align: center;">SRCC ↑</th>
<th style="text-align: center;">PLCC ↑</th>
<th style="text-align: center;">RMSE ↓</th>
<th style="text-align: center;">SRCC ↑</th>
<th style="text-align: center;">PLCC ↑</th>
<th style="text-align: center;">RMSE ↓</th>
<th style="text-align: center;">SRCC ↑</th>
<th style="text-align: center;">PLCC ↑</th>
<th style="text-align: center;">RMSE ↓</th>
<th style="text-align: center;">SRCC ↑</th>
<th style="text-align: center;">PLCC ↑</th>
<th style="text-align: center;">RMSE ↓</th>
<th style="text-align: center;">SRCC ↑</th>
<th style="text-align: center;">PLCC ↑</th>
<th style="text-align: center;">RMSE ↓</th>
<th style="text-align: center;">SRCC ↑</th>
<th style="text-align: center;">PLCC ↑</th>
<th style="text-align: center;">RMSE ↓</th>
</tr>
</thead>
<tbody>
<tr><td colspan="19" style="font-weight: bold; text-align: left; padding-top: 8px; padding-bottom: 4px; border-bottom: 1px solid #ddd;"><em>基于人类视觉 (HVS-based)</em></td></tr>
<tr>
<td style="text-align: left;">PSNR</td>
<td style="text-align: center;">0.2034</td>
<td style="text-align: center;">0.1620</td>
<td style="text-align: center;">0.3541</td>
<td style="text-align: center;">0.2927</td>
<td style="text-align: center;">0.2812</td>
<td style="text-align: center;">0.2692</td>
<td style="text-align: center;">0.2234</td>
<td style="text-align: center;">0.2449</td>
<td style="text-align: center;">0.2747</td>
<td style="text-align: center;">0.3712</td>
<td style="text-align: center;">0.3933</td>
<td style="text-align: center;">0.2839</td>
<td style="text-align: center;">0.2182</td>
<td style="text-align: center;">0.2398</td>
<td style="text-align: center;">0.2616</td>
<td style="text-align: center;">0.3796</td>
<td style="text-align: center;">0.4061</td>
<td style="text-align: center;">0.2770</td>
</tr>
<tr>
<td style="text-align: left;">SSIM</td>
<td style="text-align: center;">0.2529</td>
<td style="text-align: center;">0.2101</td>
<td style="text-align: center;">0.3509</td>
<td style="text-align: center;">0.3740</td>
<td style="text-align: center;">0.3663</td>
<td style="text-align: center;">0.2610</td>
<td style="text-align: center;">0.3434</td>
<td style="text-align: center;">0.3419</td>
<td style="text-align: center;">0.2662</td>
<td style="text-align: center;">0.5128</td>
<td style="text-align: center;">0.5130</td>
<td style="text-align: center;">0.2651</td>
<td style="text-align: center;">0.3271</td>
<td style="text-align: center;">0.3284</td>
<td style="text-align: center;">0.2545</td>
<td style="text-align: center;">0.5174</td>
<td style="text-align: center;">0.5204</td>
<td style="text-align: center;">0.2589</td>
</tr>
<tr>
<td style="text-align: left;">VSI</td>
<td style="text-align: center;">0.3020</td>
<td style="text-align: center;">0.2515</td>
<td style="text-align: center;">0.3473</td>
<td style="text-align: center;">0.4392</td>
<td style="text-align: center;">0.4336</td>
<td style="text-align: center;">0.2528</td>
<td style="text-align: center;">0.3799</td>
<td style="text-align: center;">0.3685</td>
<td style="text-align: center;">0.2634</td>
<td style="text-align: center;">0.5700</td>
<td style="text-align: center;">0.5571</td>
<td style="text-align: center;">0.2565</td>
<td style="text-align: center;">0.3703</td>
<td style="text-align: center;">0.3645</td>
<td style="text-align: center;">0.2509</td>
<td style="text-align: center;">0.5757</td>
<td style="text-align: center;">0.5749</td>
<td style="text-align: center;">0.2481</td>
</tr>
<tr>
<td style="text-align: left;">LPIPS</td>
<td style="text-align: center;">0.2680</td>
<td style="text-align: center;">0.2355</td>
<td style="text-align: center;">0.3488</td>
<td style="text-align: center;">0.3927</td>
<td style="text-align: center;">0.4032</td>
<td style="text-align: center;">0.2567</td>
<td style="text-align: center;">0.4064</td>
<td style="text-align: center;">0.3987</td>
<td style="text-align: center;">0.2598</td>
<td style="text-align: center;"><strong>0.6196</strong></td>
<td style="text-align: center;"><strong>0.6232</strong></td>
<td style="text-align: center;"><strong>0.2415</strong></td>
<td style="text-align: center;">0.3972</td>
<td style="text-align: center;">0.3941</td>
<td style="text-align: center;">0.2476</td>
<td style="text-align: center;"><strong>0.6300</strong></td>
<td style="text-align: center;"><strong>0.6344</strong></td>
<td style="text-align: center;"><strong>0.2344</strong></td>
</tr>
<tr>
<td style="text-align: left;">DISTS</td>
<td style="text-align: center;"><strong>0.3291</strong></td>
<td style="text-align: center;"><strong>0.2768</strong></td>
<td style="text-align: center;"><strong>0.3448</strong></td>
<td style="text-align: center;"><strong>0.4683</strong></td>
<td style="text-align: center;"><strong>0.4628</strong></td>
<td style="text-align: center;"><strong>0.2487</strong></td>
<td style="text-align: center;"><strong>0.4089</strong></td>
<td style="text-align: center;"><strong>0.3999</strong></td>
<td style="text-align: center;"><strong>0.2597</strong></td>
<td style="text-align: center;">0.6174</td>
<td style="text-align: center;">0.6178</td>
<td style="text-align: center;">0.2429</td>
<td style="text-align: center;"><strong>0.4069</strong></td>
<td style="text-align: center;"><strong>0.4012</strong></td>
<td style="text-align: center;"><strong>0.2468</strong></td>
<td style="text-align: center;">0.6255</td>
<td style="text-align: center;">0.6270</td>
<td style="text-align: center;">0.2362</td>
</tr>
<tr>
<td style="text-align: left;">HyperIQA</td>
<td style="text-align: center;">0.2100</td>
<td style="text-align: center;">0.1649</td>
<td style="text-align: center;">0.3540</td>
<td style="text-align: center;">0.2966</td>
<td style="text-align: center;">0.2777</td>
<td style="text-align: center;">0.2695</td>
<td style="text-align: center;">0.3646</td>
<td style="text-align: center;">0.3545</td>
<td style="text-align: center;">0.2649</td>
<td style="text-align: center;">0.5009</td>
<td style="text-align: center;">0.4943</td>
<td style="text-align: center;">0.2684</td>
<td style="text-align: center;">0.3486</td>
<td style="text-align: center;">0.3442</td>
<td style="text-align: center;">0.2530</td>
<td style="text-align: center;">0.5056</td>
<td style="text-align: center;">0.4995</td>
<td style="text-align: center;">0.2626</td>
</tr>
<tr>
<td style="text-align: left;">MANIQA</td>
<td style="text-align: center;">0.2924</td>
<td style="text-align: center;">0.2435</td>
<td style="text-align: center;">0.3481</td>
<td style="text-align: center;">0.3963</td>
<td style="text-align: center;">0.3870</td>
<td style="text-align: center;">0.2587</td>
<td style="text-align: center;">0.3839</td>
<td style="text-align: center;">0.3823</td>
<td style="text-align: center;">0.2618</td>
<td style="text-align: center;">0.4991</td>
<td style="text-align: center;">0.4975</td>
<td style="text-align: center;">0.2679</td>
<td style="text-align: center;">0.3755</td>
<td style="text-align: center;">0.3749</td>
<td style="text-align: center;">0.2498</td>
<td style="text-align: center;">0.5096</td>
<td style="text-align: center;">0.5098</td>
<td style="text-align: center;">0.2608</td>
</tr>
<tr><td colspan="19" style="font-weight: bold; text-align: left; padding-top: 8px; padding-bottom: 4px; border-bottom: 1px solid #ddd;"><em>基于机器 (Machine-based)</em></td></tr>
<tr>
<td style="text-align: left;">ResNet-50</td>
<td style="text-align: center;">0.4734</td>
<td style="text-align: center;">0.4411</td>
<td style="text-align: center;">0.3221</td>
<td style="text-align: center;">0.5989</td>
<td style="text-align: center;">0.6551</td>
<td style="text-align: center;">0.2119</td>
<td style="text-align: center;">0.6955</td>
<td style="text-align: center;">0.6898</td>
<td style="text-align: center;">0.2051</td>
<td style="text-align: center;">0.8252</td>
<td style="text-align: center;">0.8457</td>
<td style="text-align: center;">0.1648</td>
<td style="text-align: center;">0.6863</td>
<td style="text-align: center;">0.6847</td>
<td style="text-align: center;">0.1964</td>
<td style="text-align: center;">0.8320</td>
<td style="text-align: center;">0.8480</td>
<td style="text-align: center;">0.1607</td>
</tr>
<tr>
<td style="text-align: left;">EfficientNet-b5</td>
<td style="text-align: center;">0.5586</td>
<td style="text-align: center;">0.5149</td>
<td style="text-align: center;">0.3076</td>
<td style="text-align: center;">0.6774</td>
<td style="text-align: center;">0.7168</td>
<td style="text-align: center;">0.1956</td>
<td style="text-align: center;">0.7042</td>
<td style="text-align: center;">0.6991</td>
<td style="text-align: center;">0.2026</td>
<td style="text-align: center;">0.8353</td>
<td style="text-align: center;">0.8530</td>
<td style="text-align: center;">0.1612</td>
<td style="text-align: center;">0.6933</td>
<td style="text-align: center;">0.6949</td>
<td style="text-align: center;">0.1938</td>
<td style="text-align: center;">0.8419</td>
<td style="text-align: center;">0.8564</td>
<td style="text-align: center;">0.1565</td>
</tr>
<tr>
<td style="text-align: left;">ViT-small</td>
<td style="text-align: center;">0.5788</td>
<td style="text-align: center;">0.5197</td>
<td style="text-align: center;">0.3066</td>
<td style="text-align: center;">0.6798</td>
<td style="text-align: center;">0.7189</td>
<td style="text-align: center;">0.1950</td>
<td style="text-align: center;">0.7121</td>
<td style="text-align: center;">0.7052</td>
<td style="text-align: center;">0.2008</td>
<td style="text-align: center;">0.8459</td>
<td style="text-align: center;">0.8620</td>
<td style="text-align: center;">0.1566</td>
<td style="text-align: center;">0.7168</td>
<td style="text-align: center;">0.7146</td>
<td style="text-align: center;">0.1885</td>
<td style="text-align: center;">0.8487</td>
<td style="text-align: center;">0.8616</td>
<td style="text-align: center;">0.1539</td>
</tr>
<tr style="background-color: #f0f8ff;">
<td style="text-align: left;"><strong>RA-MIQA</strong></td>
<td style="text-align: center;"><strong>0.6573</strong></td>
<td style="text-align: center;"><strong>0.5823</strong></td>
<td style="text-align: center;"><strong>0.2917</strong></td>
<td style="text-align: center;"><strong>0.7707</strong></td>
<td style="text-align: center;"><strong>0.7866</strong></td>
<td style="text-align: center;"><strong>0.1732</strong></td>
<td style="text-align: center;"><strong>0.7448</strong></td>
<td style="text-align: center;"><strong>0.7370</strong></td>
<td style="text-align: center;"><strong>0.1915</strong></td>
<td style="text-align: center;"><strong>0.8526</strong></td>
<td style="text-align: center;"><strong>0.8692</strong></td>
<td style="text-align: center;"><strong>0.1527</strong></td>
<td style="text-align: center;"><strong>0.7363</strong></td>
<td style="text-align: center;"><strong>0.7327</strong></td>
<td style="text-align: center;"><strong>0.1834</strong></td>
<td style="text-align: center;"><strong>0.8632</strong></td>
<td style="text-align: center;"><strong>0.8756</strong></td>
<td style="text-align: center;"><strong>0.1464</strong></td>
</tr>
</tbody>
</table>
</details> 


## 📚 引用

如果本研究对您的工作有所帮助，请考虑引用我们的论文：

```bibtex
@article{wang2025miqa,
  title={Image Quality Assessment for Machines: Paradigm, Large-scale Database, and Models},
  author={Wang, Xiaoqi and Zhang, Yun and Lin, Weisi},
  journal={arXiv preprint arXiv:2508.19850},
  year={2025}
}
```

## 🤝 贡献指南 （待办事项）

我们非常欢迎来自社区的贡献！如果您有兴趣改进 MIQA，请查阅我们的 [贡献指南](CONTRIBUTING.md) 以了解提交错误报告、功能请求和代码合并请求（Pull Request）的详细流程。

<details>
<summary>开发环境配置</summary>
请按照以下步骤配置您的本地开发环境：

1.  **克隆本仓库：**
    ```bash
    git clone https://github.com/XiaoqiWang/MIQA.git
    cd MIQA
    ```

2.  **以可编辑模式安装依赖：**
    此命令会安装项目本身以及开发所需的额外依赖（如测试工具）。
    ```bash
    pip install -e ".[dev]"
    ```

3.  **设置 pre-commit 钩子：**
    这能确保您提交的代码自动符合我们的代码风格规范。
    ```bash
    pre-commit install
    ```

### 运行测试

我们使用 `pytest` 进行测试。您可以运行测试套件以确保您的更改没有破坏现有功能。

1.  **运行所有测试：**
    ```bash
    pytest tests/
    ```

2.  **运行测试并生成覆盖率报告：**
    要检查您的测试覆盖了多少代码，请运行：
    ```bash
    python -m pytest tests/ --cov=miqa # 请将 `miqa` 替换为您项目源码包的实际名称。
    ```
    
</details> 

## 📧 联系我们

- **项目维护者**: [Xiaoqi Wang](mailto:wangxq79@mail2.sysu.edu.cn)
- **问题反馈**: 请通过 [GitHub Issues](https://github.com/XiaoqiWang/MIQA/issues) 提交错误报告和功能建议。

---
**⭐ 如果这个项目对您有帮助，请点亮 Star！😊**

*最后更新于: [09/26/2025]*