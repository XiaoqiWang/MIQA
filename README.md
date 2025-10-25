
<div align="center">
  <img src="./assets/logo1.png" alt="IQA Logo" width="1200"/>
  
  <h3><strong> Image Quality Assessment for Machines: Paradigm, Large-scale Database, and Models
</strong></h3> 

  [![Database](https://img.shields.io/badge/Database-Available-green?style=flat-square)](https://github.com/XiaoqiWang/MIQD-2.5M)
  [![Paper](https://img.shields.io/badge/arXiv-Paper-red?style=flat-square)](https://arxiv.org/abs/2508.19850)
  [![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow?style=flat-square)](https://huggingface.co/)
  [![Colab](https://img.shields.io/badge/Colab-Tutorial-blue?style=flat-square)](link-to-colab)
  [![GitHub Stars](https://img.shields.io/github/stars/XiaoqiWang/MIQA?style=social)](https://github.com/XiaoqiWang/MIQA)
  
[中文](README_CN.md) | [English](README.md) | [Colab](colab-link) | [博客](blog-link)
</div>
<div style="font-size: 13px;">
🎯 Project Overview

- 🤖 Machine-Centric: We bypass human perception to evaluate images from the perspective of the deep learning models that use them.  
- 📈 Task-Driven Metrics: Directly measure how degradations like blur, noise, or compression artifacts impact the performance of downstream vision tasks.  
- 💡 A New Paradigm: MIQA offers a new lens for optimizing image processing pipelines where machines make the final decision.
</div>

---
<!--
## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [📰 News & Announcements](#-news--announcements)
- [🔬 Research Background](#-research-background)
- [✨ Key Features](#-key-features)
- [🛠️ Installation](#️-installation)
- [📦 Model Weights](#-model-weights)
- [🚀 Quick Start](#-quick-start)
- [📊 Evaluation](#-evaluation)
- [📈 Benchmarks](#-benchmarks)
- [📚 Citation](#-citation)
- [🤝 Contributing](#-contributing)
- [📧 Contact](#-contact)
-->
<!--
## 🔥 **Latest Updates**
> To do list:
> 
> **[2025-XX-XX]** 📊 [Dataset release and download instructions](link-to-dataset)
>
> **[2025-XX-XX]** 📖 [Interactive Colab tutorial now available](colab-link)
>
> **[2025-XX-XX]** 🤗 [Models uploaded to HuggingFace Hub](huggingface-link)
-->

## 🔬 Research Background
- **Background**: Machine vision systems excel in controlled environments but suffer severe performance degradation from image distortions in real-world deployment. Traditional image quality assessment prioritizes human perceptual fidelity, creating a fundamental mismatch with machine sensitivities.

- **Key Benefits**: The machine-centric framework enables quality monitoring across acquisition, transmission, and processing stages, ensuring reliable machine vision performance and supporting optimization of automated visual systems in adverse conditions.


## ✨ Does MIQA Work?
<div align="center">
  <img src="./assets/cls_ratio.png" alt="Classification Performance" width="32%"/>
  <img src="./assets/det_ratio_ap75.png" alt="Detection Performance" width="32%"/>
  <img src="./assets/ins_ratio_ap75.png" alt="Instance Segmentation Performance" width="32%"/>
  <p><em>Performance improvement across tasks when filtering low-quality images using MIQA scores</em></p>
</div>

<details>
<summary> 🗝️ Key Results</summary>

Our results provide clear evidence of MIQA's effectiveness across three representative computer vision tasks: classification, detection, and segmentation.
The framework consistently identifies images that degrade model performance. By filtering these detrimental samples, MIQA directly leads to improved outcomes and demonstrates the universal utility of a machine-centric approach. This transforms quality assessment from a passive metric into a proactive tool, safeguarding downstream models against the unpredictable image quality of real-world conditions and ensuring robust performance when it matters most.
</details>

---
## 🛠️ Installation Guide

#### Step 1: Install Dependencies

To get started, you'll need to install two essential libraries: **mmcv** and **mmsegmentation**.
<details>
<summary> Install mmcv and mmsegmentation</summary>

* For the latest version of **mmsegmentation**, follow the installation guide here:
  [MMsegmentation Installation Guide](https://mmsegmentation.readthedocs.io/en/main/get_started.html)

* Alternatively, you can install a specific version of **mmsegmentation** based on your CUDA and PyTorch versions. You can find the version compatibility details here:
  [MMCV Installation Guide](https://mmcv.readthedocs.io/zh-cn/latest/get_started/installation.html)

</details>

#### Step 2: Handle CUDA Version Compatibility

If your CUDA version is relatively high, such as 12.7 or higher, you might encounter a version mismatch with **mmcv**. In this case, you may need to install a compatible version of **mmcv**.
<details>
<summary> Install a compatible version of mmcv</summary>

For example, if you need a specific version of **mmcv**, you can uninstall the existing versions and install a compatible one as follows:


```bash
pip uninstall mmcv mmcv-full -y
mim install "mmcv>=2.0.0rc4,<2.2.0"  # The version specified here is just an example. You should choose a version that is compatible with your CUDA and PyTorch setup.*
```
</details>

#### Step 3: Install Required Libraries

```bash
pip install -r requirements.txt
```

## 📦 Model Weights & Performance
<details open>
<summary> Composite Metric </summary>

| **Method**          |                                      **Image Classification** <br> (SRCC / PLCC & Download)                                       |                                        **Object Detection** <br> (SRCC / PLCC & Download)                                         |                                      **Instance Segmentation** <br> (SRCC / PLCC & Download)                                      | **Training Label Type** |
| :------------------ |:---------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------:|:---------------------:|
| **ResNet-18**       |   `0.5131 / 0.5427` <br> [**Google Drive**](https://drive.google.com/file/d/1zq03_TRYbg1zYEilP66x6HXpUUQ2sV_H/view?usp=sharing)   |   `0.7541 / 0.7734` <br> [**Google Drive**](https://drive.google.com/file/d/1_5mP7nOc2kla6l4QaTBBs5Xlj4hSu9dE/view?usp=sharing)   |   `0.7582 / 0.7790` <br> [**Google Drive**](https://drive.google.com/file/d/1umqAI4MiqfPK7dPiro6im_vDA_zrNfRO/view?usp=sharing)   |     Composite Score   |
| **ResNet-50**       |   `0.5581 / 0.5797` <br> [**Google Drive**](https://drive.google.com/file/d/1y8cV_iOOVNIa66WaAxESqqaOLiCv-GAY/view?usp=sharing)   |   `0.7743 / 0.7925` <br> [**Google Drive**](https://drive.google.com/file/d/1qLiznF02he6VHEGUDkNr9p0M2-4xO3kr/view?usp=sharing)   |   `0.7729 / 0.7933` <br> [**Google Drive**](https://drive.google.com/file/d/1Q-zgOoUvXQb3cKtxgC8B9YtbH5YVtYyg/view?usp=sharing)   |     Composite Score   |
| **EfficientNet-b1** |   `0.5901 / 0.6130` <br> [**Google Drive**](https://drive.google.com/file/d/1ERKTGO18AD2G1J-fr8zjvzoQpSbx6lAo/view?usp=sharing)   |   `0.7766 / 0.7950` <br> [**Google Drive**](https://drive.google.com/file/d/1vTKaEI_AG7Vnhmrn2B9Rkfblay-GyKvu/view?usp=sharing)   |   `0.7808 / 0.7999` <br> [**Google Drive**](https://drive.google.com/file/d/1aqun7dmtALkYwvhOSWzlnJByDHTPMQVn/view?usp=sharing)   |     Composite Score   |
| **EfficientNet-b5** |   `0.6330 / 0.6440` <br> [**Google Drive**](https://drive.google.com/file/d/1utE5Rd8onzSlHeve0WYvgDwq4Kctl4zf/view?usp=sharing)   |   `0.7866 / 0.8041` <br> [**Google Drive**](https://drive.google.com/file/d/1Vx4KcZfisyrfoiZ5zHfBMJpugsFgB82p/view?usp=sharing)   |   `0.7899 / 0.8074` <br> [**Google Drive**](https://drive.google.com/file/d/1pi2-5Iat1qq0xP9H1vDdlcZBpN5-EUwB/view?usp=sharing)   |     Composite Score   |
| **ViT-small**       |   `0.5998 / 0.6161` <br> [**Google Drive**](https://drive.google.com/file/d/11YSVK8rrjMfw3N8XAK_CqzQiL30SuOYZ/view?usp=sharing)   |   `0.7992 / 0.8142` <br> [**Google Drive**](https://drive.google.com/file/d/1-KUxxK3j0JflRp2oTKROLEVCBl5q21eF/view?usp=sharing)   |   `0.7968 / 0.8139` <br> [**Google Drive**](https://drive.google.com/file/d/10HcI61FEISLbmXME4knZEMBzQmOR8MVs/view?usp=sharing)   |   Composite Score     |
| **RA-MIQA (Ours)**  | **`0.7003 / 0.6989`** <br> [**Google Drive**](https://drive.google.com/file/d/1n_NhJcnVpb8dC3B2UZ5ETl2-a96uK0Js/view?usp=sharing) | **`0.8125 / 0.8264`** <br> [**Google Drive**](https://drive.google.com/file/d/1zUcrPOvvYd4rquAm1Wilnh03d8Hj1EDe/view?usp=sharing) | **`0.8188 / 0.8340`** <br> [**Google Drive**](https://drive.google.com/file/d/1uvN9jEFuGK5PFQzjiuS9s7A0H9NXyOyc/view?usp=sharing) |     Composite Score   |

</details>

<details>
<summary> Accuracy Metric</summary>
 
| **Method**          |                                      **Image Classification** <br> (SRCC / PLCC & Download)                                       |                                        **Object Detection** <br> (SRCC / PLCC & Download)                                         |                                      **Instance Segmentation** <br> (SRCC / PLCC & Download)                                      | **Training Label Type** |
| :------------------ |:---------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------:|:-----------------------:|
| **ResNet-50**       |   `0.4734 / 0.4411` <br> [**Google Drive**](https://drive.google.com/file/d/1mXzm-EuKhLY6zRW0jeVoBAi-kfGfGU0a/view?usp=sharing)   |   `0.6955 / 0.6898` <br> [**Google Drive**](https://drive.google.com/file/d/1e01vieTy4Fdgpqepoi1a1qpenpQLyfei/view?usp=sharing)   |   `0.6863 / 0.6847` <br> [**Google Drive**](https://drive.google.com/file/d/1qi9uCv_i3fAN6WVoYEHn6mI-BguFYEd-/view?usp=sharing)   |     Accuracy Score      |
| **EfficientNet-b5** |   `0.5586 / 0.5149` <br> [**Google Drive**](https://drive.google.com/file/d/1qz7Qwrpa6PSwtSgPczADsYf5tVOdujw3/view?usp=sharing)   |   `0.7042 / 0.6991` <br> [**Google Drive**](https://drive.google.com/file/d/1rH36SwceDQ4zSr_exWCvpL_G2AOnCLT-/view?usp=sharing)   |   `0.6933 / 0.6949` <br> [**Google Drive**](https://drive.google.com/file/d/1DzgEkhFB182XshMBrh_MsWNHQWOYB3Ea/view?usp=sharing)   |     Accuracy Score     |
| **ViT-small**       |   `0.5788 / 0.5197` <br> [**Google Drive**](https://drive.google.com/file/d/1fkROk-dQ63PdIeqiSIyrs7suDm_sJSFH/view?usp=sharing)   |   `0.7121 / 0.7052` <br> [**Google Drive**](https://drive.google.com/file/d/1K_b29iBLIx1AHCCNaNJUHYx_LT-1Rcwh/view?usp=sharing)   |   `0.7168 / 0.7146` <br> [**Google Drive**](https://drive.google.com/file/d/1Ft90uII_kfMLIHsIFJ4X8D4kI_jaxWC3/view?usp=sharing)   |     Accuracy Score     |
| **RA-MIQA (Ours)**  | **`0.6573 / 0.5823`** <br> [**Google Drive**](https://drive.google.com/file/d/1zVhc8Jl1TJYC7Th_4WvwpFiTwac6D6X0/view?usp=sharing) | **`0.7448 / 0.7370`** <br> [**Google Drive**](https://drive.google.com/file/d/1gGAM7Wr-65CtN4gUdoLU0ZvN-fdFbosD/view?usp=sharing) | **`0.7363 / 0.7327`** <br> [**Google Drive**](https://drive.google.com/file/d/1eR3ba5E-rbv6d08VBOXJ_EAUCDkVNGa9/view?usp=sharing) |     Accuracy Score     |

</details>

 
<details>
<summary> Consistency Metric </summary>

| **Method**          |                                      **Image Classification** <br> (SRCC / PLCC & Download)                                       |                                        **Object Detection** <br> (SRCC / PLCC & Download)                                         |                                      **Instance Segmentation** <br> (SRCC / PLCC & Download)                                      |  **Training Label Type**  |
| :------------------ |:---------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------:|:-------------------------:|
| **ResNet-50**       |   `0.5989 / 0.6551` <br> [**Google Drive**](https://drive.google.com/file/d/1VUPGUNatYPTvF_q9iNJ0WUAMLmeCNdPi/view?usp=sharing)   |   `0.8252 / 0.5457` <br> [**Google Drive**](https://drive.google.com/file/d/1HV_YiDcMGd2GNQDZiJBjq9oJQ4mmkWXs/view?usp=sharing)   |   `0.8320 / 0.8480` <br> [**Google Drive**](https://drive.google.com/file/d/1IYpjSy2Mbr0EMw8kagPrMy3ZFd7ggNUw/view?usp=sharing)   |     Consistency Score     |
| **EfficientNet-b5** |   `0.6774 / 0.7168` <br> [**Google Drive**](https://drive.google.com/file/d/1gao45m88gRzlY6jbcB3C0B3Y25eJpjvW/view?usp=sharing)   |   `0.8353 / 0.8530` <br> [**Google Drive**](https://drive.google.com/file/d/1stlveb-l4YfDW7Jd5HxqAvtkKoSpBVlO/view?usp=sharing)   |   `0.8419 / 0.8564` <br> [**Google Drive**](https://drive.google.com/file/d/1mbbalTCfZGvxR9zD03BhZCoOCfKOHYhp/view?usp=sharing)   |     Consistency Score     |
| **ViT-small**       |   `0.6798 / 0.7189` <br> [**Google Drive**](https://drive.google.com/file/d/1ZoRfSGJzu4NrIg7LZ03cLZ5Pwml1Di4o/view?usp=sharing)   |   `0.8459 / 0.8620` <br> [**Google Drive**](https://drive.google.com/file/d/1yx7hMh3Bt0qEE_9oNcP5LO_SeBre7sde/view?usp=sharing)   |   `0.8487 / 0.8616` <br> [**Google Drive**](https://drive.google.com/file/d/10VmxqqvpWnd7uxE7mx8WcRqJQNM8dbFo/view?usp=sharing)   |     Consistency Score     |
| **RA-MIQA (Ours)**  | **`0.7707 / 0.7866`** <br> [**Google Drive**](https://drive.google.com/file/d/1bJrNFAz4hWAP9wO680Kq36EhQ0oCl1sj/view?usp=sharing) | **`0.8526 / 0.8692`** <br> [**Google Drive**](https://drive.google.com/file/d/1TvyiN-DPtol0B7k2mo9bPXUoMjJ8F0Xn/view?usp=sharing) | **`0.8632 / 0.8756`** <br> [**Google Drive**](https://drive.google.com/file/d/1E9H7zerQgf2CUtLhttQBk70AsGb04hih/view?usp=sharing) |     Consistency Score     |

</details>


## 🚀 Quick Start

### 🖼️ Assess Image
#### Evaluate a Single Image
Run MIQA inference for a single image using command-line interface:

```
# Evaluate a single image for classification-oriented MIQA

python img_inference.py --input path/to/image.jpg --task cls --model ra_miqa
```

### Evaluate a Directory of Images

Process all images within a directory

```
# Assess all images in a directory (e.g., detection-oriented MIQA)

python img_inference.py --input ./assets/demo_images/coco_demo --task det --model ra_miqa
```

### Save Results and Visualizations
To save outputs and generate visualized results:
```
# Save the predicted scores and visualization for a single image
python img_inference.py --input path/to/image.jpg --task cls --model ra_miqa --save-results --visualize

# Save batch results and generate visualization for a directory
python img_inference.py --input ./assets/demo_images/imagenet_demo --task ins --save-results --visualize
```
<details open>
<summary>📸 <b>Results of MIQA Prediction</b></summary>

<small>

<p align="center">
<img src="inference_results/image/cls/composite/miqa_ra_miqa_ILSVRC2012_val_00024142_motion_blur_1.png" width="18%">
<img src="inference_results/image/cls/composite/miqa_ra_miqa_ILSVRC2012_val_00024142_motion_blur_2.png" width="18%">
<img src="inference_results/image/cls/composite/miqa_ra_miqa_ILSVRC2012_val_00024142_motion_blur_3.png" width="18%">
<img src="inference_results/image/cls/composite/miqa_ra_miqa_ILSVRC2012_val_00024142_motion_blur_4.png" width="18%">
<img src="inference_results/image/cls/composite/miqa_ra_miqa_ILSVRC2012_val_00024142_motion_blur_5.png" width="18%">
</p>

<p align="center">
<small><sub><em>
Pretrained model: <b>RA-MIQA</b> | Trained label: <b>Composite Score</b> | Distortion type: <b>Motion Blur</b> | Task: <b>Classification-oriented MIQA</b>
</em></sub></small>
</p>

<p align="center">
<img src="inference_results/image/det/composite/miqa_ra_miqa_000000258883_jpeg_compression_1.png" width="18%">
<img src="inference_results/image/det/composite/miqa_ra_miqa_000000258883_jpeg_compression_2.png" width="18%">
<img src="inference_results/image/det/composite/miqa_ra_miqa_000000258883_jpeg_compression_3.png" width="18%">
<img src="inference_results/image/det/composite/miqa_ra_miqa_000000258883_jpeg_compression_4.png" width="18%">
<img src="inference_results/image/det/composite/miqa_ra_miqa_000000258883_jpeg_compression_5.png" width="18%">
</p>

<p align="center">
<small><sub><em>
Pretrained model: <b>RA-MIQA</b> | Trained label: <b>Composite Score</b> | Distortion type: <b>JPEG Compression</b> | Task: <b>Detection-oriented MIQA</b>
</em></sub></small>
</p>

</small>
</details>


### 🎬 Video Assessment

Video Quality Assessment offers two workflows: **(1) Frame-by-Frame Annotation**: Generates fully annotated videos for detailed visual inspection. Suitable for demos and qualitative analysis but computationally intensive.
**(2) Selective Sampling & Aggregation**: Samples frames to produce plots and structured data (.json) for efficient, quantitative analysis. Ideal for batch processing and reporting.
#### Analyze a Single Video (**Frame-by-Frame Annotation**)

Run MIQA video inference for one video and save the annotated output. 
```bash
# Evaluate a single video using RA-MIQA (classification-oriented MIQA)
python video_annotator_inference.py --input assets/demo_video/brightness_distorted.mp4 --task cls --model ra_miqa
```

#### Evaluate a Directory of Videos (**Frame-by-Frame Annotation**)
Process all videos within a given folder:

```bash
# Assess all videos in a directory for object detection-oriented MIQA
python video_annotator_inference.py --input assets/demo_video/ --task det --model ra_miqa
```

The primary output is a new `.mp4` video file. This video shows the original footage playing alongside a dynamic side panel that displays the real-time quality score and a line chart that grows as the video progresses.

<details open>

<summary>🎥 <b>Example: Frame-wise MIQA Predictions on Videos</b></summary> 

| Brightness Variation | Compression Artifacts | Minimal Perceptual Distortion |
| :---: | :---: | :---: |
| <video src="https://github.com/user-attachments/assets/9b20cbc4-3baf-4d57-8d5f-49acd6873725" width="280" controls></video> | <video src="https://github.com/user-attachments/assets/c2fc142b-6889-4451-8a05-fb93e0ec0656" width="280" controls></video> | <video src="https://github.com/user-attachments/assets/14f4fc37-5ae5-4068-81f2-6f86bec30a27" width="280" controls></video> | 

</details>

#### Analyze a Single Video (**Selective Sampling & Aggregation**)

For efficient, quantitative analysis, this script samples frames from the video instead of processing all of them. It is significantly faster and designed for generating analytical reports.

```bash
# Analyze a video, sample frames, and create a dual-granularity plot
python video_analytics_inference.py --input assets/demo_video/gaussian_distorted.mp4 --task ins --visualize --viz-granularity both
```

#### Evaluate a Directory of Videos (**Selective Sampling & Aggregation**)

This workflow is highly optimized for batch processing.

```bash
# Analyze all videos in a directory, sampling 120 frames from each
python video_analytics_inference.py --input assets/demo_video/ --task det --video-frames 120 --visualize


python video_analytics_inference.py --input assets/demo_video/jpeg_distorted.mp4  --task det --visualize --viz-granularity both
# viz-granularity both : Specifies the type of plot to generate. 'composite' creates a comprehensive, side-by-side comparison chart showing:
#1. The raw, frame-level quality scores. 2. The smoothed, per-second average quality scores.
```

This process **does not create a new video**. Instead, it generates two key outputs for each video analyzed:
1.  A **`.png` image**: A detailed time-series plot showing the quality score fluctuation over the video's duration.
2.  A **`.json` file**: A structured data file containing per-second aggregated scores, overall statistics (average, min, max, std. dev), and video metadata.



<details>
<summary>🎥 <b>Example: Selective Sampling MIQA Predictions on Videos</b></summary>

| Brightness Variation | Compression Artifacts |         Minimal Perceptual Distortion          |
| :---: | :---: |:----------------------------------------------:|
| <img src="inference_results/brightness_distorted_composite_quality_comparison.png" width="280"> | <img src="inference_results/jpeg_distorted_composite_quality_comparison.png" width="280"> | <img src="inference_results/B314_composite_quality_comparison.png" width="280"> |
 
</details>

## 🏃 Training and Evaluation

### Training 

```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py \
      --dataset 'miqa_cls' \
      --path_miqa_cls 'path/to/datasets_miqa_cls' \
      --train_split_file '../data/dataset_splitting/miqa_cls_train.csv' \
      --val_split_file '../data/dataset_splitting//miqa_cls_val.csv' \
      --metric_type 'composite' --loss_name 'mse' --is_two_transform \
      -a 'RA-MIQA' --pretrained --transform_type 'simple_transform' \
      -b 256 --epochs 5 --warmup_epochs 1 --validate_num 2 --lr 1e-4 \
      --image_size 288 --crop_size 224 --workers 8 -p 100 \
      --multiprocessing-distributed --world-size 1 --rank 0
```
More training scripts are available in the "**scripts**" directory.

### Evaluation on Benchmarks

```bash
# Evaluate on miqa_cls val set
python evaluate.py --model_name ra_miqa  --train_dataset cls  --test_dataset cls  --metric_type composite

# Cross-dataset evaluation: evaluate the RA-MIQA model trained on miqa_cls dataset and tested on miqa_det dataset
python evaluate.py --model_name ra_miqa  --train_dataset cls  --test_dataset det  --metric_type composite
```

## 📈 Benchmarks

<details>
<summary>Tabel 1: Performance Benchmark on Composite Score</summary>

<table>
<thead>
<tr>
<th rowspan="2" style="text-align: center;">Category</th>
<th rowspan="2" style="text-align: left;">Method</th>
<th colspan="4" style="text-align: center;">Image Classification</th>
<th colspan="4" style="text-align: center;">Object Detection</th>
<th colspan="4" style="text-align: center;">Instance Segmentation</th>
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
<td rowspan="7" style="text-align: center; vertical-align: middle;"><strong>HVS-based</strong></td>
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
<td colspan="14" style="border-bottom: 1px solid #ddd;"></td>
</tr>
<tr>
<td rowspan="6" style="text-align: center; vertical-align: middle;"><strong>Machine-based</strong></td>
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
<td style="text-align: left;"><strong>RA-MIQA</strong></td>
<td style="text-align: center;"><strong>0.7003</strong><br><small style="color: #c00;"></small></td>
<td style="text-align: center;"><strong>0.6989</strong><br><small style="color: #c00;"></small></td>
<td style="text-align: center;"><strong>0.5255</strong><br><small style="color: #c00;"></small></td>
<td style="text-align: center;"><strong>0.2152</strong><br><small style="color: #c00;"></small></td>
<td style="text-align: center;"><strong>0.8125</strong><br><small style="color: #c00;"></small></td>
<td style="text-align: center;"><strong>0.8264</strong><br><small style="color: #c00;"></small></td>
<td style="text-align: center;"><strong>0.6263</strong><br><small style="color: #c00;"></small></td>
<td style="text-align: center;"><strong>0.1596</strong><br><small style="color: #c00;"></small></td>
<td style="text-align: center;"><strong>0.8188</strong><br><small style="color: #c00;"></small></td>
<td style="text-align: center;"><strong>0.8340</strong><br><small style="color: #c00;"></small></td>
<td style="text-align: center;"><strong>0.6333</strong><br><small style="color: #c00;"></small></td>
<td style="text-align: center;"><strong>0.1505</strong><br><small style="color: #c00;"></small></td>
</tr>
</tbody>
</table>

</details>

<details>
<summary>Table 2: Consistency & Accuracy Score Benchmark</summary>
<table>
<thead>
<tr>
<th rowspan="3" style="text-align: center; vertical-align: middle;">Method</th>
<th colspan="6" style="text-align: center;">Image Classification</th>
<th colspan="6" style="text-align: center;">Object Detection</th>
<th colspan="6" style="text-align: center;">Instance Segmentation</th>
</tr>
<tr>
<th colspan="3" style="text-align: center;">Accuracy Score</th>
<th colspan="3" style="text-align: center;">Consistency Score</th>
<th colspan="3" style="text-align: center;">Accuracy Score</th>
<th colspan="3" style="text-align: center;">Consistency Score</th>
<th colspan="3" style="text-align: center;">Accuracy Score</th>
<th colspan="3" style="text-align: center;">Consistency Score</th>
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
<tr><td colspan="19" style="font-weight: bold; text-align: left; padding-top: 8px; padding-bottom: 4px; border-bottom: 1px solid #ddd;"><em>HVS-based Methods</em></td></tr>
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
<tr><td colspan="19" style="font-weight: bold; text-align: left; padding-top: 8px; padding-bottom: 4px; border-bottom: 1px solid #ddd;"><em>Machine-based Methods</em></td></tr>
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
<td style="text-align: center;"><strong>0.6573</strong><br></td>
<td style="text-align: center;"><strong>0.5823</strong><br></td>
<td style="text-align: center;"><strong>0.2917</strong><br></td>
<td style="text-align: center;"><strong>0.7707</strong><br></td>
<td style="text-align: center;"><strong>0.7866</strong><br></td>
<td style="text-align: center;"><strong>0.1732</strong><br></td>
<td style="text-align: center;"><strong>0.7448</strong><br></td>
<td style="text-align: center;"><strong>0.7370</strong><br></td>
<td style="text-align: center;"><strong>0.1915</strong><br></td>
<td style="text-align: center;"><strong>0.8526</strong><br></td>
<td style="text-align: center;"><strong>0.8692</strong><br></td>
<td style="text-align: center;"><strong>0.1527</strong><br></td>
<td style="text-align: center;"><strong>0.7363</strong><br></td>
<td style="text-align: center;"><strong>0.7327</strong><br></td>
<td style="text-align: center;"><strong>0.1834</strong><br></td>
<td style="text-align: center;"><strong>0.8632</strong><br></td>
<td style="text-align: center;"><strong>0.8756</strong><br></td>
<td style="text-align: center;"><strong>0.1464</strong><br></td>
</tr>
</tbody>
</table>
</details> 


## 📚 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{wang2025miqa,
  title={Image Quality Assessment for Machines: Paradigm, Large-scale Database, and Models},
  author={Wang, Xiaoqi and Zhang, Yun and Lin, Weisi},
  journal={arXiv preprint arXiv:2508.19850},
  year={2025}
}
```



## 🤝 Contributing
* [ ] TO DO : 

We welcome contributions from the community! If you're interested in improving MIQA, please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit bug reports, feature requests, and pull requests.

</details>

<details>
<summary>Development Setup</summary>
To set up your environment for local development, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/XiaoqiWang/MIQA.git
    cd MIQA
    ```

2.  **Install dependencies in editable mode:**
    This command installs the project and the development-specific dependencies (like testing tools).
    ```bash
    pip install -e ".[dev]"
    ```

3.  **Set up pre-commit hooks:**
    This ensures your code contributions adhere to our style guidelines automatically.
    ```bash
    pre-commit install
    ```

### Running Tests

We use `pytest` for testing. You can run the test suite to ensure your changes are working correctly.

1.  **Run all tests:**
    ```bash
    pytest tests/
    ```

2.  **Run tests with coverage report:**
    To check how much of the code is covered by tests, run:
    ```bash
    python -m pytest tests/ --cov=miqa # Replace `miqa` with the actual name of your source package if it's different.
    ```
    
</details> 

## 📧 Contact

- **Project Maintainer**: [Xiaoqi Wang](mailto:wangxq79@mail2.sysu.edu.cn)
- **Issues**: Please use [GitHub Issues](https://github.com/XiaoqiWang/MIQA/issues) for bug reports and feature requests

---
**⭐ Star this repository if you find it helpful!😊**

*Last updated: [10/25/2025]* 
