<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->

<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!-- [![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url] -->
<!-- 
***[![MIT License][license-shield]][license-url]
-->

<!-- PROJECT LOGO -->

<div align="center">
<h2><a href="https://arxiv.org/abs/2310.03013">SemiReward: A General Reward Model for Semi-supervised Learning (ICLR 2024)</a> </h2>

[Siyuan Li](https://lupin1998.github.io/)<sup>\*,1,2</sup>, [Weiyang Jin](https://scholar.google.co.id/citations?hl=zh-CN&user=cazmdIMAAAAJ)<sup>\*,1</sup>, [Zedong Wang](https://zedongwang.netlify.app/)<sup>1,2</sup>, [Fang Wu](https://smiles724.github.io/)<sup>1,2</sup>, [Zicheng Liu](https://pone7.github.io/)<sup>1,2</sup>, [Chen Tan](https://chengtan9907.github.io/)<sup>1,2</sup>, [Stan Z. Li](https://scholar.google.com/citations?user=Y-nyLGIAAAAJ&hl=zh-CN)<sup>†,1</sup>

<sup>1</sup>[Westlake University](https://westlake.edu.cn/), <sup>2</sup>[Zhejiang University](https://www.zju.edu.cn/english/)
</div>

<p align="center">
<a href="https://arxiv.org/abs/2310.03013" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2310.03013-b31b1b.svg?style=flat" /></a>
<a href="https://github.com/Westlake-AI/SemiReward/blob/main/LICENSE.txt" alt="license">
    <img src="https://img.shields.io/badge/license-Apache--2.0-%23B7A800" /></a>
<a href="https://openreview.net/forum?id=dnqPvUjyRI" alt="Colab">
    <img src="https://img.shields.io/badge/openreview-SemiReward-blue" /></a>
</p>

Semi-supervised Reward framework (SemiReward) is designed to predict reward scores to evaluate and filter out high-quality pseudo labels, which is pluggable to mainstream Semi-Supervised Learning (SSL) methods in wide task types and scenarios. The results and details are reported in [our paper](https://arxiv.org/abs/2310.03013). The implementations and models of **SemiReward** are based on **USB** codebase.
_**USB** is a Pytorch-based Python package for SSL. It is easy-to-use/extend, *affordable* to small groups, and comprehensive for developing and evaluating SSL algorithms. USB provides the implementation of 14 SSL algorithms based on Consistency Regularization, and 15 tasks for evaluation from CV, NLP, and Audio domain. More details can be seen in [Semi-supervised Learning](https://github.com/microsoft/Semi-supervised-learning)._

<p align="center">
<img src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/276408256-f860de7b-bb3c-42c3-8ef9-2f1a91dac55b.png" width=100% height=100% 
class="center">
</p>

<!-- TABLE OF CONTENTS -->

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#news-and-updates">News and Updates</a></li>
    <li><a href="#intro">Introduction</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Community</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- Introduction -->

## Introduction

Semi-supervised learning (SSL) has witnessed great progress with various improvements in the self-training framework with pseudo labeling. The main challenge is how to distinguish high-quality pseudo labels against the confirmation bias. However, existing pseudo-label selection strategies are limited to pre-defined schemes or complex hand-crafted policies specially designed for classification, failing to achieve high-quality labels, fast convergence, and task versatility simultaneously. To these ends, we propose a Semi-supervised Reward framework (SemiReward) that predicts reward scores to evaluate and filter out high-quality pseudo labels, which is pluggable to mainstream SSL methods in wide task types and scenarios. To mitigate confirmation bias, SemiReward is trained online in two stages with a generator model and subsampling strategy. With classification and regression tasks on 13 standard SSL benchmarks of three modalities, extensive experiments verify that SemiReward achieves significant performance gains and faster convergence speeds upon Pseudo Label, FlexMatch, and Free/SoftMatch.

<p align="center">
<img src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/276409517-7a7907f5-01b9-4953-818d-767c0dfb9c6b.png" width=90% 
class="center">
</p>

<!-- News and Updates -->

## News and Updates

- [01/16/2024] SemiReward v0.2.0 has been updated and accepted by [ICLR'2024](https://openreview.net/forum?id=dnqPvUjyRI).
- [10/18/2023] SemiReward v0.1.0 has been released.

<p align="right">(<a href="#top">back to top</a>)</p>


## Getting Started

First, you need to set up USB locally.
To get a local copy up, running follow these simple example steps.

### Prerequisites

USB is built on pytorch, with torchvision, torchaudio, and transformers.

To install the required packages, you can create a conda environment:

```sh
conda create --name semireward python=3.8
```

then use pip to install required packages:

```sh
pip install -r requirements.txt
```

From now on, you can start use USB by typing 

```sh
python train.py --c config/usb_cv/fixmatch/fixmatch_cifar100_200_0.yaml
```

### Installation

USB provide a Python package *semilearn* of USB for users who want to start training/testing the supported SSL algorithms on their data quickly:

```sh
pip install semilearn
```

You can also develop your own SSL algorithm and evaluate it by cloning SemiReward (USB):
```sh
git clone https://github.com/Westlake-AI/SemiReward.git
```

<p align="right">(<a href="#top">back to top</a>)</p>


### Prepare Datasets

The detailed instructions for downloading and processing are shown in [Dataset Download](./preprocess/). Please follow it to download datasets before running or developing algorithms.

<p align="right">(<a href="#top">back to top</a>)</p>


## Usage

### Start with Docker
The following steps to train your own SemiReward model just as same with USB.

**Step1: Check your environment**

You need to properly install Docker and nvidia driver first. To use GPU in a docker container
You also need to install nvidia-docker2 ([Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)).
Then, Please check your CUDA version via `nvidia-smi`

**Step2: Clone the project**

```shell
git clone https://github.com/microsoft/Semi-supervised-learning.git
```

**Step3: Build the Docker image**

Before building the image, you may modify the [Dockerfile](Dockerfile) according to your CUDA version.
The CUDA version we use is 11.6. You can change the base image tag according to [this site](https://hub.docker.com/r/nvidia/cuda/tags).
You also need to change the `--extra-index-url` according to your CUDA version in order to install the correct version of Pytorch.
You can check the url through [Pytorch website](https://pytorch.org).

Use this command to build the image

```shell
cd Semi-supervised-learning && docker build -t semilearn .
```

Job done. You can use the image you just built for your own project. Don't forget to use the argument `--gpu` when you want
to use GPU in a container.

### Training

Here is an example to train one of baselines FlexMatch on CIFAR-100 with 200 labels. Training other supported algorithms (on other datasets with different label settings) can be specified by a config file:

```sh
python train.py --c config/usb_cv/flexmatch/flexmatch_cifar100_200_0.yaml
```

Here is an example to train FlexMatch with SemiReward on CIFAR-100 with 200 labels. Training other baselines with SemiReward can be specified by a config file:

```sh
python train.py --c config/SemiReward/usb_cv/flexmatch/flexmatch_cifar100_200_0.yaml
```
You can change hyperparameters for SemiReward by configurations (.yaml files) like other baselines. If you want to change loss or something is fixed in our method for SemiReward, it is recommanded to open flie from:

```sh
semilearn/algorithms/srflexmatch/srflexmatch.py
```

**Tips:** Semireward use **4GPUs** for training by default. Also, for users in some areas of China, huggingface region locking occurs, so local pre-training weights need to be used when using the Bert and huBert models. Take the Bert model as an example, you need to focus on `./semilearn/datasets/collactors/nlp_collactor.py`, find line 102 to change it's address into your local folder for Bert. Also, in file `./semilearn/nets/bert/bert.py` line 13, it need to the same way to adjust.

### Evaluation

After training, you can check the evaluation performance on training logs, or running evaluation script:

```
python eval.py --dataset cifar100 --num_classes 100 --load_path /PATH/TO/CHECKPOINT
```

<p align="center">
<img src="https://github.com/Westlake-AI/openmixup/assets/44519745/266f5667-9e5f-44c9-ba63-f3e8b733d5a9" width=95% 
class="center">
</p>

### Develop

Check the developing documentation for creating your own SSL algorithm!

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#top">back to top</a>)</p>


## Contributing

If you have any ideas to improve SemiReward, we welcome your contributions! Feel free to fork the repository and submit a pull request. Alternatively, you can open an issue and label it as "enhancement." Don't forget to show your support by giving the project a star! Thank you once more!

1. Fork the project
2. Create your branch (`git checkout -b your_name/your_branch`)
3. Commit your changes (`git commit -m 'Add some features'`)
4. Push to the branch (`git push origin your_name/your_branch`)
5. Open a Pull Request


## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>


## Citation

Please consider citing us if you find this project helpful for your project/paper:

```
@inproceedings{iclr2024semireward,
  title={SemiReward: A General Reward Model for Semi-supervised Learning},
  author={Siyuan Li and Weiyang Jin and Zedong Wang and Fang Wu and Zicheng Liu and Cheng Tan and Stan Z. Li},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```


## Acknowledgments

SemiReward's implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works:

- [USB](https://github.com/microsoft/Semi-supervised-learning)
- [TorchSSL](https://github.com/TorchSSL/TorchSSL)
- [FixMatch](https://github.com/google-research/fixmatch)
- [CoMatch](https://github.com/salesforce/CoMatch)
- [SimMatch](https://github.com/KyleZheng1997/simmatch)
- [HuggingFace](https://huggingface.co/docs/transformers/index)
- [Pytorch Lighting](https://github.com/Lightning-AI/lightning)
- [README Template](https://github.com/othneildrew/Best-README-Template)

## Contribution and Contact

For adding new features, looking for helps, or reporting bugs associated with `SemiReward`, please open a [GitHub issue](https://github.com/Westlake-AI/SemiReward/issues) and [pull request](https://github.com/Westlake-AI/SemiReward/pulls) with the tag "new features" or "help wanted". Feel free to contact us through email if you have any questions.

- Siyuan Li (lisiyuan@westlake.edu.cn), Westlake University & Zhejiang University
- Weiyang Jin (wayneyjin@gmail.com), Westlake University & Beijing Jiaotong University

<p align="right">(<a href="#top">back to top</a>)</p>
