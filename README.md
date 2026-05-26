<h1 align="center" style="margin-top: 10px;">AT<sup>2</sup>PO: Agentic Turn-based Policy Optimization via Tree Search</h1>



<div align="center"> 

[![Paper](https://img.shields.io/badge/Paper-arXiv-b5212f.svg?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2601.04767)
[![Paper](https://img.shields.io/badge/Paper-Hugging%20Face-yellow?style=flat-square&logo=huggingface)](https://huggingface.co/papers/2601.04767)
<!-- [![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](https://swanlab.cn/@yux1ang/Tree-GRPO/overview) -->

</div>

<!-- ## News
- [Sep 25, 2025]: Codebase released. (work in progress) -->

## Table of contents

- [Overview](#overview)
- [Quick start](#quick-start)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)


## Overview
Agentic reinforcement learning (RL) has proven effective for training LLM-based agents with external tool-use capabilities. However, we identify that agentic RL training induces increasing redundant tool calls and blurs the model's intrinsic knowledge boundary, where the model fails to distinguish when tools are needed versus when parametric knowledge suffices. Existing solutions based on reward shaping create coarse-grained optimization targets that tend to incentivize indiscriminate tool-call suppression, leading to reward hacking. In this paper, we propose **AKBE** (**A**gentic **K**nowledge **B**oundary **E**nhancement), an on-policy method that dynamically probes the model's intrinsic knowledge boundary through dual-path (with-tool and no-tool) rollouts during training. We define the knowledge boundary as the per-instance determination of whether tools are required and the minimum tool calls necessary. By comparing correctness across paths, AKBE categorizes trajectories and constructs targeted supervisory signals that guide efficient tool-use patterns for each question. These signals are integrated seamlessly into the agentic RL training loop. Experiments on seven QA benchmarks demonstrate that AKBE improves task accuracy by +1.85 on average and reduces tool calls by 18% over standard agentic RL, yielding 25% higher tool productivity without any accuracy-efficiency trade-off. Further analysis suggests its plug-and-play compatibility across different RL algorithms and the mechanism of each signal category.

<p align="center">
  <img alt="intro" src="assets/method_framework.pdf" />
  <i>
  The overview of AKBE framework.
  </i>
</p>

*Evaluation on seven benchmarks shows consistent improvement against existing strongest baselines.*

## Quick Start

### Local Retriever Tool Initialization

#### Environment
```bash
conda create -y -n retriever python=3.10
conda activate retriever
conda install -y pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini
conda install -y -c pytorch -c nvidia faiss-gpu=1.8.0
pip install uvicorn fastapi
```

#### Download Retriever Data
```bash
save_path=/the/path/to/save
python rag_server/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

#### Initialize Retriever API
```bash
conda activate retriever
# edit save_path in rag_server/launch.sh
bash rag_server/launch.sh
```

### Dataset
```bash
# Process training set of multi-hop QA benchmarks with dual path rollout
python data_process/hotpotqa_dual_path.py
# Process test set of multi-hop QA benchmarks
python data_process/multihop_test_merge.py
# Process training set of single-hop QA benchmarks with dual path rollout
python data_process/nq_dual_path.py
# Process test set of single-hop QA benchmarks
python data_process/singlehop_test_merge.py
```


### Training Environment Installation

```bash
#create env
conda create -n akbe python==3.10
conda activate akbe

# install torch & flash-atten
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation

# install RL basic env
cd AKBE

# This is our RL env freeze file. You can install it as a supplement or use it for checking.
pip install -r requirements.txt

```

### RL Training

Run AKBE training with Qwen3-4B on multi-hop QA setting.
```bash
conda activate akbe
bash AKBE/scripts/AKBE_multihop_qwen3_4B.sh
```

## Acknowledgement
The codebase is built upon [veRL](https://github.com/volcengine/verl).
The implementation is inspired by [AEPO](https://github.com/RUC-NLPIR/ARPO).
We express our gratitude to these open-source projects.

## Citation
```bibtex

```
