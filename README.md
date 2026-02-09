# CurvFormer: Boundary-Guided Differential Geometry for Topology-Preserving Retinal Layer Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Conference](https://img.shields.io/badge/MICCAI-2026-blue)](https://miccai.org/)



## 📌 Abstract
[cite_start]This repository contains the official PyTorch implementation of **CurvFormer**, a boundary-guided geometric transformer for volumetric OCT segmentation[cite: 33]. [cite_start]Unlike standard transformers, CurvFormer injects explicit **differential geometric priors** (via a differentiable Structure Tensor) into an **Axial Attention** bottleneck[cite: 34]. [cite_start]This ensures topological continuity and precise boundary delineation, even in sparse clinical protocols ($D=11$ slices)[cite: 26, 32].

## 🌟 Key Features
* [cite_start]**Geometry-Aware Attention:** Injects curvature and anisotropy priors directly into the transformer bottleneck[cite: 34].
* [cite_start]**Structure Tensor Module:** Computes differentiable edge strength ($e$) and coherence ($c$) to guide learning[cite: 54].
* [cite_start]**Fine-Grained Segmentation:** Designed for dense **11-layer** retinal phenotyping (vs. standard 7-9 layers)[cite: 27].
* [cite_start]**Boundary-Guided Decoding:** Explicit boundary heads reduce leakage between thin layers[cite: 34].
* [cite_start]**OCT-Specific Design:** Anisotropic encoder-decoder preserves depth resolution for short stacks[cite: 96].

## 🚀 Usage

### Installation
```bash
git clone [https://github.com/yourusername/CurvFormer.git](https://github.com/yourusername/CurvFormer.git)
cd CurvFormer
pip install -r requirements.txt
