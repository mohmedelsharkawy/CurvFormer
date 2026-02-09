# CurvFormer: Boundary-Guided Differential Geometry for Topology-Preserving Retinal Layer Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Conference: MICCAI](https://img.shields.io/badge/MICCAI-2026-blue)](https://miccai.org/)

## 📌 Abstract
This repository contains the official PyTorch implementation of **CurvFormer**, a geometry-aware volumetric network designed for fine-grained OCT analysis.

Retinal layer segmentation in sparse clinical protocols is often compromised by pathological deformations and limited depth resolution. **CurvFormer** addresses this by injecting explicit **differential geometric priors**—specifically a differentiable Structure Tensor—into an **Axial Attention** bottleneck. This mechanism allows the model to learn topological continuity and anisotropy, ensuring robust separation of **11 distinct retinal layers**, even in the presence of fluid or drusen.

## 🌟 Key Features

* **Differentiable Structure Tensor:** Computes explicit curvature ($e$) and coherence ($c$) cues to guide the attention mechanism.
* **Geometric Axial Attention:** A transformer bottleneck that utilizes learned geometric biases to preserve global topology.
* **Boundary-Guided Decoding:** A dual-stream decoder that enforces crisp transitions between thin retinal layers.
* **Anisotropic Design:** optimized for clinical OCT stacks ($D=11$) to prevent depth information loss.

## 🚀 Usage

### Installation
```bash
git clone [https://github.com/yourusername/CurvFormer.git](https://github.com/yourusername/CurvFormer.git)
cd CurvFormer
pip install -r requirements.txt
