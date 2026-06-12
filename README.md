# 🎧 AudioX-Turbo: A Unified Framework for Efficient Anything-to-Audio Generation

[![arXiv](https://img.shields.io/badge/arXiv-2606.12555-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2606.12555)
[![Project Page](https://img.shields.io/badge/GitHub.io-Project-blue?logo=Github&style=flat-square)](https://zeyuet.github.io/AudioX-Turbo/)




---

## ✨ Abstract

Audio and music generation based on flexible multimodal control signals is a widely applicable topic, with the following key challenges: 1) a unified multimodal modeling framework, 2) large-scale, high-quality training data, and 3) the prohibitive inference cost of multi-step diffusion sampling.

As such, we propose **AudioX-Turbo**, a unified and efficient framework for anything-to-audio generation that integrates varied multimodal conditions (i.e., text, video, and audio signals). AudioX-Turbo follows a *teacher–student* paradigm. The teacher **AudioX-Base** is built on a Multimodal Diffusion Transformer with a Multimodal Adaptive Fusion module that aligns diverse multimodal inputs for high-fidelity synthesis, and is then distilled into the few-step student **AudioX-Turbo** via Distribution Matching Distillation adapted to flow matching, complemented by a diffusion-based discriminator for high-quality few-step generation.

To support training, we construct a large-scale, high-quality dataset, **IF-caps-Pro**, comprising approximately **9.2M** samples curated through a two-stage data collection and annotation pipeline. We benchmark AudioX-Turbo across a wide range of tasks, finding that our model achieves superior performance, especially on text-to-audio and text-to-music generation, while operating at only **4 sampling steps** and requiring up to **~25×** fewer function evaluations (NFE) than multi-step baselines. These results demonstrate that our method is capable of audio generation under flexible multimodal control, showing efficient and powerful instruction-following capabilities.

## ✨ Teaser

<p align="center">
  <img src="https://github.com/user-attachments/assets/9a4ddcbb-fa6b-4dd1-acc9-f5f0484e84e9" width="100%" alt="teaser"/>
</p>
<p align="center">
  Performance comparison against baselines: (a) Inception Score across benchmarks, (b) instruction-following results, (c) quality–efficiency trade-off.
</p>

## ✨ Method

<p align="center">
  <img src="https://github.com/user-attachments/assets/7cb50c13-afd0-49a7-8945-e050bb59981e" width="100%" alt="method"/>
</p>
<p align="center">
  Overview of the AudioX-Turbo framework.
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/306d75a4-ac12-40f0-b78c-26658ee2c201" width="100%" alt="distillation_framework"/>
</p>
<p align="center">
  Few-step distillation with DMD, fake-model diffusion loss, and adversarial supervision.
</p>

---

## 🚧 Status

Code, pretrained models, dataset, and demo are **coming soon**.

---

## 🚀 Citation

If you find our work useful, please consider citing:

```bibtex
@article{tian2026audioxturbo,
  title={AudioX-Turbo: A Unified Framework for Efficient Anything-to-Audio Generation},
  author={Tian, Zeyue and Ke, Lei and Liu, Zhaoyang and Yuan, Ruibin and Xue, Liumeng and Yang, Yujiu and Chen, Weijia and Tan, Xu and Chen, Qifeng and Xue, Wei and Guo, Yike},
  journal={arXiv preprint arXiv:2606.12555},
  year={2026}
}
@inproceedings{tian2026audiox,
  title={AudioX: a unified framework for anything-to-audio generation},
  author={Tian, Zeyue and Jin, Y and Liu, Z and others},
  booktitle={Proceedings of the Fourteenth International Conference on Learning Representations},
  year={2026}
}
@inproceedings{tian2025vidmuse,
  title={Vidmuse: A simple video-to-music generation framework with long-short-term modeling},
  author={Tian, Zeyue and Liu, Zhaoyang and Yuan, Ruibin and Pan, Jiahao and Liu, Qifeng and Tan, Xu and Chen, Qifeng and Xue, Wei and Guo, Yike},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={18782--18793},
  year={2025}
}
```

---

## 📭 Contact

- **Zeyue Tian**: ztianad@connect.ust.hk

---

## 📄 License

Please follow [CC-BY-NC 4.0](./LICENSE).

**Note:** The models are watermarked and are strictly for non-commercial use only.
