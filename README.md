
# CoTj

## 🧭 Description

**CoTj (Chain-of-Trajectories)** is a **graph-theoretic trajectory planning framework** for diffusion models. 
It upgrades the standard, fixed-step denoising schedules (System 1) into **condition-adaptive, optimally planned trajectories (System 2)**, enabling flexible, high-fidelity image generation under varying prompts and constraints.

CoTj establishes an **offline graph for each condition**, searches for optimal denoising paths, and supports both **fixed-step optimal sequences** and **adaptive-length planning** for fewer steps without sacrificing output quality. 
The full methodology, theoretical analysis, and experiments are detailed in a research work **currently under submission**.
<img width="1453" height="1185" alt="image" src="https://github.com/user-attachments/assets/1457d497-72f5-4dd2-a771-4736ef5e4f48" />

---

## 🚀 Quick Start

CoTj can be directly used with the **Qwen-Image pipeline**. Example usage:

```python
from CoTj_pipeline_qwenimage import CoTjQwenImagePipeline
import os

model_path = '~/.cache/modelscope/hub/models/Qwen/Qwen-Image/'
mlp_path = './prompt_models/qwenimage_mlp_models/'
device = 'cuda:0'

pipe = None
cotj = CoTjQwenImagePipeline(model_path=model_path, mlp_path=mlp_path, pipe=pipe, device=device)

prompt = "一位身着深蓝色Polo衫的年轻女性研究员，胸前印有“Unicom”的红色Logo，正对镜头自信微笑，在充满科技感的数据中心透明的玻璃幕墙上，用黑色马克笔清晰地写着：“CoTj 让生成式 AI 从‘盲人摸象’的固定模式，迈入‘智能规划’的自适应时代。”"


num_inference_steps = 10

# Baseline Euler sampling
pipe_image = cotj.get_pipe_image(prompt, 
                                 num_inference_steps=num_inference_steps, 
                                 width=1664, 
                                 height=928,
                                 seed=42)

# Fixed-Step Planning
prompt_cotj_image_fixed = cotj.get_prompt_cotj_image_fixed_step(prompt, 
                                                                num_inference_steps=num_inference_steps, 
                                                                width=1664, 
                                                                height=928,
                                                                seed=42)

# Adaptive-Length Planning
prompt_cotj_image_adaptive = cotj.get_prompt_cotj_image_adaptive_step(prompt, 
                                                                      inference_steps_max=50, 
                                                                      fidelity_target=0.99, 
                                                                      width=1664, 
                                                                      height=928,
                                                                      seed=42)

```

For a complete demo, see `CoTj_qwenimage_demo.ipynb`.

**Note:** This example uses Qwen-Image with the default Euler sampler.

## 🌟 Acknowledgements

This implementation is built upon the Hugging Face **Diffusers** library. 

---

## 📖 Citation

If you find CoTj useful, please consider citing:

```bibtex
@article{chen2026cotj,
  title   = {Chain-of-Trajectories: Unlocking the Intrinsic Generative Optimality of Diffusion Models via Graph-Theoretic Planning},
  author  = {Ping Chen and Xiang Liu and Xingpeng Zhang and Fei Shen and Xun Gong and Zhaoxiang Liu and Zezhou Chen and Huan Hu and Kai Wang and Shiguo Lian},
  journal = {arXiv preprint},
  year    = {2026},
  note    = {arXiv:XXXX.XXXXX [cs.CV]}
}

```

