
# CoTj (Chain-of-Trajectories: Unlocking the Intrinsic Generative Optimality of Diffusion Models via Graph-Theoretic Planning)

## 🧭 Description

**CoTj (Chain-of-Trajectories)** is a **graph-theoretic trajectory planning framework** for diffusion models. 
It upgrades the standard, fixed-step denoising schedules (System 1) into **condition-adaptive, optimally planned trajectories (System 2)**, enabling flexible, high-fidelity image generation under varying prompts and constraints.

CoTj establishes an **offline graph for each condition**, searches for optimal denoising paths, and supports both **fixed-step optimal sequences** and **adaptive-length planning** to reduce sampling steps without sacrificing output quality.

**The latest full paper PDF (CoTj_v20260305.pdf) is included in this repository, and we recommend reading the repo version for the most up-to-date manuscript**. The paper is also available via the [**ChinaXiv**](https://chinaxiv.org/abs/202603.00028), with a reprint soon to appear on arXiv.


<img width="1453" height="1185" alt="image" src="https://github.com/user-attachments/assets/1457d497-72f5-4dd2-a771-4736ef5e4f48" />



---

## 💡 Core Highlights & Breakthroughs

* 🧠 **"System 2" Global Planning**: CoTj ends the "blind-box" generation of traditional diffusion models. By extracting a **Diffusion DNA** in just **0.073ms** to quantify generation difficulty, it transforms high-dimensional generation into a graph-theoretic shortest path problem. It takes shortcuts for simple scenes and meticulously refines complex ones, enabling truly deliberate, planned generation.

* ⚡ **Trajectory Reachability & Emergent Acceleration**: Fewer steps don’t imply lower quality. Following geometrically optimal paths ensures high-fidelity **latent endpoints** remain reachable. A 10-step CoTj reconstruction can surpass multi-step baselines. This precise trajectory optimization naturally produces emergent inference acceleration and seamlessly integrates with **cache-adaptive acceleration**, reusing computation in high-information-density regions.
  
* 🛣️ **Trajectory Routing > Solvers**: Choosing the right path matters more than stacking high-order solvers. Even under low computational budgets, CoTj demonstrates superior image quality and proves that **optimal trajectory planning outweighs solver complexity**.

* 🎬 **Robust Video Generation**: Validated on **Wan2.2**, CoTj reveals the **Generative Hierarchy** principle: stabilize structure first, then animate. By prioritizing fidelity, it eliminates frame collapse and "pseudo-motion" seen in low-step baselines, producing smooth and coherent motion dynamics.

* 🩺 **Model "X-Ray" Diagnostics**: Diffusion DNA also functions as a structural diagnostic tool, transparently revealing hidden issues like **over-cooking** and non-convergence in the late stages of certain distilled models.

<details>
<summary><b>📢 Highlights </b></summary>
<br>

🚀 **Diffusion models officially enter the "System 2" global planning era!**

The newly open-sourced, train-free **CoTj** framework from China Unicom AI Institute enables diffusion models to leave behind "blind-box" generation and gain human-like global planning capability. By extracting **Diffusion DNA** in just **0.073ms** to quantify generation difficulty, high-dimensional generation is transformed into a graph-theoretic shortest path problem. Simple prompts take shortcuts, while complex descriptions are refined meticulously — achieving truly deliberate, planned generation.

1️⃣ **Trajectory reachability & emergent acceleration**: Fewer steps don’t mean lower quality. By following geometrically optimal paths, high-fidelity latent endpoints remain fully reachable. A 10-step CoTj reconstruction can surpass a baseline with dozens of steps. This precise trajectory optimization directly produces **emergent inference acceleration**, eliminating redundant computation. It also naturally supports **cache-adaptive acceleration**, targeting high-information-density regions for computation reuse.

2️⃣ **Right path, exponential effect**: Even at low computational budgets, image quality is dramatically improved. Data proves that **finding the right trajectory outweighs merely stacking high-order solvers**.

3️⃣ **Robust video generation**: Tested on **Wan2.2**, CoTj reveals the **Generative Hierarchy** principle: stabilize structure first, then animate. This approach eliminates frame collapse and "pseudo-motion" seen in low-step baselines, prioritizing fidelity to produce smooth dynamic content.

4️⃣ **Model "X-Ray" diagnostics**: Diffusion DNA can also serve as a structural diagnostic tool, exposing issues in certain distilled models such as **over-cooking** and late-stage non-convergence.

</details>

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
  title        = {Chain-of-Trajectories: Unlocking the Intrinsic Generative Optimality of Diffusion Models via Graph-Theoretic Planning},
  author       = {Ping Chen and Xiang Liu and Xingpeng Zhang and Fei Shen and Xun Gong and Zhaoxiang Liu and Zezhou Chen and Huan Hu and Kai Wang and Shiguo Lian},
  year         = {2026},
  publisher    = {ChinaXiv},
  doi          = {10.12074/202603.00028},
  url          = {https://chinaxiv.org/abs/202603.00028},
  note         = {ChinaXiv preprint}
}

```

