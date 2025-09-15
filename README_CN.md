# TextFlux: An OCR-Free DiT Model for High-Fidelity Multilingual Scene Text Synthesis

<div style="display: flex; justify-content: center; align-items: center;">
  <a href="https://arxiv.org/abs/2505.17778">
    <img src='https://img.shields.io/badge/arXiv-2505.17778-red?style=flat&logo=arXiv&logoColor=red' alt='arxiv'>
  </a>
  <a href='https://huggingface.co/yyyyyxie/textflux'>
    <img src='https://img.shields.io/badge/Hugging Face-ckpts-orange?style=flat&logo=HuggingFace&logoColor=orange' alt='huggingface'>
  </a>
  <a href="https://github.com/yyyyyxie/textflux">
    <img src='https://img.shields.io/badge/GitHub-Repo-blue?style=flat&logo=GitHub' alt='GitHub'>
  </a>
  <a href="https://huggingface.co/yyyyyxie/textflux" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Demo-Gradio-gold?style=flat&logo=Gradio&logoColor=red' alt='Demo'>
  </a>
  <a href='https://yyyyyxie.github.io/textflux-site/'>
    <img src='https://img.shields.io/badge/Webpage-Project-silver?style=flat&logo=&logoColor=orange' alt='webpage'>
  </a>
  <a href="https://modelscope.cn/models/xieyu20001003/textflux">
  <img src="https://img.shields.io/badge/🤖_ModelScope-ckpts-ffbd45.svg" alt="ModelScope">
  </a>
</div>
  <p align="left">
    <strong>中文简体</strong> | <a href="./README.md"><strong>English</strong></a>
  </p>
得益于 ComfyUI 针对 Flux 的量化优化，可以直接在 4090 等消费级显卡上成功运行该模型。

## 使用方法

1. 下载模型权重

   您需要手动下载 Flux 相关的模型权重。与 Flux 官方工作流一致，请将模型放置在 ComfyUI/models 目录下。需要下载的权重内容如下，其中只有 LoRA 权重是额外需要下载的。您也可以下载全参数模型 [textflux](https://huggingface.co/yyyyyxie/textflux-beta) 来替换 flux1-fill-dev 和 LoRA，以体验效果更好的文本编辑。

   - **Diffusion Model**: diffusion_models/[flux1-fill-dev.safetensors](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev/blob/main/flux1-fill-dev.safetensors)
   - **Text Encoder**: clip/[t5xxl_fp16.safetensors](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp16.safetensors), clip/[clip_l.safetensors](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/clip_l.safetensors)
   - **VAE**: ae/[ae.safetensors](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev/blob/main/ae.safetensors)
   - **LoRA**: lora/[pytorch_lora_weights.safetensors](https://huggingface.co/yyyyyxie/textflux-lora-beta/blob/main/pytorch_lora_weights.safetensors)

2. 安装并使用工作流

   克隆此项目到您的 ComfyUI 自定义节点目录中。您可以直接加载并使用 example_workflows/textflux_lora_inpaint_example.json 工作流。

   Bash

   ```
   # 进入 ComfyUI 自定义节点目录
   cd ComfyUI/custom_nodes
   
   # 克隆仓库
   git clone https://github.com/yyyyyxie/textflux_comfyui.git
   
   # (可选) 复制示例工作流到你的 ComfyUI 用户工作流目录
   cp example_workflows/textflux_lora_inpaint_example.json ComfyUI/user/default/workflows
   ```

3) 直接启动ComfyUI
