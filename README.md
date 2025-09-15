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
    <strong>English</strong> | <a href="./README_CN.md"><strong>中文简体</strong></a>
  </p>

Thanks to ComfyUI's quantization optimizations for Flux, this model can be run directly on consumer-grade GPUs like the 4090.

## How to Use

1. You need to manually download the Flux-related model weights. Following the official Flux workflow, place the models in the `ComfyUI/models` directory. The required weights are listed below. Only the LoRA weights are an additional download. You can also download the full parameter model [textflux](https://huggingface.co/yyyyyxie/textflux-beta) to replace `flux1-fill-dev` and the LoRA for an even better text editing experience.

   - **Diffusion Model**: diffusion_models/[flux1-fill-dev.safetensors](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev/blob/main/flux1-fill-dev.safetensors)
   - **Text Encoder**: clip/[t5xxl_fp16.safetensors](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp16.safetensors)
   - **VAE**: ae/[ae.safetensors](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev/blob/main/ae.safetensors)
   - **LoRA**: lora/[pytorch_lora_weights.safetensors](https://huggingface.co/yyyyyxie/textflux-lora-beta/blob/main/pytorch_lora_weights.safetensors)

   

2. Clone this project. You can directly use the workflow from `example_workflows/textflux_lora_inpaint_example.json`.

   Bash

   ```
   # Navigate to your ComfyUI custom nodes directory
   cd ComfyUI/custom_nodes
   
   # Clone the repository
   git clone https://github.com/yyyyyxie/textflux_comfyui.git
   
   # Optional: Copy the example workflow to your ComfyUI user workflows
   # Note: Adjust the path if your user workflow directory is different
   cp example_workflows/textflux_lora_inpaint_example.json ComfyUI/user/default/workflows
   ```

3) Start ComfyUI !