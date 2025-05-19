# Image-Layout Generation

## 1. [POSTA: A Go-to Framework for Customized Artistic Poster Generation](https://arxiv.org/abs/2503.14908)
- **Based-on**: DM+MLLM (FLUX+LLAVA)
- **Training**: Training-free  
- **Star Count**: Closed Source  
- **Release Date**: 2025.03
- **Tech**: a three-stage pipeline that uses diffusion models and a multimodal LLM to generate artistic posters by combining background synthesis, layout planning, and stylized text generation.

## 2. [BannerAgency: Advertising Banner Design with Multimodal LLM Agents](https://www.arxiv.org/abs/2503.11060)
- **Based-on**: T2I+MLLM (SVG+Claude3.5-Sonnect)
- **Training**: Training-free  
- **Star Count**: Closed Source  
- **Release Date**: 2025.03
- **Tech**: a modular banner generation system that uses a strategist, background and foreground designers, and a developer to create editable, context-aware banners based on user requests and external knowledge.

## 3. [Prompt2Poster: Automatically Artistic Chinese Poster Creation from Prompt Only](https://openreview.net/forum?id=BQbfGk3JPY&referrer=%5Bthe%20profile%20of%20Li%20Yuan%5D(%2Fprofile%3Fid%3D~Li_Yuan2))
- **Based-on**: T2I+LLM (diffusion+GPT)
- **Training**: Training (layout generator+text generator)  
- **Star Count**: Closed Source  
- **Release Date**: 2024.10
- **Tech**: a system that generates artistic Chinese posters from natural language prompts by combining llm and T2I model, controllable layout planning, and stylized graphical text rendering to ensure semantic alignment and visual harmony.

## 4. [Desigen: A Pipeline for Controllable Design Template Generation](https://openaccess.thecvf.com/content/CVPR2024/papers/Weng_Desigen_A_Pipeline_for_Controllable_Design_Template_Generation_CVPR_2024_paper.pdf)
- **Based-on**: SD
- **Training**: Training (layout generator+background generator) 
- **Star Count**: [⭐69](https://github.com/whaohan/desigen)
- **Release Date**: 2024.06
- **Tech**: generates design templates by first synthesizing a background from a text description, applying a layout mask to preserve key regions, and then generating a layout based on specified element types and quantities.

## 5. [OpenCOLE: Towards Reproducible Automatic Graphic Design Generation](https://arxiv.org/abs/2406.08232)
- **Based-on**: LLM+SD (GPT-3.5+SDXL+Llava1.5)
- **Training**: Training
- **Star Count**: [⭐68](https://github.com/CyberAgentAILab/OpenCOLE)  
- **Release Date**: 2024.06
- **Tech**: a design generation framework that transforms user intentions into a structured design plan using GPT-3.5 and in-context learning, then synthesizes visual and typographic elements via diffusion and vision-language models to render the final graphic.

## 6. [Mastering Text-to-Image Diffusion:  Recaptioning, Planning, and Generating with Multimodal LLMs](https://github.com/YangLing0818/RPG-DiffusionMaster)
- **Based-on**: MLLM+SD (GPT-4+SDXL)
- **Training**: Training-free
- **Star Count**: [⭐1.8k](https://github.com/CyberAgentAILab/OpenCOLE)  
- **Release Date**: 2024.06
- **Tech**: a training-free text-to-image generation and editing framework that enhances compositionality by using a multimodal LLM for prompt decomposition and planning, and introduces complementary regional diffusion to guide region-wise image synthesis based on subprompts.


# Layout Generation
## 1. [Smaller But Better: Unifying Layout Generation with Smaller Large Language Models](https://arxiv.org/abs/2502.14005)
- **Based-on**: LLM (customized)
- **Training**: Training  
- **Star Count**: ⭐21 
- **Release Date**: 2025.02
- **Tech**: integrates Arbitrary Layout Instructions (ALI), Interval Quantization Encoding (IQE), and Universal Layout Response (ULR) into a unified LLM framework to enable precise and compact layout generation from diverse layout prompts.

## 2. [From Elements to Design: A Layered Approach for Automatic Graphic Design Composition](https://arxiv.org/abs/2412.19712)
- **Based-on**: LLM (Llama-3.1-8B)
- **Training**: Training  
- **Star Count**: Closed Source  
- **Release Date**: 2024.12
- **Tech**:  enables layered design composition by annotating input elements with GPT-4o, fine-tuning LMMs for hierarchical generation, and iteratively feeding back rendered intermediate designs to guide subsequent layer synthesis.

## 3. [TextLap: Customizing Language Models for Text-to-Layout Planning](https://arxiv.org/abs/2410.12844)
- **Based-on**: LLM (Vicuna-1.5-7B) 
- **Training**: Training  
- **Star Count**: ⭐15  
- **Release Date**: 2024.10
- **Tech**: generates graphic designs and outputs element coordinates based on text prompts and element descriptions, leveraging fine-tuning on the InstLap dataset.

## 4. [Layout-Corrector: Alleviating Layout Sticking Phenomenon in Discrete Diffusion Model](https://arxiv.org/abs/2409.16689)
- **Based-on**: DDM (LayoutDM, discrete diffusion models) 
- **Training**: Training  
- **Star Count**: ⭐6  
- **Release Date**: 2024.09
- **Tech**:  iteratively refines layout generation by combining a fixed pretrained diffusion model (DDM) with a correction module that adjusts outputs during the sampling process.

## 5. [PosterLlama: Bridging Design Ability of Language Model to Content-Aware Layout Generation](https://arxiv.org/abs/2404.00995)
- **Based-on**: LLM (CodeLlaMA)
- **Training**: Training  
- **Star Count**: ⭐37  
- **Release Date**: 2024.07
- **Tech**: trained in two stages: first, it learns visual alignment through adapter training, and then it fine-tunes the model to generate HTML-based poster layouts using the frozen adapter.

## 6. [Visual Layout Composer: Image-Vector Dual Diffusion Model for Design Layout Generation](https://openaccess.thecvf.com/content/CVPR2024/html/Shabani_Visual_Layout_Composer_Image-Vector_Dual_Diffusion_Model_for_Design_Layout_CVPR_2024_paper.html)
- **Based-on**: DM+VDM (SD1.5)
- **Training**: Training  
- **Star Count**: Closed Source  
- **Release Date**: 2024.06
- **Tech**: employs a dual diffusion process to generate coherent image layouts and bounding boxes, integrating image and attribute encoders, attention modulation, and a U-Net-based decoder pipeline.

## 7. [Retrieval-Augmented Layout Transformer  for Content-Aware Layout Generation](https://arxiv.org/abs/2311.13602)
- **Based-on**: Transformer (customized)
- **Training**: Training  
- **Star Count**: ⭐112 
- **Release Date**: 2024.06
- **Tech**: uses retrieval-augmented examples and optional user constraints to generate image-aligned, controllable layouts from a canvas and saliency map.

## 8. [Graphic Design with Large Multimodal Model](https://arxiv.org/abs/2404.14368)
- **Based-on**: LLM (Qwen1.5)
- **Training**: Training  
- **Star Count**: ⭐115 
- **Release Date**: 2024.02
- **Tech**: encodes design elements using an RGB-A encoder and visual shrinker, then leverages a large language model to generate graphic compositions in JSON format.

## 9. [Spot the Error: Non-autoregressive Graphic Layout Generation with Wireframe Locator](https://arxiv.org/abs/2401.16375)
- **Based-on**: Network (customized)
- **Training**: Training  
- **Star Count**: Closed Source 
- **Release Date**: 2024.01
- **Tech**: iteratively refines attribute tokens by using a locator to detect errors and a non-autoregressive decoder to predict the corrected tokens.

## 10. [LayoutPrompter: Awaken the Design Ability of Large Language Models](https://arxiv.org/abs/2311.06495)
- **Based-on**: LLM (GPT-3)
- **Training**: Training-free  
- **Star Count**: ⭐182
- **Release Date**: 2023.11
- **Tech**: uses in-context learning with dynamically retrieved exemplars and a layout ranker to generate and select the best content-aware poster layout.

## 11. [LayoutGPT: Compositional Visual Planning and Generation with Large Language Models](https://arxiv.org/abs/2305.15393)
- **Based-on**: LLM (GPT-3)
- **Training**: Training-free  
- **Star Count**: ⭐342
- **Release Date**: 2023.10
- **Tech**: leverages in-context learning with exemplars to perform 2D or 3D layout planning from text prompts


## 12. [AutoPoster: A Highly Automatic and Content-aware Design System for Advertising Poster Generation](https://arxiv.org/abs/2308.01095)
- **Based-on**: Transformer (customized)
- **Training**: Training  
- **Star Count**: Closed Source 
- **Release Date**: 2023.08
- **Tech**: a four-stage system that automatically generates product posters by processing images and titles through image retargeting, layout generation, tagline creation, and style prediction.

## 13. [LayoutDM: Discrete Diffusion Model for Controllable Layout Generation](https://arxiv.org/abs/2303.08137)
- **Based-on**: DDM (VQDiffusion)
- **Training**: Training  
- **Star Count**: ⭐265 
- **Release Date**: 2023.03
- **Tech**: diffusion-based model that generates layouts from scratch in discrete space and supports flexible conditional generation without extra training.


# Image Generation

## 1. [VerbDiff: Text-Only Diffusion Models with Enhanced Interaction Awareness](https://arxiv.org/abs/2503.16406)
- **Based-on**: DM (SD v1.4)
- **Training**: Training
- **Star Count**: Closed Source  
- **Release Date**: 2025.03
- **Tech**: a human-object interaction-aware text-to-image generation framework that disentangles relational features using Relation Disentanglement Guidance and localizes interaction regions via an Interaction Region module to enhance compositional generation.

## 2. [MS-DIFFUSION: MULTI-SUBJECT ZERO-SHOT IMAGE  PERSONALIZATION WITH LAYOUT GUIDANCE](https://arxiv.org/abs/2406.07209)
- **Based-on**: SDXL
- **Training**: Training  
- **Star Count**: [⭐267](https://github.com/MS-Diffusion/MS-Diffusion)
- **Release Date**: 2025.03
- **Tech**: enhances diffusion models with a grounding resampler that aligns visual inputs with entities and spatial constraints, and a multi-subject cross-attention mechanism that enables precise interaction between image conditions and latent variables.

## 3. [DesignDiffusion: High-Quality Text-to-Design Image Generation  with Diffusion Models](https://arxiv.org/abs/2503.01645)
- **Based-on**: SDXL
- **Training**: Training  
- **Star Count**: Closed Source
- **Release Date**: 2025.03
- **Tech**: enhances text-to-image generation by fine-tuning a trainable CLIP text encoder and UNet with prompt augmentation and character localization loss, and further improves visual text quality using a self-refining DPO strategy.

## 4. [ART: Anonymous Region Transformer for  Variable Multi-Layer Transparent Image Generation](https://arxiv.org/abs/2502.18364)
- **Based-on**: FLUX.1[dev]
- **Training**: Training  
- **Star Count**: [⭐296](https://github.com/microsoft/art-msra)
- **Release Date**: 2025.02
- **Tech**: combines a Multi-Layer Transparent Image Autoencoder that encodes and decodes layered RGBA images into latent space, with an Anonymous Region Transformer (ART) that denoises multi-layer latent variables corresponding to a variable number of transparent image layers.

## 5. [ITERCOMP: ITERATIVE COMPOSITION-AWARE  FEEDBACK LEARNING FROM MODEL GALLERY FOR  TEXT-TO-IMAGE GENERATION](https://arxiv.org/abs/2410.07171)
- **Based-on**: T2I models
- **Training**: Training  
- **Star Count**: [⭐184](https://github.com/YangLing0818/IterComp)
- **Release Date**: 2025.02
- **Tech**: leverages composition-aware preferences from multiple models and employs iterative feedback learning to progressively optimize both the base diffusion model and the reward model.

## 6. [Transparent Image Layer Diffusion using Latent Transparency](https://arxiv.org/abs/2402.17113)
- **Based-on**: SDXL
- **Training**: Training  
- **Star Count**: [⭐2.1k](https://github.com/lllyasviel/LayerDiffuse)
- **Release Date**: 2024.06
- **Tech**: encodes transparency information into the latent space of Stable Diffusion by adjusting it with a trainable model, enabling accurate reconstruction of both color and alpha channels from transparent images.

## 7. [LayerDiff: Exploring Text-guided Multi-layered  Composable Image Synthesis via  Layer-Collaborative Diffusion Model](https://arxiv.org/abs/2403.11929)
- **Based-on**: SD1.5
- **Training**: Training  
- **Star Count**: Closed Source
- **Release Date**: 2024.03
- **Tech**: enables multi-layered composable image synthesis by jointly generating layer images and masks using a collaborative diffusion model guided by both global and layer-specific prompts, enhanced by a prompt enhancer and cross-layer attention.


## 8. [InstanceDiffusion: Instance-level Control for Image Generation](https://arxiv.org/abs/2402.03290)
- **Based-on**: SD
- **Training**: Training  
- **Star Count**: [⭐568](https://github.com/frank-xwang/InstanceDiffusion)
- **Release Date**: 2024.02
- **Tech**: enhances text-to-image generation by enabling instance-level control through paired prompts and spatial inputs—such as points, boxes, scribbles, or masks—alongside global text prompts, allowing for flexible and precise composition.


## 9. [GLIGEN: Open-Set Grounded Text-to-Image Generation](https://arxiv.org/abs/2301.07093)
- **Based-on**: SD
- **Training**: Training  
- **Star Count**: [⭐2.1k](https://github.com/gligen/GLIGEN)
- **Release Date**: 2024.02
- **Tech**: enhances a pretrained text-to-image model by introducing a gated self-attention layer that incorporates new conditional information alongside standard cross-attention with text features.