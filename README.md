## NeurIPS2024 Spotlight: Diffusion Priors for Variational Likelihood Estimation and Image Denoising

> **Abstract** Real-world noise removal is crucial in low-level computer vision. Due to the remarkable generation capabilities of diffusion models, recent attention has shifted towards leveraging diffusion priors for image restoration tasks. However, existing diffusion priors-based methods either consider simple noise types or rely on approximate posterior estimation, limiting their effectiveness in addressing structured and signal-dependent noise commonly found in real-world images. In this paper, we build upon diffusion priors and propose adaptive likelihood estimation and MAP inference during the reverse diffusion process to tackle real-world noise. We introduce an independent, non-identically distributed likelihood combined with the noise precision (inverse variance) prior and dynamically infer the precision posterior using variational Bayes during the generation process. Meanwhile, we rectify the estimated noise variance through local Gaussian convolution. The final denoised image is obtained by propagating intermediate MAP solutions that balance the updated likelihood and diffusion prior. Additionally, we explore the local diffusion prior inherent in low-resolution diffusion models, enabling direct handling of high-resolution noisy images. Extensive experiments and analyses on diverse real-world datasets demonstrate the effectiveness of our method.

Arxiv link: https://arxiv.org/abs/2410.17521

### Inference

To conduct real-world denoising, first prepare real-world noisy datasets [PolyU, CC and SIDD](https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset) and  [FMDD](https://github.com/yinhaoz/denoising-fluorescence). Then download pre-trained 256x256 unconditional diffusion model: [256x256_diffusion_uncond.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt). Finally, run the following scripts, and the result will be stored in ./result folder:

```
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"

python inference_cond.py $MODEL_FLAGS
```

### Citation

Please consider citing this paper if it helps you:

```
@inproceedings{Jun2024Diffusion,
    title={Diffusion Priors for Variational Likelihood Estimation and Image Denoising}, 
    author={Jun Cheng and Shan Tan},
    booktitle={NeurIPS},
    year={2024}
}
```

**Acknowledgment**: This code is based on [guided-diffusion](https://github.com/openai/guided-diffusion)
