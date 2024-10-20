import argparse
from glob import glob
import os
import cv2
import numpy as np
import torch as th
from psnr_ssim import calculate_psnr, calculate_ssim

from guided_diffusion import logger
from guided_diffusion.script_util import (
                                          add_dict_to_argparser, args_to_dict,
                                          create_model_and_diffusion,
                                          model_and_diffusion_defaults,
                                          )
def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    device = th.device('cuda')
    
    args = create_argparser().parse_args()
    out_dir = f'{args.out_dir}-seed{args.seed}'
    logger.configure(dir=out_dir)
    os.makedirs(out_dir, exist_ok=True)

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        th.load(args.model_path, map_location="cpu")
    )
    model.to(device)
    model.eval()

    seed = args.seed
    th.manual_seed(seed)
    np.random.seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)

    # logger.log("Sampling...")
    if args.testset == 'CC':
        path = './datasets/CC/'
        noises = sorted(glob(path + '*real.png'))
        cleans = sorted(glob(path + '*mean.png'))

    elif args.testset == 'PolyU':
        path = './datasets/PolyU/'
        noises = sorted(glob(path + '*real.JPG'))
        cleans = sorted(glob(path + '*mean.JPG'))
        
    elif args.testset == 'FMDD':    
        path = './datasets/FMDD/'
        noises = sorted(glob(path + 'raw' + '/*.png'))
        cleans = sorted(glob(path + 'gt' + '/*.png'))
        
    elif args.testset == 'SIDD_val':    
        path = './datasets/SIDD_val/'
        noises = sorted(glob(path + 'noisy' + '/*.png'))
        cleans = sorted(glob(path + 'GT' + '/*.png'))

    for lq_path, hq_path in zip(noises, cleans):

        model_kwargs = {}
        if args.testset == 'FMDD':
            y0 = cv2.imread(lq_path, flags=cv2.IMREAD_GRAYSCALE).astype(np.float32) / 127.5 - 1 #[-1, 1]
            y0 = th.from_numpy(y0).unsqueeze(0).unsqueeze(0).cuda()

            clean = cv2.imread(hq_path, flags=cv2.IMREAD_GRAYSCALE).astype(np.uint8)
            clean = th.from_numpy(clean).unsqueeze(0).unsqueeze(0).cuda()
        else:
            y0 = cv2.imread(lq_path).astype(np.float32)[:, :, [2, 1, 0]] / 127.5 - 1 #[-1, 1]
            y0 = th.from_numpy(y0).permute(2,0,1).unsqueeze(0).cuda()

            clean = cv2.imread(hq_path).astype(np.uint8)[:, :, [2, 1, 0]]
            clean = th.from_numpy(clean).permute(2,0,1).unsqueeze(0).cuda()
        
        sample_fn = (
            diffusion.p_sample_loop_vb if not args.use_ddim else None 
        )
        sample, _, _ = sample_fn(
            model,
            y0.shape,
            y0,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=None,
            device=device,
            progress=True,
            b0=args.b0,
            T=args.T,
            scale=args.scale,
            TS=1000
        )

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        if args.testset != 'FMDD':
            psnr = calculate_psnr(sample, clean, crop_border=0)
            ssim = calculate_ssim(sample, clean, crop_border=0, ssim3d=False)
        else:
            # print(sample.dtype)
            psnr = calculate_psnr(sample.float().mean(dim=1, keepdim=True), clean, crop_border=0)
            ssim = calculate_ssim(sample.float().mean(dim=1, keepdim=True), clean, crop_border=0, ssim3d=False)

        sample = sample.permute(0, 2, 3, 1).contiguous()
        img_name = lq_path.split('/')[-1]
        logger.log('{}, psnr:{}, ssim:{}'.format(img_name, psnr, ssim))

        # cv2.imwrite(f'{out_dir}/{img_name}', sample.cpu().numpy()[-1][...,[2,1,0]])

    logger.log("Sampling complete!")

def create_argparser():
    defaults = dict(
        seed=1314,
        testset='CC',
        out_dir='./result/CC_T5b001_s1.0',
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path=256x256_diffusion_uncond.pt",
        b0=0.01,
        T=1/5,
        scale=1.0
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
