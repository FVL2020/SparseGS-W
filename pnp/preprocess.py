import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as T
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, logging

from pnp.pnp_utils import seed_everything

logging.set_verbosity_error()


def get_timesteps(scheduler, num_inference_steps, strength, device):
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    return scheduler.timesteps[t_start:], num_inference_steps - t_start


class Preprocess(nn.Module):
    def __init__(self, device, sd_version="2.0", hf_key=None):
        super().__init__()
        self.device = device
        self.use_depth = False

        print("[INFO] loading stable diffusion...")
        model_key = sd_version

        self.vae = AutoencoderKL.from_pretrained(
            model_key, subfolder="vae", revision="fp16", torch_dtype=torch.float16
        ).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_key, subfolder="text_encoder", revision="fp16", torch_dtype=torch.float16
        ).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            model_key, subfolder="unet", revision="fp16", torch_dtype=torch.float16
        ).to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        print("[INFO] loaded stable diffusion!")

        self.inversion_func = self.ddim_inversion

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, device="cuda"):
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
        uncond_input = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]
        return torch.cat([uncond_embeddings, text_embeddings])

    @torch.no_grad()
    def decode_latents(self, latents):
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            latents = 1 / 0.18215 * latents
            imgs = self.vae.decode(latents).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def load_img(self, image_path):
        image_pil = T.Resize(512)(Image.open(image_path).convert("RGB"))
        return T.ToTensor()(image_pil).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def encode_imgs(self, imgs):
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            imgs = 2 * imgs - 1
            posterior = self.vae.encode(imgs).latent_dist
            latents = posterior.mean * 0.18215
        return latents

    @torch.no_grad()
    def ddim_inversion(self, cond, latent, save_path, save_latents=True,
                       timesteps_to_save=None):
        timesteps = reversed(self.scheduler.timesteps)
        all_latents = {}
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            for i, t in enumerate(tqdm(timesteps)):
                cond_batch = cond.repeat(latent.shape[0], 1, 1)

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i - 1]]
                    if i > 0
                    else self.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = self.unet(latent, t, encoder_hidden_states=cond_batch).sample
                pred_x0 = (latent - sigma_prev * eps) / mu_prev
                latent = mu * pred_x0 + sigma * eps
                if save_latents:
                    all_latents[t.item()] = latent.clone()
        all_latents[t.item()] = latent.clone()
        torch.save(all_latents, os.path.join(save_path, "all_noisy_latents.pt"))
        return latent

    @torch.no_grad()
    def ddim_sample(self, x, cond, save_path, save_latents=False, timesteps_to_save=None):
        timesteps = self.scheduler.timesteps
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            for i, t in enumerate(tqdm(timesteps)):
                cond_batch = cond.repeat(x.shape[0], 1, 1)
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i + 1]]
                    if i < len(timesteps) - 1
                    else self.scheduler.final_alpha_cumprod
                )
                mu = alpha_prod_t ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = self.unet(x, t, encoder_hidden_states=cond_batch).sample
                pred_x0 = (x - sigma * eps) / mu
                x = mu_prev * pred_x0 + sigma_prev * eps

            if save_latents:
                torch.save(x, os.path.join(save_path, f"noisy_latents_{t}.pt"))
        return x

    @torch.no_grad()
    def extract_latents(self, num_steps, data_path, save_path, timesteps_to_save,
                        inversion_prompt="", extract_reverse=False):
        self.scheduler.set_timesteps(num_steps)

        cond = self.get_text_embeds(inversion_prompt, "")[1].unsqueeze(0)
        image = self.load_img(data_path)
        T.ToPILImage()(image[0]).save(os.path.join(save_path, "original.png"))
        latent = self.encode_imgs(image)

        inverted_x = self.inversion_func(
            cond,
            latent,
            save_path,
            save_latents=not extract_reverse,
            timesteps_to_save=timesteps_to_save,
        )
        latent_reconstruction = self.ddim_sample(
            inverted_x,
            cond,
            save_path,
            save_latents=extract_reverse,
            timesteps_to_save=timesteps_to_save,
        )
        return self.decode_latents(latent_reconstruction)


def run(opt):
    model_key = opt.sd_version
    toy_scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
    toy_scheduler.set_timesteps(opt.save_steps)
    timesteps_to_save, _ = get_timesteps(
        toy_scheduler, num_inference_steps=opt.save_steps, strength=1.0, device=opt.device
    )

    seed_everything(opt.seed)

    extraction_path_prefix = "_reverse" if opt.extract_reverse else "_forward"
    save_path = os.path.join(
        opt.save_dir + extraction_path_prefix,
        os.path.splitext(os.path.basename(opt.data_path))[0],
    )
    os.makedirs(save_path, exist_ok=True)

    model = Preprocess(opt.device, sd_version=model_key, hf_key=None)
    recon_image = model.extract_latents(
        data_path=opt.data_path,
        num_steps=opt.steps,
        save_path=save_path,
        timesteps_to_save=timesteps_to_save,
        inversion_prompt=opt.inversion_prompt,
        extract_reverse=opt.extract_reverse,
    )

    T.ToPILImage()(recon_image[0]).save(os.path.join(save_path, "recon.png"))
    print(os.path.join(save_path, "recon.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./latents/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sd_version", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=999)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--inversion_prompt", type=str, default="")
    parser.add_argument("--extract-reverse", default=False, action="store_true")
    opt = parser.parse_args()
    run(opt)
