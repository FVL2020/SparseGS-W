import os

import torch
import torch.nn as nn
from diffusers import DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from torch.cuda.amp import autocast
from torchvision import transforms
from tqdm import tqdm
from transformers import logging

from pnp.pnp_utils import (
    register_attention_control_efficient,
    register_conv_control_efficient,
    register_time,
)

logging.set_verbosity_error()

MODEL_KEYS = {
    "2.1": "stabilityai/stable-diffusion-2-1-base",
    "2.0": "stabilityai/stable-diffusion-2-base",
    "1.5": "runwayml/stable-diffusion-v1-5",
    "xl_turbo": "stabilityai/sdxl-turbo",
}


class StableDiffusion(nn.Module):
    def __init__(
        self,
        device,
        args,
        fp16=True,
        vram_O=False,
        sd_version="1.5",
        hf_key=None,
    ):
        super().__init__()

        self.device = device
        self.sd_version = sd_version
        self.config = args

        if hf_key is not None:
            print(f"[INFO] using hugging face custom model key: {hf_key}")
            model_key = hf_key
        else:
            model_key = MODEL_KEYS.get(self.sd_version)
            if model_key is None:
                raise ValueError(f"Stable-diffusion version {self.sd_version} not supported.")

        self.dtype = torch.float16 if fp16 else torch.float32

        if self.config.use_finetune_unet:
            print(f"Loading finetuned UNet from {self.config.use_finetune_unet}!")
            unet_dreambooth = UNet2DConditionModel.from_pretrained(
                self.config.use_finetune_unet,
                subfolder="unet",
                revision="fp16",
                torch_dtype=self.dtype,
            ).to(self.device)
            pipe = StableDiffusionPipeline.from_pretrained(
                model_key, unet=unet_dreambooth, torch_dtype=self.dtype
            ).to(self.device)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.dtype).to(self.device)

        pipe.enable_xformers_memory_efficient_attention()

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
        else:
            pipe.to(device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.unet_sd15 = UNet2DConditionModel.from_pretrained(
            model_key, subfolder="unet", revision="fp16", torch_dtype=torch.float16
        ).to(self.device)

        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype
        )

        del pipe

        self.text_embeds = self.get_text_embeds(self.config.prompt, self.config.negative_prompt)
        self.pnp_guidance_embeds = self.get_text_embeds("", "").chunk(2)[0]

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, batch_size=1):
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        uncond_input = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        return torch.cat([uncond_embeddings] * batch_size + [text_embeddings] * batch_size)

    @torch.no_grad()
    def denoise_step(self, x, t, noise_latents):
        source_latents = noise_latents[t.item()].to(self.device)
        latent_model_input = torch.cat([source_latents] + ([x] * 2))

        register_time(self, t.item())

        text_embed_input = torch.cat([self.pnp_guidance_embeds, self.text_embeds], dim=0)
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input)["sample"]

        _, noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
        noise_pred = noise_pred_uncond + self.config.guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        return self.scheduler.step(noise_pred, t, x)["prev_sample"]

    def init_pnp(self, conv_injection_t, qk_injection_t):
        self.qk_injection_timesteps = (
            self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        )
        self.conv_injection_timesteps = (
            self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        )
        register_attention_control_efficient(self, self.qk_injection_timesteps)
        register_conv_control_efficient(self, self.conv_injection_timesteps)

    def run_pnp(self, noise_latents):
        self.scheduler.set_timesteps(self.config.num_inference_steps, device=self.device)
        pnp_f_t = int(self.config.num_inference_steps * self.config.pnp_f_t)
        pnp_attn_t = int(self.config.num_inference_steps * self.config.pnp_attn_t)
        self.init_pnp(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t)
        return self.sample_loop(noise_latents)

    def sample_loop(self, noise_latents):
        x = noise_latents[self.scheduler.timesteps[0].item()]
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            for _, t in enumerate(tqdm(self.scheduler.timesteps, desc="Sampling")):
                x = self.denoise_step(x, t, noise_latents)
            decoded_latent = self.decode_latents(x)

        return decoded_latent, x

    @torch.no_grad()
    def ddim_inversion2(self, cond, latent, num_inference_steps=1000, save_latents=True,
                        save_latent_path=""):
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = reversed(self.scheduler.timesteps)
        all_latents = {}
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            for i, t in enumerate(tqdm(timesteps, desc="DDIM Inversion")):
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

                eps = self.unet_sd15(latent, t, encoder_hidden_states=cond_batch).sample
                pred_x0 = (latent - sigma_prev * eps) / mu_prev
                latent = mu * pred_x0 + sigma * eps
                if save_latents:
                    all_latents[t.item()] = latent.clone()
        if save_latent_path != "":
            torch.save(all_latents, save_latent_path)
        return all_latents

    def train_step(
        self,
        pred_rgb,
        num_inference_steps=50,
        novelview_idx=None,
        noisy_latent_folder="",
    ):
        pred_rgb = pred_rgb.to(self.dtype)

        if pred_rgb.shape[2] == 512 and pred_rgb.shape[3] == 688:
            pred_rgb_512 = pred_rgb
        else:
            pred_rgb_512 = transforms.Resize((512, 688), antialias=True)(pred_rgb)
        latents = self.encode_imgs(pred_rgb_512)

        if noisy_latent_folder != "":
            os.makedirs(noisy_latent_folder, exist_ok=True)
            noisy_latent_path = os.path.join(
                noisy_latent_folder, f"noisy_latents_{novelview_idx}.pt"
            )
        else:
            noisy_latent_path = ""

        cond = self.get_text_embeds(self.config.inversion_prompt, "")[1].unsqueeze(0)
        ddim_latents = self.ddim_inversion2(
            cond=cond,
            latent=latents.clone(),
            num_inference_steps=num_inference_steps,
            save_latent_path=noisy_latent_path,
        )

        with torch.no_grad():
            with autocast(device_type="cuda", dtype=torch.float32):
                edited_img, _ = self.run_pnp(ddim_latents)

        return edited_img

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        imgs = self.vae.decode(latents).sample
        return (imgs / 2 + 0.5).clamp(0, 1)

    def encode_imgs(self, imgs):
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        return posterior.sample() * self.vae.config.scaling_factor
