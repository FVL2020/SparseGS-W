import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as T
import yaml
from diffusers import DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from PIL import Image
from tqdm import tqdm
from transformers import logging

from pnp.pnp_utils import (
    register_attention_control_efficient,
    register_conv_control_efficient,
    register_time,
    seed_everything,
)

logging.set_verbosity_error()


class PNP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config["device"]

        model_key = config["sd_version"]
        print("Loading SD model")
        unet = UNet2DConditionModel.from_pretrained(
            model_key, subfolder="unet", revision="fp16", torch_dtype=torch.float16
        ).to(self.device)
        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, unet=unet, torch_dtype=torch.float16
        ).to(self.device)
        pipe.enable_xformers_memory_efficient_attention()

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.scheduler.set_timesteps(config["n_timesteps"], device=self.device)
        print("SD model loaded")

        self.image, self.eps = self.get_data()
        self.text_embeds = self.get_text_embeds(config["prompt"], config["negative_prompt"])
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
    def decode_latent(self, latent):
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            latent = 1 / 0.18215 * latent
            img = self.vae.decode(latent).sample
            img = (img / 2 + 0.5).clamp(0, 1)
        return img

    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def get_data(self):
        image = Image.open(self.config["image_path"]).convert("RGB")
        image = image.resize((512, 512), resample=Image.Resampling.LANCZOS)
        image = T.ToTensor()(image).to(self.device)
        latents_path = os.path.join(
            self.config["latents_path"],
            os.path.splitext(os.path.basename(self.config["image_path"]))[0],
            f"noisy_latents_{self.scheduler.timesteps[0]}.pt",
        )
        noisy_latent = torch.load(latents_path).to(self.device)
        return image, noisy_latent

    @torch.no_grad()
    def denoise_step(self, x, t):
        latents_path = os.path.join(
            self.config["latents_path"],
            os.path.splitext(os.path.basename(self.config["image_path"]))[0],
            "noisy_latents.pt",
        )
        assert os.path.exists(latents_path), f"Missing latents at path {latents_path}"
        all_latents = torch.load(latents_path)
        source_latents = all_latents[t.item()].to(self.device)

        latent_model_input = torch.cat([source_latents] + ([x] * 2))

        register_time(self, t.item())

        text_embed_input = torch.cat([self.pnp_guidance_embeds, self.text_embeds], dim=0)
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input)["sample"]

        _, noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
        noise_pred = noise_pred_uncond + self.config["guidance_scale"] * (
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

    def run_pnp(self):
        pnp_f_t = int(self.config["n_timesteps"] * self.config["pnp_f_t"])
        pnp_attn_t = int(self.config["n_timesteps"] * self.config["pnp_attn_t"])
        self.init_pnp(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t)
        self.sample_loop(self.eps)

    def sample_loop(self, x):
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            for _, t in enumerate(tqdm(self.scheduler.timesteps, desc="Sampling")):
                x = self.denoise_step(x, t)

            decoded_latent = self.decode_latent(x)
            T.ToPILImage()(decoded_latent[0]).save(
                os.path.join(self.config["output_path"], f"output-{self.config['prompt']}.png")
            )

        return decoded_latent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="pnp/config_pnp.yaml")
    opt = parser.parse_args()
    with open(opt.config_path, "r") as f:
        config = yaml.safe_load(f)
    os.makedirs(config["output_path"], exist_ok=True)
    with open(os.path.join(config["output_path"], "config.yaml"), "w") as f:
        yaml.dump(config, f)

    seed_everything(config["seed"])
    print(config)
    PNP(config).run_pnp()
