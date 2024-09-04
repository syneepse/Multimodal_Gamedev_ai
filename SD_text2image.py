import torch
from diffusers import StableDiffusion3Pipeline


def runModel(prompt):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float32)
    pipe = pipe.to(device)
    image = pipe(
        prompt,
        negative_prompt="",
        num_inference_steps=28,
        guidance_scale=7.0,
    ).images[0]
    return image
