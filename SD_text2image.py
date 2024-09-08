import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


def runModel(prompt):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float32)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    image = pipe(
        prompt,
        negative_prompt="",
        num_inference_steps=2,
        guidance_scale=7.0,
    ).images[0]
    ImagePath = "text2img.png"
    image.save(ImagePath)
    return ImagePath
