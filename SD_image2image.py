# import requests
# import torch
from PIL import Image
from io import BytesIO

# from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# def read_image(file_path):
#     with open(file_path, "rb") as file:
#         image_bytes = file.read()
#     return image_bytes

# def runModel(input_image, prompt):
#     device = "cpu"
#     if torch.cuda.is_available():
#         device = "cuda"

#     model_id_or_path = "stabilityai/stable-diffusion-2-1"
#     pipe = StableDiffusionPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float32)
#     pipe = pipe.to(device)

#     #url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
#     init_image = Image.open(input_image).convert("RGB")
#     #init_image = init_image.resize((768, 512))

#     # prompt = "A fantasy landscape, trending on artstation"

#     pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
#     pipe = pipe.to(device)
#     image = pipe(
#         prompt,
#         image=init_image,
#         negative_prompt="",
#         num_inference_steps=4,
#         guidance_scale=9.0,
#     ).images[0]
#     ImagePath = "image2img.png"
#     image.save(ImagePath)
#     return ImagePath
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid

def runModel(input_image, prompt):
    pipeline = AutoPipelineForImage2Image.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder",
        torch_dtype=torch.float32,
        use_safetensors=True,
    )
    pipeline.enable_sequential_cpu_offload()
    # pipeline.max_memory({})
    # remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
    #pipeline.enable_xformers_memory_efficient_attention()

    # Load an image to pass to the pipeline:

    # Pass a prompt and image to the pipeline to generate an image:
    init_image = Image.open(input_image).convert("RGB")
    image = pipeline(prompt, image=init_image, strength = 0.8).images[0]
    
    ImagePath = "image2img.png"
    image.save(ImagePath)
    return ImagePath