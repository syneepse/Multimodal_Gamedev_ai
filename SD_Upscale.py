import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch
# import streamlit as st
from diffusers.models.attention_processor import AttnProcessor2_0
import torch._dynamo
import ImageProcess as ipr
from glob import glob
import os

torch._dynamo.config.suppress_errors = True






def generate(uploaded_file, prompt):
    
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float32)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    pipeline = pipeline.to(device)

    resizedImg = ipr.reSizeImg(uploaded_file)
    chunkDir,Width_size, Height_size = ipr.splitImages(resizedImg)

    chunkArray = glob(chunkDir + '*.png')
    upscaledChunkDirectory = './UpscaledChunks'
    if not os.path.exists(upscaledChunkDirectory):
        os.makedirs(upscaledChunkDirectory)
    for i in range(len(chunkArray)):
        print(i)
        low_res_img = Image.open(chunkArray[i],mode='r').convert("RGB")
        print(low_res_img.size)
        upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
        upscaled_image.save(upscaledChunkDirectory + '/' + str(i)+".png")

    UpscaledImageFinal = ipr.reconstruct(upscaledChunkDirectory + '/' , Width_size, Height_size, 128*4)
    return UpscaledImageFinal
    # upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]

    #st.image(upscaled_image)
    #upscaled_image.save("upsampled_image.png")
# load model and scheduler


#pipeline.enable_sequential_cpu_offload()

#pipe.enable_attention_slicing("max")

# let's upload an  image
#uplaoded_file = st.file_uploader("choose an image", type=['png','jpg','bmp'], accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
#if uplaoded_file is not None:





#st.image(low_res_img)
#st.text_area("prompt", value="", height=None, max_chars=None, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder="discrible the image", disabled=False, label_visibility="visible")
#if st.button("generate", key=None, help=None, on_click=None, args=None, kwargs=None, type="secondary", disabled=False, use_container_width=False):
# generate(low_res_img)

