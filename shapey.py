import torch
import os
import shutil
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
#from shap_e.util.notebooks import create_pan_cameras, decode_latent_images
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
device = torch.device(device)



batch_size = 1 # this is the size of the models, higher values take longer to generate.
guidance_scale = 15.0 # this is the scale of the guidance, higher values make the model look more like the prompt.



def getShape(inputString):
    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    prompt = inputString
    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1E-3,
        sigma_max=160,
        s_churn=0,
    )
    
    from shap_e.util.notebooks import decode_latent_mesh

    t = decode_latent_mesh(xm, latents[0]).tri_mesh()
    outputFile = '3dOutput.obj'
    with open(outputFile, 'w') as f:
        t.write_obj(f)
    return outputFile
    # render_mode = 'nerf' # you can change this to 'stf'
    # size = 64 # this is the size of the renders, higher values take longer to render.

    # cameras = create_pan_cameras(size, device)
    # for i, latent in enumerate(latents):
    #     images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)

    
    # modelDir = './shap_e_3d_models'
    # if not os.path.exists(modelDir):
    #     os.makedirs(modelDir)

    # for i, latent in enumerate(latents):
    #     t = decode_latent_mesh(xm, latent).tri_mesh()
    #     with open(f'shap_e_3d_models/example_mesh_{i}.obj', 'w') as f: # we will use this file to customize in Blender Studio later.
    #         t.write_obj(f)
    #archived = shutil.make_archive('./modelszip', 'zip', modelDir)
    # return modelDir
