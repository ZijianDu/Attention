import numpy as np
from PIL import Image
import torch
def create_noisy_images(self, configs):
    # read noiseless image
    n = configs.batch_size
    image = Image.read(configs.image_path)
    img = image.to(configs.device)
    img = image.unsqueeze(dim = 0)
    img = img.repeat(n, 1, 1, 1)
    x0 = img
    x0 = (x0 - 0.5)*2.
    for it in range(configs.sample_step):
        e = torch.randn_like(x0)
        total_noise_levels = configs.t
        a = (1-self.betas).cumprod(dim = 0)
        x = x0 * a[total_noise_levels-1].sqrt() + e*(1.0 - a[total_noise_levels-1]).sqrt()
        tvu.save_image((x+1.0)*0.5, configs.noised_images_folder, filename)
        # now we denoise the randomly noised images
        with tqdm(total = total_noise_levels, desc ="Iteration {}".format(it)) as progress_bar:
            for i in reversed(range(configs.total_noise_levels)):
                t = (torch.ones(n) * i).to(configs.device)























#imarray = np.ones(shape = (100,100,3)) * 125
#im = Image.fromarray(imarray.astype('uint8')).convert('RGBA')
im = Image.open("angel.png")
import requests
import torch
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline
model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float32) 
pipe = pipe.to("cuda")
url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((768, 512))

#prompt = "A fantasy landscape, trending on artstation"
prompt = 'an angel'

images = pipe(prompt=prompt, image=im, strength=0.75, guidance_scale=7.5).images
images[0].save("angel_again.png")
