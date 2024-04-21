from diffusers import StableDiffusionPipeline

import torch

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",  torch_dtype=torch.float16)
pipe.to("cuda")
prompt = "a photograph of an astronaut riding a horse"

image = pipe(prompt).images[0]

# you can save the image with
# image.save(f"astronaut_rides_horse.png")
