import gradio as gr

from io import BytesIO
import requests
import PIL
from PIL import Image
import numpy as np
import os
import uuid
import torch
from torch import autocast
import cv2
from matplotlib import pyplot as plt
from torchvision import transforms
from diffusers import DiffusionPipeline

device = "cuda" if torch.cuda.is_available() else "cpu" 

if device == "cuda":
    pipe = DiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", 
        torch_dtype=torch.float16
    ).to(device)
else:
    print("CUDA is not available.")

def predict(dict, prompt=""):
    init_image = dict["image"].convert("RGB").resize((512, 512))
    mask = dict["mask"].convert("RGB").resize((512, 512))
    output = pipe(prompt = prompt, image=init_image, mask_image=mask,guidance_scale=7.5)
    return output.images[0]

css = '''
    #button{max-width: 169px;max-height: 60px;border-radius: 0px 10px 10px 0px !important;}
    #image_upload{min-height:400px}
    #image_upload [data-testid="image"], #image_upload [data-testid="image"] > div{min-height: 400px}
    #mask_radio .gr-form{background:transparent; border: none}
    #image_upload .touch-none{display: flex}
'''

with gr.Blocks(css=css) as demo:
    with gr.Row(elem_id="prompt-container", equal_height=True):
        prompt = gr.Textbox(placeholder = 'Prompt', elem_id="input-text", show_label=False, lines=2, scale=4)
        btn = gr.Button("Generate", variant="primary", elem_id="button", margin=False, full_width=False, min_width=20, scale=1)
    with gr.Row(elem_id="image-container", equal_height=True):  
        with gr.Column():
            image = gr.Image(source='upload', tool='sketch', elem_id="image_upload", type="pil", label="Upload", height=400)
        with gr.Column():
            mage_out = gr.Image(label="Output", elem_id="output-img", height=400)

    btn.click(fn=predict, inputs=[image, prompt], outputs=[image_out])

demo.queue(concurrency_count=3)
demo.launch()