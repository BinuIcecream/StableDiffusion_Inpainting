from diffusers import StableDiffusionXLInpaintPipeline
import gradio as gr
import numpy as np
import imageio
from PIL import Image
import torch
import modin.pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionXLInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", safety_checker=None)
pipe = pipe.to(device)

def resize(value,img):
    img = Image.open(img)
    img = img.resize((value,value))
    return img

def predict(input_image, prompt, negative_prompt):
    imageio.imwrite("data.png", input_image["image"])
    imageio.imwrite("data_mask.png", input_image["mask"])
    src = resize(768, "data.png")
    src.save("src.png")
    mask = resize(768, "data_mask.png")  
    mask.save("mask.png")
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, image=src, mask_image=mask, num_inference_steps=20).images[0]
    return image

# Gradio UI
css = '''
    @media (min-width: 1280px) {.gradio-container{max-width: 900px !important;}}
    h1{text-align: center;}
    textarea.scroll-hide{border: none !important;box-sizing: initial;overflow-y: auto !important;height: 8px !important;line-height: 0.8 !important;}
    .container.svelte-1f354aw>textarea.svelte-1f354aw{border-radius: 10px 0px 0px 0px;border-right: none !important;box-shadow: none !important;}
'''

with gr.Blocks(css=css) as demo:
    title = gr.Markdown(''' # Stable Diffusion XL Inpainting''')
    with gr.Row(elem_id="prompt-row", equal_height=True) as promot_row:
        prompt = gr.Textbox(label="Prompt", placeholder="Write your desired prompt here...", elem_id="prompt", show_label=False, lines=1, scale=4)
        submit_button = gr.Button("Generate", variant="primary", elem_id="button", min_width=20, scale=1)
    with gr.Row(elem_id="image-row", equal_height=True) as image_row:  
        with gr.Column(elem_id="input-image-container") as input_image_container:
            input_image = gr.Image(label="Input", source="upload", type="numpy", tool="sketch", elem_id="input-image")
        with gr.Column(elem_id="output-image-container") as output_image_container:
            output_image = gr.Image(label="Output", elem_id="output_container")
    with gr.Row(elem_id="additional-container") as additional_container: 
        with gr.Accordion("Additional", elem_id="accordion-container", open=False) as accordion_container:
            negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Write your negative prompt here...", elem_id="negative-prompt", show_label=True, lines=1)
    
    submit_button.click(fn=predict, inputs=[prompt, init_image], outputs=output_image)

demo.queue(concurrency_count=3)
demo.launch()