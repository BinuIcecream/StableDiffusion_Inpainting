import gradio as gr
from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the Stable Diffusion Inpainting pipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
).to(device)

# Prediction function
def predict(prompt, pil_image):
    # Extract and resize the image and mask from the Gradio sketch tool
    image = pil_image["image"].convert("RGB").resize((512, 512))

    # Perform inpainting
    result = pipe(prompt=prompt, image=image).images[0]
    return result

# Gradio UI
css = '''
    @media (min-width: 1280px) {.gradio-container{max-width: 900px !important;}}
    h1{text-align: center;}
    textarea.scroll-hide{border: none !important;box-sizing: initial;overflow-y: auto !important;height: 8px !important;line-height: 0.8 !important;}
    .container.svelte-1f354aw>textarea.svelte-1f354aw{border-radius: 10px 0px 0px 0px;border-right: none !important;box-shadow: none !important;}
'''

with gr.Blocks(css=css) as demo:
    gr.Markdown(''' # Stable Diffusion Inpainting''')
    with gr.Row(elem_id="prompt-container", equal_height=True):
        prompt = gr.Textbox(placeholder='Prompt', elem_id="input-text", show_label=False, lines=1, scale=4)
        submit_button = gr.Button("Generate", variant="primary", elem_id="button", min_width=20, scale=1)
    with gr.Row(elem_id="image-container", equal_height=True):  
        with gr.Column():
            init_image = gr.Image(source='upload', tool='sketch', elem_id="image_upload", type="pil", label="Upload", height=400)
        with gr.Column():
            output_image = gr.Image(label="Output", elem_id="output-img", height=400)
    
    submit_button.click(fn=predict, inputs=[prompt, init_image], outputs=output_image)

demo.queue(concurrency_count=3)
demo.launch()
