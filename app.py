import gradio as gr
from PIL import Image
import torch
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
    output = pipe(prompt = prompt, image=init_image, mask_image=mask, guidance_scale=7.5)
    return output.images[0]

css = '''
    @media (min-width: 1280px) {.gradio-container{max-width: 900px !important;}}
    h1{text-align: center;}
    textarea.scroll-hide{border: none !important;box-sizing: initial;max-height: 20px;overflow-y: auto !important;}
    #component-1{border-bottom: 1px solid var(--border-color-primary);}
    .container.svelte-1f354aw>textarea.svelte-1f354aw{border-radius: 10px 0px 0px 0px;border-right: none !important;box-shadow: none !important;}
'''

with gr.Blocks(css=css) as demo:
    gr.Markdown(''' # Stable Diffusion Inpainting''')
    with gr.Row(elem_id="prompt-container", equal_height=True):
        prompt = gr.Textbox(placeholder = 'Prompt', elem_id="input-text", show_label=False, lines=2, scale=4)
        btn = gr.Button("Generate", variant="primary", elem_id="button", min_width=20, scale=1)
    with gr.Row(elem_id="image-container", equal_height=True):  
        with gr.Column():
            image = gr.Image(source='upload', tool='sketch', elem_id="image_upload", type="pil", label="Upload", height=400)
        with gr.Column():
            image_out = gr.Image(label="Output", elem_id="output-img", height=400)
    
    btn.click(fn=predict, inputs=[image, prompt], outputs=[image_out])

demo.queue(concurrency_count=3)
demo.launch()