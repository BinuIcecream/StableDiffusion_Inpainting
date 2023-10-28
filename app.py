import gradio as gr
import torch
from diffusers import AutoPipelineForInpainting, UNet2DConditionModel
import diffusers

device = "cuda" if torch.cuda.is_available() else "cpu"

if device != "cpu":
    print("Your Space is running on GPU hardware.")
    pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to(device)
else:
    print("Your Space is running on CPU hardware.")
    print("CUDA is currently unavailable as this Space is running on CPU hardware. To enable GPU acceleration, you can upgrade your Space by clicking the 'Settings' button located in the top navigation bar of the Space.")


def read_content(file_path: str) -> str:
    """read the content of target file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return content

def predict(dict, prompt="", negative_prompt="", guidance_scale=7.5, steps=20, strength=1.0, scheduler="EulerDiscreteScheduler"):
    if device != "cuda":
        print("CUDA is currently unavailable as this Space is running on CPU hardware. To enable GPU acceleration, you can upgrade your Space by clicking the 'Settings' button located in the top navigation bar of the Space.")
    else:
        if negative_prompt == "":
            negative_prompt = None
        scheduler_class_name = scheduler.split("-")[0]
    
        add_kwargs = {}
        if len(scheduler.split("-")) > 1:
            add_kwargs["use_karras"] = True
        if len(scheduler.split("-")) > 2:
            add_kwargs["algorithm_type"] = "sde-dpmsolver++"
    
        scheduler = getattr(diffusers, scheduler_class_name)
        pipe.scheduler = scheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler", **add_kwargs)
        
        init_image = dict["image"].convert("RGB").resize((1024, 1024))
        mask = dict["mask"].convert("RGB").resize((1024, 1024))
        
        output = pipe(prompt = prompt, negative_prompt=negative_prompt, image=init_image, mask_image=mask, guidance_scale=guidance_scale, num_inference_steps=int(steps), strength=strength)
    
        print("Image Generated:")
        print(output.images[0])
        
        return output.images[0]

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
            image = gr.Image(label="Input", source="upload", type="pil", tool="sketch", elem_id="input-image")
        with gr.Column(elem_id="output-image-container") as output_image_container:
            output_image = gr.Image(label="Output", elem_id="output_container")
    with gr.Row(elem_id="additional-container") as additional_container: 
        with gr.Accordion("Additional Inputs", elem_id="accordion-container", open=False) as accordion_container:
            negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Write your negative prompt here...", elem_id="negative-prompt", show_label=True, lines=1)
            guidance_scale = gr.Number(value=7.5, minimum=1.0, maximum=20.0, step=0.1, label="guidance_scale")
            steps = gr.Number(value=20, minimum=10, maximum=30, step=1, label="steps")
            strength = gr.Number(value=0.99, minimum=0.01, maximum=1.0, step=0.01, label="strength")
            schedulers = ["DEISMultistepScheduler", "HeunDiscreteScheduler", "EulerDiscreteScheduler", "DPMSolverMultistepScheduler", "DPMSolverMultistepScheduler-Karras", "DPMSolverMultistepScheduler-Karras-SDE"]
            scheduler = gr.Dropdown(label="Schedulers", choices=schedulers, value="EulerDiscreteScheduler")
    
    submit_button.click(fn=predict, inputs=[image, prompt, negative_prompt, guidance_scale, steps, strength, scheduler], outputs=output_image)
    prompt.submit(fn=predict, inputs=[image, prompt, negative_prompt, guidance_scale, steps, strength, scheduler], outputs=output_image)

demo.queue(max_size=25)
demo.launch()