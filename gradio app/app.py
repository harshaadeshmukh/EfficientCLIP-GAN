import os
import random
import gradio as gr
import requests
from PIL import Image

from utils import read_css_from_file
from inference import generate_image_from_text, generate_image_from_text_with_persistent_storage
# Read CSS from file
css = read_css_from_file("style.css")

DESCRIPTION = '''
    <div id="content_align">
        <span style="color:darkred;font-size:32px:font-weight:bold">
        EfficientCLIP-GAN Models Image Generation Demo
        </span>
    </div>
    <div id="content_align">
        <span style="color:blue;font-size:16px:font-weight:bold">
        Generate images directly from text prompts
        </span>
    </div>
    <div id="content_align" style="margin-top: 10px;">
    </div>
'''

# Creating Gradio interface
with gr.Blocks(css=css) as app:
    gr.Warning("This 💻 demo uses the EfficientCLIP-GAN model which is trained on CUB dataset 🐦🐥.\nKeep your prompt coherent to the birds domain.\nIf you like the demo, don't forget to click on the like 💖 button.")
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            text_prompt = gr.Textbox(label="Input Prompt", placeholder="", lines=3)
    generate_button = gr.Button("Generate Image", variant='primary')

    with gr.Row():
        with gr.Column():
            image_output1 = gr.Image(type="pil", label="Image Output 1")
            image_output2 = gr.Image(type="pil", label="Image Output 2")

        with gr.Column():
            image_output3 = gr.Image(type="pil", label="Image Output 3")
            image_output4 = gr.Image(type="pil", label="Image Output 4")

    generate_button.click(generate_image_from_text, inputs=[text_prompt], outputs=[image_output1, image_output2, image_output3, image_output4])

# Launch the app
app.launch()
