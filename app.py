import platform
import subprocess
import openvino_genai as ov_genai
from pathlib import Path
from PIL import Image
import gradio as gr
import sys

# Install dependencies if not installed
def install_dependencies():
    packages = [
        'diffusers>=0.30.0',
        'torch>=2.1',
        'huggingface-hub>=0.9.1',
        'Pillow',
        'opencv-python',
        'tqdm',
        'gradio>=4.19',
        'openvino>=2025.0',
        'openvino_genai>=2025.0',
        'openvino_tokenizers>=2025.0',
        'git+https://github.com/huggingface/optimum-intel.git'
    ]
    for package in packages:
        subprocess.run(['pip', 'install', '--quiet', '--extra-index-url', 'https://download.pytorch.org/whl/cpu', package])

# Run installation (only needed once)
install_dependencies()

if platform.system() == "Darwin":  # macOS compatibility fix
    subprocess.run(['pip', 'install', '--quiet', 'numpy<2.0'])

# Model setup
MODEL_ID = "prompthero/openjourney"
MODEL_DIR = Path("diffusion_pipeline")

if not MODEL_DIR.exists():
    from cmd_helper import optimum_cli
    optimum_cli(MODEL_ID, MODEL_DIR, additional_args={"weight-format": "fp16"})

DEVICE = "CPU"
pipe = ov_genai.Text2ImagePipeline(MODEL_DIR, DEVICE)

# Image Generation Function
def generate_image(prompt, steps, seed):
    generator = ov_genai.TorchGenerator(seed)
    result = pipe.generate(prompt, num_inference_steps=steps, generator=generator)
    img = Image.fromarray(result.data[0])
    return img

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## Anirudh's Text-to-Image Generator")
    
    with gr.Row():
        prompt = gr.Textbox(label="Enter Your Text Prompt", value="")
    
    with gr.Row():
        steps = gr.Slider(1, 50, value=20, step=1, label="Number of Steps")
        seed = gr.Slider(0, 10000000, value=42, step=1, label="Seed")
    
    generate_btn = gr.Button("Generate Image")
    output_img = gr.Image(label="Generated Image")
    
    generate_btn.click(fn=generate_image, inputs=[prompt, steps, seed], outputs=output_img)

# Run the Web UI
if __name__ == "__main__":
    demo.launch(share=True)
