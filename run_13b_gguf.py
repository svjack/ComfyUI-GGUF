#### 通过 comfyui model manager 进行安装
https://huggingface.co/mcmonkey/google_t5-v1_1-xxl_encoderonly

git clone https://huggingface.co/datasets/svjack/Aesthetics_X_Phone_Images_Rec_Captioned_5120x2880

vim run_single_gguf.py

from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
with Workflow():
    #_ = AnySwitchRgthree()
    #_2 = AnySwitchRgthree()
    #model, _2 = PowerLoraLoaderRgthree(_, _2)
    model = UnetLoaderGGUF('ltxv-13b-0.9.7-distilled-Q8_0.gguf')
    _2 = ClipLoaderGGUF('t5/google_t5-v1_1-xxl_encoderonly-fp8_e4m3fn.safetensors', 'ltxv', 'default')
    vae = VAELoader('ltxv-13b-0.9.7-vae-BF16.safetensors')
    conditioning = CLIPTextEncode('', _2)
    conditioning2 = CLIPTextEncode('low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly', _2)
    positive, negative = LTXVConditioning(conditioning, conditioning2, 24.000000000000004)
    stg_advanced_preset = STGAdvancedPresets('13b Dynamic')
    guider = STGGuiderAdvanced(model, positive, negative, 0.9970000000000002, True, '1.0, 0.9933, 0.9850, 0.9767, 0.9008, 0.6180', '1,16,8,8,4,1', '0, 4, 4, 2, 1, 1', '1, 1, 1, 1, 1, 1', '[35], [35], [35], [42], [42], [42]', stg_advanced_preset)
    sampler = KSamplerSelect('euler')
    sigmas = BasicScheduler(model, 'beta', 8, 1)
    noise = RandomNoise(109)
    image, _ = LoadImage('image (89).jpg')
    denoised_output = LTXVBaseSampler(model, vae, 768, 512, 97, guider, sampler, sigmas, noise, image, '0', 0.8, 'center', 30, 0)
    vae2 = SetVAEDecoderNoise(vae, 0.05, 0.025, 44731965495089)
    image2 = VAEDecode(denoised_output, vae2)
    _ = VHSVideoCombine(image2, 24, 0, 'ltxv-base', 'video/h264-mp4', False, True, None, None, None)

vim run_ltxv_13b_gguf.py

import os
import time
import subprocess
import shutil
from pathlib import Path

# Configuration
SEED = 661695664686456
SOURCE_DIR = 'Aesthetics_X_Phone_Images_Rec_Captioned_5120x2880'
INPUT_DIR = 'ComfyUI/input'
OUTPUT_DIR = 'ComfyUI/output'
PYTHON_PATH = '/environment/miniconda3/bin/python'

def copy_image_pairs_to_input():
    """Copy all image-text pairs from source to ComfyUI/input"""
    os.makedirs(INPUT_DIR, exist_ok=True)

    # Clear input directory first
    for file in Path(INPUT_DIR).glob('*'):
        try:
            if file.is_file():
                file.unlink()
        except Exception as e:
            print(f"Error deleting {file}: {e}")

    # Copy new image-text pairs
    for img_file in Path(SOURCE_DIR).glob('*.png'):
        txt_file = img_file.with_suffix('.txt')
        if txt_file.exists():
            try:
                shutil.copy(img_file, INPUT_DIR)
                shutil.copy(txt_file, INPUT_DIR)
            except Exception as e:
                print(f"Error copying {img_file} or {txt_file}: {e}")

def get_latest_video_count():
    """Return the number of MP4 files in the output directory"""
    try:
        return len(list(Path(OUTPUT_DIR).glob('*.mp4')))
    except:
        return 0

def wait_for_new_video(initial_count):
    """Wait until a new MP4 file appears in the output directory"""
    timeout = 3000  # seconds (increased for video generation)
    start_time = time.time()

    while time.time() - start_time < timeout:
        current_count = get_latest_video_count()
        if current_count > initial_count:
            time.sleep(1)  # additional 1 second delay
            return True
        time.sleep(1)  # check less frequently for videos
    return False

def read_prompt_from_txt(txt_path):
    """Read the prompt from the text file"""
    try:
        with open(txt_path, 'r') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading prompt from {txt_path}: {e}")
        return ""

def generate_script(input_image, prompt, SEED):
    """Generate the ComfyUI script for the given input image and prompt"""
    script_content = f"""from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
with Workflow():
    model = UnetLoaderGGUF('ltxv-13b-0.9.7-distilled-Q8_0.gguf')
    _2 = ClipLoaderGGUF('t5/google_t5-v1_1-xxl_encoderonly-fp8_e4m3fn.safetensors', 'ltxv', 'default')
    vae = VAELoader('ltxv-13b-0.9.7-vae-BF16.safetensors')
    conditioning = CLIPTextEncode('{prompt}', _2)
    conditioning2 = CLIPTextEncode('low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly', _2)
    positive, negative = LTXVConditioning(conditioning, conditioning2, 24.000000000000004)
    stg_advanced_preset = STGAdvancedPresets('13b Dynamic')
    guider = STGGuiderAdvanced(model, positive, negative, 0.9970000000000002, True, '1.0, 0.9933, 0.9850, 0.9767, 0.9008, 0.6180', '1,16,8,8,4,1', '0, 4, 4, 2, 1, 1', '1, 1, 1, 1, 1, 1', '[35], [35], [35], [42], [42], [42]', stg_advanced_preset)
    sampler = KSamplerSelect('euler')
    sigmas = BasicScheduler(model, 'beta', 8, 1)
    noise = RandomNoise({SEED})
    image, _ = LoadImage('{input_image}')
    denoised_output = LTXVBaseSampler(model, vae, 768, 512, 97, guider, sampler, sigmas, noise, image, '0', 0.8, 'center', 30, 0)
    vae2 = SetVAEDecoderNoise(vae, 0.05, 0.025, 44731965495089)
    image2 = VAEDecode(denoised_output, vae2)
    _ = VHSVideoCombine(image2, 24, 0, 'ltxv-base', 'video/h264-mp4', False, True, None, None, None)
"""
    return script_content

def main():
    SEED = 661695664686456
    # Ensure directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Copy all image-text pairs to input directory
    copy_image_pairs_to_input()

    # Get list of input images
    input_images = list(Path(INPUT_DIR).glob('*.png'))
    if not input_images:
        print(f"No images found in {INPUT_DIR}")
        return

    print(f"Found {len(input_images)} image-text pairs to process")

    # Process each image
    for img_path in input_images:
        # Get corresponding text file
        txt_path = img_path.with_suffix('.txt')
        if not txt_path.exists():
            print(f"No matching text file found for {img_path.name}. Skipping.")
            continue

        # Read prompt from text file
        prompt = read_prompt_from_txt(txt_path)
        if not prompt:
            print(f"Empty prompt for {img_path.name}. Using empty string.")
            prompt = ""

        # Get current video count before running
        initial_count = get_latest_video_count()

        # Generate script for this image
        script = generate_script(
            str(img_path.name),  # Just the filename, not full path
            prompt.replace("'", "\\'"),  # Escape single quotes in prompt
            SEED
        )

        # Write script to file
        with open('run_ltxv_generation.py', 'w') as f:
            f.write(script)

        print(f"Processing image: {img_path.name} with prompt: {prompt[:50]}...")
        subprocess.run([PYTHON_PATH, 'run_ltxv_generation.py'])

        # Wait for new video
        if not wait_for_new_video(initial_count):
            print(f"Timeout waiting for video generation for {img_path.name}. Continuing to next image.")

        # Increment seed for next generation
        SEED -= 1

if __name__ == "__main__":
    main()
