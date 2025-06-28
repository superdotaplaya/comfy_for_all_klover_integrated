import requests
import json
import os,time,sys
import pynvml
import threading
from datetime import datetime
import hashlib
from PIL import Image
import base64
import io

idle_time_setting = 30
polling_interval = 15
gpu_id_setting = 0
gpu_idle_timer = idle_time_setting
hashes = {}
checkpoint_db = 'checkpoints.json'
forge_model_directory = "F:\\new-forge\\webui\\models\\Stable-diffusion"


def load_hashes():
    if os.path.exists(checkpoint_db):
        with open(checkpoint_db, 'r') as file:
            return json.load(file).get('hashes', [])
    return []

def save_hashes(hashes):
    with open(checkpoint_db, 'w') as file:
        json.dump({'hashes': hashes}, file, indent=4)

def hash_file(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def add_file_hash_if_new(filepath):
    hashes = load_hashes()
    
    filename = os.path.basename(filepath)

    # Check if hash already exists in any [hash, filename] entry
    if not any(entry[1] == filename for entry in hashes):
        file_hash = hash_file(filepath)
        hashes.append([file_hash, filename])
        save_hashes(hashes)
        print(f"✅ Added new hash for {filename}")
    else:
        print(f"⚠️ Hash for {filename} already exists.")

print("-- Checking Checkpoint Hashes --")
# 🔍 Hash all safetensors in the directory
for fname in os.listdir(forge_model_directory):
    if fname.endswith('.safetensors'):
        full_path = os.path.join(forge_model_directory, fname)
        add_file_hash_if_new(full_path)
print("-- Checkpoint Hashing Completed! --")
def get_job():
    """try:"""
def get_job():
    resp = requests.get('http://192.168.5.199:5000/api/get-job')
    if resp.status_code != 200:
        print("no job or error", resp.status_code)
        return

    job = resp.json()          # now a dict!
    job_type   = job['request_type']
    job_id     = job['job_id']
    prompt     = job['requested_prompt']
    channel_id = job['channel']
    model_hash = job['model']
    image_link = job.get('image_link')   # only face-fix has this populated
    steps      = job['steps']
    neg_prompt = job.get('neg_prompt', "")
    resolution = job.get('resolution')
    batch_size = job.get('batch_size')
    cfg_scale  = job.get('cfg_scale')

    if job_type in ["generate", "generate chat"]:
        generate_image(
           channel_id, prompt, model_hash, neg_prompt,
           resolution, job_type, batch_size,
           cfg_scale, steps, job_id
        )
    elif job_type == "face fix":
        face_fix_post(
           image_link, channel_id, model_hash,
           prompt, job_id
        )
    else:
        print("unknown job_type", job_type)
    """except requests.RequestException as e:
        print(f"Request failed: {e}")
        get_job()"""



def submit_results(images,channel_id,job_id):
    url = f'http://192.168.5.199:5000/api/upload?channel={channel_id}&job_id={job_id}'


    files = images

    try:
        response = requests.post(url, files=files)
        print(response.json())
    except Exception as e:
        print(f"Failed to submit images: {e}")

    for _, (_, file, _) in images:
        file.close()
    
    get_job()


def is_gpu_idle(gpu_id, threshold=10):
    global gpu_idle_timer
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    if utilization.gpu > threshold:
        gpu_idle_timer = idle_time_setting
        pynvml.nvmlShutdown()
        return False  # GPU is not idle
    
    pynvml.nvmlShutdown()
    return True  # All GPUs are idle

def main_loop():
    global gpu_idle_timer
    if is_gpu_idle(gpu_id_setting) and gpu_idle_timer > 0:
        gpu_idle_timer -= polling_interval
        print("gpu time remaining until idle: " + str(gpu_idle_timer) + " seconds")
    elif gpu_idle_timer <= 0:
        get_job()
    threading.Timer(polling_interval, main_loop).start()

def get_newest_files(directory, count=1):
    # Get all files in the directory with their full paths
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # Sort files by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)
    
    # Return the specified number of newest files
    return files[:count]






def generate_image(channel_id,
                   prompt,
                   model,
                   neg_prompt,
                   resolution,
                   job_type,
                   batch_size,
                   cfg_scale,
                   steps,
                   job_id):

    # 1) Coerce None → sane defaults
    if cfg_scale is None:
        cfg_scale = 7.5
    if neg_prompt is None:
        neg_prompt = ""
    if resolution is None:
        resolution = "512x512"

    # 2) Parse resolution safely
    try:
        w, h = resolution.lower().split("x")
        width  = min(int(w), 1536)
        height = min(int(h), 1536)
    except Exception:
        width, height = 512, 512

    # 3) Build payload
    payload = {
        "prompt": prompt,
        "sampler_index": "Euler",
        "alwayson_scripts": {
            "ADetailer": {
            "args": [
                True,
                True,
                {
                "ad_model": "face_yolov8s.pt",
                "ad_model_classes": "",
                "ad_tab_enable": True,
                "ad_prompt": prompt,
                "ad_negative_prompt": "",
                "ad_confidence": 0.3,
                "ad_mask_filter_method": "Area",
                "ad_mask_k": 0,
                "ad_mask_min_ratio": 0.0,
                "ad_mask_max_ratio": 1.0,
                "ad_dilate_erode": 4,
                "ad_x_offset": 0,
                "ad_y_offset": 0,
                "ad_mask_merge_invert": "None",
                "ad_mask_blur": 4,
                "ad_denoising_strength": 0.4,
                "ad_inpaint_only_masked": True,
                "ad_inpaint_only_masked_padding": 32,
                "ad_use_inpaint_width_height": False,
                "ad_inpaint_width": 512,
                "ad_inpaint_height": 512,
                "ad_use_steps": False,
                "ad_steps": 28,
                "ad_use_cfg_scale": False,
                "ad_cfg_scale": 7.0,
                "ad_use_checkpoint": False,
                "ad_use_vae": False,
                "ad_vae": "None",
                "ad_use_sampler": False,
                "ad_sampler": "DPM++ 2M Karras",
                "ad_scheduler": "Use same scheduler",
                "ad_use_noise_multiplier": False,
                "ad_noise_multiplier": 1.0,
                "ad_use_clip_skip": False,
                "ad_clip_skip": 1,
                "ad_restore_face": False,
                "ad_controlnet_model": "None",
                "ad_controlnet_module": "None",
                "ad_controlnet_weight": 1.0,
                "ad_controlnet_guidance_start": 0.0,
                "ad_controlnet_guidance_end": 1.0
                }
            ]
            }
        },
        "scheduler": "Automatic",
        "steps": steps,
        "sd_model_checkpoint": model,
        "batch_size": batch_size,
        "distilled_cfg_scale": 1,
        "cfg_scale": cfg_scale,
        "height": height,
        "width": width,
        "filter_nsfw": False,
        "negative_prompt": neg_prompt,
        "save_images": True,
    }

    # 4) Optionally set the model on the WebUI first
    opts_url = "http://127.0.0.1:7860/sdapi/v1/options"
    requests.post(opts_url, json={
        "forge_additional_modules": [],
        "sd_model_checkpoint": model,
    })

    # 5) Fire off the txt2img call
    txt2img_url = "http://127.0.0.1:7860/sdapi/v1/txt2img"
    r = requests.post(txt2img_url, json=payload)
    try:
        result = r.json()
    except ValueError:
        raise RuntimeError(f"WebUI returned non-JSON: {r.text}")

    # 6) Grab today’s folder
    today = datetime.now().strftime("%Y-%m-%d")
    output_dir = fr"F:\new-forge\webui\outputs\txt2img-images\{today}"

    images = []
    files = get_newest_files(output_dir, batch_size)  # your helper
    for path in files:
        f = open(path, "rb")
        images.append(("images", (os.path.basename(path), f, "image/png")))

    # 7) Push results back to your API
    submit_results(images, channel_id, job_id)

def face_fix_post(image_link, channel, checkpoint, prompt, job_id):
    print(channel)

    payload = {
        "init_images": [image_link],
        "prompt": "",
        "sampler_name": "Euler",
        "alwayson_scripts": {
            "ADetailer": {
            "args": [
                True,
                True,
                {
                "ad_model": "face_yolov8s.pt",
                "ad_model_classes": "",
                "ad_tab_enable": True,
                "ad_prompt": prompt,
                "ad_negative_prompt": "",
                "ad_confidence": 0.3,
                "ad_mask_filter_method": "Area",
                "ad_mask_k": 0,
                "ad_mask_min_ratio": 0.0,
                "ad_mask_max_ratio": 1.0,
                "ad_dilate_erode": 4,
                "ad_x_offset": 0,
                "ad_y_offset": 0,
                "ad_mask_merge_invert": "None",
                "ad_mask_blur": 4,
                "ad_denoising_strength": 0.4,
                "ad_inpaint_only_masked": True,
                "ad_inpaint_only_masked_padding": 32,
                "ad_use_inpaint_width_height": False,
                "ad_inpaint_width": 512,
                "ad_inpaint_height": 512,
                "ad_use_steps": False,
                "ad_steps": 28,
                "ad_use_cfg_scale": False,
                "ad_cfg_scale": 7.0,
                "ad_use_checkpoint": True,
                "ad_checkpoint": checkpoint,
                "ad_use_vae": False,
                "ad_vae": "None",
                "ad_use_sampler": False,
                "ad_sampler": "DPM++ 2M Karras",
                "ad_scheduler": "Use same scheduler",
                "ad_use_noise_multiplier": False,
                "ad_noise_multiplier": 1.0,
                "ad_use_clip_skip": False,
                "ad_clip_skip": 1,
                "ad_restore_face": False,
                "ad_controlnet_model": "None",
                "ad_controlnet_module": "None",
                "ad_controlnet_weight": 1.0,
                "ad_controlnet_guidance_start": 0.0,
                "ad_controlnet_guidance_end": 1.0
                }
            ]
            }
        }
        }
    response = requests.post(url=f'http://127.0.0.1:7860/sdapi/v1/img2img', json=payload)

    r = response.json()
    print(r)
    image = Image.open(io.BytesIO(base64.b64decode(r['images'][0])))
    image.save('output.png')
    files = [('images', ('output.png', open('output.png', 'rb'), 'image/png'))]
    submit_results(files,channel,job_id)
    os.remove("output.png")
    

# Start the first execution
main_loop()