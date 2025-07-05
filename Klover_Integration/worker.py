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
worker_id = ""
accepted_job_types = ["generate", "facefix", "img2img", "upscale"]
max_batch_size = 5

#HASH CALC AND CHECKING BELOW
#Load list of all hashes
def load_hashes():
    if os.path.exists(checkpoint_db):
        with open(checkpoint_db, 'r') as file:
            return json.load(file).get('hashes', [])
    return []
#Convert hash to local systems file name
def hash_to_model_name(hash):
    hashes = load_hashes()
    for file_hash in hashes:
        if file_hash[0] == hash:
            print(file_hash[1])
            return(file_hash[1])
#Save all hashes with respective file name in 'checkpoints.json'
def save_hashes(hashes):
    with open(checkpoint_db, 'w') as file:
        json.dump({'hashes': hashes}, file, indent=4)
#Calculate hashes of all files in given directory
def hash_file(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()
#Add hash and filename to list if its not already there
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
# 🔍 Check all safetensors in the directory
for fname in os.listdir(forge_model_directory):
    if fname.endswith('.safetensors'):
        full_path = os.path.join(forge_model_directory, fname)
        add_file_hash_if_new(full_path)
print("-- Checkpoint Hashing Completed! --")

def worker_login():
    global worker_id
    global accepted_job_types
    try:
        with open("worker_auth.json", "rb") as worker_auth_file:
            worker_id = json.load(worker_auth_file).get('worker_id')
            hashes_list = []
            raw_hashes = load_hashes()
            for hash in raw_hashes:
                    hashes_list.append(hash[0])
            resp = requests.get('https://7690-97-119-124-11.ngrok-free.app/api/init', json = {'worker_id': worker_id, 'checkpoints':hashes_list, 'acceptable_job_types':accepted_job_types})
            if not resp.json().get("created"):
                print(f"Worker authenticated successfully\nCheckpoint Hashes: {str(len(hashes_list))}\nAcceptable Job Types: {accepted_job_types}")
    except:
        resp = requests.get('https://7690-97-119-124-11.ngrok-free.app/api/init', json = {'worker_id': "N/A"})
        if resp.json().get("created"):
            worker_id = resp.json().get("worker_id")
            print(f'New user created: {worker_id}...')
            json.dump({'worker_id': worker_id}, open('worker_auth.json', 'w'), indent=4)

# JOB PROCESSING BELOW
def get_job():
    global worker_id
    today = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    """ try:"""
    resp = requests.get('https://7690-97-119-124-11.ngrok-free.app/api/get-job', json = {'worker_id':worker_id})
    if resp.status_code != 200:
        print(f"[{today}] || No accceptable job found. Status:", resp.status_code)
        return
    
    job = resp.json()          # now a dict!
    job_type   = job['request_type']
    job_id     = job['job_id']
    prompt     = job['requested_prompt']
    channel_id = job['channel']
    model_hash = hash_to_model_name(job['model'])
    image_link = job.get('image_link')   
    steps      = job['steps']
    neg_prompt = job.get('neg_prompt', "")
    resolution = job.get('resolution')
    batch_size = job.get('batch_size')
    cfg_scale  = job.get('config_scale')
    requester = job.get('requester')
    print(f"[{today}] || Job recieved: \nJob ID: {job_id}\nJob Type: {job_type}\nModel/Checkpoint: {model_hash}")
    if job_type in ["generate", "generate chat"]:
        generate_image(
        channel_id, prompt, model_hash, neg_prompt,
        resolution, job_type, batch_size,
        cfg_scale, steps, job_id, requester
        )
    elif job_type == "face fix":
        face_fix_post(
        image_link, channel_id, model_hash,
        prompt, job_id, requester
        )
    elif job_type == "upscale":
        upscale_image(image_link,channel_id,cfg_scale,job_id, requester)
    elif job_type == "img2img":
        img2imggen(image_link,channel_id,model_hash,prompt,neg_prompt,resolution,batch_size, job_id, requester)
    """except requests.RequestException as e:
        print(f"Request failed: {e}")
        get_job()"""
    """except:
        print("No applicable jobs found")"""


def submit_results(images,channel_id,requester,job_id):
    global worker_id
    url = f'https://7690-97-119-124-11.ngrok-free.app/api/upload?channel={channel_id}&job_id={job_id}'


    files = images

    try:
        response = requests.post('https://7690-97-119-124-11.ngrok-free.app/api/upload', files=files, data = {'worker_id':worker_id, 'channel':channel_id, 'job_id':job_id, 'requester':requester})
        print(response.json())
    except Exception as e:
        print(f"Failed to submit images: {e}")

    for _, (_, file, _) in images:
        file.close()
    hashes_list = []
    raw_hashes = load_hashes()
    for hash in raw_hashes:
            hashes_list.append(hash[0])
    if response.status_code == 200:
        print(f"✅ Job ID: {job_id} completed with {len(files)} files submitted to server successfully! ✅")
    for file in files:
        os.remove(str(file[1][1]).split("=")[1].replace("'","").replace(">",""))
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
        hashes_list = []
        raw_hashes = load_hashes()
        for hash in raw_hashes:
            hashes_list.append(hash[0])
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
                   job_id, requester):

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
    submit_results(images, channel_id, requester,job_id)

def face_fix_post(image_link, channel, checkpoint, prompt, job_id, requester):
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
    submit_results(files,channel,requester,job_id)
    os.remove("output.png")
    


def upscale_image(image_link,channel,upscale_by,job_id, requester):
    payload = {
        "resize_mode": 0,
        "show_extras_results": True,
        "gfpgan_visibility": 0,
        "codeformer_visibility": 0,
        "codeformer_weight": 0,
        "upscaling_resize": upscale_by,
        "upscaling_resize_w": 3072,
        "upscaling_resize_h": 3072,
        "upscaling_crop": True,
        "upscaler_1": "R-ESRGAN 4x+",
        "upscaler_2": "None",
        "extras_upscaler_2_visibility": 0,
        "upscale_first": False,
        "image": image_link,

    }
    response = requests.post(url=f'http://127.0.0.1:7860/sdapi/v1/extra-single-image', json=payload)
    r = response.json()
    print(r)
    image = Image.open(io.BytesIO(base64.b64decode(r['image'])))
    image.save('output.png')
    files = [('images', ('output.png', open('output.png', 'rb'), 'image/png'))]
    submit_results(files,channel,requester,job_id)
    os.remove("output.png")


def img2imggen(image_link, channel_id, checkpoint, prompt, negative_prompt, resolution, batch_size, job_id, requester):
    payload = {
        "init_images": [image_link],
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "sampler_name": "Euler",
        "sd_model_checkpoint": checkpoint,
        "width": int(resolution.split("x")[0]),
        "height": int(resolution.split("x")[1]),
        "batch_size": batch_size,
        "steps": 25,
        "cfg_scale": 5,
        "denoising_strength": 0.75,
        "alwayson_scripts": {
            "ADetailer": {
            "args": [
                True,
                False,
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
                "ad_denoising_strength": 0.5,
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
    submit_results(files,channel_id,requester,job_id)
    os.remove("output.png")
worker_login()
# Start the first execution
main_loop()