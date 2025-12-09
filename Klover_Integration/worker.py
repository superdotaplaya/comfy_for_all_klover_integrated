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
import re
import tkinter as tk
from tkinter import ttk, messagebox
from urllib import request
import PIL.Image as Image
import websocket
import uuid
from urllib.parse import urlparse
import random 


idle_time_setting = 30
polling_interval = 15
gpu_id_setting = 0
gpu_idle_timer = idle_time_setting
hashes = {}
checkpoint_db = 'checkpoints.json'
lora_db = 'loras.json'
forge_model_directory = "F:\\stable-diffusion-webui-reForge\\models\\Stable-diffusion"
lora_model_directory = "F:\\stable-diffusion-webui-reForge\\models\\Lora"
civit_api_token = ""
worker_id = ""
accepted_job_types = ["generate", "facefix", "img2img", "upscale", "img2vid"]
max_batch_size = 5
server_url = "https://c3bc6e2471d1.ngrok-free.app"
auto_dl_lora = True
auto_dl_checkpoints = True
output_directory = "F:\\stable-diffusion-webui-reForge\\outputs\\txt2img-images"
worker_type = "forge"
worker_main_loop = None
worker_name = "TBD"


def run_app():
    check_checkpoint_hashes()
    check_lora_hashes()
    worker_login()
    # Start the first execution
    main_loop()
def load_user_interface():
    global forge_model_directory, lora_model_directory, server_url, civit_api_token, auto_dl_lora, auto_dl_checkpoints, aria2_path, output_directory, accepted_job_types, worker_name
    root = tk.Tk()
    root.title("Forge SD Worker")

    # Initialize Tkinter variables
    auto_dl_lora = tk.BooleanVar()
    auto_dl_checkpoints = tk.BooleanVar()

    # Use current globals as defaults (fall back to empty string)
    forge_model_directory = forge_model_directory if 'forge_model_directory' in globals() else ""
    lora_model_directory  = lora_model_directory  if 'lora_model_directory' in globals() else ""
    server_url            = server_url            if 'server_url' in globals() else ""
    civit_api_token       = civit_api_token       if 'civit_api_token' in globals() else ""
    aria2_path            = aria2_path            if 'aria2_path' in globals() else ""
    output_directory      = output_directory      if 'output_directory' in globals() else ""
    worker_name         = worker_name          if 'worker_name' in globals() else "TBD" 

    # Load config if it exists
    if os.path.exists("worker_config.json"):
        try:
            with open("worker_config.json", "r") as config_file:
                config = json.load(config_file)
                forge_model_directory = config.get("forge_model_directory", forge_model_directory)
                lora_model_directory  = config.get("lora_model_directory", lora_model_directory)
                server_url            = config.get("server_url", server_url)
                civit_api_token       = config.get("civit_api_token", civit_api_token)
                aria2_path            = config.get("aria2_path", aria2_path)
                output_directory      = config.get("output_directory", output_directory)
                worker_name           = config.get("worker_name", worker_name)
                auto_dl_lora.set(config.get("auto_dl_lora", False))
                auto_dl_checkpoints.set(config.get("auto_dl_checkpoints", False))
                # Load acceptable job types from config if present so UI reflects current selections
                accepted_job_types = config.get("accepted_job_types", accepted_job_types)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config:\n{e}")

    frm = ttk.Frame(root, padding=10)
    frm.grid()

    # Labels and Inputs
    ttk.Label(frm, text="Worker Name:").grid(column=0, row=0, sticky="w")
    Worker_Name = ttk.Entry(frm, width=100)
    Worker_Name.insert(0, worker_name)
    Worker_Name.grid(column=1, row=0)

    ttk.Label(frm, text="Forge Model Directory:").grid(column=0, row=1, sticky="w")
    SD_Model_Directory = ttk.Entry(frm, width=100)
    SD_Model_Directory.insert(0, forge_model_directory)
    SD_Model_Directory.grid(column=1, row=1)

    ttk.Label(frm, text="Lora Model Directory:").grid(column=0, row=2, sticky="w")
    Lora_Model_Directory = ttk.Entry(frm, width=100)
    Lora_Model_Directory.insert(0, lora_model_directory)
    Lora_Model_Directory.grid(column=1, row=2)

    ttk.Label(frm, text="Server URL:").grid(column=0, row=3, sticky="w")
    Server_URL = ttk.Entry(frm, width=100)
    Server_URL.insert(0, server_url)
    Server_URL.grid(column=1, row=3)

    ttk.Label(frm, text="Civit API Token:").grid(column=0, row=4, sticky="w")
    Civit_API_Token = ttk.Entry(frm, width=100)
    Civit_API_Token.insert(0, civit_api_token)
    Civit_API_Token.grid(column=1, row=4)

    ttk.Label(frm, text="Aria2 Executable Path:").grid(column=0, row=5, sticky="w")
    Aria2_Path = ttk.Entry(frm, width=100)
    Aria2_Path.insert(0, aria2_path)
    Aria2_Path.grid(column=1, row=5)

    ttk.Label(frm, text="Image Output Directory:").grid(column=0, row=6, sticky="w")
    Output_Directory = ttk.Entry(frm, width=100)
    Output_Directory.insert(0, output_directory)
    Output_Directory.grid(column=1, row=6)

    ttk.Label(frm, text="Auto Download Lora:").grid(column=0, row=7, sticky="w")
    Auto_DL_Lora = ttk.Checkbutton(frm, variable=auto_dl_lora)
    Auto_DL_Lora.grid(column=1, row=7, sticky="w")

    ttk.Label(frm, text="Auto Download Checkpoints:").grid(column=0, row=8, sticky="w")
    Auto_DL_Checkpoints = ttk.Checkbutton(frm, variable=auto_dl_checkpoints)
    Auto_DL_Checkpoints.grid(column=1, row=8, sticky="w")

    # Acceptable job types multi-select
    ttk.Label(frm, text="Acceptable Job Types:").grid(column=0, row=9, sticky="nw")
    job_options = ["txt2img", "upscale", "face fix", "img2img", "img2vid"]
    job_listbox = tk.Listbox(frm, selectmode='multiple', height=len(job_options), exportselection=False)
    for opt in job_options:
        job_listbox.insert(tk.END, opt)
    job_listbox.grid(column=1, row=9, sticky="w")

    # Map user-facing labels to internal job identifiers used by the worker/server
    label_to_internal = {
        "txt2img": "generate",
        "upscale": "upscale",
        "face fix": "face fix",
        "img2img": "img2img",
        "img2vid": "img2vid"
    }

    # Pre-select entries based on current accepted_job_types
    try:
        for idx, label in enumerate(job_options):
            internal = label_to_internal.get(label)
            if internal in accepted_job_types:
                job_listbox.selection_set(idx)
    except Exception:
        pass

    # Helpers to apply selections
    def apply_job_selection():
        global accepted_job_types
        sel = job_listbox.curselection()
        selected_labels = [job_listbox.get(i) for i in sel]
        # Map labels to internal names and update global
        accepted_job_types = [label_to_internal[l] for l in selected_labels]

    # Save button (updates accepted_job_types before saving)
    def save_and_update():
        apply_job_selection()
        save_settings(
            SD_Model_Directory.get(),
            Lora_Model_Directory.get(),
            Server_URL.get(),
            Civit_API_Token.get(),
            Aria2_Path.get(),
            Output_Directory.get(),
            auto_dl_lora.get(),
            auto_dl_checkpoints.get(),
            Worker_Name.get()
        )
        messagebox.showinfo("Saved", f"Acceptable job types set to: {accepted_job_types}")

    ttk.Button(frm, text="Save Settings", command=save_and_update).grid(column=1, row=10)

    # Update globals with current entries so values are available immediately
    forge_model_directory = SD_Model_Directory.get()
    lora_model_directory = Lora_Model_Directory.get()
    server_url = Server_URL.get()
    civit_api_token = Civit_API_Token.get()
    aria2_path = Aria2_Path.get()
    output_directory = Output_Directory.get()
    worker_name = Worker_Name.get()

    # Start / Quit buttons
    def start_worker():
        apply_job_selection()
        run_app()

    ttk.Button(frm, text="Start Worker!", command=start_worker).grid(column=1, row=11)
    ttk.Button(frm, text="Quit", command=root.destroy).grid(column=1, row=12)

    root.mainloop()

def save_settings(forge_model_dir, lora_model_dir, server, token, aria2, out_dir, dl_lora, dl_checkpoints, worker):
    global forge_model_directory, lora_model_directory, server_url, civit_api_token, aria2_path, output_directory, auto_dl_lora, auto_dl_checkpoints, accepted_job_types, worker_name
    config = {
        "forge_model_directory": forge_model_dir,
        "lora_model_directory": lora_model_dir,
        "server_url": server,
        "civit_api_token": token,
        "aria2_path": aria2,
        "output_directory": out_dir,
        "auto_dl_lora": dl_lora,
        "auto_dl_checkpoints": dl_checkpoints,
        "accepted_job_types": accepted_job_types,
        "worker_name": worker
    }
    forge_model_directory = forge_model_dir
    lora_model_directory = lora_model_dir
    server_url = server
    civit_api_token = token
    aria2_path = aria2
    output_directory = out_dir
    worker_name = worker
    try:
        with open("worker_config.json", "w") as config_file:
            json.dump(config, config_file, indent=4)
        messagebox.showinfo("Success", "Settings saved successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save settings:\n{e}")
#HASH CALC AND CHECKING BELOW
#Load list of all hashes
def load_hashes(type):
    if os.path.exists(checkpoint_db) and type == "checkpoints":
        with open(checkpoint_db, 'r') as file:
            return json.load(file).get('hashes', [])
    elif os.path.exists(lora_db) and type == "loras":
        with open(lora_db, 'r') as file:
            return json.load(file).get('hashes', [])
    return []
#Convert hash to local systems file name
def hash_to_model_name(hash,type):
    hashes = load_hashes(type)
    for file_hash in hashes:
        if file_hash[0] == hash:
            print(file_hash[1])
            return(file_hash[1])
#Save all hashes with respective file name in 'checkpoints.json'
def save_hashes(hashes, type):
    if type == "checkpoints":
        with open(checkpoint_db, 'w') as file:
            json.dump({'hashes': hashes}, file, indent=4)
    else:
        with open(lora_db, 'w') as file:
            json.dump({'hashes': hashes}, file, indent=4)

#Calculate hashes of all files in given directory
def hash_file(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()
#Add hash and filename to list if its not already there
def add_file_hash_if_new(filepath,type):
    hashes = load_hashes(type)
    
    filename = os.path.basename(filepath)

    # Check if hash already exists in any [hash, filename] entry
    if not any(entry[1] == filename for entry in hashes):
        file_hash = hash_file(filepath)
        hashes.append([file_hash, filename])
        save_hashes(hashes,type)
        print(f"✅ Added new hash for {filename}")
    else:
        print(f"⚠️ Hash for {filename} already exists.")
def check_checkpoint_hashes():
    global hashes
    hashes = {}
    print("-- Checking Checkpoint Hashes --")
    # 🔍 Check all safetensors in the directory
    for root, dirs, files in os.walk(forge_model_directory):
        for fname in files:
            if fname.endswith('.safetensors'):
                full_path = os.path.join(root, fname)
                add_file_hash_if_new(full_path,"checkpoints")
    print("-- Checkpoint Hashing Completed! --")


# 🔍 Check all safetensors in the directory
def check_lora_hashes():
    global hashes
    hashes = {}
    print("-- Checking Lora Hashes --")

    for root, dirs, files in os.walk(lora_model_directory):
        for fname in files:
            if fname.endswith('.safetensors'):
                full_path = os.path.join(root, fname)
                add_file_hash_if_new(full_path,"loras")
    print("-- Lora Hashing Completed! --")


def lora_conversion(prompt):
    lora_hashes_list = load_hashes("loras")
    lora_matches = re.findall(r'<(.*?)>', prompt)
    lora_hash = ""
    for lora in lora_matches:
        lora_string = lora.split(":")
        lora_hash = lora_string[1]
        for lora_hash_in_dict in lora_hashes_list:
            if lora_hash.lower() == lora_hash_in_dict[0]: 
                prompt = prompt.replace(lora_hash,lora_hash_in_dict[1].replace(".safetensors",""))
            else:
                continue
    return(prompt)


def worker_login():
    global worker_id
    global accepted_job_types
    global auto_dl_lora
    global auto_dl_checkpoints
    global worker_name
    print("🛜🛜 Attempting to login... NOTE: This may take a bit if you have a lot of models installed! 🛜🛜")
    checkpoint_hashes_list = []
    lora_hashes_list = []
    raw_checkpoint_hashes = load_hashes("checkpoints")
    raw_lora_hashes = load_hashes("loras")
    try:
        with open("worker_auth.json", "rb") as worker_auth_file:
            worker_id = json.load(worker_auth_file).get('worker_id')
            try:
                for hash in raw_checkpoint_hashes:
                    checkpoint_hashes_list.append(hash[0])
                for hash in raw_lora_hashes:
                    lora_hashes_list.append(hash[0])
                resp = requests.get(f'{server_url}/api/init', json = {'worker_id': worker_id, 'checkpoints':checkpoint_hashes_list, 'acceptable_job_types':accepted_job_types, 'loras':lora_hashes_list, 'dl_lora':auto_dl_lora.get(), 'dl_checkpoint': auto_dl_checkpoints.get(), 'worker_name': worker_name}, timeout=5)
                print(f"Worker authenticated successfully! Acceptable Job Types: {accepted_job_types}")
                return
            except:
                print("Unable to connect to the server, retrying in 15 seconds...")
                time.sleep(15)
                worker_login()
    except:
        worker_id = None
        resp = requests.get(f'{server_url}/api/init', json = {'worker_id': "N/A", 'checkpoints':checkpoint_hashes_list, 'acceptable_job_types':accepted_job_types, 'loras':lora_hashes_list, 'dl_lora':auto_dl_lora.get(), 'dl_checkpoint': auto_dl_checkpoints.get(), 'worker_name': worker_name}, timeout=5)
        worker_id = resp.json().get("worker_id")
        json.dump({'worker_id': worker_id}, open('worker_auth.json', 'w'), indent=4)
        print(f"New worker registered, worker ID: {worker_id}")
        worker_login()


# JOB PROCESSING BELOW
def get_job():
    global worker_id
    global forge_model_directory
    global aria2_path
    today = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    try:
        resp = requests.get(f'{server_url}/api/get-job', json = {'worker_id':worker_id}, timeout=5)
        #Download missing checkpoint if job is found to require it
        if resp.status_code == 201:
            job = resp.json()
            print(job)
            job_id     = job['job_id']
            channel_id = job['channel']
            requester = job['requester']
            download_hash = resp.json()['status'].split(': ')[1]
            print(download_hash)
            '''try:'''
            resp1 = requests.get(f'https://civitai.com/api/v1/model-versions/by-hash/{download_hash}')
            resp_dict = resp1.json()
            print(resp_dict)
            style_id = resp_dict['id']
            url = f"https://civitai.com/api/download/models/{style_id}?token={civit_api_token}"
            os.system(f"{aria2_path}/aria2c.exe -d {forge_model_directory} {url} --conf {aria2_path}/config.cfg --allow-overwrite=true")
            resp2 = requests.post('http://127.0.0.1:7860/sdapi/v1/refresh-checkpoints')
            check_checkpoint_hashes()
            time.sleep(5)
            worker_login()
            return
            '''except:

                    resp = requests.get(f'{server_url}/api/download-fail', json = {'worker_id': worker_id, 'job_id': job_id, 'requester': requester, 'channel': channel_id})
            return'''
        if resp.status_code == 202:
            job = resp.json()
            job_id     = job['job_id']
            channel_id = job['channel']
            requester = job['requester']
            download_hashes = job['download_loras']
            print(download_hashes)
            for missing_hash in download_hashes:
                print(f'Downloading model: {missing_hash}')
                try:
                    resp1 = requests.get(f'https://civitai.com/api/v1/model-versions/by-hash/{missing_hash.replace(" ","")}')
                    resp_dict = resp1.json()
                    print(resp_dict)
                    style_id = resp_dict['id']
                    url = f"https://civitai.com/api/download/models/{style_id}?token={civit_api_token}"
                    os.system(f"{aria2_path}/aria2c.exe -d {lora_model_directory} {url} --conf {aria2_path}/config.cfg --allow-overwrite=true")
                    resp = requests.post('http://127.0.0.1:7860/sdapi/v1/refresh-loras')
                    check_lora_hashes()
                    print("Relogging to update available models and job types!")
                    worker_login()
                except:
                    resp2 = requests.get(f'{server_url}/api/download-fail', json = {'worker_id': worker_id, 'job_id': job_id, 'requester': requester, 'channel': channel_id})

            return
        if resp.status_code == 500:
            print(f"[{today}] || No accceptable job found. Status:", resp.status_code)
            return
        elif resp.status_code != 200:
            print("unable to connect to server, relogging in 15 seconds...")
            pass
        """try:"""
        job = resp.json()          # now a dict!
        try:
            job_type   = job['request_type']
            job_id     = job['job_id']
            prompt     = job['requested_prompt']
            prompt = lora_conversion(prompt)
            print(prompt)
            channel_id = job['channel']
        except:
            print("recieved bad request, retrying in 15 seconds...")
        if job_type != "img2vid":
            model_hash = hash_to_model_name(job['model'],"checkpoints")
            steps      = job['steps']
            neg_prompt = job.get('negative_prompt', "")
            resolution = job.get('resolution')
            batch_size = job.get('batch_size')
            cfg_scale  = job.get('cfg_scale')
            sampler = job.get('sampler', "Euler")
            clip_skip = job.get('clip_skip', 2)
            face_fix = job.get('face_fix', "False")
            hires_fix = job.get('hires_fix', "False")
        image_link = job.get('image_link')   
        

        requester = job.get('requester')
        try:
            print(f"[{today}] || Job recieved: \nJob ID: {job_id}\nJob Type: {job_type}\nModel/Checkpoint: {model_hash}")
        except:
            print(f"[{today}] || Job recieved: \nJob ID: {job_id}\nJob Type: {job_type}\nModel/Checkpoint: None")
        if job_type in ["generate", "generate chat"]:
            if worker_type == "forge":
                generate_image(
                channel_id, prompt, model_hash, neg_prompt,
                resolution, job_type, batch_size,
                cfg_scale, steps, job_id, requester, sampler, clip_skip, face_fix, hires_fix
                )
            elif worker_type == "comfy":
                generate_image_comfy(
                channel_id, prompt, model_hash, neg_prompt,
                resolution, job_type, batch_size,
                cfg_scale, steps, job_id, requester, sampler, clip_skip, face_fix, hires_fix
                )
        elif job_type == "face fix":
            face_fix_post(
            image_link, channel_id, model_hash,
            prompt, job_id, requester
            )
        elif job_type == "upscale":
            upscale_image(image_link,channel_id,cfg_scale,job_id, requester)
        elif job_type == "img2img":
            if worker_type == "forge":
                img2imggen(image_link,channel_id,model_hash,prompt,neg_prompt,resolution,batch_size, job_id, requester)
            elif worker_type == "comfy":
                img2img_comfy(image_link,channel_id,model_hash,prompt,neg_prompt,resolution,batch_size, job_id, requester)
        elif job_type == "img2vid":
            cfg_scale = job.get('config_scale')
            img2vidgen(image_link,channel_id,prompt, job_id, requester, cfg_scale)
    except requests.RequestException as e:
        print(f"Unable to connect to server, relogging in 15 seconds...")
        worker_login()
    """except:
        print("No applicable jobs found")"""
    """except:
        print("Unable to connect to get job, if this continues to occur, please contact superdotaplaya!")"""


def submit_results(images,channel_id,requester,job_id,prompt,model):
    global worker_id
    url = f'{server_url}/api/upload?channel={channel_id}&job_id={job_id}'


    files = images

    try:
        response = requests.post(f'{server_url}/api/upload', files=files, data = {'worker_id':worker_id, 'channel':channel_id, 'job_id':job_id, 'requester':requester, 'prompt':prompt, 'model':model})
        print(response.json())
    except Exception as e:
        print(f"Failed to submit images: {e}")

    for _, (_, file, _) in images:
        file.close()

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
    global worker_main_loop
    if is_gpu_idle(gpu_id_setting) and gpu_idle_timer > 0:
        gpu_idle_timer -= polling_interval
        print("gpu time remaining until idle: " + str(gpu_idle_timer) + " seconds")
    elif gpu_idle_timer <= 0:
        get_job()

    worker_main_loop = threading.Timer(polling_interval, main_loop).start()

def get_newest_files(directory, count=1):
    # Get all files in the directory with their full paths
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # Sort files by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)
    
    # Return the specified number of newest files
    return files[:count]

def str_to_bool(s):
    return s.lower() in ['true', '1', 'yes', 'y']





def generate_image(channel_id,
                   prompt,
                   model,
                   neg_prompt,
                   resolution,
                   job_type,
                   batch_size,
                   cfg_scale,
                   steps,
                   job_id, requester, sampler,
                   clip_skip,
                   face_fix,
                   hires_fix):
    global output_directory
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
        "sampler_index": sampler,
        "alwayson_scripts": {
            "ADetailer": {
            "args": [
                str_to_bool(face_fix),
                str_to_bool(face_fix),
                {
                "ad_model": "face_yolov8s.pt",
                "ad_model_classes": "",
                "ad_tab_enable": True,
                "ad_prompt": prompt,
                "ad_negative_prompt": neg_prompt,
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
        "enable_hr": str_to_bool(hires_fix),
        "hr_upscaler": "R-ESRGAN 4x+",
        "hr_checkpoint_name": "Use same checkpoint",
        "hr_scale": 1.5,
        "hr_prompt": "",
        "denoising_strength": 0.4,
        "hr_second_pass_steps": 15,
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
        "clip_skip": int(clip_skip),
    }

    # 4) Optionally set the model on the WebUI first
    opts_url = "http://127.0.0.1:7860/sdapi/v1/options"
    requests.post(opts_url, json={
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
    output_dir = fr"{output_directory}\\{today}"
    backslash_character = "\\"
    grid_dir = fr"{backslash_character.join(output_directory.split(backslash_character)[:3])}\txt2img-grids\{today}"
    print(grid_dir)
    images = []
    grid_files = get_newest_files(grid_dir,batch_size)
    for path in grid_files:
        os.remove(path)
    files = get_newest_files(output_dir, batch_size)  # your helper
    for path in files:
        f = open(path, "rb")
        images.append(("images", (os.path.basename(path), f, "image/png")))

    # 7) Push results back to your API
    submit_results(images, channel_id, requester,job_id, prompt, model)

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
    submit_results(files,channel,requester,job_id,prompt,"Face Fix")
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
        "upscaler_1": "R-ESRGAN 4x+ Anime6B",
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
    submit_results(files,channel,requester,job_id,"upscaled image","Upscale")
    os.remove("output.png")
def generate_image_comfy(channel_id, user_prompt, model_hash, neg_prompt,
            resolution, job_type, batch_size,
            cfg_scale, steps, job_id, requester, sampler, clip_skip, face_fix, hires_fix):
         # Get prompt_id for tracking the execution
    if sampler.lower() == "dpm2":
        sampler = "dpm_2"
    client_id = str(uuid.uuid4())  # Generate a unique client ID
    seed = random.randint(1, 999999999999999)
    loras = re.findall(r"<.*?>", user_prompt)
    neg_loras = re.findall(r"<.*?>", neg_prompt)
    completed_loras = []
    for lora in loras:
        lora = lora.replace("<","").replace(">","").split(":")
        completed_loras.append([True,lora[1],lora[2]])
    for lora in neg_loras:
        lora = lora.replace("<","").replace(">","").split(":")
        completed_loras.append([False,lora[1],lora[2]])
    user_prompt = re.sub(r"<.*?>", "", user_prompt)
    neg_prompt = re.sub(r"<.*?>", "", neg_prompt)
    i = 1
    prompt = json.load(open("Klover_SDXL.json", "r", encoding="utf-8"))
    for lora in completed_loras :
        prompt['63']['inputs'][f'lora_{i}'] = {'on': True, 'lora': f'{lora[1]}.safetensors', 'strength': float(lora[2])}
        i+= 1
    prompt['5']['inputs']['width'] = int(resolution.lower().split('x')[0])
    prompt['5']['inputs']['height'] = int(resolution.lower().split('x')[1])
    prompt['5']["inputs"]["batch_size"] = batch_size
    prompt['96']['inputs']['text'] = user_prompt
    prompt['97']['inputs']['text'] = neg_prompt
    prompt['4']['inputs']['ckpt_name'] = model_hash
    prompt['98']['inputs']['steps'] = steps
    prompt['98']['inputs']['sampler_name'] = sampler.lower()
    prompt['98']['inputs']['noise_seed'] = seed
    prompt['98']['inputs']['cfg'] = cfg_scale
    prompt['76']['inputs']['seed'] = seed
    prompt['76']['inputs']['cfg'] = cfg_scale
    prompt['56']['inputs']['seed'] = seed
    prompt['56']['inputs']['cfg'] = cfg_scale
    print(prompt)
    #set the text prompt for our positive CLIPTextEncode
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = request.Request(f"http://127.0.0.1:8188/prompt", data=data)
    prompt_id = json.loads(request.urlopen(req).read())['prompt_id']
    ws = websocket.WebSocket()
    ws.connect(f"ws://127.0.0.1:8188/ws?clientId={client_id}")
    # Ensure all referenced node IDs exist and are correct
    # For example, check that all "model", "positive", "negative", "vae", "image" references point to valid node IDs
    # If you change the image or prompt, make sure downstream nodes are updated accordingly



    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    today = datetime.now().strftime("%Y-%m-%d")
                    output_dir = rf"C:\Users\main\Documents\ComfyUI\output"

                    images = []
                    files = get_newest_files(output_dir, batch_size)  # your helper
                    for path in files:
                        try:
                            f = open(path, "rb")
                            for path in files:
                                f = open(path, "rb")
                                mime = "image/png" if path.lower().endswith(('.png')) else "application/octet-stream"
                                images.append(("images", (os.path.basename(path), f, mime)))
                                print(images)
                            # upload videos (submit_results expects same tuple shape as images)
                            submit_results(images, channel_id, requester, job_id, str(prompt) if not isinstance(prompt, str) else prompt, "Video")
                            ws.close()
                        except:
                            ws.close()
                            break
                    break  # Execution complete
        else:
            # Binary data (preview images)
            continue


def img2img_comfy(image_link, channel_id, checkpoint, user_prompt, negative_prompt, resolution, batch_size, job_id, requester):
    try:
        # Download image from image_link and save it into ComfyUI input folder
        print(image_link)
        resp = requests.get(image_link, stream=True, timeout=30)
        resp.raise_for_status()

        parsed = urlparse(image_link)
        ext = os.path.splitext(parsed.path)[1]
        if not ext:
            ctype = resp.headers.get("content-type", "")
            if "jpeg" in ctype or "jpg" in ctype:
                ext = ".jpg"
            elif "png" in ctype:
                ext = ".png"
            elif "webp" in ctype:
                ext = ".webp"
            else:
                ext = ".png"

        fname = f"img2img{ext}"
        input_dir = rf"C:\Users\main\Documents\ComfyUI\input"
        os.makedirs(input_dir, exist_ok=True)
        out_path = os.path.join(input_dir, fname)

        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                if chunk:
                    f.write(chunk)

        # Tell the prompt to use the downloaded file (ComfyUI expects the filename)
        print(f"Downloaded image to {out_path}")
    except Exception as e:
        print(f"Failed to download image: {e}")
    
    
    loras = re.findall(r"<.*?>", user_prompt)
    neg_loras = re.findall(r"<.*?>", negative_prompt)
    completed_loras = []
    for lora in loras:
        lora = lora.replace("<","").replace(">","").split(":")
        completed_loras.append([True,lora[1],lora[2]])
    for lora in neg_loras:
        lora = lora.replace("<").replace(">").split(":")
        completed_loras.append([False,lora[1],lora[2]])
    user_prompt = re.sub(r"<.*?>", "", user_prompt)
    negative_prompt = re.sub(r"<.*?>", "", negative_prompt)
    i = 1
    prompt = json.load(open("Klover_img2img.json", "r", encoding="utf-8"))
    for lora in completed_loras :
        prompt['33']['inputs'][f'lora_{i}'] = {'on': True, 'lora': f'{lora[1]}.safetensors', 'strength': float(lora[2])}
        i+=1
    
    
    seed = random.randint(1, 999999999999999)
    client_id = str(uuid.uuid4())  # Generate a unique client ID
    
    print(prompt)
    prompt["29"]["inputs"]["image"] = 'img2img.png'
    prompt['26']['inputs']['seed'] = seed
    prompt['23']['inputs']['ckpt_name'] = checkpoint
    prompt['24']['inputs']['text'] = user_prompt
    prompt['25']['inputs']['text'] = negative_prompt
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = request.Request(f"http://127.0.0.1:8188/prompt", data=data)
    prompt_id = json.loads(request.urlopen(req).read())['prompt_id']
    ws = websocket.WebSocket()
    ws.connect(f"ws://127.0.0.1:8188/ws?clientId={client_id}")

    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    today = datetime.now().strftime("%Y-%m-%d")
                    output_dir = rf"C:\Users\main\Documents\ComfyUI\output"

                    images = []
                    files = get_newest_files(output_dir, batch_size)  # your helper
                    for path in files:
                        try:
                            f = open(path, "rb")
                            for path in files:
                                f = open(path, "rb")
                                mime = "image/png" if path.lower().endswith(('.png')) else "application/octet-stream"
                                images.append(("images", (os.path.basename(path), f, mime)))
                                print(images)
                            # upload videos (submit_results expects same tuple shape as images)
                            submit_results(images, channel_id, requester, job_id, str(prompt) if not isinstance(prompt, str) else prompt, "Video")
                            ws.close()
                        except:
                            ws.close()
                            break
                    break  # Execution complete
        else:
            # Binary data (preview images)
            continue




def img2vidgen(image_link,channel_id,user_prompt, job_id, requester, vid_length):

        # Get prompt_id for tracking the execution
    client_id = str(uuid.uuid4())  # Generate a unique client ID

    prompt = json.load(open("E:\klover_for_all\Main\Klover_Integration\Klover_img2vid.json", "r", encoding="utf-8"))
    print(prompt)
    #set the text prompt for our positive CLIPTextEncode
    prompt["88"]["inputs"]["value"] = user_prompt
    
    try:
        # Download image from image_link and save it into ComfyUI input folder
        print(image_link)
        resp = requests.get(image_link, stream=True, timeout=30)
        resp.raise_for_status()

        parsed = urlparse(image_link)
        ext = os.path.splitext(parsed.path)[1]
        if not ext:
            ctype = resp.headers.get("content-type", "")
            if "jpeg" in ctype or "jpg" in ctype:
                ext = ".jpg"
            elif "png" in ctype:
                ext = ".png"
            elif "webp" in ctype:
                ext = ".webp"
            else:
                ext = ".png"

        fname = f"img2vid{ext}"
        input_dir = rf"C:\Users\main\Documents\ComfyUI\input"
        os.makedirs(input_dir, exist_ok=True)
        out_path = os.path.join(input_dir, fname)

        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                if chunk:
                    f.write(chunk)

        # Tell the prompt to use the downloaded file (ComfyUI expects the filename)
        prompt["52"]["inputs"]["image"] = fname
        print(f"Downloaded image to {out_path}")
    except Exception as e:
        print(f"Failed to download image: {e}")
    try:
        prompt["52"]["inputs"]["image"] = 'img2vid.png'
        today = datetime.now().strftime("%Y-%m-%d")
        prompt["77"]["inputs"]["filename_prefix"] = f'Video/{today}/140806'
        prompt["50"]["inputs"]["length"] = vid_length*16
        p = {"prompt": prompt, "client_id": client_id}
        data = json.dumps(p).encode('utf-8')
        req = request.Request(f"http://127.0.0.1:8188/prompt", data=data)
        prompt_id = json.loads(request.urlopen(req).read())['prompt_id']
        ws = websocket.WebSocket()
        ws.connect(f"ws://127.0.0.1:8188/ws?clientId={client_id}")
        # Ensure all referenced node IDs exist and are correct
        # For example, check that all "model", "positive", "negative", "vae", "image" references point to valid node IDs
        # If you change the image or prompt, make sure downstream nodes are updated accordingly



        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        today = datetime.now().strftime("%Y-%m-%d")
                        output_dir = rf"C:\Users\main\Documents\ComfyUI\output\Video\{today}"

                        videos = []
                        files = get_newest_files(output_dir, 1)  # your helper
                        for path in files:
                            try:
                                f = open(path, "rb")
                                for path in files:
                                    f = open(path, "rb")
                                    mime = "video/mp4" if path.lower().endswith(('.mp4', '.mov', '.mkv', '.webm')) else "application/octet-stream"
                                    videos.append(("images", (os.path.basename(path), f, mime)))
                                    print(videos)
                                # upload videos (submit_results expects same tuple shape as images)
                                submit_results([videos[0]], channel_id, requester, job_id, str(prompt) if not isinstance(prompt, str) else prompt, "Video")
                                ws.close()
                            except:
                                break
                        break  # Execution complete
            else:
                # Binary data (preview images)
                continue
    except:
        print("An error occured, relogging in 15 seconds...")
        time.sleep(15)
        worker_login()


    
def img2imggen(image_link, channel_id, checkpoint, prompt, negative_prompt, resolution, batch_size, job_id, requester):
    payload = {
        "init_images": [image_link],
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "sampler_name": "Euler",
        "sd_model_checkpoint": checkpoint,
        "width": int(resolution.lower().split("x")[0]),
        "height": int(resolution.lower().split("x")[1]),
        "batch_size": batch_size,
        "steps": 25,
        "cfg_scale": 5,
        "denoising_strength": 0.6,
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
        },
        "scheduler": "Automatic",
        "enable_hr": True,
        "hr_upscaler": "R-ESRGAN 4x+",
        "hr_checkpoint_name": "Use same checkpoint",
        "hr_cfg": 1,
        "hr_scale": 1.5,
        }
    response = requests.post(url=f'http://127.0.0.1:7860/sdapi/v1/img2img', json=payload)

    r = response.json()
    print(r)
    image = Image.open(io.BytesIO(base64.b64decode(r['images'][0])))
    image.save('output.png')
    files = [('images', ('output.png', open('output.png', 'rb'), 'image/png'))]
    submit_results(files,channel_id,requester,job_id,prompt,"Img2Img")


load_user_interface()
