import requests
import json
import os,time,sys,threading
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import discord
from discord.ext import commands, tasks
from discord.commands import Option
import ngrok
import asyncio
import random,string
import pygsheets
from flask import Flask, request, jsonify

app = Flask(__name__)
bot = discord.Bot()

worker_hashes = {}
lora_hashes = {}
worker_job_types = {}

gc = pygsheets.authorize(service_file='credentials.json')
sh = gc.open('Klover_Styles')
if not os.path.isfile("config.py"):
    sys.exit("'config.py' not found! Please add it and try again.")
else:
    import config

def authenticate_worker(worker_id):
    worker_list = load_worker_ids()
    for worker in worker_list:
        if worker[0] == worker_id:
            return(True)
    return(False)
def load_worker_ids():
    if os.path.exists('worker_list.json'):
        with open('worker_list.json', 'r') as file:
            return json.load(file).get('workers', [])
    return []

@app.route('/api/init', methods=['GET'])
def login():
    global worker_hashes
    global worker_job_types
    global lora_hashes
    worker_id = request.get_json().get("worker_id")
    if  worker_id != "N/A":
        worker_id_list = load_worker_ids()
        print(worker_id_list)
        if authenticate_worker(worker_id):
            print("Worker id in list")
            worker_hashes[worker_id] = request.get_json().get("checkpoints")
            lora_hashes[worker_id] = request.get_json().get("loras")
            worker_job_types[worker_id] = request.get_json().get("acceptable_job_types")
            print(f'WORKER HASHES: {worker_hashes}' )
            print(f"LORA HASHES: {lora_hashes}" )



        #update checkpoint list to list available models
            
            for checkpoint_hash in worker_hashes[worker_id]:
                print(checkpoint_hash)
                logged_checkpoints = []
                try:
                    wks = sh.worksheet('title','New Checkpoints')
                    logged_checkpoints = wks.get_values_batch(['D2:D10000'])
                    filtered_list = [x[0] for x in logged_checkpoints[0] if x[0] != ""]
                    print(filtered_list)
                    if checkpoint_hash not in filtered_list:
                        hash_request = requests.get(f"https://civitai.com/api/v1/model-versions/by-hash/{checkpoint_hash}")
                        resp_dict = hash_request.json()
                        wks = sh.worksheet('title','New Checkpoints')
                        style_id = resp_dict['id']
                        model_id = resp_dict['modelId']
                        preview_image_cell = [f'''=IMAGE("{resp_dict['images'][0]['url']}")''']
                        base_model = [resp_dict['baseModel']]
                        if resp_dict['images'][0]['nsfwLevel'] >= 5:
                            nsfw = ["NSFW"]
                        else:
                            nsfw = ["SFW"]
                        model_link = [f'''=HYPERLINK("https://www.civitai.com/models/{str(model_id)}?modelVersionId={style_id}","{resp_dict['model']['name']}")''']

                        wks.append_table(values=[preview_image_cell, model_link,base_model,[checkpoint_hash],nsfw], start="A2", end="D10000", dimension="COLUMNS", overwrite=False)
                except:
                    continue


            for lora_hash in lora_hashes[worker_id]:
                print(lora_hash)
                try:
                    wks = sh.worksheet('title','New Loras')
                    logged_loras = wks.get_values_batch(['F2:F10000'])
                    filtered_list = [x[0] for x in logged_loras[0] if x[0] != ""]
                    if lora_hash not in filtered_list:
                        hash_request = requests.get(f"https://civitai.com/api/v1/model-versions/by-hash/{lora_hash}")
                        resp_dict = hash_request.json()                 
                        style_id = resp_dict['id']
                        model_id = resp_dict['modelId']
                        preview_image_cell = [f'''=IMAGE("{resp_dict['images'][0]['url']}")''']
                        base_model = [resp_dict['baseModel']]
                        if resp_dict['images'][0]['nsfwLevel'] >= 5:
                            nsfw = ["NSFW"]
                        else:
                            nsfw = ["SFW"]
                        try:
                            trigger_words = resp_dict['trainedWords'][0]
                        except:
                            trigger_words = ""
                        filtered_list = wks.get_values_batch(['D2:D10000'])
                        model_link = [f'''=HYPERLINK("https://www.civitai.com/models/{str(model_id)}?modelVersionId={style_id}","{resp_dict['model']['name']}")''']
                        activation = [f"<lora:{lora_hash}:1> {trigger_words}"]
                        wks.append_table(values=[preview_image_cell,model_link, activation, base_model,nsfw,[lora_hash]], start="A2", end="F10000", dimension="COLUMNS", overwrite=False)
                except:
                    continue





        return(jsonify({"worker_id": worker_id, "created": False}))
    elif request.get_json().get("worker_id") == "N/A":
        worker_id = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(24))
        worker_id_list = load_worker_ids()
        worker_id_list.append([worker_id])
        worker_hashes[worker_id] = request.get_json().get("checkpoints")
        worker_job_types[worker_id] = request.get_json().get("acceptable_job_types")
        json.dump({'workers': worker_id_list}, open('worker_list.json', 'w'), indent=4)
        return(jsonify({"worker_id": worker_id,"created": True}))
    


@app.route('/api/get-job', methods=['GET'])

def get_job():
    global worker_hashes
    global worker_job_types
    worker_id = request.get_json().get("worker_id")
    if authenticate_worker(worker_id):
        i = 0
        mydb = mysql.connector.connect(
            host=config.db_host,
            user="root",
            password=config.db_pass,
            database="user_requests"
            )
        mycursor = mydb.cursor()
        data = request.get_json()
        checkpoint_list = worker_hashes[worker_id]
        mycursor.execute("""
            SELECT job_id, requested_prompt, steps, model, channel,
                request_type, image_link, requested_at, started_at,
                negative_prompt, resolution, batch_size, config_scale,
                requester
            FROM requests
            WHERE started_at IS NULL
            ORDER BY job_id ASC
        """)
        job_list = mycursor.fetchall()
        for job in job_list:
            print(job)
            if not job:
                return jsonify({"error":"no jobs"}), 404

            # unpack into named vars so you never mix up the order later:
            (job_id, prompt, steps, model_hash, channel_id,
            job_type, image_link, created_at, started_at,
            neg_prompt, resolution, batch_size, cfg_scale, requester) = job
        print(worker_job_types[worker_id])
        if job_type in worker_job_types[worker_id]:
            now = datetime.now()
            if model_hash in checkpoint_list:
                mycursor.execute(
                    "UPDATE requests SET started_at=%s WHERE job_id=%s",
                    (now, job_id)
                )
                mydb.commit()

                # build a dict to jsonify
                result = {
                    "job_id": job_id,
                    "requested_prompt": prompt,
                    "steps": steps,
                    "model": str(model_hash),
                    "channel": channel_id,
                    "request_type": job_type,
                    "image_link": image_link,
                    "requested_at": created_at.isoformat() if created_at else None,
                    "started_at": now.isoformat(),
                    "negative_prompt": neg_prompt,
                    "resolution": resolution,
                    "batch_size": batch_size,
                    "config_scale": int(cfg_scale),
                    "requester": requester
                }
                print(result)
                return jsonify(result)
            elif job_type == "upscale":
                mycursor.execute(
                    "UPDATE requests SET started_at=%s WHERE job_id=%s",
                    (now, job_id)
                )
                mydb.commit()

                # build a dict to jsonify
                result = {
                    "job_id": job_id,
                    "requested_prompt": prompt,
                    "steps": steps,
                    "model": str(model_hash),
                    "channel": channel_id,
                    "request_type": job_type,
                    "image_link": image_link,
                    "requested_at": created_at.isoformat() if created_at else None,
                    "started_at": now.isoformat(),
                    "negative_prompt": neg_prompt,
                    "resolution": resolution,
                    "batch_size": batch_size,
                    "config_scale": float(cfg_scale),
                    "requester": requester
                }
                print(result)
                return jsonify(result)
        
            else:
                i+=1
        else:
            i += 1
        return jsonify({'status': 'No job found'}), 404
    else:
        return(jsonify({'status':'Could not authenticate worker!'}))
    
async def upload_to_discord(files, channel, requester, job_id):
    print("sending to discord")
    await bot.get_channel(1366919874194178108).send(content=f"<@{requester}>\nJob ID: {job_id}",files=files)
@app.route('/api/upload', methods=['POST'])
def upload_images():
    if authenticate_worker(request.form.get("worker_id")):
        requester = request.form.get("requester")
        mydb = mysql.connector.connect(
            host=config.db_host,
            user="root",
            password=config.db_pass,
            database="user_requests"
            )
        mycursor = mydb.cursor()
        channel = request.form.get('channel')
        job_id = request.form.get('job_id')
        if 'images' not in request.files:
            return jsonify({"error": "No 'images' field in request"}), 400

        files = request.files.getlist('images')
        saved_files = []
        upload_files = []
        i = 0
        for file in files:
            if file.filename:
                file.save(f'uploads/{channel}_{i}.png')
                saved_files.append(file.filename)
                upload_files.append(discord.File(f'uploads/{channel}_{i}.png'))
                i += 1
        
        mycursor.execute("""DELETE FROM requests WHERE job_id = %s""",(int(job_id),))
        mydb.commit()
        asyncio.run_coroutine_threadsafe(upload_to_discord(upload_files,channel, requester,job_id),bot.loop)
        return jsonify({"status": "success", "message": "Images uploaded"})
    else:
        return(jsonify({'status':'Could not authenticate worker!'}))
@bot.event
async def on_ready():
    print(f"🤖 Bot logged in as {bot.user}")
def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False)
    

threading.Thread(target=run_flask, daemon=True).start()

@bot.slash_command(description = "Generate an AI image")
async def gen(ctx, prompt: Option(str, "What prompt are you going to use?"), negative_prompt: Option(str, "What negative prompt are you going to use?"), checkpoint_hash: Option(str, "What checkpoint are you going to use?"), batch_size: Option(int,choices=[1,2,3,4,5]), resolution: Option(str, "What size do you want the image to be? (WxH) MAXIMUM is 1536x1536", required = False), config_scale: Option(int, "What CFG scale do you want to use? (Default is 5 for SD and SDXL, 3.5 for FLUX)", default = 5,choices=[1,2,3,4,5,6,7,8,9,10],required=False), steps: Option(int, "How many steps do you want to use? (Default is 25, Max is 35)", default = 25,required=False)):
    print("added to queue")

"""@bot.slash_command(description = "Generate an AI QR code image")
async def qrcode(ctx, prompt: Option(str, "What prompt are you going to use?"), qr_code: discord.Attachment):
    print("qr code added to queue")"""

"""@bot.slash_command(description = "Remove the background from an image")
async def bgremove(ctx, image: discord.Attachment):
    print("background removal queued")"""

@bot.slash_command(description = "Download a style for use with the bot")
async def aidownload(ctx, version_id: Option(str, "Version ID of the checkpoint/lora you would like to download"), type: Option(str, choices=['Checkpoint', 'LorA', "Wan I2V Lora"])):
    print("downloading")

@bot.slash_command(description = "Generate an AI image")
async def img2img(ctx, image: Option(discord.Attachment, "What image are you going to face fix?"), prompt: Option(str, "What prompt are you going to use?"), negative_prompt: Option(str, "What negative prompt are you going to use?"), checkpoint_hash: Option(str, "What checkpoint are you going to use?"), resolution: Option(str, "What size do you want the image to be? (WxH) MAXIMUM is 1536x1536", required = True)):
    print("img2img")
@bot.slash_command(description = "Get a list of available checkpoints and LorAs that Klover has access to")
async def models(ctx):
    print("sharing models")

@bot.slash_command(description = "Download a style for use with the bot")
async def facefix(ctx, image: Option(discord.Attachment, "What image are you going to face fix?"), prompt: Option(str, "What prompt do you want to run for face fix?"), checkpoint_hash: Option(str, "What checkpoint do you want to sue for face fix? (check /models for a full list!)")):
    print("face fix")

@bot.slash_command(description = "Download a style for use with the bot")
async def upscale(ctx, image: Option(discord.Attachment, "What image are you going to upscale?"), upscale_by: Option(float, choices=[1.5, 2, 2.5, 3, 3.5, 4])):
    print("upscale")
# 2) run the bot
bot.run(config.discord_token)