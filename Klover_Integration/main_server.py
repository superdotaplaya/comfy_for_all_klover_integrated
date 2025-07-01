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

from flask import Flask, request, jsonify

app = Flask(__name__)
bot = discord.Bot()
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
    if request.get_json().get("worker_id") != "N/A":
        worker_id_list = load_worker_ids()
        print(worker_id_list)
        if authenticate_worker(request.get_json().get("worker_id")):
            print("Worker id in list")
        return(jsonify({"worker_id": request.get_json().get("worker_id"), "created": False}))
    elif request.get_json().get("worker_id") == "N/A":
        worker_id = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(24))
        worker_id_list = load_worker_ids()
        worker_id_list.append([worker_id])
        json.dump({'workers': worker_id_list}, open('worker_list.json', 'w'), indent=4)
        return(jsonify({"worker_id": worker_id,"created": True}))
    


@app.route('/api/get-job', methods=['GET'])

def get_job():
    print(authenticate_worker(request.get_json().get("worker_id")))
    if authenticate_worker(request.get_json().get("worker_id")):
        i = 0
        mydb = mysql.connector.connect(
            host=config.db_host,
            user="root",
            password=config.db_pass,
            database="user_requests"
            )
        mycursor = mydb.cursor()
        data = request.get_json()
        checkpoint_list = data.get("checkpoints", [])
        mycursor.execute("""
            SELECT job_id, requested_prompt, steps, model, channel,
                request_type, image_link, requested_at, started_at,
                negative_prompt, resolution, batch_size, config_scale
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
            neg_prompt, resolution, batch_size, cfg_scale) = job

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
                    "config_scale": int(cfg_scale)
                }
                print(result)
                return jsonify(result)
            else:
                i+=1
        return jsonify({'status': 'No job found'}), 404
    else:
        return(jsonify({'status':'Could not authenticate worker!'}))
    
async def upload_to_discord(files, channel):
    print("sending to discord")
    await bot.get_channel(1366919874194178108).send(files=files)
@app.route('/api/upload', methods=['POST'])
def upload_images():
    if authenticate_worker(request.form.get("worker_id")):
        mydb = mysql.connector.connect(
            host=config.db_host,
            user="root",
            password=config.db_pass,
            database="user_requests"
            )
        mycursor = mydb.cursor()
        channel = request.args.get('channel')
        job_id = request.args.get('job_id')
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
        asyncio.run_coroutine_threadsafe(upload_to_discord(upload_files,channel),bot.loop)
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
async def img2img(ctx, image: Option(discord.Attachment, "What image are you going to face fix?"), prompt: Option(str, "What prompt are you going to use?"), negative_prompt: Option(str, "What negative prompt are you going to use?"), checkpoint: Option(str, "What checkpoint are you going to use?"), resolution: Option(str, "What size do you want the image to be? (WxH) MAXIMUM is 1536x1536", required = True)):
    print("img2img")
@bot.slash_command(description = "Get a list of available checkpoints and LorAs that Klover has access to")
async def models(ctx):
    print("sharing models")

@bot.slash_command(description = "Download a style for use with the bot")
async def facefix(ctx, image: Option(discord.Attachment, "What image are you going to face fix?"), prompt: Option(str, "What prompt do you want to run for face fix?"), checkpoint: Option(str, "What checkpoint do you want to sue for face fix? (check /models for a full list!)")):
    print("face fix")
# 2) run the bot
bot.run(config.discord_token)