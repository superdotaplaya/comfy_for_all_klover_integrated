import requests
import json
import os,time,sys,threading
import pymysql.cursors
from datetime import datetime
import discord
from discord.ext import commands, tasks
from discord.commands import Option
import asyncio
import random,string
import pygsheets
from flask import Flask, request, jsonify
import re

app = Flask(__name__)
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
bot = discord.Bot(loop=loop)

worker_hashes = {}
lora_hashes = {}
download_handling = {}
worker_job_types = {}

gc = pygsheets.authorize(service_file='credentials.json')
sh = gc.open('Klover_Styles')
if not os.path.isfile("config.py"):
    sys.exit("'config.py' not found! Please add it and try again.")
else:
    import config
from werkzeug.utils import secure_filename

def authenticate_worker(worker_id):
    try:
        with open('worker_list.json', 'r') as f:
            data = json.load(f)
            workers = data.get("workers", {})
    except (FileNotFoundError, json.JSONDecodeError):
        workers = {}

    # workers is a dict keyed by worker_id
    return worker_id in workers


def check_for_loras(worker_id, prompt):
    loras_not_found = []
    global lora_hashes
    worker_loras = lora_hashes[worker_id]
    lora_matches = re.findall(r'<(.*?)>', prompt)
    print(lora_matches)
    for lora in lora_matches:
        lora_string = lora.split(":")
        lora_hash = lora_string[1].lower().strip()
        if lora_hash.lower() not in worker_loras:
            print(f"lora not found: {lora_hash.lower()}")
            loras_not_found.append(lora_hash.lower().strip())
    if len(loras_not_found) == 0:
        return(True, False)
    else:
        return(False, loras_not_found)
        

@app.route('/api/init', methods=['GET'])
def login():
    global worker_hashes
    global worker_job_types
    global lora_hashes
    global download_handling

    data = request.get_json()
    worker_id = data.get("worker_id")
    worker_name = data.get("worker_name")

    # Load existing workers dictionary
    try:
        with open('worker_list.json', 'r') as f:
            worker_dict = json.load(f).get("workers", {})
    except (FileNotFoundError, json.JSONDecodeError):
        worker_dict = {}

    if worker_id == "N/A":
        worker_id = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(24))
        worker_dict[worker_id] = {"worker_name": worker_name}

        worker_hashes[worker_id] = data.get("checkpoints")
        worker_job_types[worker_id] = data.get("acceptable_job_types")


        json.dump({'workers': worker_dict}, open('worker_list.json', 'w'), indent=4)

        return jsonify({"worker_id": worker_id, "worker_name": worker_name, "created": True})

    else:
        if authenticate_worker(worker_id):
            worker_dict[worker_id] = {"worker_name": worker_name}
            
            worker_hashes[worker_id] = data.get("checkpoints")
            lora_hashes[worker_id] = data.get("loras")
            download_handling[worker_id] = {
            'lora': data.get("dl_lora"),
            'checkpoints': data.get('dl_checkpoint')
            }
            worker_job_types[worker_id] = data.get("acceptable_job_types")

            thread = threading.Thread(target=update_hashes_sheet, args=(worker_hashes[worker_id], lora_hashes[worker_id]))
            thread.start()

            json.dump({'workers': worker_dict}, open('worker_list.json', 'w'), indent=4)

            return jsonify({"worker_id": worker_id, "worker_name": worker_name, "created": False})

    # Fallback if authentication fails
    return jsonify({"error": "Invalid worker_id or authentication failed"}), 400


def remove_bad_job(requester,job_id,reason,channel):
    message = ""
    if reason == "Download Failed":
        message = f"Job ID {job_id} has failed! \n\n failure reason: Unable to download resource, please double check the model hash, and all lora hashes and their tags! If you believe this is an error on my end, reach out to Shellerman!"
        mydb = pymysql.connect(
            host=config.db_host,
            user="root",
            password=config.db_pass,
            database="user_requests"
        )
        mycursor = mydb.cursor()
        mycursor.execute("""DELETE FROM requests WHERE job_id = %s""", (job_id,))
        mydb.commit()
        asyncio.run_coroutine_threadsafe(job_failed_discord_message(channel,requester,message),bot.loop)
    return("Done!")


def update_hashes_sheet(checkpoint_hashes,lora_hashes):
    model_types = ['concept','character','style','clothing','objects','animal','poses', 'background','vehicle']
    logged_checkpoints = []
    wks = sh.worksheet('title','New Checkpoints')
    logged_checkpoints = wks.get_values_batch(['D2:D10000'])
    filtered_list = [x[0] for x in logged_checkpoints[0] if x[0] != ""]
    for checkpoint_hash in checkpoint_hashes:
        try:
            if checkpoint_hash not in filtered_list:
                model_type = ["N/A"]
                hash_request = requests.get(f"https://civitai.com/api/v1/model-versions/by-hash/{checkpoint_hash}")
                resp_dict = hash_request.json()
                wks = sh.worksheet('title','New Checkpoints')
                style_id = resp_dict['id']
                model_id = resp_dict['modelId']
                preview_image_cell = []
                nsfw = ['nsfw']
                for image in resp_dict['images']:
                    if image['url'].endswith('.webp') == False and image['url'].endswith('.mp4') == False:
                        preview_image_cell = [f'''=IMAGE("{image['url']}")''']
                        break
                        if resp_dict['images'][0]['nsfwLevel'] >= 5:
                            nsfw = ["NSFW"]
                        else:
                            nsfw = ["SFW"]
                base_model = [resp_dict['baseModel']]
                
                model_link = [f'''=HYPERLINK("https://www.civitai.com/models/{str(model_id)}?modelVersionId={style_id}","{resp_dict['model']['name']}")''']

                wks.append_table(values=[preview_image_cell, model_link,base_model,[checkpoint_hash],nsfw], start="A2", end="D10000", dimension="COLUMNS", overwrite=False)
        except Exception as e:
            continue

    wks = sh.worksheet('title','New Loras')
    logged_loras = wks.get_values_batch(['G2:G10000'])
    filtered_list = [x[0] for x in logged_loras[0] if x[0] != ""]
    for lora_hash in lora_hashes:
        try:
            if lora_hash not in filtered_list:
                hash_request = requests.get(f"https://civitai.com/api/v1/model-versions/by-hash/{lora_hash}")
                resp_dict = hash_request.json()                 
                style_id = resp_dict['id']
                model_id = resp_dict['modelId']
                tags_request = requests.get(f"https://civitai.com/api/v1/models/{model_id}")
                model_tags = tags_request.json()['tags']
                for tag in model_tags:
                    if tag.lower() in model_types:
                        model_type = [tag.title()]
                        break
                preview_image_cell = []
                for image in resp_dict['images']:
                    if image['url'].endswith('.webp') == False and image['url'].endswith('.mp4') == False:
                        preview_image_cell = [f'''=IMAGE("{image['url']}")''']
                        break
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
                wks.append_table(values=[preview_image_cell,model_link, activation, base_model,model_type,nsfw,[lora_hash]], start="A2", end="G10000", dimension="COLUMNS", overwrite=False)
        except Exception as e:
            continue
        

@app.route('/api/download-fail', methods=['GET'])
def download_failed():
    data = request.get_json() or {}
    print(data)
    if authenticate_worker(data.get('worker_id')):
        reason = "Download Failed"
        requester = data.get("requester")
        job_id = data.get("job_id")
        channel = data.get("channel")
        remove_bad_job(requester,job_id,reason,channel)
        return jsonify({'status': 'Job removed!'}), 500

@app.route('/api/get-job', methods=['GET'])
def get_job():
    data = request.get_json() or {}
    worker_id = data.get("worker_id")

    if not authenticate_worker(worker_id):
        return jsonify({'status': 'Could not authenticate worker!'}), 401

    checkpoint_list = worker_hashes.get(worker_id, [])
    acceptable_types = worker_job_types.get(worker_id, [])
    download_cfg = download_handling.get(worker_id, {})

    mydb = pymysql.connect(
        host=config.db_host,
        user="root",
        password=config.db_pass,
        database="user_requests"
    )
    try:
        mycursor = mydb.cursor()
        mycursor.execute("""
            SELECT job_id, requested_prompt, steps, model, channel,
                   request_type, image_link, requested_at, started_at,
                   negative_prompt, resolution, batch_size, config_scale,
                   requester, attempts, sampler, clip_skip, face_fix, hires_fix
            FROM requests
            WHERE started_at IS NULL
            ORDER BY job_id ASC
        """)
        job_list = mycursor.fetchall()

        if not job_list:
            return jsonify({"error": "no jobs"}), 500

        for job in job_list:
            (job_id, prompt, steps, model_hash, channel_id,
             job_type, image_link, created_at, started_at,
             neg_prompt, resolution, batch_size, cfg_scale,
             requester, attempts, sampler, clip_skip, face_fix, hires_fix) = job

            # only consider jobs this worker can run
            if job_type not in acceptable_types:
                continue

            now = datetime.now()

            # If worker will auto-download checkpoints and the checkpoint isn't present, notify
            if download_cfg.get('checkpoints') is True and str(model_hash).lower() not in checkpoint_list and model_hash != None:
                return jsonify({'status':f'Model not found: {model_hash}', 'job_id': job_id, 'requester':requester, 'channel': channel_id}), 201

            # Check lorAs required by the prompt
            lora_check = check_for_loras(worker_id, prompt)
            if lora_check[0] is False:
                return jsonify({'status':f'Loras not found: {lora_check[1]}', 'job_id': job_id, 'requester':requester, 'channel': channel_id, 'download_loras':lora_check[1]}), 202

            # If model present and lorAs satisfied, claim the job
            if str(model_hash).lower() in checkpoint_list and lora_check[0] is True:
                mycursor.execute(
                    "UPDATE requests SET started_at=%s, attempts = attempts + 1 WHERE job_id=%s",
                    (now, job_id)
                )
                mydb.commit()
                result = {
                    "job_id": job_id,
                    "requested_prompt": prompt,
                    "steps": steps,
                    "model": str(model_hash).lower(),
                    "channel": channel_id,
                    "request_type": job_type,
                    "image_link": image_link,
                    "requested_at": created_at.isoformat() if created_at else None,
                    "started_at": now.isoformat(),
                    "negative_prompt": neg_prompt,
                    "resolution": resolution,
                    "batch_size": batch_size,
                    "config_scale": int(cfg_scale) if cfg_scale is not None else None,
                    "requester": requester,
                    "sampler": sampler,
                    "clip_skip": clip_skip,
                    "face_fix": face_fix,
                    "hires_fix": hires_fix
                }
                return jsonify(result)

            # Upscale jobs do not require checkpoint presence
            if job_type == "upscale":
                mycursor.execute(
                    "UPDATE requests SET started_at=%s WHERE job_id=%s",
                    (now, job_id)
                )
                mydb.commit()
                result = {
                    "job_id": job_id,
                    "requested_prompt": prompt,
                    "steps": steps,
                    "model": str(model_hash).lower(),
                    "channel": channel_id,
                    "request_type": job_type,
                    "image_link": image_link,
                    "requested_at": created_at.isoformat() if created_at else None,
                    "started_at": now.isoformat(),
                    "negative_prompt": neg_prompt,
                    "resolution": resolution,
                    "batch_size": batch_size,
                    "config_scale": float(cfg_scale) if cfg_scale is not None else None,
                    "requester": requester
                }
                return jsonify(result)
            if job_type == "img2vid":
                mycursor.execute(
                    "UPDATE requests SET started_at=%s WHERE job_id=%s",
                    (now, job_id)
                )
                mydb.commit()
                result = {
                    "job_id": job_id,
                    "requested_prompt": prompt,
                    "channel": channel_id,
                    "request_type": job_type,
                    "image_link": image_link,
                    "requester": requester,
                    "config_scale": cfg_scale # Using config_scale field to pass video length
                }
                return jsonify(result)
        return jsonify({'status': 'No job found'}), 500

    finally:
        try:
            mycursor.close()
        except Exception:
            pass
        try:
            mydb.close()
        except Exception:
            pass
    
async def upload_to_discord(files, channel, requester, job_id, prompt, model, worker_id):
    print("sending to discord")
    worker_name = ""
    with open('worker_list.json', 'r') as f:
        worker_dict = json.load(f).get("workers", {})
        worker_name = worker_dict.get(worker_id, {}).get("worker_name", worker_id)
    try:
        await bot.get_channel(int(channel)).send(content=f"<@{requester}>\nJob ID: {job_id}\n\nJob completed by: {worker_name}",files=files)
    except Exception as e:
        print(f"Error sending to Discord: {e}")

async def job_failed_discord_message(channel, requester, message):
    print(channel)
    print("sending failure message to discord")
    await bot.get_channel(int(channel)).send(content=f"<@{requester}>\n {message}")
    
@app.route('/api/upload', methods=['POST'])
def upload_images():
    if authenticate_worker(request.form.get("worker_id")):
        requester = request.form.get("requester")
        mydb = pymysql.connect(
            host=config.db_host,
            user="root",
            password=config.db_pass,
            database="user_requests"
        )
        mycursor = mydb.cursor()
        channel = request.form.get('channel')
        job_id = request.form.get('job_id')
        prompt = request.form.get('prompt')
        model = request.form.get('model') or "none:"
        if 'images' not in request.files:
            return jsonify({"error": "No 'images' field in request"}), 400

        files = request.files.getlist('images')
        saved_files = []
        upload_files = []
        os.makedirs('uploads', exist_ok=True)
        i = 0
        for file in files:
            if file and file.filename:
                original_name = secure_filename(file.filename)
                _, ext = os.path.splitext(original_name)
                save_path = f'uploads/{channel}_{i}{ext}' if ext else f'uploads/{channel}_{job_id}_{i}'
                file.save(save_path)
                saved_files.append(original_name)
                upload_files.append(discord.File(save_path))
                i += 1
                # Copy job entry into completed_jobs before deletion
        worker_id = request.form.get("worker_id")
        mycursor.execute("SELECT * FROM requests WHERE job_id = %s", (int(job_id),))
        job_row = mycursor.fetchone()

        if job_row:
            # Assuming completed_jobs has the same columns as requests plus 'worker'
            # You may need to adjust column names depending on your schema
            columns = [desc[0] for desc in mycursor.description]
            placeholders = ', '.join(['%s'] * (len(columns) + 1))  # +1 for worker
            insert_query = f"""
                INSERT INTO completed_jobs ({', '.join(columns)}, worker)
                VALUES ({placeholders})
            """
            mycursor.execute(insert_query, job_row + (worker_id,))
            mydb.commit()

        # Now remove from requests
        mycursor.execute("""DELETE FROM requests WHERE job_id = %s""", (int(job_id),))
        mydb.commit()
        asyncio.run_coroutine_threadsafe(upload_to_discord(upload_files, channel, requester, job_id, prompt, model, worker_id), bot.loop)

        return jsonify({"status": "success", "message": "Images uploaded", "files": saved_files})
    else:
        return jsonify({'status': 'Could not authenticate worker!'})
@bot.event
async def on_ready():
    print(f"🤖 Bot logged in as {bot.user}")
def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False)

def db_cleanup():
    print("DB cleanup running")
    mydb = pymysql.connect(
        host=config.db_host,
        user="root",
        password=config.db_pass,
        database="user_requests"
        )
    mycursor = mydb.cursor()
    query = """
        UPDATE requests
        SET started_at = NULL
        WHERE started_at <= NOW() - INTERVAL 10 MINUTE AND attempts <= 2
    """
    mycursor.execute(query)
    mydb.commit()
    select_query = "SELECT * FROM requests WHERE attempts >= 3"
    mycursor.execute(select_query)
    rows_to_delete = mycursor.fetchall()
    for rows in rows_to_delete:
        channel = rows[9]
        requester = rows[4]
        job_id = rows[0]
        asyncio.run_coroutine_threadsafe(job_failed_discord_message(channel, requester, message=f"Job ID: {job_id} has failed! \n\nThis job has times out on 3 occasions, please double check your requests format!"), bot.loop)
        print(f"Deleting job with ID: {rows[0]} due to too many attempts.")
    print(f"{mycursor.rowcount} rows updated while checking job status.")
    mycursor = mydb.cursor()
    query = """
        UPDATE requests
        SET started_at = NULL
        WHERE started_at <= NOW() - INTERVAL 180 MINUTE
    """
    mycursor.execute(query)
    mydb.commit()
    mycursor.execute(select_query)
    rows_to_delete = mycursor.fetchall()
    for rows in rows_to_delete:
        channel = rows[9]
        requester = rows[4]
        job_id = rows[0]
        asyncio.run_coroutine_threadsafe(job_failed_discord_message(channel, requester, job_id), bot.loop)
        print(f"Deleting job with ID: {rows[0]} due to too many attempts.")
    print(f"{mycursor.rowcount} rows updated while checking job status.")



    delete_query = """
    DELETE FROM requests
    WHERE attempts >= 3
    """
    mycursor.execute(delete_query)
    mydb.commit()
    print(f"{mycursor.rowcount} jobs failed.")
    threading.Timer(600, db_cleanup).start()
async def update_leaderboard_message(completed_jobs):
    try:
        print("Updating leaderboard")
        leaderboard_channel_id = 1443043944136183970
        leaderboard_message_id = 1443052331477237983
        leaderboard_channel = bot.get_channel(leaderboard_channel_id)
        leaderboard_message = await leaderboard_channel.fetch_message(leaderboard_message_id)

        # Load worker_id → worker_name mapping
        try:
            with open('worker_list.json', 'r') as f:
                worker_dict = json.load(f).get("workers", {})
        except (FileNotFoundError, json.JSONDecodeError):
            worker_dict = {}

        # Count jobs per worker_id
        job_counts = {}
        for user_id in completed_jobs:
            job_counts[user_id] = job_counts.get(user_id, 0) + 1

        # Sort by job count
        sorted_leaderboard = sorted(job_counts.items(), key=lambda x: x[1], reverse=True)

        # Build leaderboard text
        leaderboard_text = "**🏆 Klover Leaderboard 🏆**\n\nThank you to each and every person contributing their computer's power to giving others the ability to run AI workloads!\n\n"
        rank = 1
        for user_id, count in sorted_leaderboard:
            # Look up worker name, fall back to ID if not found
            worker_name = worker_dict.get(user_id, {}).get("worker_name", user_id)
            leaderboard_text += f"**{rank}. {worker_name}** - {count} jobs completed\n"
            rank += 1
            if rank > 10:
                break

        await leaderboard_message.edit(content=leaderboard_text)
        threading.Timer(20, leaderboard_update).start()

    except Exception as e:
        print(f"Error updating leaderboard: {e}")
        threading.Timer(20, leaderboard_update).start()
def leaderboard_update():
    mydb = pymysql.connect(
        host=config.db_host,
        user="root",
        password=config.db_pass,
        database="user_requests"
        )
    mycursor = mydb.cursor()
    mycursor.execute("""SELECT * FROM completed_jobs""")
    rows = mycursor.fetchall()
    completed_jobs = []
    for row in rows:
        completed_jobs.append(row[20])
    asyncio.run_coroutine_threadsafe(update_leaderboard_message(completed_jobs), bot.loop)
        
threading.Thread(target=run_flask, daemon=True).start()
threading.Timer(600, db_cleanup).start()
threading.Timer(20, leaderboard_update).start()

@bot.slash_command(description = "Generate an AI image")
async def gen(ctx, prompt: Option(str, "What prompt are you going to use?"), checkpoint_hash: Option(str, "Sumit the HASH of the model you would like to use!"), batch_size: Option(int,choices=[1,2,3,4,5]), resolution: Option(str, "What size do you want the image to be? (WxH) MAXIMUM is 1536x1536", required = False), config_scale: Option(int, "What CFG scale do you want to use? (Default is 5 for SD and SDXL, 3.5 for FLUX)", default = 5,choices=[1,2,3,4,5,6,7,8,9,10],required=False), steps: Option(int, "How many steps do you want to use? (Default is 25, Max is 35)", default = 25,required=False),negative_prompt: Option(str, "What negative prompt are you going to use?", default = "",required=False),sampler: Option(str, "What sampler should we use? (Default Euler)", default = "Euler", choices = ["Euler", "Euler A", "DPM2","DPM++ 2M","DPM 2a", "DDPM"],required=False),clip_skip: Option(int, "What clip skip should we use? (Default Euler)", choices=[1,2,3,4,5,6,7,8,9,10], default = 2,required=False),facefix: Option(str, "Use facefix? (Default False)",choices=["True","False"], default = False,required=False), hires: Option(str, "Use highres fix? (Default False)", choices=["True","False"], default = False,required=False)):
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
@bot.slash_command(description = "Generate an AI video from an image!")
async def img2vid(ctx, image: Option(discord.Attachment, "What image are you going to fuse?"), prompt: Option(str, "What prompt are you going to use?"), length: Option(int, "How long do you want the video to be? (in seconds)", default = 5, choices=[5,6,7,8], required = False)):
    print("img2vid")
@bot.slash_command(description = "Download a style for use with the bot")
async def facefix(ctx, image: Option(discord.Attachment, "What image are you going to face fix?"), prompt: Option(str, "What prompt do you want to run for face fix?"), checkpoint_hash: Option(str, "What checkpoint do you want to sue for face fix? (check /models for a full list!)")):
    print("face fix")

@bot.slash_command(description = "Download a style for use with the bot")
async def upscale(ctx, image: Option(discord.Attachment, "What image are you going to upscale?"), upscale_by: Option(float, choices=[1.5, 2, 2.5, 3, 3.5, 4])):
    print("upscale")
# 2) run the bot
bot.run(config.discord_token)

