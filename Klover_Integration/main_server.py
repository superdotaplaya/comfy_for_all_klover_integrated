import requests
import json
import os,time,sys,threading
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import discord
from discord.ext import commands, tasks
from discord.commands import Option
import asyncio

from flask import Flask, request, jsonify

app = Flask(__name__)
bot = discord.Bot()
if not os.path.isfile("config.py"):
    sys.exit("'config.py' not found! Please add it and try again.")
else:
    import config

mydb = mysql.connector.connect(
        host=config.db_host,
        user="root",
        password=config.db_pass,
        database="user_requests"
        )
mycursor = mydb.cursor()

@app.route('/api/get-job', methods=['GET'])
def get_job():
    mycursor.execute("""
        SELECT job_id, requested_prompt, steps, model, channel,
               request_type, image_link, requested_at, started_at,
               negative_prompt, resolution, batch_size, config_scale
          FROM requests
         WHERE started_at IS NULL
         ORDER BY job_id ASC
         LIMIT 1
    """)
    row = mycursor.fetchone()
    if not row:
        return jsonify({"error":"no jobs"}), 404

    # unpack into named vars so you never mix up the order later:
    (job_id, prompt, steps, model_hash, channel_id,
     job_type, image_link, created_at, started_at,
     neg_prompt, resolution, batch_size, cfg_scale) = row

    now = datetime.now()
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
async def upload_to_discord(files, channel):
    print("sending to discord")
    await bot.get_channel(1366919874194178108).send(files=files)
@app.route('/api/upload', methods=['POST'])
def upload_images():
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

@bot.event
async def on_ready():
    print(f"🤖 Bot logged in as {bot.user}")
def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False)

threading.Thread(target=run_flask, daemon=True).start()

# 2) run the bot
bot.run(config.discord_token)