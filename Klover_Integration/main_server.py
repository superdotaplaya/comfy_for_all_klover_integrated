import requests
import json
import os,time,sys
import mysql.connector
from mysql.connector import Error

from flask import Flask, request, jsonify

app = Flask(__name__)
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
    mycursor.execute("SELECT * FROM requests LIMIT 1")
    row = mycursor.fetchone()

    return jsonify(row)

@app.route('/api/upload', methods=['POST'])
def upload_images():
    channel = request.args.get('channel')
    if 'images' not in request.files:
        return jsonify({"error": "No 'images' field in request"}), 400

    files = request.files.getlist('images')
    saved_files = []
    i = 0
    for file in files:
        if file.filename:
            file.save(f'uploads/{channel}_{i}.png')
            saved_files.append(file.filename)
            i += 1

    return jsonify({"message": "Images uploaded", "files": saved_files})



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)