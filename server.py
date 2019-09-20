import argparse
import datetime
import pickle
import cv2
import numpy
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import face_recognition
import myICP
from detect_word import detect_word_from_video
from hand_recognition.utils import detector_utils

app = Flask(__name__)


@app.route('/hello', methods=['GET'])
def debug():
    return "Hello "


@app.route('/video_prediction', methods=['POST'])
def get_video_prediction():
    start_time = datetime.datetime.now()
    print(start_time)
    json = request.get_json()

    chosenWord = detect_word_from_video(json['name'])
    end_time = datetime.datetime.now()
    print(end_time - start_time)

    return jsonify({"word": chosenWord})


@app.route('/upload_file', methods=['POST'])
def upload_file():
    f = request.files['file']
    f.save("users_videos/" + secure_filename(f.filename))
    return jsonify({"filename": f.filename})


if __name__ == '__main__':
    app.run(debug=True, host="192.168.56.1", port=5000)
    print("server is on")

