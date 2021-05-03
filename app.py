import os.path
import sys 
import pandas as pd 
import matplotlib.pyplot as plt 

# CNN + opencv
import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing import image 
from threading import Thread
from queue import Queue 

#flask 
from flask import Flask, render_template, Response, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import sqlite3
import time 

# Flask upload video
UPLOAD_FOLDER = 'D:/FYP/Prototype/'
ALLOWED_EXTENSIONS = {'mp4'}

# Define emotion category, engagemnet mapping and load model
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
Engagement = {'surprise': 'strong', 'angry':'high','fear':'high', 'happy':'medium','disgust':'medium','sad':'low','neutral':'disengaged'}
model = load_model("D:/FYP/Prototype/FER2013.hdf5")

# Load face detector
face_cascade = cv2.CascadeClassifier('D:/FYP/Prototype/haarcascade_frontalface_default.xml')

# Database
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, "ferdb.db")

# Database Connector
connector = sqlite3.connect(db_path, check_same_thread=False)
cursor = connector.cursor()

# set video path
path = 'D:/dataset_classroom_Trim.mp4'

class Recognition():

    def __init__(self): 
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
    
    def threading(self):
        # threading to accelerate FPS
        thread = Thread(target = self.update, args = ())
        thread.daemon = True
        thread.start()

        return self 

    def update(self):

        while True:
        	# get frame from camera
            ret, frame = self.capture.read()
            # convert RBG frame to Grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # detect faces in frame
            faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
            # count number of faces detected
            face_count = len(faces)
			
			# coordinates of square box around faces: x = left, y = top, right = x + w, bottom = y + h
            for (x,y,w,h) in faces: 

            	# crop face
                roi = frame[int(y):int(y+h), int(x):int(x+w)]
                # to grayscale 
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) 
                # resize to 48x48 pixels
                roi = cv2.resize(roi, (48,48)) 

                # convert to array
                face_pixels = image.img_to_array(roi)
                # expand the dimension of face pixel array 
                face_pixels = np.expand_dims(face_pixels, axis = 0)
                # normalize pixels from [0,255] to [0,1]
                face_pixels /= 255 

                # get probability of 7 expressions
                prediction = model.predict(face_pixels)
                # get highest probability expression 
                emotion = emotions[np.argmax(prediction[0])] 
                # store into mapping dictionary
                engage = Engagement[emotion]

                # put engagement level text around face
                cv2.putText(img = frame, text = engage, org = (int(x), int(y)), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (255,255,255), thickness = 2) 
                # put border rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) 

                #expressions = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}
                # empty list to store detected engagement levels
                engagement_level = {'strong': 0, 'high': 0, 'medium': 0, 'low': 0, "disengaged": 0}

                # get current time
                current_time = time.localtime() 
                # formatting timestamp to YY-MM-DD HH:MM:SS
                time_stamp = time.strftime('%Y-%m-%d %H:%M:%S', current_time) 

                # count the engagement level detected
                for level in engagement_level:
                    if engage == level:
                        engagement_level[level] += 1

			    
                # SQL command to insert the detection results into SQLite database
                insert = '''INSERT INTO detection (time, detected_face, strong, high, medium, low, disengaged) VALUES (?,?,?,?,?,?,?)'''
                # insert to database
                cursor.execute(insert, (time_stamp, face_count, engagement_level["strong"], engagement_level["high"], engagement_level["medium"], engagement_level["low"], engagement_level["disengaged"]))
                connector.commit()
			
			# encoding each frames to jpg iamge
            ret, frames = cv2.imencode('.jpg',frame)
            # convert jpg image to bytes
            frames = frames.tobytes() 
	
		    # return frames in bytes
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frames + b'\r\n\r\n')




app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Home Page
@app.route("/")
def home():
    return render_template('home.html')

# Camera Page
@app.route("/camera")
def render_camera():
    return render_template('camera.html')

# Stream classification to flask
@app.route('/video_feed', methods=["GET","POST"])
def video_feed():
    return Response(Recognition().update(),mimetype='multipart/x-mixed-replace; boundary=frame')

# Video Page
@app.route("/video", methods=["GET","POST"])
def render_video():
    return render_template('video.html')


# Handle upload video
@app.route("/", methods=['GET', 'POST'])
def allowed_file(filename):
    return "." in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def upload_file():
    file = request.files['file']

    if request.method == 'GET':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash("No file part")

            return redirect(request.url)

        # if user does not select file, browser will submit an empty part without filename
        if file.filename == '':
            flash("No selected file")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(url_for('uploaded_file', filename=filename))

            return redirect(url_for('uploaded_file', filename=filename))   


# Save Result
@app.route("/save")
def save_result():
    data = pd.read_sql_query("SELECT * FROM detection", connector) 
    data['time'] = pd.to_datetime(data['time'], format="%Y-%m-%d %H:%M:%S")
    data = data.set_index(pd.DatetimeIndex(data['time']))
    data.resample("60S").mean()

    plot_graph(data)

    return None

def plot_graph(df):

    x = df['time']
    y1 = df['strong']
    y2 = df['high']
    y3 = df['medium']
    y4 = df['low']
    y5 = df['disengaged']

    plt.plot(x, y1, label = "strong")
    plt.plot(x, y2, label = "high")
    plt.plot(x, y3, label = "medium")
    plt.plot(x, y4, label = "low")
    plt.plot(x, y5, label = "disengaged")

    plt.xlabel('Time')
    plt.ylabel('Numbers')
    plt.title("Engagement detected over time")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

