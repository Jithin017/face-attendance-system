from flask import Flask, render_template, request, redirect, url_for, session, Response, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from database.db import connect
import subprocess
import os
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import openpyxl

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this for production!

# === Load Trained Model ===
model = tf.keras.models.load_model('pretrained_model/face_recognition_model.h5')

# === Load Labels ===
labels = {}
with open('pretrained_model/labels.txt') as f:
    for line in f:
        k, v = line.strip().split(',')
        labels[int(k)] = v

# === Load Face Detector and Initialize Camera ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():
    raise RuntimeError("❌ Could not open webcam at /dev/video0. Is it used by another app?")

# === Log Attendance to DB ===
def log_attendance(name):
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ?", (name,))
    user = cursor.fetchone()
    if user:
        cursor.execute("INSERT INTO attendance (user_id) VALUES (?)", (user[0],))
        conn.commit()
    conn.close()

# === ROUTES ===

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        name = request.form['name']
        password = generate_password_hash(request.form['password'])

        conn = connect()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password, name) VALUES (?, ?, ?)", (username, password, name))
        conn.commit()
        conn.close()

        subprocess.call(['python3', 'face_recognition/collect_data.py'])
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = connect()
        cursor = conn.cursor()
        cursor.execute("SELECT id, password FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[1], password):
            session['user_id'] = user[0]
            return redirect(url_for('dashboard'))
        return "Invalid login"
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    users = cursor.fetchall()
    conn.close()
    return render_template('dashboard.html', users=users)

@app.route('/attendance')
def attendance():
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("""SELECT users.name, attendance.timestamp
                      FROM attendance JOIN users ON attendance.user_id = users.id
                      ORDER BY attendance.timestamp DESC""")
    records = cursor.fetchall()
    conn.close()
    return render_template('attendance.html', records=records)

@app.route('/export_attendance')
def export_attendance():
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("""SELECT users.name, attendance.timestamp
                      FROM attendance JOIN users ON attendance.user_id = users.id""")
    records = cursor.fetchall()
    conn.close()

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(['Name', 'Timestamp'])
    for name, ts in records:
        ws.append([name, ts])

    os.makedirs("attendance", exist_ok=True)
    path = 'attendance/attendance_export.xlsx'
    wb.save(path)
    return send_file(path, as_attachment=True)

@app.route('/camera')
def camera_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('camera.html')

def gen_frames():
    seen_users = set()
    while True:
        success, frame = camera.read()
        if not success:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            img = Image.fromarray(roi).resize((100, 100))
            img = np.expand_dims(np.expand_dims(np.array(img)/255.0, axis=-1), axis=0)
            prediction = model.predict(img)
            user_id = np.argmax(prediction)
            confidence = np.max(prediction)
            name = labels[user_id] if confidence > 0.7 else "Unknown"

            if name != "Unknown" and name not in seen_users:
                seen_users.add(name)
                log_attendance(name)

            cv2.putText(frame, f"{name} ({confidence*100:.1f}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_snapshot')
def capture_snapshot():
    success, frame = camera.read()
    if success:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("snapshots", exist_ok=True)
        path = f"snapshots/snap_{ts}.jpg"
        cv2.imwrite(path, frame)
        return f"Snapshot saved: {path}"
    return "Failed to capture"

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

