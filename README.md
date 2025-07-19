# 👤 Face Recognition Attendance System (Flask + OpenCV + Docker)

This project is a real-time face recognition-based attendance system built using Flask, OpenCV, TensorFlow, and SQLite. It features a responsive Bootstrap UI, user registration and training system, attendance tracking, Excel export, and complete Docker support for easy deployment.

---

## 🚀 Features

- ✅ Real-time face detection & recognition (OpenCV + Haar Cascades)
- ✅ Add new users with webcam or online images
- ✅ Train & re-train the model dynamically
- ✅ Attendance marking with timestamp
- ✅ Export to Excel (.xlsx)
- ✅ Responsive dashboard (Bootstrap 5)
- ✅ Flash login/logout alerts
- ✅ Admin panel with delete user option
- ✅ Fully Dockerized setup for portable deployment

---

## 🖥️ Screenshots

> *(Add your screenshots here if needed)*

---

## 📁 Project Structure

face_attendance_system/
├── app.py # Main Flask app
├── face_recognition/ # Face logic: train, collect, recognize
├── dataset/ # Saved user images (per person)
├── static/ # CSS, JS, images
├── templates/ # Jinja2 HTML templates
├── database/attendance.db # SQLite DB for attendance
├── requirements.txt # Python dependencies
├── Dockerfile # Build container
└── README.md


---

## ⚙️ Installation (Without Docker)

### 1. Clone the repo

```bash
git clone https://github.com/Jithin017/face-attendance-system.git
cd face-attendance-system

2. Create & activate virtual environment

python3 -m venv venv
source venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Collect user data

python3 face_recognition/collect_data.py

5. Train the model

python3 face_recognition/train_model.py

6. Run the Flask app

python3 app.py

🐳 Docker Deployment
1. Build the Docker image

docker build -t face-attendance .

2. Run the container

docker run -p 5000:5000 --device=/dev/video0 face-attendance

    Visit: http://localhost:5000

💾 Export Attendance

    Visit Export tab on dashboard

    Click "Export" to download .xlsx file

🧠 Model & Face Recognition

    Uses Haar Cascade Classifier for face detection

    Embeddings + SVM for recognition (TensorFlow + scikit-learn)

🛠️ Admin Features

    Login system (session-based)

    Add/Delete users

    Face re-registration from dashboard

    Flash messages on login/logout/actions

📦 Requirements

See requirements.txt
📄 License

MIT License © 2025 Jithin017
