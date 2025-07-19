from bing_image_downloader import downloader
import os
import cv2
import shutil

# ✅ Person name and number of images to save
name = "Neymar"
num_images = 20

# ✅ Save cropped faces here: face_attendance_system/dataset/Neymar/
base_dir = os.path.dirname(os.path.abspath(__file__))  # /face_attendance_system/dataset
person_dir = os.path.join(base_dir, name)
os.makedirs(person_dir, exist_ok=True)

# ✅ Raw download folder
download_dir = os.path.join(base_dir, "name")

print(f"🔽 Downloading images for: {name}")
downloader.download(name + " face", limit=num_images, output_dir=download_dir, adult_filter_off=True)

# ✅ Haar cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ✅ Path to downloaded raw images
raw_folder = os.path.join(download_dir, name + " face", name + " face")
count = 0

print("📷 Cropping faces and saving...")

for img_name in os.listdir(raw_folder):
    img_path = os.path.join(raw_folder, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"❌ Invalid image: {img_name}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        print(f"⚠️ No face found in: {img_name}")
        continue

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100))
        face_path = os.path.join(person_dir, f"{count}.jpg")
        cv2.imwrite(face_path, face)
        print(f"✅ Saved: {face_path}")
        count += 1
        break

    if count >= num_images:
        break

# ✅ Clean up downloaded raw folder
shutil.rmtree(download_dir)
print(f"🎉 Done. {count} faces saved to: {person_dir}")

