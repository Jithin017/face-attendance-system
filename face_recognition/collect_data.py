import cv2
import os

def collect_images(user_id, output_dir='dataset'):
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    count = 0
    path = os.path.join(output_dir, user_id)
    os.makedirs(path, exist_ok=True)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face = gray[y:y+h, x:x+w]
            cv2.imwrite(f"{path}/img_{count}.jpg", face)
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

        cv2.imshow("Register Face", img)
        if cv2.waitKey(1) == 27 or count >= 20:  # ESC or 20 images
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    user_id = input("Enter username to register face: ")
    collect_images(user_id)

