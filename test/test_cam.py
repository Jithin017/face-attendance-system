import cv2

cam = cv2.VideoCapture(-1)
if not cam.isOpened():
    print("❌ Cannot open camera")
else:
    print("✅ Camera is working")
    cam.release()

