# capture_images.py - 拍标定照片

import cv2
import os

SAVE_DIR = "/home/yln/elc25e/target_images"
os.makedirs(SAVE_DIR, exist_ok=True)

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("无法打开摄像头")
    exit()

print("按 's' 保存照片，按 'q' 退出")
print(f"照片保存到: {SAVE_DIR}")

count = 0
while True:
    ret, frame = cam.read()
    if not ret:
        break

    cv2.imshow('Capture', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        filename = os.path.join(SAVE_DIR, f"target_{count:03d}.jpg")
        cv2.imwrite(filename, frame)
        count += 1
        print(f"已保存: {filename}")
    elif key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
print(f"共保存 {count} 张照片")