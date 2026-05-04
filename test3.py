
import cv2, time

for i in range(30, 50):
    cam = cv2.VideoCapture(i, cv2.CAP_V4L2)
    if cam.isOpened():
        print(f"找到摄像头 at index {i}")
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cam.set(cv2.CAP_PROP_FPS, 30)
        
        t = time.time()
        count = 0
        for _ in range(30):
            ret, frame = cam.read()
            if ret:
                count += 1
        elapsed = time.time() - t
        print(f"30帧耗时: {elapsed:.2f}s, FPS: {count/elapsed:.1f}")
        cam.release()
        break
    cam.release()
