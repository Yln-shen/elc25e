import cv2

class Camera:
    def __init__(self, index=0, format='MJPG', width=640, height=480, fps=30):
        self.cam = self.find_cam(index)  # Corrected method name and added index parameter
        self.cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*format))
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # Fixed width and height order
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cam.set(cv2.CAP_PROP_FPS, fps)

    def read(self):
        ret, frame = self.cam.read()
        return ret, frame

    def find_cam(self, index = 30):  # Added index parameter and proper exception handling
        max_tries = index + 20  # Maximum number of cameras to try
        for i in range(index, max_tries):
            cam = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if cam.isOpened():
                return cam
            cam.release()
        raise RuntimeError("Could not open any camera")

if __name__ == '__main__':

    cam = Camera(index=0, format='MJPG', width=640, height=480, fps=30)
    
    # # 创建可调窗口并设置大小
    # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('frame', 680, 420)
    
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()