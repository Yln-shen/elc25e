# capture_images_improved.py - 改进的拍照脚本
import cv2
import os
import time

SAVE_DIR = "/home/yln/elc25e/data/images/target"
os.makedirs(SAVE_DIR, exist_ok=True)

cam = cv2.VideoCapture(4)
if not cam.isOpened():
    print("无法打开摄像头")
    exit()

# 设置较高的分辨率
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("=== 相机标定拍照指南 ===")
print("按以下顺序拍摄（每个姿势按's'保存1-2张）：")
print()
print("1. 距离变化（3个距离）：")
print("   近距(~40cm)：靶子占画面50%")
print("   中距(~80cm)：靶子占画面25%") 
print("   远距(~120cm)：靶子占画面15%")
print()
print("2. 位置变化（5个位置）：")
print("   中央、左上、右上、左下、右下")
print()
print("3. 角度变化（4个角度）：")
print("   前倾30°、后倾30°、左倾30°、右倾30°")
print()
print("按 's' 保存，'q' 退出，'h' 显示帮助")
print(f"照片保存到: {SAVE_DIR}")

count = 0
guide_text = ""
guide_timer = 0

while True:
    ret, frame = cam.read()
    if not ret:
        break
    
    # 添加网格辅助线
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # 画九宫格
    for i in range(1, 3):
        cv2.line(overlay, (w*i//3, 0), (w*i//3, h), (0, 255, 0), 1)
        cv2.line(overlay, (0, h*i//3), (w, h*i//3), (0, 255, 0), 1)
    
    # 画中心十字
    cv2.line(overlay, (w//2, h//2-20), (w//2, h//2+20), (0, 0, 255), 1)
    cv2.line(overlay, (w//2-20, h//2), (w//2+20, h//2), (0, 0, 255), 1)
    
    # 半透明叠加
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    
    # 显示计数
    cv2.putText(frame, f"已拍: {count} 张", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 显示引导文字
    if guide_text and time.time() - guide_timer < 3:
        cv2.putText(frame, guide_text, (10, h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow('Calibration Capture', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        filename = os.path.join(SAVE_DIR, f"target_{count:03d}.jpg")
        cv2.imwrite(filename, frame)
        count += 1
        guide_text = f"已保存第 {count} 张照片"
        guide_timer = time.time()
        print(f"[{count}] 已保存: {filename}")
        
        # 拍照建议
        if count == 1:
            print("  提示：改变距离，拍摄中等距离的照片")
        elif count == 3:
            print("  提示：现在改变靶子位置，移到画面左上角")
        elif count == 5:
            print("  提示：试试倾斜靶子，让它前倾约30度")
        elif count == 10:
            print("  提示：已经10张了，继续保持不同角度")
        elif count >= 20:
            print("  提示：20张已经足够，但可以继续")
            
    elif key == ord('q'):
        break
    elif key == ord('h'):
        print("\n=== 当前位置建议 ===")
        print(f"已拍摄: {count} 张")
        if count < 3:
            print("建议：继续拍摄不同距离的照片")
        elif count < 8:
            print("建议：拍摄不同位置（四个角落）")
        elif count < 15:
            print("建议：拍摄不同倾斜角度")
        else:
            print("建议：已经足够，可以退出")
        print()

cam.release()
cv2.destroyAllWindows()
print(f"\n共保存 {count} 张照片")
if count < 15:
    print("建议再拍几张，至少15张以上效果更好")