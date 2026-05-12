# calibrate_with_target.py - 用A4靶子标定相机

import cv2
import numpy as np
import glob
import os

# ========== 配置 ==========
TARGET_WIDTH = 0.260   # A4宽（米）
TARGET_HEIGHT = 0.173  # A4高（米）
IMAGE_DIR = "/home/yln/elc25e/target_images/"  # 靶子照片目录

# ========== 靶子3D点（和PNP一样）==========
half_w = TARGET_WIDTH / 2
half_h = TARGET_HEIGHT / 2
object_points = np.array([
    [-half_w, -half_h, 0],  # 左上
    [-half_w,  half_h, 0],  # 左下
    [ half_w,  half_h, 0],  # 右下
    [ half_w, -half_h, 0],  # 右上
], dtype=np.float32)

# ========== 靶子检测函数（复用你的逻辑）==========
def find_target_corners(frame):
    """找到靶子的四个角点，返回 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    best_corners = None
    best_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 5000 or area > 500000:
            continue
        
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        if len(approx) != 4:
            continue
        
        points = approx.reshape(4, 2)
        
        # 排序
        sum_xy = points.sum(axis=1)
        diff_xy = points[:, 0] - points[:, 1]
        sorted_points = np.array([
            points[np.argmin(sum_xy)],   # 左上
            points[np.argmin(diff_xy)],  # 左下
            points[np.argmax(sum_xy)],   # 右下
            points[np.argmax(diff_xy)]   # 右上
        ], dtype=np.float32)
        
        # 检查长宽比
        w = np.linalg.norm(sorted_points[0] - sorted_points[1])
        h = np.linalg.norm(sorted_points[1] - sorted_points[2])
        ratio = max(w, h) / min(w, h)
        
        if 1.1 <= ratio <= 1.6:
            if area > best_area:
                best_area = area
                best_corners = sorted_points
    
    return best_corners


# ========== 收集数据 ==========
images = glob.glob(IMAGE_DIR + "*.jpg") + glob.glob(IMAGE_DIR + "*.jpeg") + glob.glob(IMAGE_DIR + "*.png")

if len(images) == 0:
    print(f"错误：目录 {IMAGE_DIR} 中没有找到图片")
    print("请先拍20-30张靶子在不同位置、不同角度的照片，保存到该目录")
    exit()

print(f"找到 {len(images)} 张图片")

all_obj_points = []   # 每张图对应的3D点
all_img_points = []   # 每张图对应的2D点
good_images = 0

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        continue
    
    corners = find_target_corners(img)
    
    if corners is not None:
        all_obj_points.append(object_points)
        all_img_points.append(corners)
        good_images += 1
        
        # 显示检测结果
        display = img.copy()
        for i, pt in enumerate(corners):
            cv2.circle(display, tuple(pt.astype(int)), 5, (0, 255, 0), -1)
        cv2.imshow('Detected', display)
        cv2.waitKey(100)
    else:
        print(f"在 {os.path.basename(fname)} 中未检测到靶子")

cv2.destroyAllWindows()

print(f"\n成功检测到靶子的图片: {good_images}/{len(images)}")

if good_images < 10:
    print("图片太少，至少需要10张以上才能标定")
    exit()

# ========== 标定 ==========
print("\n正在标定...")
img_size = img.shape[1], img.shape[0]

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    all_obj_points, all_img_points, img_size, None, None
)

# ========== 输出 ==========
print("\n" + "=" * 50)
print("标定完成！")
print("=" * 50)
print(f"\n重投影误差: {ret:.4f} 像素")
print("\n相机内参矩阵:")
print(mtx)
print("\n畸变系数:")
print(dist)

fx = mtx[0, 0]
fy = mtx[1, 1]
fov_h = 2 * np.degrees(np.arctan2(img_size[0] / 2, fx))
fov_v = 2 * np.degrees(np.arctan2(img_size[1] / 2, fy))

print(f"\n水平FOV: {fov_h:.1f} 度")
print(f"垂直FOV: {fov_v:.1f} 度")
print(f"焦距fx: {fx:.1f} 像素")
print(f"焦距fy: {fy:.1f} 像素")

np.savez("camera_calib_target.npz",
         camera_matrix=mtx,
         dist_coeffs=dist,
         fov_h=fov_h, fov_v=fov_v,
         fx=fx, fy=fy)

print(f"\n结果已保存到 camera_calib_target.npz")