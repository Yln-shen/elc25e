# calibrate_with_target.py - 用A4靶子标定相机（完整修正版）

import cv2
import numpy as np
import glob
import os
from datetime import datetime

# ========== 配置 ==========
TARGET_WIDTH = 0.260   # A4宽（米）
TARGET_HEIGHT = 0.173  # A4高（米）
IMAGE_DIR = "/home/yln/elc25e/target_images/"  # 靶子照片目录

# ========== 靶子3D点（匹配新的排序：左上->右上->右下->左下）==========
half_w = TARGET_WIDTH / 2
half_h = TARGET_HEIGHT / 2
object_points = np.array([
    [-half_w, -half_h, 0],  # 左上
    [ half_w, -half_h, 0],  # 右上
    [ half_w,  half_h, 0],  # 右下
    [-half_w,  half_h, 0],  # 左下
], dtype=np.float32)

# ========== 靶子检测函数（修正版）==========
def find_target_corners(frame, debug=False):
    """
    找到靶子的四个角点，返回顺序：左上->右上->右下->左下
    如果debug=True，返回检测过程的中间图像
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 使用高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 使用OTSU二值化
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 形态学操作去噪
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # OpenCV不同版本的兼容性处理
    contours_result = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours_result) == 3:
        contours = contours_result[1]  # OpenCV 3
    else:
        contours = contours_result[0]  # OpenCV 4
    
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
        
        points = approx.reshape(4, 2).astype(np.float32)
        
        # 检查是否为凸四边形
        if not cv2.isContourConvex(approx):
            continue
        
        # 修正后的排序：左上->右上->右下->左下
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # 使用坐标和来区分左上角和右下角
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]  # 左上（x+y最小）
        rect[2] = points[np.argmax(s)]  # 右下（x+y最大）
        
        # 使用坐标差来区分右上角和左下角
        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]   # 右上（x-y最小）
        rect[3] = points[np.argmax(diff)]   # 左下（x-y最大）
        
        # 检查是否为合理的矩形（长宽比验证）
        w1 = np.linalg.norm(rect[0] - rect[1])  # 上边
        w2 = np.linalg.norm(rect[2] - rect[3])  # 下边
        h1 = np.linalg.norm(rect[0] - rect[3])  # 左边
        h2 = np.linalg.norm(rect[1] - rect[2])  # 右边
        
        w = (w1 + w2) / 2
        h = (h1 + h2) / 2
        
        if w == 0 or h == 0:
            continue
            
        ratio = max(w, h) / min(w, h)
        
        # A4纸比例约1.5，这里放宽到1.2-1.8
        if 1.2 <= ratio <= 1.8:
            if area > best_area:
                best_area = area
                best_corners = rect
    
    if debug:
        return best_corners, binary
    
    return best_corners


# ========== 图像预处理函数 ==========
def preprocess_image(img):
    """对图像进行预处理以提高检测成功率"""
    # 转为灰度图
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # 直方图均衡化提高对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 转回BGR以便显示
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    return enhanced_bgr


# ========== 主程序 ==========
print("=" * 60)
print("         相机标定程序 - A4靶子标定")
print("=" * 60)

# ========== 收集数据 ==========
images = []
for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
    images.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))

if len(images) == 0:
    print(f"错误：目录 {IMAGE_DIR} 中没有找到图片")
    print("请先拍20-30张靶子在不同位置、不同角度的照片，保存到该目录")
    exit()

print(f"\n找到 {len(images)} 张图片")
print("正在检测靶子...")

all_obj_points = []   # 每张图对应的3D点
all_img_points = []   # 每张图对应的2D点
good_images = 0
detection_failed = []

# 创建调试目录
debug_dir = os.path.join(IMAGE_DIR, "debug")
os.makedirs(debug_dir, exist_ok=True)

for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    if img is None:
        print(f"无法读取: {os.path.basename(fname)}")
        continue
    
    # 尝试原始图像
    corners = find_target_corners(img)
    
    # 如果失败，尝试预处理后的图像
    if corners is None:
        print(f"原始图像未检测到，尝试预处理: {os.path.basename(fname)}")
        enhanced_img = preprocess_image(img)
        corners = find_target_corners(enhanced_img)
        
        # 保存调试信息
        _, binary_debug = find_target_corners(enhanced_img, debug=True)
        if binary_debug is None:
            _, binary_debug = find_target_corners(img, debug=True)
        
        if binary_debug is not None:
            debug_comparison = np.hstack([
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 
                binary_debug
            ])
            cv2.imwrite(os.path.join(debug_dir, f"debug_{idx:03d}.jpg"), debug_comparison)
    
    if corners is not None:
        all_obj_points.append(object_points)
        all_img_points.append(corners)
        good_images += 1
        
        # 绘制检测结果
        display = img.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
        labels = ['TL', 'TR', 'BR', 'BL']  # 左上、右上、右下、左下
        
        for i, (pt, color, label) in enumerate(zip(corners, colors, labels)):
            pt_int = tuple(pt.astype(int))
            cv2.circle(display, pt_int, 10, color, -1)
            cv2.putText(display, f"{i}-{label}", (pt_int[0] + 15, pt_int[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 连接角点显示检测框
        for i in range(4):
            cv2.line(display, 
                    tuple(corners[i].astype(int)), 
                    tuple(corners[(i+1)%4].astype(int)), 
                    (0, 255, 255), 2)
        
        # 显示图像信息
        cv2.putText(display, f"Good: {good_images}/{idx+1}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 缩放显示
        scale = min(1200 / display.shape[1], 800 / display.shape[0])
        if scale < 1:
            display = cv2.resize(display, None, fx=scale, fy=scale)
        
        cv2.imshow('Target Detection', display)
        key = cv2.waitKey(50)
        
        # 按空格暂停/继续，按ESC退出
        if key == 27:  # ESC
            print("用户中断")
            cv2.destroyAllWindows()
            exit()
        elif key == ord(' '):
            print("暂停中，按任意键继续...")
            cv2.waitKey(0)
    else:
        detection_failed.append(os.path.basename(fname))
        print(f"✗ 在 {os.path.basename(fname)} 中未检测到靶子")

cv2.destroyAllWindows()

# ========== 报告检测结果 ==========
print("\n" + "=" * 60)
print(f"检测结果: {good_images}/{len(images)} 张图片成功检测到靶子")
print("=" * 60)

if detection_failed:
    print("\n以下图片检测失败:")
    for fname in detection_failed:
        print(f"  - {fname}")
    print(f"调试图像已保存到: {debug_dir}")

if good_images < 10:
    print(f"\n错误：仅 {good_images} 张图片可用，至少需要10张以上才能标定")
    print("建议：")
    print("1. 确保靶子完整出现在图像中")
    print("2. 背景尽量简单，与白色A4纸形成对比")
    print("3. 避免强光直射或严重阴影")
    print("4. 尝试不同角度和距离拍摄")
    exit()

# ========== 标定 ==========
print("\n正在标定相机...")
print(f"使用 {good_images} 张图片进行标定")

# 获取图像尺寸
sample_img = cv2.imread(images[0])
if sample_img is None:
    print("错误：无法读取样本图像获取尺寸")
    exit()
img_size = (sample_img.shape[1], sample_img.shape[0])

# 执行标定
print("执行 calibrateCamera...")
try:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        all_obj_points, 
        all_img_points, 
        img_size, 
        None, 
        None,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    )
    print("标定成功完成！")
except Exception as e:
    print(f"标定失败: {e}")
    exit()

# ========== 验证标定结果 ==========
print("\n验证标定结果...")
mean_error = 0
individual_errors = []

for i in range(len(all_obj_points)):
    # 将3D点投影回图像平面
    imgpoints2, _ = cv2.projectPoints(
        all_obj_points[i], 
        rvecs[i], 
        tvecs[i], 
        mtx, 
        dist
    )
    
    # 将投影点从 (4, 1, 2) 重塑为 (4, 2)
    imgpoints2 = imgpoints2.reshape(-1, 2)
    
    # 将原始图像点转换为float32
    original_points = all_img_points[i].astype(np.float32)
    
    # 计算每个点的误差
    errors = np.sqrt(np.sum((original_points - imgpoints2) ** 2, axis=1))
    
    # 该图像的平均误差
    img_error = np.mean(errors)
    mean_error += img_error
    individual_errors.append(img_error)
    
    # 如果某张图误差特别大，打印警告
    if img_error > 1.0:
        print(f"  ⚠️  图片 {i} 的重投影误差较大: {img_error:.3f} 像素")

total_error = mean_error / len(all_obj_points)
std_error = np.std(individual_errors)

print(f"平均重投影误差: {total_error:.4f} 像素")
print(f"标准差: {std_error:.4f} 像素")
print(f"最大误差: {max(individual_errors):.4f} 像素")
print(f"最小误差: {min(individual_errors):.4f} 像素")

# ========== 输出结果 ==========
print("\n" + "=" * 60)
print("                    标 定 结 果")
print("=" * 60)

# 检查标定质量
quality = "优秀"
if total_error > 1.0:
    quality = "差"
    print("\n⚠️  警告：重投影误差较大，标定质量可能不佳")
    print("建议：")
    print("  1. 检查角点检测的准确性")
    print("  2. 重新拍摄更多不同角度的图片")
    print("  3. 确保靶子在图像中清晰可见")
    print("  4. 考虑使用更精确的标定板（如棋盘格）")
elif total_error > 0.5:
    quality = "一般"
    print("\n⚠️  注意：重投影误差略高，可以考虑重新标定")
else:
    quality = "良好"
    print("\n✓ 标定质量良好")

print(f"\n整体重投影误差: {ret:.4f} 像素")
print(f"平均重投影误差: {total_error:.4f} 像素")
print(f"图像尺寸: {img_size[0]} x {img_size[1]}")

print("\n--- 相机内参矩阵 ---")
np.set_printoptions(precision=3, suppress=True)
print(mtx)

print("\n--- 畸变系数 ---")
dist_flat = dist.ravel()
print("k1, k2, p1, p2, k3 =", dist_flat)

# 计算并显示FOV和焦距
fx = mtx[0, 0]
fy = mtx[1, 1]
cx = mtx[0, 2]
cy = mtx[1, 2]

fov_h = 2 * np.degrees(np.arctan2(img_size[0] / 2, fx))
fov_v = 2 * np.degrees(np.arctan2(img_size[1] / 2, fy))

print("\n--- 相机参数 ---")
print(f"焦距 fx: {fx:.2f} 像素")
print(f"焦距 fy: {fy:.2f} 像素")
print(f"光心 cx: {cx:.2f} 像素")
print(f"光心 cy: {cy:.2f} 像素")
print(f"水平FOV: {fov_h:.1f} 度")
print(f"垂直FOV: {fov_v:.1f} 度")

# 检查参数合理性
print("\n--- 参数检查 ---")
issues = []

# 检查焦距
sensor_width_guess = 3.6  # mm, 典型USB摄像头
pixel_size_guess = sensor_width_guess / img_size[0] * 1000  # µm
fx_mm = fx * pixel_size_guess / 1000  # mm
print(f"估算物理焦距: {fx_mm:.1f} mm (假设传感器宽度 {sensor_width_guess}mm)")
if fx_mm < 1 or fx_mm > 50:
    issues.append(f"物理焦距异常 ({fx_mm:.1f}mm)，通常摄像头焦距在2-6mm之间")

# 检查光心位置
center_x, center_y = img_size[0] / 2, img_size[1] / 2
offset_x = abs(cx - center_x) / img_size[0] * 100
offset_y = abs(cy - center_y) / img_size[1] * 100
print(f"光心偏移: 水平{offset_x:.1f}%, 垂直{offset_y:.1f}%")
if offset_x > 10 or offset_y > 10:
    issues.append(f"光心偏移较大")

# 检查fx和fy的比例
fx_fy_ratio = abs(fx - fy) / max(fx, fy)
print(f"fx/fy比例差异: {fx_fy_ratio:.3f}")
if fx_fy_ratio > 0.2:
    issues.append(f"fx和fy差异较大 ({fx/fy:.3f})，可能标定不准确")

# 检查畸变系数
print(f"畸变系数 k1: {dist_flat[0]:.4f}")
if abs(dist_flat[0]) > 1.0:
    issues.append(f"畸变系数k1过大 ({dist_flat[0]:.4f})，通常|k1|<1")

if issues:
    print("\n⚠️  发现潜在问题：")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("\n✓ 所有参数看起来合理")

# ========== 保存结果 ==========
output_file = "camera_calib_target.npz"
np.savez(output_file,
         camera_matrix=mtx,
         dist_coeffs=dist,
         fov_h=fov_h, 
         fov_v=fov_v,
         fx=fx, 
         fy=fy,
         cx=cx,
         cy=cy,
         reprojection_error=total_error,
         reprojection_std=std_error,
         image_size=img_size,
         num_images=good_images,
         target_size=[TARGET_WIDTH, TARGET_HEIGHT],
         quality=quality,
         individual_errors=individual_errors)

print(f"\n✓ 标定结果已保存到: {output_file}")

# 保存为可读文本格式
with open("camera_calib_target.txt", "w") as f:
    f.write("=" * 60 + "\n")
    f.write("                   相机标定结果\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"标定日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"标定质量: {quality}\n")
    f.write(f"使用图片数: {good_images}/{len(images)}\n")
    f.write(f"图像尺寸: {img_size[0]} x {img_size[1]}\n")
    f.write(f"靶子尺寸: {TARGET_WIDTH*1000:.0f}mm x {TARGET_HEIGHT*1000:.0f}mm\n")
    f.write(f"重投影误差: {total_error:.4f} ± {std_error:.4f} 像素\n\n")
    
    f.write("相机内参矩阵:\n")
    np.savetxt(f, mtx, fmt='%.6f')
    
    f.write("\n畸变系数 (k1, k2, p1, p2, k3):\n")
    np.savetxt(f, dist_flat.reshape(1, -1), fmt='%.6f')
    
    f.write(f"\n相机参数:\n")
    f.write(f"  焦距: fx={fx:.2f}, fy={fy:.2f} 像素\n")
    f.write(f"  光心: cx={cx:.2f}, cy={cy:.2f} 像素\n")
    f.write(f"  FOV: 水平={fov_h:.1f}°, 垂直={fov_v:.1f}°\n")
    f.write(f"  估算物理焦距: {fx_mm:.1f} mm\n")
    
    if issues:
        f.write("\n潜在问题:\n")
        for issue in issues:
            f.write(f"  - {issue}\n")

print("✓ 文本格式结果已保存到: camera_calib_target.txt")

# ========== 显示反投影验证 ==========
try:
    print("\n显示反投影验证（前5张图片）...")
    
    # 找出成功检测的图片索引
    good_indices = []
    img_idx = 0
    for fname in images:
        if img_idx < good_images:
            good_indices.append(fname)
            img_idx += 1
        else:
            break
    
    for i in range(min(5, good_images)):
        # 读取成功检测的图片
        img = cv2.imread(good_indices[i])
        if img is None:
            continue
        
        # 反投影3D点到图像上
        imgpoints2, _ = cv2.projectPoints(
            all_obj_points[i], rvecs[i], tvecs[i], mtx, dist
        )
        imgpoints2 = imgpoints2.reshape(-1, 2)
        
        # 绘制原始角点和反投影点
        for j, (pt_true, pt_proj) in enumerate(zip(all_img_points[i], imgpoints2)):
            pt_true_int = tuple(pt_true.astype(int))
            pt_proj_int = tuple(pt_proj.astype(int))
            
            # 原始角点（绿色圆圈）
            cv2.circle(img, pt_true_int, 10, (0, 255, 0), 2)
            cv2.putText(img, f"T{j}", (pt_true_int[0] + 15, pt_true_int[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 反投影点（红色十字）
            cv2.drawMarker(img, pt_proj_int, (0, 0, 255), 
                          markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
            
            # 连接对应点（白色线）
            cv2.line(img, pt_true_int, pt_proj_int, (255, 255, 255), 1)
        
        # 显示误差信息
        cv2.putText(img, f"Error: {individual_errors[i]:.3f} pix", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, f"Image {i+1}/{good_images}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 缩放显示
        scale = min(1200 / img.shape[1], 800 / img.shape[0])
        if scale < 1:
            display = cv2.resize(img, None, fx=scale, fy=scale)
        else:
            display = img
        
        cv2.imshow(f'Reprojection Verification', display)
        
        key = cv2.waitKey(1000)
        if key == 27:  # ESC
            break
        elif key == ord(' '):
            print("暂停中，按任意键继续...")
            cv2.waitKey(0)
    
    print("\n按任意键退出...")
    cv2.waitKey(0)
    
except Exception as e:
    print(f"反投影验证显示失败: {e}")

cv2.destroyAllWindows()

# ========== 总结 ==========
print("\n" + "=" * 60)
print("                    标定完成总结")
print("=" * 60)
print(f"标定质量: {quality}")
print(f"成功图片: {good_images}/{len(images)}")
print(f"重投影误差: {total_error:.4f} 像素")
print(f"焦距: fx={fx:.1f}, fy={fy:.1f} 像素")
print(f"FOV: {fov_h:.1f}° x {fov_v:.1f}°")
print(f"\n结果文件:")
print(f"  - camera_calib_target.npz (用于程序加载)")
print(f"  - camera_calib_target.txt (可读文本)")

if quality != "良好":
    print("\n建议重新标定以提高精度")

print("\n程序结束")