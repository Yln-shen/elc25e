# filter_calibration_images_fixed.py - 修正版：不依赖旧标定结果

import cv2
import numpy as np
import glob
import os
import shutil
from datetime import datetime

# ========== 配置 ==========
IMAGE_DIR = "/home/yln/elc25e/data/images/target"
OUTPUT_GOOD_DIR = os.path.join(IMAGE_DIR, "good_images")
OUTPUT_BAD_DIR = os.path.join(IMAGE_DIR, "bad_images")
OUTPUT_MANUAL_DIR = os.path.join(IMAGE_DIR, "manual_check")  # 需要手动检查的

# A4靶子尺寸
TARGET_WIDTH = 0.263   # 米
TARGET_HEIGHT = 0.174  # 米

# 创建输出目录
os.makedirs(OUTPUT_GOOD_DIR, exist_ok=True)
os.makedirs(OUTPUT_BAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_MANUAL_DIR, exist_ok=True)

# ========== 靶子检测函数（和标定程序一致）==========
def find_target_corners(frame, debug=False):
    """
    找到靶子的四个角点，返回顺序：左上->右上->右下->左下
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    contours_result = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours_result) == 3:
        contours = contours_result[1]
    else:
        contours = contours_result[0]
    
    best_corners = None
    best_area = 0
    best_ratio = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 5000 or area > 500000:
            continue
        
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        if len(approx) != 4:
            continue
        
        points = approx.reshape(4, 2).astype(np.float32)
        
        if not cv2.isContourConvex(approx):
            continue
        
        rect = np.zeros((4, 2), dtype=np.float32)
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]
        rect[2] = points[np.argmax(s)]
        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]
        rect[3] = points[np.argmax(diff)]
        
        # 计算长宽比
        w1 = np.linalg.norm(rect[0] - rect[1])
        w2 = np.linalg.norm(rect[2] - rect[3])
        h1 = np.linalg.norm(rect[0] - rect[3])
        h2 = np.linalg.norm(rect[1] - rect[2])
        
        w = (w1 + w2) / 2
        h = (h1 + h2) / 2
        
        if w == 0 or h == 0:
            continue
        
        ratio = max(w, h) / min(w, h)
        
        # A4纸比例约1.5，放宽到1.2-1.8
        if 1.2 <= ratio <= 1.8:
            # 检查四个角是否在图像内
            if np.all(rect >= 0) and np.all(rect[:, 0] < frame.shape[1]) and np.all(rect[:, 1] < frame.shape[0]):
                if area > best_area:
                    best_area = area
                    best_corners = rect
                    best_ratio = ratio
    
    if debug:
        return best_corners, binary
    
    return best_corners

# ========== 评估图片质量 ==========
def evaluate_image_quality(img_path):
    """评估单张图片的角点检测质量"""
    img = cv2.imread(img_path)
    if img is None:
        return None, "无法读取图片"
    
    # 尝试检测角点
    corners = find_target_corners(img)
    if corners is None:
        # 尝试预处理
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        corners = find_target_corners(enhanced_bgr)
    
    if corners is None:
        return None, "未检测到靶子"
    
    # 计算角点质量指标
    # 1. 检查四个角是否都在图像内
    h, w = img.shape[:2]
    if not (np.all(corners[:, 0] >= 0) and np.all(corners[:, 0] < w) and
            np.all(corners[:, 1] >= 0) and np.all(corners[:, 1] < h)):
        return None, "角点超出图像边界"
    
    # 2. 检查长宽比
    w1 = np.linalg.norm(corners[0] - corners[1])
    w2 = np.linalg.norm(corners[2] - corners[3])
    h1 = np.linalg.norm(corners[0] - corners[3])
    h2 = np.linalg.norm(corners[1] - corners[2])
    avg_w = (w1 + w2) / 2
    avg_h = (h1 + h2) / 2
    ratio = max(avg_w, avg_h) / min(avg_w, avg_h) if min(avg_w, avg_h) > 0 else 0
    
    if not (1.2 <= ratio <= 1.8):
        return None, f"长宽比异常: {ratio:.2f}"
    
    # 3. 检查面积（太小或太大都说明检测不准确）
    area = cv2.contourArea(corners.astype(np.float32))
    if area < 10000 or area > 300000:
        return None, f"面积异常: {area:.0f}px²"
    
    # 4. 检查凸性（四个角是否构成凸四边形）
    if not cv2.isContourConvex(corners.astype(np.int32).reshape(4, 1, 2)):
        return None, "不是凸四边形"
    
    # 5. 计算角点检测的稳定性（四个角的排序是否正确）
    # 计算四边形各边长度
    sides = []
    for i in range(4):
        side = np.linalg.norm(corners[i] - corners[(i+1)%4])
        sides.append(side)
    
    # 检查对边是否平行（近似）
    # 不严格要求平行，但差异不能太大
    if max(sides[0], sides[2]) / min(sides[0], sides[2]) > 2.0:
        return None, "对边长度差异过大"
    if max(sides[1], sides[3]) / min(sides[1], sides[3]) > 2.0:
        return None, "对边长度差异过大"
    
    # 计算综合评分（面积越大、长宽比越接近1.5，分数越高）
    area_score = min(area / 100000, 1.0)  # 面积归一化
    ratio_score = 1.0 - abs(ratio - 1.5) / 0.6  # 越接近1.5越好
    ratio_score = max(0, min(1, ratio_score))
    
    quality_score = 0.7 * area_score + 0.3 * ratio_score
    
    return corners, quality_score

# ========== 主程序 ==========
print("=" * 60)
print("         标定图片筛选工具（修正版）")
print("=" * 60)

# 收集图片
images = []
for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
    images.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))

if len(images) == 0:
    print(f"错误：目录 {IMAGE_DIR} 中没有找到图片")
    exit()

print(f"\n找到 {len(images)} 张图片")
print("正在评估图片质量...\n")

good_images = []
bad_images = []
manual_check = []
quality_scores = []

for idx, fname in enumerate(images):
    basename = os.path.basename(fname)
    
    # 评估图片
    result = evaluate_image_quality(fname)
    
    if result[0] is None:
        reason = result[1]
        bad_images.append((fname, reason))
        print(f"✗ [{idx+1:2d}/{len(images)}] {basename}: {reason}")
        continue
    
    corners, quality_score = result
    quality_scores.append(quality_score)
    
    # 绘制检测结果
    img = cv2.imread(fname)
    display = img.copy()
    
    for i, pt in enumerate(corners):
        pt_int = tuple(pt.astype(int))
        color = (0, 255, 0) if quality_score > 0.6 else (0, 255, 255)
        cv2.circle(display, pt_int, 8, color, 2)
        cv2.putText(display, str(i), (pt_int[0] + 10, pt_int[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 连接角点
    for i in range(4):
        cv2.line(display, 
                tuple(corners[i].astype(int)), 
                tuple(corners[(i+1)%4].astype(int)), 
                (0, 255, 255), 2)
    
    # 显示评分
    cv2.putText(display, f"Score: {quality_score:.3f}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(display, basename, (10, 55),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 根据评分分类
    if quality_score > 0.7:
        # 好图片
        good_images.append(fname)
        cv2.putText(display, "GOOD", (display.shape[1]-120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        dest_dir = OUTPUT_GOOD_DIR
        print(f"✓ [{idx+1:2d}/{len(images)}] {basename}: 质量 {quality_score:.3f} (好)")
    elif quality_score > 0.5:
        # 需要手动检查
        manual_check.append(fname)
        cv2.putText(display, "CHECK", (display.shape[1]-120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        dest_dir = OUTPUT_MANUAL_DIR
        print(f"? [{idx+1:2d}/{len(images)}] {basename}: 质量 {quality_score:.3f} (需检查)")
    else:
        # 差图片
        bad_images.append((fname, f"质量评分 {quality_score:.3f}"))
        cv2.putText(display, "BAD", (display.shape[1]-120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        dest_dir = OUTPUT_BAD_DIR
        print(f"✗ [{idx+1:2d}/{len(images)}] {basename}: 质量 {quality_score:.3f} (差)")
    
    # 保存标记后的图片
    cv2.imwrite(os.path.join(dest_dir, basename), display)

# ========== 统计报告 ==========
print("\n" + "=" * 60)
print("                    筛选结果")
print("=" * 60)
print(f"总图片数: {len(images)}")
print(f"好图片: {len(good_images)} (质量 > 0.7)")
print(f"需检查: {len(manual_check)} (0.5 < 质量 ≤ 0.7)")
print(f"差图片: {len(bad_images)} (质量 ≤ 0.5)")

if quality_scores:
    print(f"\n质量评分统计:")
    print(f"  平均分: {np.mean(quality_scores):.3f}")
    print(f"  最高分: {np.max(quality_scores):.3f}")
    print(f"  最低分: {np.min(quality_scores):.3f}")
    print(f"  标准差: {np.std(quality_scores):.3f}")

print(f"\n文件已保存到:")
print(f"  好图片: {OUTPUT_GOOD_DIR}")
print(f"  需检查: {OUTPUT_MANUAL_DIR}")
print(f"  差图片: {OUTPUT_BAD_DIR}")

# ========== 交互式检查 ==========
if manual_check:
    print(f"\n发现 {len(manual_check)} 张需要手动检查的图片")
    print("是否逐张查看并决定保留或删除？(y/N): ", end="")
    response = input().strip().lower()
    
    if response == 'y':
        print("\n按 [d] 删除  [s] 保留  [ESC] 退出")
        
        for fname in manual_check:
            basename = os.path.basename(fname)
            img = cv2.imread(os.path.join(OUTPUT_MANUAL_DIR, basename))
            if img is None:
                continue
            
            cv2.imshow('Manual Check', img)
            print(f"\n当前: {basename}")
            
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == 27:  # ESC
                    print("退出检查")
                    break
                elif key == ord('d') or key == ord('D'):
                    # 移动到bad目录
                    shutil.move(os.path.join(OUTPUT_MANUAL_DIR, basename),
                               os.path.join(OUTPUT_BAD_DIR, basename))
                    print(f"  ✗ 已标记为差: {basename}")
                    break
                elif key == ord('s') or key == ord('S'):
                    # 移动到good目录
                    shutil.move(os.path.join(OUTPUT_MANUAL_DIR, basename),
                               os.path.join(OUTPUT_GOOD_DIR, basename))
                    print(f"  ✓ 已标记为好: {basename}")
                    break
        
        cv2.destroyAllWindows()

# ========== 最终统计 ==========
# 重新统计文件数
good_final = glob.glob(os.path.join(OUTPUT_GOOD_DIR, "*.jpg")) + \
             glob.glob(os.path.join(OUTPUT_GOOD_DIR, "*.png"))
bad_final = glob.glob(os.path.join(OUTPUT_BAD_DIR, "*.jpg")) + \
            glob.glob(os.path.join(OUTPUT_BAD_DIR, "*.png"))

print("\n" + "=" * 60)
print("                    最终统计")
print("=" * 60)
print(f"好图片: {len(good_final)} 张")
print(f"差图片: {len(bad_final)} 张")

if len(good_final) >= 10:
    print(f"\n✓ 已有 {len(good_final)} 张好图片，可以进行标定")
    print(f"  好图片目录: {OUTPUT_GOOD_DIR}")
    print("\n建议：")
    print("  1. 将这些好图片复制到原始目录，或修改标定程序的 IMAGE_DIR")
    print("  2. 重新运行 calibrate_with_target.py")
else:
    print(f"\n✗ 好图片只有 {len(good_final)} 张（需要至少10张）")
    print("建议：")
    print("  1. 重新拍摄更多图片")
    print("  2. 确保靶子在图片中清晰可见")
    print("  3. 背景尽量简单，与白色A4纸形成对比")

print("\n程序结束")