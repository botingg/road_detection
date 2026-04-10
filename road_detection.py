import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

# 讀取圖片
image_path = r"C:\Users\User\Desktop\road_segmentation\road1.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"無法讀取圖片：{image_path}")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)



# 直方圖均衡化（Gray Image）
equalized_gray = cv2.equalizeHist(gray_image)

# 直方圖均衡化（HSV 通道中的 V）
h, s, v = cv2.split(hsv_image)
v_eq = cv2.equalizeHist(v)
hsv_eq = cv2.merge([h, s, v_eq])
image_eq = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

# 設定道路顏色的 HSV 範圍
lower_gray = np.array([0, 0, 20])
upper_gray = np.array([180, 45, 200])
road_mask = cv2.inRange(hsv_image, lower_gray, upper_gray)


# 計算 LBP 並僅保留道路區域
def compute_lbp(image, mask, P=8, R=2):
    lbp = local_binary_pattern(image, P, R, method='uniform')
    lbp = lbp * (mask // 255)  # 只保留道路區域的 LBP 值
    return lbp

lbp_image = compute_lbp(equalized_gray, road_mask)

# 對 LBP 圖像進行二值化
_, mask = cv2.threshold(lbp_image.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 尋找輪廓並保留最大區塊
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
cv2.imshow('Contours Detected', contour_image)

# 找到最大輪廓
max_contour = max(contours, key=cv2.contourArea)
max_contour_image = image.copy()
cv2.drawContours(max_contour_image, [max_contour], -1, (0, 0, 255), 2)
cv2.imshow('Max Contour', max_contour_image)


# 創建一個空白遮罩來保存最大區塊
filtered_mask = np.zeros_like(mask)
cv2.drawContours(filtered_mask, [max_contour], -1, 255, thickness=cv2.FILLED)

# 將道路部分覆蓋紅色
overlay = image.copy()
alpha = 0.5
for x in range(filtered_mask.shape[0]):
    for y in range(filtered_mask.shape[1]):
        if filtered_mask[x, y] > 0:
            overlay[x, y] = [0, 0, 255]  # 紅色
output = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

# 顯示原始圖片
cv2.imshow('Original Image', image)

# 顯示原始圖片直方圖
plt.figure()
plt.hist(gray_image.ravel(), bins=256, histtype='step', color='black')
plt.title('Histogram of Original Image')
plt.show()

# 顯示均衡化後的灰階圖片
cv2.imshow('Equalized Gray Image', equalized_gray)

# 顯示均衡化後的灰階圖片直方圖
plt.figure()
plt.hist(equalized_gray.ravel(), bins=256, histtype='step', color='black')
plt.title('Histogram of Equalized Gray Image')
plt.show()

# 顯示 LBP 圖像
cv2.imshow('LBP Image', lbp_image)

# 顯示最終覆蓋的圖片
cv2.imshow('Final Image', output)

# 等待按鍵並關閉顯示窗口
cv2.waitKey(0)
cv2.destroyAllWindows()

# 儲存結果
output_path = r"C:\Users\User\Desktop\road_segmentation\road_final.jpg"
cv2.imwrite(output_path, output)
