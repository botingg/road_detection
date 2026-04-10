import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.util import img_as_float

# =========================
# 讀圖
# =========================
image_path = r"C:\Users\User\Desktop\road_segmentation\road\road.jpg"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("圖片讀不到")

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# =========================
# SLIC Superpixel
# =========================
img_float = img_as_float(rgb)

segments = slic(
    img_float,
    n_segments=300,
    compactness=10,
    sigma=1,
    start_label=1
)

# =========================
# Road color mask (HSV)
# =========================
lower = np.array([0, 0, 50])
upper = np.array([180, 80, 220])
color_mask = cv2.inRange(hsv, lower, upper)

# =========================
# Superpixel voting
# =========================
mask = np.zeros(gray.shape, dtype=np.uint8)

for seg_val in np.unique(segments):
    region = (segments == seg_val)

    # 計算該 superpixel 中 road pixel 比例
    ratio = np.mean(color_mask[region] > 0)

    if ratio > 0.5:   # threshold 可調
        mask[region] = 255

# =========================
# Morphology clean
# =========================
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# =========================
# 最大連通區
# =========================
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

if num_labels > 1:
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    final_mask = (labels == largest).astype(np.uint8) * 255
else:
    final_mask = mask

# =========================
# overlay
# =========================
overlay = image.copy()
overlay[final_mask > 0] = [0, 0, 255]

output = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

# =========================
# 顯示結果
# =========================
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(rgb)

plt.subplot(1, 3, 2)
plt.title("SLIC Segments")
plt.imshow(segments, cmap="nipy_spectral")

plt.subplot(1, 3, 3)
plt.title("Final Road Mask")
plt.imshow(final_mask, cmap="gray")

plt.tight_layout()
plt.show()

cv2.imshow("Final Road Detection", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# =========================
# 9save
# =========================
cv2.imwrite(
    r"C:\Users\User\Desktop\road_segmentation\road_final\road_final_slic.jpg",
    output
)

print("完成：SLIC road segmentation 已輸出")