import os
import cv2
import numpy as np
from PIL import Image

source = 'samples' # dir
low_range = np.array([100, 100, 100]) # Minnum threshold
high_range = np.array([255, 255, 255]) # Maxnum threshold
files = sorted(os.listdir(source))

for img_name_index in range(len(files)):
    img_name = files[img_name_index]
    img_path = os.path.join(source, img_name)
    img = Image.open(img_path).convert('RGB')
    image = cv2.imread(img_path)
    height, width, no_use = image.shape
    empty_img = np.zeros((height, width), np.uint8)

    #K-MEANS CLUSTERING
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 6
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 15, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))
    cv2.imwrite('results/kmeans.jpg', res2)
    # cv2.imshow('res2', res2)
    # while True:
    #     if cv2.waitKey(1) & 0xFF == ord('s'):
    #         break

    # Threshold segmentation
    blur = cv2.GaussianBlur(res2, (15, 15), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    final = cv2.inRange(res2, low_range, high_range)
    cv2.imwrite('results/binary.jpg', final)

    # Connected domain analysis
    contours, hierchary = cv2.findContours(final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    final = cv2.drawContours(final, contours, -1, 0, 3)
    num, labels = cv2.connectedComponents(final)
    area_label = 0
    area = 0
    for li in range(1, num):
        area_temp = np.sum(labels == li)
        if area_temp > area:
            area = area_temp
            area_label = li
    empty_img = np.zeros((height, width), np.uint8)
    empty_img[labels == area_label] = 255
    final_masked = empty_img
    cv2.imwrite('results/hand_seg.jpg', final_masked)
