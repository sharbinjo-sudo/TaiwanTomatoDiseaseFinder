import os
import cv2
import numpy as np

TRAIN_DIR = r"C:\Users\sharb\Downloads\taiwan\taiwan\Preprocessed data\train"

def augment_image(img):
    augmented = []

    # rotations
    augmented.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    augmented.append(cv2.rotate(img, cv2.ROTATE_180))
    augmented.append(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))

    # flips
    augmented.append(cv2.flip(img, 1))   # horizontal
    augmented.append(cv2.flip(img, 0))   # vertical

    # brightness increase
    brighter = cv2.convertScaleAbs(img, alpha=1.2, beta=20)
    augmented.append(brighter)

    # brightness decrease
    darker = cv2.convertScaleAbs(img, alpha=0.8, beta=-20)
    augmented.append(darker)

    return augmented

for cls in os.listdir(TRAIN_DIR):
    cls_path = os.path.join(TRAIN_DIR, cls)
    if not os.path.isdir(cls_path):
        continue

    for img_name in os.listdir(cls_path):
        img_path = os.path.join(cls_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        base = os.path.splitext(img_name)[0]
        aug_imgs = augment_image(img)

        for i, aug in enumerate(aug_imgs):
            save_name = f"{base}_aug{i}.jpg"
            cv2.imwrite(os.path.join(cls_path, save_name), aug)

print("Augmentation completed (train only).")
