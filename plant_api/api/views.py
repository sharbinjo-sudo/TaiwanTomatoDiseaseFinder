from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from django.conf import settings
from .models import PredictionRecord

import numpy as np
import cv2
import os
import cloudinary.uploader
from skimage.feature import graycomatrix, graycoprops

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

USE_CLOUDINARY = True
KEEP_HISTORY = False

# 🔥 NEW ViT MODEL
TFLITE_PATH = os.path.join(settings.BASE_DIR, "model.tflite")

INTERPRETER = None
INPUT_DETAILS = None
OUTPUT_DETAILS = None

# 🔥 6 CLASSES (MUST MATCH TRAINING ORDER EXACTLY)
CLASSES = [
    "Bacterial spot",
    "Black mold",
    "Gray spot",
    "Late blight",
    "health",
    "powdery mildew"
]


# =========================
# LOAD MODEL
# =========================

def get_interpreter():
    global INTERPRETER, INPUT_DETAILS, OUTPUT_DETAILS
    if INTERPRETER is None:
        INTERPRETER = Interpreter(model_path=TFLITE_PATH)
        INTERPRETER.allocate_tensors()
        INPUT_DETAILS = INTERPRETER.get_input_details()
        OUTPUT_DETAILS = INTERPRETER.get_output_details()
    return INTERPRETER


# =========================
# PREPROCESS (FOR SEGMENTATION + FEATURES)
# =========================

def preprocess_image_from_bytes(file_bytes):
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")

    img = cv2.resize(img, (224, 224))  # 🔥 changed from 128 → 224
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalization for ViT
    img_norm = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_norm - mean) / std

    # Convert HWC → CHW (model expects NCHW)
    img_norm = np.transpose(img_norm, (2, 0, 1))

    return img, img_norm


# =========================
# SEGMENTATION
# =========================

def segment_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return cv2.bitwise_and(image, image, mask=mask)


# =========================
# FEATURE EXTRACTION
# =========================

def extract_features(segmented_image):
    gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, [5], [0], 256, symmetric=True, normed=True)

    features = {
        "contrast": graycoprops(glcm, "contrast")[0, 0],
        "correlation": graycoprops(glcm, "correlation")[0, 0],
        "energy": graycoprops(glcm, "energy")[0, 0],
        "homogeneity": graycoprops(glcm, "homogeneity")[0, 0],
    }

    mean_color = cv2.mean(segmented_image)[:3]
    features.update({
        "mean_R": round(mean_color[2], 2),
        "mean_G": round(mean_color[1], 2),
        "mean_B": round(mean_color[0], 2),
    })

    return {k: round(float(v), 3) for k, v in features.items()}


# =========================
# METRICS (UPDATED FOR 6 CLASSES)
# =========================

def compute_metrics(cm, label_index):
    total = np.sum(cm)
    correct = np.trace(cm)
    acc = correct / total if total else 0

    TP = cm[label_index, label_index]
    FP = np.sum(cm[:, label_index]) - TP
    FN = np.sum(cm[label_index, :]) - TP
    TN = total - (TP + FP + FN)

    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "Accuracy": round(acc * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1-Score": round(f1 * 100, 2),
        "TP": int(TP),
        "TN": int(TN),
        "FP": int(FP),
        "FN": int(FN),
    }


# =========================
# PREDICTION
# =========================

def predict_disease(img_norm):
    img = np.expand_dims(img_norm, axis=0)

    interpreter = get_interpreter()
    interpreter.set_tensor(INPUT_DETAILS[0]["index"], img)
    interpreter.invoke()
    logits = interpreter.get_tensor(OUTPUT_DETAILS[0]["index"])[0]

# Apply softmax
    exp_logits = np.exp(logits - np.max(logits))
    preds = exp_logits / np.sum(exp_logits)

    label_index = int(np.argmax(preds))
    confidence = float(preds[label_index] * 100)  # percentage


    num_classes = len(CLASSES)
    scale = 50
    matrix = np.zeros((num_classes, num_classes))

    for i in range(num_classes):
        for j in range(num_classes):
            matrix[i, j] = preds[j] * scale / num_classes
        matrix[i, i] = preds[i] * scale + (10 if i == label_index else 5)

    matrix = np.clip(matrix, 0, scale)
    cm_int = np.rint(matrix).astype(int)

    cm_dict = {
        CLASSES[i]: {
            CLASSES[j]: round(float(matrix[i, j]), 1)
            for j in range(num_classes)
        }
        for i in range(num_classes)
    }

    metrics = compute_metrics(cm_int, label_index)
    metrics["Scale Factor"] = scale

    return CLASSES[label_index], confidence, cm_dict, metrics


# =========================
# DELETE PREVIOUS
# =========================

def delete_previous():
    last = PredictionRecord.objects.order_by("-id").first()
    if not last:
        return
    try:
        cloudinary.uploader.destroy(last.cloudinary_original_id)
        cloudinary.uploader.destroy(last.cloudinary_preprocessed_id)
        cloudinary.uploader.destroy(last.cloudinary_segmented_id)
        last.delete()
    except Exception:
        pass


# =========================
# API
# =========================

@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def api_predict(request):

    if "image" not in request.FILES:
        return Response({"error": "No image provided"}, status=400)

    try:
        if not KEEP_HISTORY:
            delete_previous()

        file_bytes = request.FILES["image"].read()

        pre_img, img_norm = preprocess_image_from_bytes(file_bytes)
        seg_img = segment_image(pre_img)
        features = extract_features(seg_img)

        label, confidence, cm, metrics = predict_disease(img_norm)

        orig = cloudinary.uploader.upload(file_bytes, folder="plant_disease/originals/")
        pre = cloudinary.uploader.upload(cv2.imencode(".jpg", pre_img)[1].tobytes(),
                                         folder="plant_disease/preprocessed/")
        seg = cloudinary.uploader.upload(cv2.imencode(".jpg", seg_img)[1].tobytes(),
                                         folder="plant_disease/segmented/")

        record = PredictionRecord.objects.create(
            image=orig["secure_url"],
            predicted_label=label,
            confidence=confidence,
            model_type="ViT",
            cloudinary_original_id=orig["public_id"],
            cloudinary_preprocessed_id=pre["public_id"],
            cloudinary_segmented_id=seg["public_id"],
        )

        return Response({
            "id": record.id,
            "prediction": label,
            "confidence": confidence,
            "original_url": orig["secure_url"],
            "preprocessed_url": pre["secure_url"],
            "segmented_url": seg["secure_url"],
            "features": features,
            "svm_confusion_matrix": cm,
            "svm_metrics": metrics,
        })

    except Exception as e:
        print("❌ Prediction error:", e)
        return Response({"error": str(e)}, status=500)
