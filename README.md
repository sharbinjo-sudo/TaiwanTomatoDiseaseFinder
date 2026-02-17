# 🍅 Tomato Disease Detector (ViT-Based)

A cross-platform tomato leaf disease detection system using **Vision Transformer (ViT)** and a Django backend API.
The application supports Android and Web interfaces with real-time image prediction.

---

## 🚀 Overview

This project detects tomato leaf diseases from uploaded or captured images using a deep learning model trained on 6 disease classes.

The system consists of:

* 🧠 **Vision Transformer (ViT) model** (PyTorch → TFLite)
* 🌐 **Django REST API backend**
* 📱 **Flutter frontend (Android + Web)**
* ☁️ **Cloudinary image storage**
* 📊 Feature extraction + confusion matrix visualization

---

## 🏗 System Architecture

```
Flutter App (Android/Web)
        ↓
   Django REST API
        ↓
  TFLite ViT Model
        ↓
Prediction + Features
        ↓
Cloudinary Storage
```

---

## 🧠 Model Details

* Architecture: `vit_base_patch16_224`
* Framework: PyTorch (timm)
* Fine-tuned on 6 tomato leaf classes
* Converted to TFLite using LiteRT
* Input size: `224x224`
* Normalization:

  * Mean: [0.485, 0.456, 0.406]
  * Std: [0.229, 0.224, 0.225]

---

## 🏷 Disease Classes

1. Bacterial spot
2. Black mold
3. Gray spot
4. Late blight
5. Health
6. Powdery mildew

(Class order matches training dataset exactly.)

---

## ✨ Features

* 📷 Camera capture (Android)
* 🖼 Gallery selection (Android)
* 📁 File upload (Web & Desktop)
* 🧪 Image preprocessing & segmentation
* 📊 GLCM feature extraction
* 📉 Confusion matrix visualization
* 📈 Performance metrics (Accuracy, Precision, Recall, F1)
* ☁️ Cloudinary storage for processed images
* 🎨 Custom tomato-themed UI

---

## 🛠 Tech Stack

### Frontend

* Flutter
* image_picker
* file_picker
* http

### Backend

* Django
* Django REST Framework
* TensorFlow Lite
* OpenCV
* scikit-image
* Cloudinary

### ML Training

* PyTorch
* timm
* LiteRT conversion

---

## 📂 Project Structure

```
TomatoDiseaseFinder/
│
├── plant_api/              # Django backend
│   ├── api/
│   ├── model.tflite
│   └── manage.py
│
├── plant_disease/          # Flutter frontend
│   ├── lib/
│   ├── android/
│   ├── windows/
│   └── web/
│
└── Training&Data/          # Model training scripts
```

---

## ⚙️ Backend Setup (Django)

### 1️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 2️⃣ Set environment variables (.env)

```
DJANGO_SECRET_KEY=your_secret
DATABASE_URL=your_database_url
CLOUDINARY_CLOUD_NAME=your_cloud
CLOUDINARY_API_KEY=your_key
CLOUDINARY_API_SECRET=your_secret
```

### 3️⃣ Run server

For local mobile testing:

```
python manage.py runserver 0.0.0.0:8000
```

---

## 📱 Frontend Setup (Flutter)

### 1️⃣ Install dependencies

```
flutter pub get
```

### 2️⃣ Update API base URL

In `ApiService`:

```
http://YOUR_LOCAL_IP:8000/api
```

### 3️⃣ Build APK

```
flutter build apk --release
```

---

## 🌐 API Endpoint

### POST `/api/predict/`

**Form-data:**

```
image: <file>
```

**Response:**

```json
{
  "prediction": "Powdery mildew",
  "confidence": 97.54,
  "original_url": "...",
  "preprocessed_url": "...",
  "segmented_url": "...",
  "features": {...},
  "svm_confusion_matrix": {...},
  "svm_metrics": {...}
}
```

---

## 🧪 Preprocessing Pipeline

1. Image resize to 224x224
2. RGB conversion
3. Normalization
4. Leaf segmentation using Otsu threshold
5. GLCM feature extraction

---

## 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix (scaled visualization)

---

## 📸 Demo Workflow

1. Capture or upload tomato leaf image
2. Image sent to Django backend
3. Model predicts disease
4. Processed images uploaded to Cloudinary
5. Results visualized in UI

---

## 🚧 Limitations

* Requires stable network for backend access
* Model size may impact performance
* Confusion matrix is visualization-based (not dataset-wide evaluation)

---

## 🔮 Future Improvements

* Quantized TFLite model (int8)
* Offline inference inside Flutter
* Real-time camera preview
* Dataset expansion
* Model calibration for better confidence reliability

---

## 📌 Authors

Tomato Disease Detection System
Built using Vision Transformer and Flutter-Django architecture.

---