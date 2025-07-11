# 🚀 Eye Drowsiness Detection System

This is a computer vision system for detecting driver drowsiness by monitoring whether both eyes are closed for prolonged periods. It uses:

- **Convolutional Neural Networks (CNNs)** trained to classify eye state (open or closed)
- **OpenCV Haar cascades** for real-time face and eye detection
- **An alarm system** that triggers if eyes remain closed beyond a safety threshold

This project can help reduce accidents caused by driver fatigue.

---

## 🎯 Features

✅ Detects:
- **Open eyes**
- **Closed eyes**

✅ Maintains a drowsiness **score**:
- Score increases if both eyes are detected closed
- Score decreases if eyes are open
- Triggers an alarm if score exceeds a safety threshold

✅ Runs in real-time using a webcam feed.

✅ Easy to train your own model on new datasets.

---

## 📂 Dataset

This model is trained using the publicly available dataset:

> [Yawn Eye Dataset (New) on Kaggle](https://www.kaggle.com/datasets/serenaraju/yawn-eye-dataset-new)

It contains labeled images of:
- Open eyes
- Closed eyes
- Yawning mouths
- Non-yawning mouths

Only the eye images were used for the basic drowsiness detection implemented here. The same dataset can be extended to add mouth/yawn detection.

---

## 💻 Project Structure

```
.
├── haar cascade files/
│   ├── haarcascade_frontalface_alt.xml
│   ├── haarcascade_lefteye_2splits.xml
│   └── haarcascade_righteye_2splits.xml
├── data/
│   ├── train/
│   │   ├── Closed/
│   │   └── Open/
│   └── test/
│       ├── Closed/
│       └── Open/
├── models/
│   └── eye_model.keras
├── alarm.wav
├── model.py
├── drowsiness_detection.py
├── requirements.txt
└── README.md
```



---

## 🧠 How It Works

1. **Train a model:**

Your `model.py` script:
- Loads images from folders:
    - `Closed/` → eyes closed
    - `Open/` → eyes open
- Trains a CNN to classify open vs closed eyes
- Saves the trained model as `models/eye_model.keras`

2. **Real-time detection:**

`drowsiness_detection.py`:
- Captures webcam frames
- Detects faces and eyes
- Predicts eye state
- Increases score if eyes are closed
- Plays alarm if score exceeds threshold

---

## 📦 Installation

Clone the repo:

```bash
git clone https://github.com/YOUR_USERNAME/drowsiness-detection.git
cd drowsiness-detection
```
Install dependencies:

```bash
pip install -r requirements.txt
```

🎓 Training
Organize your data as:

```
data/train/Closed/
data/train/Open/
data/test/Closed/
data/test/Open/
```

Then run:

```bash
python model.py
```

This will save the trained model:

```
models/eye_model.keras
```

🚘 Running Detection
Ensure:

-Your webcam is connected
-Your alarm file alarm.wav exists

Run:

```bash
python drowsiness_detection.py
```
Press q to quit the webcam window.

## 📷 Demo

<img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/feae5812-c8b2-4699-8fe9-31c36873dc63" />
<img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/28c43020-fc13-4806-bd15-f0cfb01e1f26" />


