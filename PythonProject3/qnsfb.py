import os
import time

import cv2
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder
import joblib
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

DATA_DIR   = "data"
MODEL_PATH = "lstm_model.h5"
LE_PATH    = "label_encoder.pkl"
FPS        = 30
DURATION   = 2
CAP_FRAMES = FPS * DURATION

mp_holistic = mp.solutions.holistic
holistic    = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_keypoints(results):
    lh = np.array(
        [
            [lm.x, lm.y, lm.z]
            for lm in (
                results.left_hand_landmarks.landmark
                if results.left_hand_landmarks
                else []
            )
        ]
    ).flatten()
    rh = np.array(
        [
            [lm.x, lm.y, lm.z]
            for lm in (
                results.right_hand_landmarks.landmark
                if results.right_hand_landmarks
                else []
            )
        ]
    ).flatten()
    pose = np.array(
        [
            [lm.x, lm.y, lm.z]
            for lm in (
                results.pose_landmarks.landmark
                if results.pose_landmarks
                else []
            )
        ]
    ).flatten()

    lh   = lh   if lh.size   else np.zeros(21 * 3)
    rh   = rh   if rh.size   else np.zeros(21 * 3)
    pose = pose if pose.size else np.zeros(33 * 3)

    return np.concatenate([lh, rh, pose])

def load_training_data(data_dir):
    X, y = [], []
    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if not fname.endswith(".npy"):
                continue
            seq = np.load(os.path.join(class_dir, fname))
            X.append(seq)
            y.append(label)
    return np.array(X), np.array(y)

def build_lstm_model(seq_len, feat_dim, num_classes):
    model = Sequential([
        LSTM(128, input_shape=(seq_len, feat_dim)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model(model, X, y_enc):
    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    early = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        mode='max',
        verbose=1
    )

    history = model.fit(
        X, y_enc,
        epochs=30,
        batch_size=16,
        validation_split=0.2,
        callbacks=[checkpoint, early]
    )
    return history

def recognize_live(model, le):
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("dlstlr")
    print("\n▶ Press 'i' to recognize, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('i'):
            print("  ▶ Recognizing in 1 second...")
            time.sleep(1)
            seq = []
            for i in range(CAP_FRAMES):
                ret2, frm2 = cap.read()
                if not ret2:
                    continue
                frm2 = cv2.flip(frm2, 1)

                img = cv2.cvtColor(frm2, cv2.COLOR_BGR2RGB)
                res = holistic.process(img)
                seq.append(extract_keypoints(res))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            feat = np.array(seq).reshape(1, CAP_FRAMES, feat_dim)
            probs = model.predict(feat)[0]
            idx = np.argmax(probs)
            label = le.inverse_transform([idx])[0]
            print(f"  ▶ Prediction: {label} ({probs[idx]*100:.1f}%)")

            cv2.putText(frame, f"{label} ({probs[idx]*100:.1f}%)",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 255, 0), 3)

        cv2.imshow("dlstlr", frame)

    cap.release()
    cv2.destroyAllWindows()

def main():
    print("▶ Loading data...")
    X, y = load_training_data(DATA_DIR)
    print(f"   Loaded {len(X)} sequences, each {CAP_FRAMES} frames × {X.shape[2]} features.")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    joblib.dump(le, LE_PATH)
    num_classes = len(le.classes_)
    print(f"   {num_classes} classes: {list(le.classes_)}")

    seq_len, feat_dim = X.shape[1], X.shape[2]
    model = build_lstm_model(seq_len, feat_dim, num_classes)
    model.summary()
    train_model(model, X, y_enc)
    print(f"\n▶ Model saved at '{MODEL_PATH}'")

    recognize_live(load_model(MODEL_PATH), joblib.load(LE_PATH))

if __name__ == "__main__":
    main()
