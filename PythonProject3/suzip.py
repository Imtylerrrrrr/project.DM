import os
import cv2
import time
import numpy as np
import mediapipe as mp

label = input("Enter sign label: ").strip()
num_samples = int(input("Enter number of samples: "))

BASE_DIR = "data"
class_dir = os.path.join(BASE_DIR, label)
os.makedirs(class_dir, exist_ok=True)

# 3) Mediapipe 세팅
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 4) 키포인트 추출 함수
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


    if not lh.size:
        lh = np.zeros(63)
    if not rh.size:
        rh = np.zeros(63)
    if not pose.size:
        pose = np.zeros(99)

    return np.concatenate([lh, rh, pose])

# 5) 카메라 & 수집 파라미터
cap = cv2.VideoCapture(0)
fps = 30
duration = 2  # seconds
frames_to_capture = fps * duration
sample_count = 0

# 6) 창 생성
cv2.namedWindow("tnwlq")

print("Press 'k' to start each recording, 'q' to quit.")

while sample_count < num_samples:
    ret0, frame0 = cap.read()
    if ret0:
        frame0 = cv2.flip(frame0, 1)
        cv2.imshow("tnwlq", frame0)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('k'):
        print(f"\nRecording sample {sample_count+1}/{num_samples} in 1 second...")
        time.sleep(1)
        data_seq = []

        for i in range(frames_to_capture):
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)

            # 녹화 상태 표시
            cv2.putText(frame, f"Recording ({i+1}/{frames_to_capture})", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("tnwlq", frame)

            # Mediapipe 처리
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            keypoints = extract_keypoints(results)
            data_seq.append(keypoints)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 저장
        data_seq = np.array(data_seq)
        filepath = os.path.join(class_dir, f"{label}_{sample_count}.npy")
        np.save(filepath, data_seq)
        print(f"Saved {filepath}  shape={data_seq.shape}")
        sample_count += 1

cap.release()
cv2.destroyAllWindows()
