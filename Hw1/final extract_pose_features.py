import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ===== 路徑設定 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "image")
OUTPUT_CSV = os.path.join(BASE_DIR, "pose_features.csv")
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")

# ===== 類別名稱 =====
# 請確認這裡和你的資料夾名稱完全一致
CLASSES = ["tree", "warrior2", "goddess", "plank", "downdog"]

# ===== MediaPipe Pose Landmarker 模型 =====
MODEL_DIR = r"C:\temp\mediapipe_models"
MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_lite.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"


# ===== 常用 landmark index =====
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


def ensure_model_exists():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if (not os.path.exists(MODEL_PATH)) or os.path.getsize(MODEL_PATH) == 0:
        print(f"[Info] Downloading model to: {MODEL_PATH}")
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[Info] Model downloaded successfully.")

    print("MODEL_PATH =", MODEL_PATH)
    print("Model exists =", os.path.exists(MODEL_PATH))
    if os.path.exists(MODEL_PATH):
        print("Model size =", os.path.getsize(MODEL_PATH), "bytes")


def create_pose_landmarker():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def read_image_unicode(img_path):
    if not os.path.exists(img_path):
        print("[Path not found]", img_path)
        return None

    data = np.fromfile(img_path, dtype=np.uint8)
    if data.size == 0:
        print("[Empty file or unreadable]", img_path)
        return None

    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        print("[cv2.imdecode failed]", img_path)

    return img


def calculate_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def calculate_angle(a, b, c):
    """
    angle ABC, with b as the vertex
    a, b, c are (x, y)
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0

    cos_angle = np.dot(ba, bc) / denom
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return angle


def normalize_landmarks(landmarks):
    """
    用骨架中心 + 身體尺度做正規化
    中心點：左右髖中點
    尺度：左右肩中點到左右髖中點的 torso length
    """
    left_hip = np.array([landmarks[LEFT_HIP].x, landmarks[LEFT_HIP].y])
    right_hip = np.array([landmarks[RIGHT_HIP].x, landmarks[RIGHT_HIP].y])
    hip_center = (left_hip + right_hip) / 2.0

    left_shoulder = np.array([landmarks[LEFT_SHOULDER].x, landmarks[LEFT_SHOULDER].y])
    right_shoulder = np.array([landmarks[RIGHT_SHOULDER].x, landmarks[RIGHT_SHOULDER].y])
    shoulder_center = (left_shoulder + right_shoulder) / 2.0

    torso_size = np.linalg.norm(shoulder_center - hip_center)
    if torso_size < 1e-6:
        torso_size = 1.0

    norm_points = []
    for lm in landmarks:
        x = (lm.x - hip_center[0]) / torso_size
        y = (lm.y - hip_center[1]) / torso_size
        z = lm.z / torso_size
        vis = getattr(lm, "visibility", 1.0)
        norm_points.append((x, y, z, vis))

    return norm_points


def extract_pose_features(image_path, landmarker):
    image = read_image_unicode(image_path)
    if image is None:
        print("[Read failed]", image_path)
        return None

    print("shape =", image.shape, "path =", image_path)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        print("[No pose detected]", image_path)
        return None

    landmarks = result.pose_landmarks[0]
    norm_points = normalize_landmarks(landmarks)

    features = {}

    # 1. normalized landmarks
    for i, (x, y, z, v) in enumerate(norm_points):
        features[f"lm_{i}_x"] = x
        features[f"lm_{i}_y"] = y
        features[f"lm_{i}_z"] = z
        features[f"lm_{i}_vis"] = v

    # 只抓 2D 點做角度
    pts = [(landmarks[i].x, landmarks[i].y) for i in range(len(landmarks))]

    # 2. joint angles
    features["angle_left_elbow"] = calculate_angle(
        pts[LEFT_SHOULDER], pts[LEFT_ELBOW], pts[LEFT_WRIST]
    )
    features["angle_right_elbow"] = calculate_angle(
        pts[RIGHT_SHOULDER], pts[RIGHT_ELBOW], pts[RIGHT_WRIST]
    )
    features["angle_left_shoulder"] = calculate_angle(
        pts[LEFT_ELBOW], pts[LEFT_SHOULDER], pts[LEFT_HIP]
    )
    features["angle_right_shoulder"] = calculate_angle(
        pts[RIGHT_ELBOW], pts[RIGHT_SHOULDER], pts[RIGHT_HIP]
    )
    features["angle_left_hip"] = calculate_angle(
        pts[LEFT_SHOULDER], pts[LEFT_HIP], pts[LEFT_KNEE]
    )
    features["angle_right_hip"] = calculate_angle(
        pts[RIGHT_SHOULDER], pts[RIGHT_HIP], pts[RIGHT_KNEE]
    )
    features["angle_left_knee"] = calculate_angle(
        pts[LEFT_HIP], pts[LEFT_KNEE], pts[LEFT_ANKLE]
    )
    features["angle_right_knee"] = calculate_angle(
        pts[RIGHT_HIP], pts[RIGHT_KNEE], pts[RIGHT_ANKLE]
    )

    # 3. geometry / body ratios
    left_shoulder = pts[LEFT_SHOULDER]
    right_shoulder = pts[RIGHT_SHOULDER]
    left_hip = pts[LEFT_HIP]
    right_hip = pts[RIGHT_HIP]
    left_elbow = pts[LEFT_ELBOW]
    right_elbow = pts[RIGHT_ELBOW]
    left_wrist = pts[LEFT_WRIST]
    right_wrist = pts[RIGHT_WRIST]
    left_knee = pts[LEFT_KNEE]
    right_knee = pts[RIGHT_KNEE]
    left_ankle = pts[LEFT_ANKLE]
    right_ankle = pts[RIGHT_ANKLE]

    shoulder_width = calculate_distance(left_shoulder, right_shoulder)
    hip_width = calculate_distance(left_hip, right_hip)
    torso_len = calculate_distance(
        ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2),
        ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
    )

    left_upper_arm = calculate_distance(left_shoulder, left_elbow)
    left_lower_arm = calculate_distance(left_elbow, left_wrist)
    right_upper_arm = calculate_distance(right_shoulder, right_elbow)
    right_lower_arm = calculate_distance(right_elbow, right_wrist)

    left_thigh = calculate_distance(left_hip, left_knee)
    left_calf = calculate_distance(left_knee, left_ankle)
    right_thigh = calculate_distance(right_hip, right_knee)
    right_calf = calculate_distance(right_knee, right_ankle)

    eps = 1e-6
    features["shoulder_width"] = shoulder_width
    features["hip_width"] = hip_width
    features["torso_len"] = torso_len
    features["shoulder_hip_ratio"] = shoulder_width / (hip_width + eps)
    features["left_arm_ratio"] = left_upper_arm / (left_lower_arm + eps)
    features["right_arm_ratio"] = right_upper_arm / (right_lower_arm + eps)
    features["left_leg_ratio"] = left_thigh / (left_calf + eps)
    features["right_leg_ratio"] = right_thigh / (right_calf + eps)

    return features


def main():
    print("DATASET_DIR =", DATASET_DIR)

    if not os.path.exists(DATASET_DIR):
        print(f"[Error] Dataset folder not found: {DATASET_DIR}")
        return

    ensure_model_exists()

    rows = []
    failed_images = []

    with create_pose_landmarker() as landmarker:
        for label in CLASSES:
            class_dir = os.path.join(DATASET_DIR, label)

            if not os.path.exists(class_dir):
                print(f"[Warning] Class folder not found: {class_dir}")
                continue

            for fname in os.listdir(class_dir):
                if not fname.lower().endswith(IMG_EXTENSIONS):
                    continue

                img_path = os.path.join(class_dir, fname)
                feat = extract_pose_features(img_path, landmarker)

                if feat is None:
                    failed_images.append(img_path)
                    continue

                row = {
                    "filename": fname,
                    "filepath": img_path,
                    "label": label
                }
                row.update(feat)
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved features to: {OUTPUT_CSV}")
    print(f"Valid images: {len(df)}")
    print(f"Failed images: {len(failed_images)}")

    if failed_images:
        print("\nExamples of failed images:")
        for p in failed_images[:20]:
            print(p)


if __name__ == "__main__":
    main()