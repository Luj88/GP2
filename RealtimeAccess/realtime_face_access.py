import pickle
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from deepface import DeepFace


GP2_DIR = Path(__file__).resolve().parents[1]
DATABASE_PATH = GP2_DIR / "university.db"
MODEL_NAME = "VGG-Face"
MATCH_THRESHOLD = 0.35
TARGET_FPS = 30
IDENTIFICATION_COOLDOWN_SECONDS = 10.0
MAX_FACES_PER_FRAME = 25


def _load_cascade(filename: str) -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / filename
    if not cascade_path.exists():
        return cv2.CascadeClassifier()
    return cv2.CascadeClassifier(str(cascade_path))


FACE_CASCADE = _load_cascade("haarcascade_frontalface_default.xml")
EYE_CASCADE = _load_cascade("haarcascade_eye.xml")
NOSE_CASCADE = _load_cascade("haarcascade_mcs_nose.xml")
MOUTH_CASCADE = _load_cascade("haarcascade_smile.xml")


@dataclass
class StudentRecord:
    student_id: str
    name: str
    role: str
    embedding: np.ndarray


@dataclass
class FrameDecision:
    label: str
    color: tuple[int, int, int]
    bbox: tuple[int, int, int, int]
    outcome: str
    student_id: Optional[str] = None
    student_name: Optional[str] = None
    role: Optional[str] = None
    accessory_state: str = "clear"
    log_status: str = "Unknown"
    cooldown_key: str = "unknown"
    event_type: Optional[str] = None
    should_alert: bool = False
    should_alarm: bool = False
    should_capture: bool = False
    matched_until: float = 0.0


def load_database(database_path: Path = DATABASE_PATH) -> list[StudentRecord]:
    if not database_path.exists():
        return []

    connection = sqlite3.connect(database_path)
    try:
        rows = connection.execute("SELECT id, name, role, embedding FROM students").fetchall()
    except sqlite3.Error:
        rows = []
    finally:
        connection.close()

    students: list[StudentRecord] = []
    for student_id, name, role, embedding_blob in rows:
        embedding = np.array(pickle.loads(embedding_blob), dtype=np.float32)
        students.append(StudentRecord(student_id, name, role, embedding))
    return students


def cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denominator == 0:
        return 1.0
    similarity = float(np.dot(vec1, vec2) / denominator)
    return 1.0 - similarity


def match_face(embedding: np.ndarray, students: list[StudentRecord]) -> Optional[StudentRecord]:
    best_match: Optional[StudentRecord] = None
    best_distance = float("inf")

    for student in students:
        distance = cosine_distance(embedding, student.embedding)
        if distance < best_distance:
            best_distance = distance
            best_match = student

    if best_match and best_distance <= MATCH_THRESHOLD:
        return best_match
    return None


def _detect_with_cascade(
    cascade: cv2.CascadeClassifier,
    gray: np.ndarray,
    *,
    scale_factor: float,
    min_neighbors: int,
) -> int:
    if cascade.empty():
        return 0
    detections = cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    return len(detections)


def detect_feature_coverage(face_region: np.ndarray) -> tuple[bool, bool, bool]:
    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    eye_count = _detect_with_cascade(EYE_CASCADE, gray, scale_factor=1.1, min_neighbors=5)
    nose_count = _detect_with_cascade(NOSE_CASCADE, gray, scale_factor=1.1, min_neighbors=4)
    mouth_count = _detect_with_cascade(MOUTH_CASCADE, gray, scale_factor=1.7, min_neighbors=20)
    return eye_count > 0, nose_count > 0, mouth_count > 0


def analyze_accessory_state(face_region: np.ndarray) -> str:
    has_eyes, has_nose, has_mouth = detect_feature_coverage(face_region)
    if not has_eyes and not has_nose and not has_mouth:
        return "niqab"
    if has_eyes and not has_nose and not has_mouth:
        return "mask_only"
    if not has_eyes and not has_nose and has_mouth:
        return "mask_and_sunglasses"
    if not has_eyes and (has_nose or has_mouth):
        return "sunglasses_only"
    if has_eyes and has_nose and has_mouth:
        return "medical_glasses_or_clear"
    return "clear"


def _representation_to_embedding(
    representation: dict,
    frame: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    facial_area = representation.get("facial_area", {})
    bbox = (
        int(facial_area.get("x", 0)),
        int(facial_area.get("y", 0)),
        int(facial_area.get("w", frame.shape[1])),
        int(facial_area.get("h", frame.shape[0])),
    )
    embedding = np.array(representation["embedding"], dtype=np.float32)
    return embedding, bbox


def extract_embeddings(
    frame: np.ndarray,
    max_faces: int = MAX_FACES_PER_FRAME,
) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    representations = DeepFace.represent(
        img_path=frame,
        model_name=MODEL_NAME,
        detector_backend="opencv",
        enforce_detection=True,
    )
    if not representations:
        raise ValueError("No face embedding returned by DeepFace.")
    if isinstance(representations, dict):
        representations = [representations]

    faces = [
        _representation_to_embedding(representation, frame)
        for representation in representations
    ]
    faces.sort(key=lambda face: face[1][2] * face[1][3], reverse=True)
    return faces[:max_faces]


def extract_embedding(frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    return extract_embeddings(frame, max_faces=1)[0]


def fallback_person_boxes(
    frame: np.ndarray,
    max_faces: int = MAX_FACES_PER_FRAME,
) -> list[tuple[int, int, int, int]]:
    if FACE_CASCADE.empty():
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    boxes = [
        (int(x), int(y), int(w), int(h))
        for x, y, w, h in faces
    ]
    boxes.sort(key=lambda box: box[2] * box[3], reverse=True)
    return boxes[:max_faces]


def fallback_person_box(frame: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    boxes = fallback_person_boxes(frame, max_faces=1)
    return boxes[0] if boxes else None


def unidentified_cooldown_key(
    *,
    outcome: str,
    accessory_state: str,
    log_status: str,
    bbox: tuple[int, int, int, int],
) -> str:
    x, y, w, h = bbox
    center_x = (x + w // 2) // 80
    center_y = (y + h // 2) // 80
    return (
        f"{outcome}:{accessory_state}:{log_status}:"
        f"{center_x}:{center_y}:{w // 40}:{h // 40}"
    )


def clip_bbox(bbox: tuple[int, int, int, int], frame_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    frame_h, frame_w = frame_shape[:2]
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = max(1, min(w, frame_w - x))
    h = max(1, min(h, frame_h - y))
    return x, y, w, h


def trigger_alarm() -> None:
    print("ALARM: Unauthorized entry detected.")


def _make_decision(
    *,
    bbox: tuple[int, int, int, int],
    now: float,
    outcome: str,
    label: str,
    color: tuple[int, int, int],
    student: Optional[StudentRecord] = None,
    accessory_state: str = "clear",
    log_status: Optional[str] = None,
    event_type: Optional[str] = None,
    should_alert: bool = False,
    should_alarm: bool = False,
    should_capture: bool = False,
    cooldown_key: Optional[str] = None,
) -> FrameDecision:
    resolved_log_status = log_status or label
    resolved_key = cooldown_key or (
        f"student:{student.student_id}" if student else f"{outcome}:{accessory_state}:{resolved_log_status}"
    )
    return FrameDecision(
        label=label,
        color=color,
        bbox=bbox,
        outcome=outcome,
        student_id=None if student is None else student.student_id,
        student_name=None if student is None else student.name,
        role=None if student is None else student.role,
        accessory_state=accessory_state,
        log_status=resolved_log_status,
        cooldown_key=resolved_key,
        event_type=event_type,
        should_alert=should_alert,
        should_alarm=should_alarm,
        should_capture=should_capture,
        matched_until=now + IDENTIFICATION_COOLDOWN_SECONDS,
    )


def evaluate_face(
    frame: np.ndarray,
    students: list[StudentRecord],
    *,
    bbox: tuple[int, int, int, int],
    embedding: Optional[np.ndarray],
    now: float,
) -> Optional[FrameDecision]:
    x, y, w, h = clip_bbox(bbox, frame.shape)
    face_region = frame[y : y + h, x : x + w]
    if face_region.size == 0:
        return None

    accessory_state = analyze_accessory_state(face_region)
    if accessory_state == "niqab":
        return _make_decision(
            bbox=(x, y, w, h),
            now=now,
            outcome="manual_review",
            label="Face Covered - Manual ID Required",
            color=(0, 165, 255),
            accessory_state=accessory_state,
            log_status="Manual ID Required",
            event_type="Manual ID Required",
            should_alert=True,
            should_alarm=True,
            should_capture=True,
            cooldown_key=unidentified_cooldown_key(
                outcome="manual_review",
                accessory_state=accessory_state,
                log_status="Manual ID Required",
                bbox=(x, y, w, h),
            ),
        )

    if accessory_state == "mask_and_sunglasses":
        return _make_decision(
            bbox=(x, y, w, h),
            now=now,
            outcome="denied",
            label="Mask + Sunglasses - Access Denied",
            color=(0, 0, 255),
            accessory_state=accessory_state,
            log_status="Denied",
            event_type="Intruder Alert",
            should_alert=True,
            should_alarm=True,
            should_capture=True,
            cooldown_key=unidentified_cooldown_key(
                outcome="denied",
                accessory_state=accessory_state,
                log_status="Denied",
                bbox=(x, y, w, h),
            ),
        )

    matched_student = match_face(embedding, students) if embedding is not None else None
    if matched_student:
        accessory_label = {
            "mask_only": "Mask Only",
            "sunglasses_only": "Sunglasses Only",
            "medical_glasses_or_clear": "Face / Medical Glasses Verified",
            "clear": "Face / Medical Glasses Verified",
        }.get(accessory_state, "Face Verified")
        return _make_decision(
            bbox=(x, y, w, h),
            now=now,
            outcome="allowed",
            label=f"{matched_student.name} - {accessory_label} - Access Allowed",
            color=(0, 255, 0),
            student=matched_student,
            accessory_state=accessory_state,
            log_status="Allowed",
        )

    if accessory_state == "mask_only":
        return _make_decision(
            bbox=(x, y, w, h),
            now=now,
            outcome="allowed",
            label="Mask Only - Access Allowed",
            color=(0, 255, 0),
            accessory_state=accessory_state,
            log_status="Allowed",
            cooldown_key=unidentified_cooldown_key(
                outcome="allowed",
                accessory_state=accessory_state,
                log_status="Allowed",
                bbox=(x, y, w, h),
            ),
        )

    if accessory_state == "sunglasses_only":
        return _make_decision(
            bbox=(x, y, w, h),
            now=now,
            outcome="allowed",
            label="Sunglasses Only - Access Allowed",
            color=(0, 255, 0),
            accessory_state=accessory_state,
            log_status="Allowed",
            cooldown_key=unidentified_cooldown_key(
                outcome="allowed",
                accessory_state=accessory_state,
                log_status="Allowed",
                bbox=(x, y, w, h),
            ),
        )

    return _make_decision(
        bbox=(x, y, w, h),
        now=now,
        outcome="unknown",
        label="Unknown Face - Intruder Alert",
        color=(0, 0, 255),
        accessory_state=accessory_state,
        log_status="Unknown",
        event_type="Intruder Alert",
        should_alert=True,
        should_alarm=True,
        should_capture=True,
        cooldown_key=unidentified_cooldown_key(
            outcome="unknown",
            accessory_state=accessory_state,
            log_status="Unknown",
            bbox=(x, y, w, h),
        ),
    )


def evaluate_frame(frame: np.ndarray, students: list[StudentRecord]) -> list[FrameDecision]:
    now = time.time()
    face_inputs: list[tuple[Optional[np.ndarray], tuple[int, int, int, int]]] = []

    try:
        face_inputs = [
            (embedding, bbox)
            for embedding, bbox in extract_embeddings(frame, max_faces=MAX_FACES_PER_FRAME)
        ]
    except Exception:
        face_inputs = [
            (None, bbox)
            for bbox in fallback_person_boxes(frame, max_faces=MAX_FACES_PER_FRAME)
        ]

    decisions: list[FrameDecision] = []
    for embedding, bbox in face_inputs[:MAX_FACES_PER_FRAME]:
        decision = evaluate_face(
            frame,
            students,
            bbox=bbox,
            embedding=embedding,
            now=now,
        )
        if decision is not None:
            decisions.append(decision)

    return decisions
