import json
import pickle
import queue
import sqlite3
import tempfile
import threading
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_from_directory, session
from flask_socketio import SocketIO
from werkzeug.security import check_password_hash, generate_password_hash

from realtime_face_access import (
    DATABASE_PATH,
    IDENTIFICATION_COOLDOWN_SECONDS,
    TARGET_FPS,
    FrameDecision,
    evaluate_face_candidate,
    evaluate_frame,
    extract_embedding,
    extract_face_embeddings,
    fallback_person_box,
    load_database,
    trigger_alarm,
)


REALTIME_DIR = Path(__file__).resolve().parent
SCREENSHOT_DIR = REALTIME_DIR / "captures"
STREAM_JPEG_QUALITY = 72
STREAM_FPS = 20
STREAM_FRAME_INTERVAL_SECONDS = 1 / STREAM_FPS
CAPTURE_MAX_WIDTH = 1280
FRAME_SKIP_INTERVAL = 30
KNOWN_FACE_RECHECK_COOLDOWN_SECONDS = 30.0
CAMERA_INDEX = 0
CAMERA_INDICES = (0, 1, 2, 3, 4)
CAMERA_MIRROR = True
CAMERA_FRAME_WIDTH = 1280
CAMERA_FRAME_HEIGHT = 720
CAMERA_BACKENDS: tuple[tuple[str, Optional[int]], ...] = (
    ("DirectShow", cv2.CAP_DSHOW),
    ("Media Foundation", cv2.CAP_MSMF),
    ("Default", None),
)
CAMERA_DISABLED_MESSAGE = "Camera is turned off by security."
MEDIA_SCAN_MAX_UPLOAD_BYTES = 250 * 1024 * 1024
MEDIA_SCAN_MAX_VIDEO_FRAMES = 45
MEDIA_SCAN_MIN_VIDEO_SAMPLE_SECONDS = 1.0
MEDIA_SCAN_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEDIA_SCAN_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
VALID_ROLES = {"Student", "Staff", "Admin", "Security"}
ACCOUNT_ROLES = {"admin", "security"}
ADMISSION_SYNC_INTERVAL_SECONDS = 15
ADMISSION_SYNC_BATCH_SIZE = 25

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = "university-face-access"
app.config["MAX_CONTENT_LENGTH"] = MEDIA_SCAN_MAX_UPLOAD_BYTES
socketio = SocketIO(app, cors_allowed_origins="*")


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def _table_sql(connection: sqlite3.Connection, table_name: str) -> str:
    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return "" if row is None or row["sql"] is None else row["sql"]


def _column_names(connection: sqlite3.Connection, table_name: str) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row["name"] for row in rows}


def ensure_database_schema() -> None:
    connection = get_connection()
    try:
        students_sql = _table_sql(connection, "students")
        if not students_sql:
            connection.execute(
                """
                CREATE TABLE students (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('Student', 'Staff', 'Admin', 'Security')),
                    embedding BLOB NOT NULL,
                    image_path TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        else:
            student_columns = _column_names(connection, "students")
            if "created_at" not in student_columns:
                connection.execute(
                    """
                    ALTER TABLE students
                    ADD COLUMN created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    """
                )
            if "image_path" not in student_columns:
                connection.execute("ALTER TABLE students ADD COLUMN image_path TEXT")

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS graduated_students (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                role TEXT NOT NULL,
                requirements_completed INTEGER NOT NULL DEFAULT 1,
                reason TEXT NOT NULL DEFAULT 'Graduated',
                graduated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        graduate_columns = _column_names(connection, "graduated_students")
        if "requirements_completed" not in graduate_columns:
            connection.execute(
                "ALTER TABLE graduated_students ADD COLUMN requirements_completed INTEGER NOT NULL DEFAULT 1"
            )
        if "reason" not in graduate_columns:
            connection.execute(
                "ALTER TABLE graduated_students ADD COLUMN reason TEXT NOT NULL DEFAULT 'Graduated'"
            )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS app_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('admin', 'security')),
                full_name TEXT,
                email TEXT,
                phone TEXT,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_login_at TEXT,
                UNIQUE(username, role)
            )
            """
        )
        app_user_columns = _column_names(connection, "app_users")
        if "full_name" not in app_user_columns:
            connection.execute("ALTER TABLE app_users ADD COLUMN full_name TEXT")
        if "email" not in app_user_columns:
            connection.execute("ALTER TABLE app_users ADD COLUMN email TEXT")
        if "phone" not in app_user_columns:
            connection.execute("ALTER TABLE app_users ADD COLUMN phone TEXT")

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS admission_students (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'Student',
                image_path TEXT,
                image_blob BLOB,
                is_graduated INTEGER NOT NULL DEFAULT 0,
                synced_at TEXT,
                sync_error TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        access_logs_sql = _table_sql(connection, "access_logs")
        needs_rebuild = not access_logs_sql
        if access_logs_sql:
            required_columns = {
                "student_id",
                "student_name",
                "role",
                "timestamp",
                "date",
                "status",
                "label",
                "event_type",
            }
            if not required_columns.issubset(_column_names(connection, "access_logs")):
                needs_rebuild = True
            if "Manual ID Required" not in access_logs_sql or "Denied" not in access_logs_sql:
                needs_rebuild = True

        if needs_rebuild:
            if access_logs_sql:
                connection.execute("ALTER TABLE access_logs RENAME TO access_logs_legacy")

            connection.execute(
                """
                CREATE TABLE access_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT,
                    student_name TEXT,
                    role TEXT,
                    timestamp TEXT NOT NULL,
                    date TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('Allowed', 'Unknown', 'Denied', 'Manual ID Required')),
                    label TEXT NOT NULL,
                    event_type TEXT
                )
                """
            )

            if access_logs_sql:
                connection.execute(
                    """
                    INSERT INTO access_logs (
                        student_id,
                        student_name,
                        role,
                        timestamp,
                        date,
                        status,
                        label,
                        event_type
                    )
                    SELECT
                        legacy.student_id,
                        students.name,
                        students.role,
                        legacy.timestamp,
                        legacy.date,
                        CASE
                            WHEN legacy.status = 'Covered' THEN 'Manual ID Required'
                            ELSE legacy.status
                        END,
                        CASE
                            WHEN legacy.status = 'Covered' THEN 'Manual ID Required'
                            WHEN legacy.status = 'Allowed' THEN 'Access Allowed'
                            WHEN legacy.status = 'Unknown' THEN 'Unknown Face - Intruder Alert'
                            ELSE legacy.status
                        END,
                        CASE
                            WHEN legacy.status = 'Covered' THEN 'Manual ID Required'
                            WHEN legacy.status = 'Unknown' THEN 'Intruder Alert'
                            ELSE NULL
                        END
                    FROM access_logs_legacy AS legacy
                    LEFT JOIN students ON students.id = legacy.student_id
                    """
                )
                connection.execute("DROP TABLE access_logs_legacy")

        connection.commit()
    finally:
        connection.close()


def get_current_user() -> Optional[dict[str, Any]]:
    user = session.get("user")
    return user if isinstance(user, dict) else None


def require_auth(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if get_current_user() is None:
            return jsonify({"error": "Authentication required"}), 401
        return view(*args, **kwargs)

    return wrapped


def require_role(*allowed_roles: str):
    normalized_roles = {role.lower() for role in allowed_roles}

    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            user = get_current_user()
            if user is None:
                return jsonify({"error": "Authentication required"}), 401
            if user.get("role") not in normalized_roles:
                return jsonify({"error": "You do not have permission for this area"}), 403
            return view(*args, **kwargs)

        return wrapped

    return decorator


def authenticate_user(role: str, username: str, password: str) -> dict[str, Any]:
    connection = get_connection()
    try:
        row = connection.execute(
            """
            SELECT id, username, role, full_name, email, phone, password_hash
            FROM app_users
            WHERE username = ? AND role = ?
            """,
            (username, role),
        ).fetchone()

        now = datetime.now().isoformat(timespec="seconds")
        if row is None:
            raise ValueError("Invalid username or password")

        if not check_password_hash(row["password_hash"], password):
            raise ValueError("Invalid username or password")

        connection.execute(
            "UPDATE app_users SET last_login_at = ? WHERE id = ?",
            (now, row["id"]),
        )
        connection.commit()
        return {
            "id": row["id"],
            "username": row["username"],
            "role": row["role"],
            "full_name": row["full_name"],
            "email": row["email"],
            "phone": row["phone"],
        }
    finally:
        connection.close()


def create_app_user(
    role: str,
    full_name: str,
    username: str,
    password: str,
    email: str,
    phone: str,
) -> dict[str, Any]:
    if role not in ACCOUNT_ROLES:
        raise ValueError("Unsupported account role")

    connection = get_connection()
    try:
        existing = connection.execute(
            "SELECT id FROM app_users WHERE username = ? AND role = ?",
            (username, role),
        ).fetchone()
        if existing is not None:
            if role == "admin":
                raise ValueError("This username is already used by another admissions employee")
            raise ValueError("This username is already used by another security employee")

        now = datetime.now().isoformat(timespec="seconds")
        password_hash = generate_password_hash(password)
        cursor = connection.execute(
            """
            INSERT INTO app_users (username, role, full_name, email, phone, password_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (username, role, full_name, email or None, phone or None, password_hash, now),
        )
        connection.commit()
        return {
            "id": cursor.lastrowid,
            "username": username,
            "role": role,
            "full_name": full_name,
            "email": email,
            "phone": phone,
        }
    finally:
        connection.close()


def create_admissions_employee(
    full_name: str,
    username: str,
    password: str,
    email: str,
    phone: str,
) -> dict[str, Any]:
    return create_app_user("admin", full_name, username, password, email, phone)


def optimize_image_for_web(frame: np.ndarray, max_width: int = CAPTURE_MAX_WIDTH) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / float(width)
    return cv2.resize(frame, (max_width, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)


def capture_screenshot(frame: np.ndarray, prefix: str) -> str:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_prefix = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in prefix).strip("_")
    filename = f"{safe_prefix or 'capture'}_{timestamp}.jpg"
    full_path = SCREENSHOT_DIR / filename
    cv2.imwrite(
        str(full_path),
        optimize_image_for_web(frame),
        [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY],
    )
    return f"/captures/{filename}"


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("The uploaded image could not be read.")
    return frame


def load_admission_image(row: sqlite3.Row) -> np.ndarray:
    if row["image_blob"]:
        return decode_image_bytes(row["image_blob"])

    raw_path = str(row["image_path"] or "").strip()
    if not raw_path:
        raise ValueError("Admission record has no image_path or image_blob.")

    image_path = Path(raw_path)
    if not image_path.is_absolute():
        image_path = DATABASE_PATH.parent / raw_path
    if not image_path.exists():
        raise ValueError(f"Admission image was not found: {image_path}")

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Admission image could not be read: {image_path}")
    return frame


def admission_sync_status() -> dict[str, Any]:
    connection = get_connection()
    try:
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN synced_at IS NULL AND COALESCE(is_graduated, 0) = 0 THEN 1 ELSE 0 END) AS pending,
                SUM(CASE WHEN synced_at IS NOT NULL THEN 1 ELSE 0 END) AS synced,
                SUM(CASE WHEN sync_error IS NOT NULL AND sync_error != '' THEN 1 ELSE 0 END) AS errors
            FROM admission_students
            """
        ).fetchone()
        return {
            "connected": True,
            "table": "admission_students",
            "total": int(row["total"] or 0),
            "pending": int(row["pending"] or 0),
            "synced": int(row["synced"] or 0),
            "errors": int(row["errors"] or 0),
            "sync_interval_seconds": ADMISSION_SYNC_INTERVAL_SECONDS,
        }
    finally:
        connection.close()


def sync_admission_students(limit: int = ADMISSION_SYNC_BATCH_SIZE) -> dict[str, Any]:
    connection = get_connection()
    try:
        rows = connection.execute(
            """
            SELECT id, name, role, image_path, image_blob
            FROM admission_students
            WHERE synced_at IS NULL
              AND COALESCE(is_graduated, 0) = 0
            ORDER BY datetime(created_at) ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    finally:
        connection.close()

    imported = 0
    failed = 0
    errors: list[dict[str, str]] = []

    for row in rows:
        student_id = str(row["id"]).strip()
        role = str(row["role"] or "Student").strip().title()
        if role not in VALID_ROLES:
            role = "Student"

        try:
            frame = load_admission_image(row)
            embedding, _ = extract_embedding(frame)
            image_path = capture_screenshot(frame, f"admission_{student_id}")
            insert_student_record(student_id, str(row["name"]).strip(), role, embedding, image_path)

            connection = get_connection()
            try:
                connection.execute(
                    """
                    UPDATE admission_students
                    SET synced_at = ?, sync_error = NULL
                    WHERE id = ?
                    """,
                    (datetime.now().isoformat(timespec="seconds"), student_id),
                )
                connection.commit()
            finally:
                connection.close()
            imported += 1
        except Exception as error:
            failed += 1
            message = str(error)
            errors.append({"student_id": student_id, "error": message})
            connection = get_connection()
            try:
                connection.execute(
                    "UPDATE admission_students SET sync_error = ? WHERE id = ?",
                    (message, student_id),
                )
                connection.commit()
            finally:
                connection.close()

    if imported:
        processor.refresh_students()

    status = admission_sync_status()
    return {
        "imported": imported,
        "failed": failed,
        "errors": errors,
        "status": status,
    }


def insert_student_record(
    student_id: str,
    name: str,
    role: str,
    embedding: np.ndarray,
    image_path: Optional[str] = None,
) -> None:
    connection = get_connection()
    try:
        connection.execute(
            """
            INSERT INTO students (id, name, role, embedding, image_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                role = excluded.role,
                embedding = excluded.embedding,
                image_path = COALESCE(excluded.image_path, students.image_path),
                created_at = excluded.created_at
            """,
            (
                student_id,
                name,
                role,
                pickle.dumps(embedding.tolist()),
                image_path,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        connection.execute("DELETE FROM graduated_students WHERE id = ?", (student_id,))
        connection.commit()
    finally:
        connection.close()


def insert_access_log(decision: FrameDecision) -> None:
    connection = get_connection()
    now = datetime.now()
    try:
        connection.execute(
            """
            INSERT INTO access_logs (
                student_id, student_name, role, timestamp, date, status, label, event_type
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision.student_id,
                decision.student_name,
                decision.role,
                now.isoformat(timespec="seconds"),
                now.date().isoformat(),
                decision.log_status,
                decision.label,
                decision.event_type,
            ),
        )
        connection.commit()
    finally:
        connection.close()


def list_students() -> list[dict[str, Any]]:
    connection = get_connection()
    try:
        rows = connection.execute(
            """
            SELECT id, name, role, image_path, created_at
            FROM students
            ORDER BY datetime(created_at) DESC, name COLLATE NOCASE ASC
            """
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def list_graduated_students() -> list[dict[str, Any]]:
    connection = get_connection()
    try:
        rows = connection.execute(
            """
            SELECT id, name, role, requirements_completed, reason, graduated_at
            FROM graduated_students
            ORDER BY datetime(graduated_at) DESC, name COLLATE NOCASE ASC
            """
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def move_student_to_graduated(student_id: str) -> bool:
    connection = get_connection()
    try:
        row = connection.execute(
            "SELECT id, name, role FROM students WHERE id = ?",
            (student_id,),
        ).fetchone()
        if row is None:
            return False

        connection.execute(
            """
            INSERT INTO graduated_students (id, name, role, requirements_completed, reason, graduated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                role = excluded.role,
                requirements_completed = excluded.requirements_completed,
                reason = excluded.reason,
                graduated_at = excluded.graduated_at
            """,
            (
                row["id"],
                row["name"],
                row["role"],
                1,
                "Graduated - all requirements completed",
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        connection.execute("DELETE FROM students WHERE id = ?", (student_id,))
        connection.commit()
        return True
    finally:
        connection.close()


def bulk_delete_graduates(student_ids: list[str]) -> int:
    connection = get_connection()
    try:
        placeholders = ", ".join("?" for _ in student_ids)
        deleted_count = connection.execute(
            f"SELECT COUNT(*) AS count FROM graduated_students WHERE id IN ({placeholders})",
            student_ids,
        ).fetchone()["count"]
        connection.execute(f"DELETE FROM graduated_students WHERE id IN ({placeholders})", student_ids)
        connection.execute(f"DELETE FROM students WHERE id IN ({placeholders})", student_ids)
        connection.execute(f"DELETE FROM access_logs WHERE student_id IN ({placeholders})", student_ids)
        connection.commit()
        return int(deleted_count)
    finally:
        connection.close()


def emit_security_event(decision: FrameDecision, image_path: str) -> None:
    severity = "orange" if decision.log_status == "Manual ID Required" else "red"
    payload = {
        "event_type": decision.event_type,
        "status": decision.log_status,
        "label": decision.label,
        "student_id": decision.student_id,
        "student_name": decision.student_name,
        "role": decision.role,
        "accessory_state": decision.accessory_state,
        "severity": severity,
        "alarm": decision.should_alarm,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "image_path": image_path,
    }
    socketio.emit("security_event", payload)
    print(f"EVENT: {json.dumps(payload)}")


def placeholder_frame(message: str) -> np.ndarray:
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    frame[:] = (12, 18, 32)
    cv2.putText(frame, "Camera Offline", (60, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (59, 130, 246), 3)
    cv2.putText(frame, message[:70], (60, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def detect_media_upload_type(media_file: Any) -> str:
    filename = str(getattr(media_file, "filename", "") or "")
    content_type = str(getattr(media_file, "mimetype", "") or "").lower()
    extension = Path(filename).suffix.lower()

    if content_type.startswith("image/") or extension in MEDIA_SCAN_IMAGE_EXTENSIONS:
        return "image"
    if content_type.startswith("video/") or extension in MEDIA_SCAN_VIDEO_EXTENSIONS:
        return "video"
    raise ValueError("Unsupported media type. Please upload an image or video file.")


def media_decision_severity(decision: FrameDecision) -> str:
    if decision.log_status in {"Unknown", "Denied"}:
        return "red"
    if decision.log_status == "Manual ID Required":
        return "orange"
    if decision.student_id is None:
        return "orange"
    return "green"


def analyze_media_frame(frame: np.ndarray, students: list[Any]) -> list[FrameDecision]:
    detections: list[tuple[Optional[np.ndarray], tuple[int, int, int, int]]] = []
    try:
        detections = [
            (embedding, bbox)
            for embedding, bbox in extract_face_embeddings(frame)
        ]
    except Exception:
        fallback_bbox = fallback_person_box(frame)
        if fallback_bbox is not None:
            detections = [(None, fallback_bbox)]

    now = time.time()
    decisions: list[FrameDecision] = []
    for embedding, bbox in detections:
        decision = evaluate_face_candidate(frame, bbox, embedding, students, now)
        if decision is not None:
            decisions.append(decision)
    return decisions


def annotate_media_frame(frame: np.ndarray, decisions: list[FrameDecision]) -> np.ndarray:
    annotated = frame.copy()
    for decision in decisions:
        x, y, w, h = decision.bbox
        cv2.rectangle(annotated, (x, y), (x + w, y + h), decision.color, 2)
        label = (decision.student_name or decision.label)[:48]
        cv2.putText(
            annotated,
            label,
            (x, max(28, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            decision.color,
            2,
        )
    return annotated


def add_media_scan_decisions(
    *,
    decisions: list[FrameDecision],
    entered_by_id: dict[str, dict[str, Any]],
    alerts_by_key: dict[str, dict[str, Any]],
    frame_number: int,
    second: Optional[float],
) -> int:
    detected_faces = 0
    rounded_second = None if second is None else round(second, 2)

    for decision in decisions:
        detected_faces += 1
        if decision.student_id:
            entry = entered_by_id.setdefault(
                decision.student_id,
                {
                    "student_id": decision.student_id,
                    "name": decision.student_name,
                    "role": decision.role,
                    "first_frame": frame_number,
                    "first_second": rounded_second,
                    "detections": 0,
                },
            )
            entry["detections"] += 1
            continue

        severity = media_decision_severity(decision)
        alert_key = f"{decision.log_status}:{decision.accessory_state}:{decision.label}"
        alert = alerts_by_key.setdefault(
            alert_key,
            {
                "status": decision.log_status,
                "label": decision.label,
                "event_type": decision.event_type or decision.log_status,
                "accessory_state": decision.accessory_state,
                "severity": severity,
                "first_frame": frame_number,
                "first_second": rounded_second,
                "detections": 0,
            },
        )
        alert["detections"] += 1

    return detected_faces


def build_media_scan_payload(
    *,
    filename: str,
    media_type: str,
    students: list[Any],
    entered_by_id: dict[str, dict[str, Any]],
    alerts_by_key: dict[str, dict[str, Any]],
    processed_frames: int,
    detected_faces: int,
    duration_seconds: Optional[float] = None,
    snapshot_path: Optional[str] = None,
) -> dict[str, Any]:
    entered_ids = set(entered_by_id)
    missing = [
        {
            "student_id": student.student_id,
            "name": student.name,
            "role": student.role,
        }
        for student in students
        if student.student_id not in entered_ids
    ]
    entered = sorted(
        entered_by_id.values(),
        key=lambda item: (item["first_frame"], str(item["name"] or "").lower()),
    )
    alerts = sorted(
        alerts_by_key.values(),
        key=lambda item: (item["first_frame"], str(item["label"] or "").lower()),
    )

    return {
        "message": "Media scan completed.",
        "filename": filename,
        "media_type": media_type,
        "processed_frames": processed_frames,
        "detected_faces": detected_faces,
        "duration_seconds": None if duration_seconds is None else round(duration_seconds, 2),
        "entered": entered,
        "entered_count": len(entered),
        "missing": missing,
        "missing_count": len(missing),
        "alerts": alerts,
        "alert_count": len(alerts),
        "snapshot_path": snapshot_path,
    }


def scan_image_media(frame: np.ndarray, filename: str, students: list[Any]) -> dict[str, Any]:
    entered_by_id: dict[str, dict[str, Any]] = {}
    alerts_by_key: dict[str, dict[str, Any]] = {}
    decisions = analyze_media_frame(frame, students)
    detected_faces = add_media_scan_decisions(
        decisions=decisions,
        entered_by_id=entered_by_id,
        alerts_by_key=alerts_by_key,
        frame_number=0,
        second=0.0,
    )
    snapshot_path = capture_screenshot(annotate_media_frame(frame, decisions), "media_scan_image") if decisions else None

    return build_media_scan_payload(
        filename=filename,
        media_type="image",
        students=students,
        entered_by_id=entered_by_id,
        alerts_by_key=alerts_by_key,
        processed_frames=1,
        detected_faces=detected_faces,
        duration_seconds=None,
        snapshot_path=snapshot_path,
    )


def iter_video_sample_frames(video_path: Path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("The uploaded video could not be opened.")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or TARGET_FPS or 30)
        if fps <= 0:
            fps = float(TARGET_FPS or 30)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        sample_every = max(1, int(fps * MEDIA_SCAN_MIN_VIDEO_SAMPLE_SECONDS))
        duration_seconds = frame_count / fps if frame_count > 0 else None

        if frame_count > 0:
            estimated_samples = max(1, (frame_count + sample_every - 1) // sample_every)
            if estimated_samples > MEDIA_SCAN_MAX_VIDEO_FRAMES:
                sample_every = max(sample_every, frame_count // MEDIA_SCAN_MAX_VIDEO_FRAMES)

            emitted = 0
            for frame_number in range(0, frame_count, sample_every):
                if emitted >= MEDIA_SCAN_MAX_VIDEO_FRAMES:
                    break
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ok, frame = capture.read()
                if not ok:
                    continue
                emitted += 1
                yield frame, frame_number, frame_number / fps, duration_seconds
            return

        frame_number = 0
        emitted = 0
        while emitted < MEDIA_SCAN_MAX_VIDEO_FRAMES:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_number % sample_every == 0:
                emitted += 1
                yield frame, frame_number, frame_number / fps, duration_seconds
            frame_number += 1
    finally:
        capture.release()


def scan_video_media(video_path: Path, filename: str, students: list[Any]) -> dict[str, Any]:
    entered_by_id: dict[str, dict[str, Any]] = {}
    alerts_by_key: dict[str, dict[str, Any]] = {}
    processed_frames = 0
    detected_faces = 0
    snapshot_path: Optional[str] = None
    duration_seconds: Optional[float] = None

    for frame, frame_number, second, video_duration in iter_video_sample_frames(video_path):
        processed_frames += 1
        duration_seconds = video_duration
        decisions = analyze_media_frame(frame, students)
        detected_faces += add_media_scan_decisions(
            decisions=decisions,
            entered_by_id=entered_by_id,
            alerts_by_key=alerts_by_key,
            frame_number=frame_number,
            second=second,
        )
        if decisions and snapshot_path is None:
            snapshot_path = capture_screenshot(annotate_media_frame(frame, decisions), "media_scan_video")

    if processed_frames == 0:
        raise ValueError("The uploaded video did not contain readable frames.")

    return build_media_scan_payload(
        filename=filename,
        media_type="video",
        students=students,
        entered_by_id=entered_by_id,
        alerts_by_key=alerts_by_key,
        processed_frames=processed_frames,
        detected_faces=detected_faces,
        duration_seconds=duration_seconds,
        snapshot_path=snapshot_path,
    )


def scan_uploaded_media(media_file: Any) -> dict[str, Any]:
    filename = str(getattr(media_file, "filename", "") or "").strip()
    if not filename:
        raise ValueError("media file is required")

    media_type = detect_media_upload_type(media_file)
    students = load_database()

    if media_type == "image":
        frame = decode_image_bytes(media_file.read())
        return scan_image_media(frame, filename, students)

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix.lower()
    if suffix not in MEDIA_SCAN_VIDEO_EXTENSIONS:
        suffix = ".mp4"
    with tempfile.NamedTemporaryFile(dir=SCREENSHOT_DIR, suffix=suffix, delete=False) as temp_file:
        media_file.save(temp_file)
        temp_path = Path(temp_file.name)

    try:
        return scan_video_media(temp_path, filename, students)
    finally:
        temp_path.unlink(missing_ok=True)


class WebFaceProcessor:
    def __init__(self) -> None:
        self.students = load_database()
        self.capture: Optional[cv2.VideoCapture] = None
        self.capture_error: Optional[str] = None
        self.camera_backend: Optional[str] = None
        self.read_failures = 0
        self.camera_enabled = True
        self.running = False
        self.current_frame: Optional[np.ndarray] = None
        self.current_decision: Optional[FrameDecision] = None
        self.state_lock = threading.Lock()
        self.frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self.frame_index = 0
        self.cooldowns: dict[str, float] = {}

    def refresh_students(self) -> None:
        self.students = load_database()

    def _drain_frame_queue(self) -> None:
        while True:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                return

    def _release_camera(self) -> None:
        if self.capture is not None:
            self.capture.release()
        self.capture = None
        self.camera_backend = None

    def set_camera_enabled(self, enabled: bool) -> dict[str, Any]:
        self.camera_enabled = enabled
        self._drain_frame_queue()
        self.cooldowns.clear()
        with self.state_lock:
            self.current_decision = None
            if not enabled:
                self.current_frame = None

        if enabled:
            self.capture_error = None
            self._open_camera()
        else:
            self._release_camera()
            self.capture_error = CAMERA_DISABLED_MESSAGE

        return self.system_status()

    def _configure_camera(self, capture: cv2.VideoCapture) -> None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
        capture.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    def _mirror_frame(self, frame: np.ndarray) -> np.ndarray:
        return cv2.flip(frame, 1) if CAMERA_MIRROR else frame

    def _open_camera(self) -> None:
        if not self.camera_enabled:
            self._release_camera()
            self.capture_error = CAMERA_DISABLED_MESSAGE
            return

        if self.capture is not None and self.capture.isOpened():
            return

        self._release_camera()
        attempted_sources: list[str] = []
        ordered_indices = (CAMERA_INDEX,) + tuple(index for index in CAMERA_INDICES if index != CAMERA_INDEX)
        for camera_index in ordered_indices:
            for backend_name, backend in CAMERA_BACKENDS:
                attempted_sources.append(f"{backend_name}:{camera_index}")
                capture = (
                    cv2.VideoCapture(camera_index)
                    if backend is None
                    else cv2.VideoCapture(camera_index, backend)
                )
                if not capture.isOpened():
                    capture.release()
                    continue

                self._configure_camera(capture)
                ok, frame = capture.read()
                if not ok or frame is None or frame.size == 0:
                    capture.release()
                    continue

                frame = self._mirror_frame(frame)
                self.capture = capture
                self.camera_backend = f"{backend_name} camera {camera_index}"
                self.capture_error = None
                self.read_failures = 0
                with self.state_lock:
                    self.current_frame = frame.copy()
                return

        self.capture_error = (
            f"Unable to open a camera. Tried: {', '.join(attempted_sources)}."
        )

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        threading.Thread(target=self._reader_loop, daemon=True).start()
        threading.Thread(target=self._worker_loop, daemon=True).start()

    def _reader_loop(self) -> None:
        while self.running:
            if not self.camera_enabled:
                self._release_camera()
                self._drain_frame_queue()
                self.capture_error = CAMERA_DISABLED_MESSAGE
                with self.state_lock:
                    self.current_frame = None
                    self.current_decision = None
                time.sleep(0.2)
                continue

            if self.capture is None or not self.capture.isOpened():
                self._open_camera()
                if self.capture is None or not self.capture.isOpened():
                    time.sleep(1.0)
                    continue

            if self.capture is None:
                time.sleep(0.2)
                continue

            ok, frame = self.capture.read()
            if not ok or frame is None or frame.size == 0:
                self.read_failures += 1
                self.capture_error = "Camera connected but no frame could be grabbed."
                if self.read_failures >= 5:
                    self._release_camera()
                    self.capture_error = "Camera stream dropped. Reconnecting..."
                    self.read_failures = 0
                time.sleep(0.1)
                continue

            frame = self._mirror_frame(frame)
            self.capture_error = None
            self.read_failures = 0
            self.frame_index += 1
            with self.state_lock:
                self.current_frame = frame.copy()

            if self.frame_index % FRAME_SKIP_INTERVAL == 0:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame)

            time.sleep(1 / TARGET_FPS)

    def _cleanup_cooldowns(self, now: float) -> None:
        expired = [key for key, expires_at in self.cooldowns.items() if expires_at <= now]
        for key in expired:
            self.cooldowns.pop(key, None)

    def _cooldown_seconds_for_decision(self, decision: FrameDecision) -> float:
        if decision.log_status == "Allowed" and decision.student_id:
            return KNOWN_FACE_RECHECK_COOLDOWN_SECONDS
        return IDENTIFICATION_COOLDOWN_SECONDS

    def _worker_loop(self) -> None:
        while self.running:
            if not self.camera_enabled:
                self._drain_frame_queue()
                time.sleep(0.2)
                continue

            try:
                frame = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if not self.camera_enabled:
                continue

            decision = evaluate_frame(frame, self.students)
            if decision is None or not self.camera_enabled:
                continue

            now = time.time()
            self._cleanup_cooldowns(now)
            is_in_cooldown = self.cooldowns.get(decision.cooldown_key, 0.0) > now

            with self.state_lock:
                self.current_decision = decision

            if is_in_cooldown:
                continue

            if not self.camera_enabled:
                continue

            self.cooldowns[decision.cooldown_key] = now + self._cooldown_seconds_for_decision(decision)
            insert_access_log(decision)

            if decision.should_capture:
                prefix = {
                    "Denied": "intruder",
                    "Manual ID Required": "manual_id",
                }.get(decision.log_status, "unknown")
                image_path = capture_screenshot(frame, prefix)
                if decision.should_alert:
                    emit_security_event(decision, image_path)
                if decision.should_alarm:
                    trigger_alarm()
    def capture_current_frame(self) -> np.ndarray:
        if not self.camera_enabled:
            raise RuntimeError(CAMERA_DISABLED_MESSAGE)

        with self.state_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()

        self._open_camera()
        if self.capture is None or not self.capture.isOpened():
            raise RuntimeError(self.capture_error or "Unable to open the camera.")

        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError("Unable to capture frame from camera.")

        frame = self._mirror_frame(frame)
        with self.state_lock:
            self.current_frame = frame.copy()
        return frame

    def get_annotated_frame(self) -> np.ndarray:
        if not self.camera_enabled:
            return placeholder_frame(CAMERA_DISABLED_MESSAGE)

        with self.state_lock:
            frame = None if self.current_frame is None else self.current_frame.copy()
            decision = self.current_decision

        if frame is None:
            return placeholder_frame(self.capture_error or "Waiting for camera...")

        if decision and decision.matched_until < time.time():
            decision = None
            with self.state_lock:
                if self.current_decision and self.current_decision.matched_until < time.time():
                    self.current_decision = None

        if decision:
            x, y, w, h = decision.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), decision.color, 2)
            cv2.putText(
                frame,
                decision.label,
                (x, max(30, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                decision.color,
                2,
            )

        return optimize_image_for_web(frame)

    def generate_stream(self):
        while True:
            frame = self.get_annotated_frame()
            success, buffer = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY],
            )
            if not success:
                time.sleep(0.03)
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frame_bytes)).encode("ascii") + b"\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )
            time.sleep(STREAM_FRAME_INTERVAL_SECONDS)

    def system_status(self) -> dict[str, Any]:
        with self.state_lock:
            camera_enabled = self.camera_enabled
            decision = self.current_decision if camera_enabled else None
        return {
            "camera_enabled": camera_enabled,
            "camera_ready": camera_enabled and self.capture is not None and self.capture.isOpened(),
            "camera_error": CAMERA_DISABLED_MESSAGE if not camera_enabled else self.capture_error,
            "camera_backend": self.camera_backend,
            "frame_skip_interval": FRAME_SKIP_INTERVAL,
            "identification_cooldown_seconds": IDENTIFICATION_COOLDOWN_SECONDS,
            "known_face_cooldown_seconds": KNOWN_FACE_RECHECK_COOLDOWN_SECONDS,
            "current_decision": None if decision is None else {
                "label": decision.label,
                "status": decision.log_status,
                "outcome": decision.outcome,
                "accessory_state": decision.accessory_state,
                "severity": (
                    "green"
                    if decision.log_status == "Allowed"
                    else "orange"
                    if decision.log_status == "Manual ID Required"
                    else "red"
                ),
            },
        }


ensure_database_schema()
processor = WebFaceProcessor()
processor.start()


def admissions_sync_loop() -> None:
    while True:
        try:
            sync_admission_students()
        except Exception as error:
            print(f"ADMISSION_SYNC_ERROR: {error}")
        time.sleep(ADMISSION_SYNC_INTERVAL_SECONDS)


threading.Thread(target=admissions_sync_loop, daemon=True).start()


@app.errorhandler(404)
def handle_not_found(error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Resource not found"}), 404
    return error


@app.errorhandler(Exception)
def handle_exception(error):
    if request.path.startswith("/api/"):
        return jsonify({"error": str(error)}), 500
    raise error


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/auth/me")
def auth_me():
    return jsonify({"user": get_current_user()})


@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    payload = request.get_json(silent=True) or {}
    role = str(payload.get("role", "")).strip().lower()
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", ""))

    if role not in ACCOUNT_ROLES or not username or not password:
        return jsonify({"error": "role, username, and password are required"}), 400

    try:
        user = authenticate_user(role, username, password)
    except ValueError as error:
        return jsonify({"error": str(error)}), 401

    session.clear()
    session["user"] = user
    return jsonify({
        "user": user,
        "created": False,
        "message": "Signed in successfully.",
    })


@app.route("/api/auth/register-admissions-employee", methods=["POST"])
def register_admissions_employee():
    payload = request.get_json(silent=True) or {}
    full_name = str(payload.get("full_name", "")).strip()
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", ""))
    email = str(payload.get("email", "")).strip()
    phone = str(payload.get("phone", "")).strip()

    if not full_name or not username or not password:
        return jsonify({"error": "full_name, username, and password are required"}), 400
    if len(password) < 4:
        return jsonify({"error": "Password must be at least 4 characters"}), 400

    try:
        user = create_admissions_employee(full_name, username, password, email, phone)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    return jsonify({
        "user": user,
        "message": "Admissions employee account created successfully.",
    }), 201


@app.route("/api/security-users", methods=["POST"])
@require_auth
@require_role("admin")
def create_security_user():
    payload = request.get_json(silent=True) or {}
    full_name = str(payload.get("full_name", "")).strip()
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", ""))
    email = str(payload.get("email", "")).strip()
    phone = str(payload.get("phone", "")).strip()

    if not full_name or not username or not password:
        return jsonify({"error": "full_name, username, and password are required"}), 400
    if len(password) < 4:
        return jsonify({"error": "Password must be at least 4 characters"}), 400

    try:
        user = create_app_user("security", full_name, username, password, email, phone)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    return jsonify({
        "user": user,
        "message": "Security employee account created successfully.",
    }), 201


@app.route("/api/auth/logout", methods=["POST"])
@require_auth
def auth_logout():
    session.clear()
    return jsonify({"message": "Signed out successfully."})


@app.route("/api/bootstrap")
@require_auth
def bootstrap():
    user = get_current_user()
    payload: dict[str, Any] = {"user": user}
    if user and user.get("role") == "admin":
        payload.update({
            "students": list_students(),
            "graduates": list_graduated_students(),
            "admissions": admission_sync_status(),
        })
    if user and user.get("role") == "security":
        payload.update({
            "system": processor.system_status(),
        })
    return jsonify(payload)


@app.route("/video_feed")
@require_role("security")
def video_feed():
    return Response(
        processor.generate_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/captures/<path:filename>")
@require_auth
def get_capture(filename: str):
    return send_from_directory(SCREENSHOT_DIR, filename)


@app.route("/api/system/status")
@require_role("security")
def get_system_status():
    return jsonify(processor.system_status())


@app.route("/api/system/camera", methods=["POST"])
@require_role("security")
def set_system_camera():
    payload = request.get_json(silent=True) or {}
    if "enabled" not in payload:
        return jsonify({"error": "enabled is required"}), 400
    return jsonify(processor.set_camera_enabled(bool(payload["enabled"])))


@app.route("/api/security/media-scan", methods=["POST"])
@require_role("security")
def security_media_scan():
    media_file = request.files.get("media")
    if media_file is None or not media_file.filename:
        return jsonify({"error": "media file is required"}), 400

    try:
        return jsonify(scan_uploaded_media(media_file))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400


@app.route("/api/admissions/status")
@require_role("admin")
def get_admissions_status():
    return jsonify(admission_sync_status())


@app.route("/api/admissions/sync", methods=["POST"])
@require_role("admin")
def sync_admissions_now():
    return jsonify(sync_admission_students())


@app.route("/api/students")
@require_role("admin")
def get_students():
    return jsonify({"items": list_students()})


@app.route("/api/students/<student_id>/graduate", methods=["POST"])
@require_role("admin")
def graduate_student(student_id: str):
    if not move_student_to_graduated(student_id):
        return jsonify({"error": "Student not found"}), 404
    processor.refresh_students()
    return jsonify({
        "message": f"Student {student_id} moved to graduated records.",
        "students": list_students(),
        "graduates": list_graduated_students(),
    })


@app.route("/api/graduates")
@require_role("admin")
def get_graduates():
    return jsonify({"items": list_graduated_students()})


@app.route("/api/graduates/delete", methods=["POST"])
@require_role("admin")
def delete_graduates():
    payload = request.get_json(silent=True) or {}
    student_ids = [
        str(student_id).strip()
        for student_id in payload.get("student_ids", [])
        if str(student_id).strip()
    ]
    if not student_ids:
        return jsonify({"error": "student_ids is required"}), 400

    deleted_count = bulk_delete_graduates(student_ids)
    processor.refresh_students()
    return jsonify({
        "message": "Selected graduates permanently deleted.",
        "deleted_count": deleted_count,
        "graduates": list_graduated_students(),
    })


@app.route("/api/logs")
@require_auth
def get_logs():
    date_value = str(request.args.get("date", "")).strip()
    search_value = str(request.args.get("search", "")).strip().lower()
    status_value = str(request.args.get("status", "")).strip()

    query = """
        SELECT id, student_id, student_name, role, timestamp, date, status, label, event_type
        FROM access_logs
        WHERE 1 = 1
    """
    params: list[Any] = []

    if date_value:
        query += " AND date = ?"
        params.append(date_value)
    if status_value:
        query += " AND status = ?"
        params.append(status_value)
    if search_value:
        query += """
            AND (
                LOWER(COALESCE(student_id, '')) LIKE ?
                OR LOWER(COALESCE(student_name, '')) LIKE ?
                OR LOWER(COALESCE(label, '')) LIKE ?
            )
        """
        wildcard = f"%{search_value}%"
        params.extend([wildcard, wildcard, wildcard])

    query += " ORDER BY timestamp DESC LIMIT 500"

    connection = get_connection()
    try:
        rows = connection.execute(query, params).fetchall()
        items = [dict(row) for row in rows]
    finally:
        connection.close()

    return jsonify({"items": items})


@app.route("/api/register", methods=["POST"])
@require_role("admin")
def register():
    if request.content_type and request.content_type.startswith("multipart/form-data"):
        student_id = str(request.form.get("student_id", "")).strip()
        name = str(request.form.get("name", "")).strip()
        role = str(request.form.get("role", "")).strip().title()
        image_file = request.files.get("image")
    else:
        payload = request.get_json(silent=True) or {}
        student_id = str(payload.get("student_id", "")).strip()
        name = str(payload.get("name", "")).strip()
        role = str(payload.get("role", "")).strip().title()
        image_file = None

    if not student_id or not name or role not in VALID_ROLES:
        return (
            jsonify({
                "error": "student_id, name, and role are required. role must be Student, Staff, Admin, or Security."
            }),
            400,
        )

    try:
        if image_file is not None and image_file.filename:
            frame = decode_image_bytes(image_file.read())
        else:
            frame = processor.capture_current_frame()
        embedding, _ = extract_embedding(frame)
    except Exception as error:
        return jsonify({"error": str(error)}), 400

    capture_path = capture_screenshot(frame, f"register_{student_id}")
    insert_student_record(student_id, name, role, embedding, capture_path)
    processor.refresh_students()

    return jsonify({
        "student_id": student_id,
        "name": name,
        "role": role,
        "capture_path": capture_path,
        "message": "Registration completed and embedding saved immediately.",
        "students": list_students(),
        "graduates": list_graduated_students(),
    })


if __name__ == "__main__":
    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )
