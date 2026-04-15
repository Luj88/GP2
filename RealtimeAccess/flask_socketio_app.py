import json
import pickle
import queue
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from flask_socketio import SocketIO

from realtime_face_access import (
    DATABASE_PATH,
    IDENTIFICATION_COOLDOWN_SECONDS,
    TARGET_FPS,
    FrameDecision,
    evaluate_frame,
    extract_embedding,
    load_database,
    trigger_alarm,
)


REALTIME_DIR = Path(__file__).resolve().parent
SCREENSHOT_DIR = REALTIME_DIR / "captures"
STREAM_JPEG_QUALITY = 72
CAPTURE_MAX_WIDTH = 960
FRAME_SKIP_INTERVAL = 10
VALID_ROLES = {"Student", "Staff", "Admin", "Security"}

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = "university-face-access"
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
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        elif "created_at" not in _column_names(connection, "students"):
            connection.execute(
                """
                ALTER TABLE students
                ADD COLUMN created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                """
            )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS graduated_students (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                role TEXT NOT NULL,
                graduated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
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


def optimize_image_for_web(frame: np.ndarray, max_width: int = CAPTURE_MAX_WIDTH) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / float(width)
    return cv2.resize(frame, (max_width, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)


def capture_screenshot(frame: np.ndarray, prefix: str) -> str:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{prefix}_{timestamp}.jpg"
    full_path = SCREENSHOT_DIR / filename
    cv2.imwrite(
        str(full_path),
        optimize_image_for_web(frame),
        [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY],
    )
    return f"/captures/{filename}"


def insert_student_record(student_id: str, name: str, role: str, embedding: np.ndarray) -> None:
    connection = get_connection()
    try:
        connection.execute(
            """
            INSERT INTO students (id, name, role, embedding, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                role = excluded.role,
                embedding = excluded.embedding,
                created_at = excluded.created_at
            """,
            (
                student_id,
                name,
                role,
                pickle.dumps(embedding.tolist()),
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
            SELECT id, name, role, created_at
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
            SELECT id, name, role, graduated_at
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
            INSERT INTO graduated_students (id, name, role, graduated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                role = excluded.role,
                graduated_at = excluded.graduated_at
            """,
            (
                row["id"],
                row["name"],
                row["role"],
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
    payload = {
        "event_type": decision.event_type,
        "status": decision.log_status,
        "label": decision.label,
        "student_id": decision.student_id,
        "student_name": decision.student_name,
        "role": decision.role,
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


class WebFaceProcessor:
    def __init__(self) -> None:
        self.students = load_database()
        self.capture: Optional[cv2.VideoCapture] = None
        self.capture_error: Optional[str] = None
        self.running = False
        self.current_frame: Optional[np.ndarray] = None
        self.current_decision: Optional[FrameDecision] = None
        self.state_lock = threading.Lock()
        self.frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self.frame_index = 0
        self.cooldowns: dict[str, float] = {}

    def refresh_students(self) -> None:
        self.students = load_database()

    def _open_camera(self) -> None:
        if self.capture is not None and self.capture.isOpened():
            return
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.capture.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        self.capture_error = None if self.capture.isOpened() else "Unable to open the default camera with DirectShow."

    def start(self) -> None:
        if self.running:
            return
        self._open_camera()
        if self.capture is None or not self.capture.isOpened():
            return
        self.running = True
        threading.Thread(target=self._reader_loop, daemon=True).start()
        threading.Thread(target=self._worker_loop, daemon=True).start()

    def _reader_loop(self) -> None:
        while self.running:
            if self.capture is None:
                time.sleep(0.1)
                continue

            ok, frame = self.capture.read()
            if not ok:
                self.capture_error = "Camera connected but no frame could be grabbed."
                time.sleep(0.05)
                continue

            self.capture_error = None
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

    def _worker_loop(self) -> None:
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            decision = evaluate_frame(frame, self.students)
            if decision is None:
                continue

            now = time.time()
            self._cleanup_cooldowns(now)
            if self.cooldowns.get(decision.cooldown_key, 0.0) > now:
                continue

            self.cooldowns[decision.cooldown_key] = now + IDENTIFICATION_COOLDOWN_SECONDS
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

            with self.state_lock:
                self.current_decision = decision

    def capture_current_frame(self) -> np.ndarray:
        with self.state_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()

        self._open_camera()
        if self.capture is None or not self.capture.isOpened():
            raise RuntimeError(self.capture_error or "Unable to open the camera.")

        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError("Unable to capture frame from camera.")

        with self.state_lock:
            self.current_frame = frame.copy()
        return frame

    def get_annotated_frame(self) -> np.ndarray:
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

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )

    def system_status(self) -> dict[str, Any]:
        with self.state_lock:
            decision = self.current_decision
        return {
            "camera_ready": self.capture is not None and self.capture.isOpened(),
            "camera_error": self.capture_error,
            "frame_skip_interval": FRAME_SKIP_INTERVAL,
            "identification_cooldown_seconds": IDENTIFICATION_COOLDOWN_SECONDS,
            "current_decision": None if decision is None else {
                "label": decision.label,
                "status": decision.log_status,
                "outcome": decision.outcome,
            },
        }


ensure_database_schema()
processor = WebFaceProcessor()
processor.start()


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


@app.route("/video_feed")
def video_feed():
    return Response(
        processor.generate_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/captures/<path:filename>")
def get_capture(filename: str):
    return send_from_directory(SCREENSHOT_DIR, filename)


@app.route("/api/system/status")
def get_system_status():
    return jsonify(processor.system_status())


@app.route("/api/students")
def get_students():
    return jsonify({"items": list_students()})


@app.route("/api/students/<student_id>/graduate", methods=["POST"])
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
def get_graduates():
    return jsonify({"items": list_graduated_students()})


@app.route("/api/graduates/delete", methods=["POST"])
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
def register():
    payload = request.get_json(silent=True) or {}
    student_id = str(payload.get("student_id", "")).strip()
    name = str(payload.get("name", "")).strip()
    role = str(payload.get("role", "")).strip().title()

    if not student_id or not name or role not in VALID_ROLES:
        return (
            jsonify({
                "error": "student_id, name, and role are required. role must be Student, Staff, Admin, or Security."
            }),
            400,
        )

    try:
        frame = processor.capture_current_frame()
        embedding, _ = extract_embedding(frame)
    except Exception as error:
        return jsonify({"error": str(error)}), 400

    capture_path = capture_screenshot(frame, f"register_{student_id}")
    insert_student_record(student_id, name, role, embedding)
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
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
