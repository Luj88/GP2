import pickle
import sqlite3
from pathlib import Path
from typing import Iterable

from deepface import DeepFace


GP2_DIR = Path(__file__).resolve().parents[1]
DATABASE_PATH = GP2_DIR / "university.db"
IMAGES_DIR = GP2_DIR / "students_images"
VALID_ROLES = {"student", "staff", "admin", "security"}


def database_exists(database_path: Path = DATABASE_PATH) -> bool:
    return database_path.exists()


def create_connection(database_path: Path = DATABASE_PATH) -> sqlite3.Connection:
    return sqlite3.connect(database_path)


def create_students_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS students (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('Student', 'Staff', 'Admin', 'Security')),
            embedding BLOB NOT NULL
        )
        """
    )
    connection.commit()


def create_access_logs_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            timestamp TEXT NOT NULL,
            date TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('Allowed', 'Unknown', 'Covered')),
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
        """
    )
    connection.commit()


def iter_image_files(images_dir: Path = IMAGES_DIR) -> Iterable[Path]:
    supported_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for path in images_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in supported_suffixes:
            yield path


def parse_student_metadata(image_path: Path) -> tuple[str, str, str]:
    role_name = image_path.parent.name.strip().lower()
    role = role_name.capitalize() if role_name in VALID_ROLES else "Student"

    stem_parts = image_path.stem.split("_", 1)
    if len(stem_parts) != 2:
        raise ValueError(
            f"Invalid filename format for '{image_path.name}'. Use 'ID_Name.ext'."
        )

    student_id, raw_name = stem_parts
    name = raw_name.replace("_", " ").strip()
    if not student_id.strip() or not name:
        raise ValueError(
            f"Invalid filename format for '{image_path.name}'. Use 'ID_Name.ext'."
        )

    return student_id.strip(), name, role


def extract_embedding(image_path: Path) -> bytes:
    representation = DeepFace.represent(
        img_path=str(image_path),
        model_name="VGG-Face",
        enforce_detection=True,
    )
    if not representation:
        raise ValueError(f"No embedding returned for '{image_path.name}'.")

    return pickle.dumps(representation[0]["embedding"])


def insert_student(
    connection: sqlite3.Connection,
    student_id: str,
    name: str,
    role: str,
    embedding: bytes,
) -> None:
    connection.execute(
        """
        INSERT INTO students (id, name, role, embedding)
        VALUES (?, ?, ?, ?)
        """,
        (student_id, name, role, embedding),
    )


def build_database(
    images_dir: Path = IMAGES_DIR,
    database_path: Path = DATABASE_PATH,
) -> None:
    connection = create_connection(database_path)
    try:
        create_students_table(connection)
        create_access_logs_table(connection)

        if database_exists(database_path):
            existing_students = connection.execute("SELECT COUNT(*) FROM students").fetchone()[0]
            if existing_students > 0:
                print(f"Database already exists at '{database_path}'. Skipping creation.")
                return

        if not images_dir.exists():
            print(f"Database schema ensured at '{database_path}'.")
            return

        image_files = list(iter_image_files(images_dir))
        if not image_files:
            print(f"Database schema ensured at '{database_path}'.")
            return

        for image_path in image_files:
            try:
                student_id, name, role = parse_student_metadata(image_path)
                embedding = extract_embedding(image_path)
                insert_student(connection, student_id, name, role, embedding)
                print(f"Saved embedding for {student_id} - {name} ({role})")
            except Exception as exc:
                print(f"Skipping '{image_path.name}': {exc}")
        connection.commit()
        print(f"Database created successfully at '{database_path}'.")
    finally:
        connection.close()


if __name__ == "__main__":
    build_database()
