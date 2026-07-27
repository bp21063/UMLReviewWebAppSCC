import random
import string
from datetime import datetime
from typing import Any, Dict, List, Optional

from . import google_drive_service as drive_svc

ASSIGNMENTS_SHEET = "assignments"
LOGS_SHEET = "logs"


def _generate_assignment_id() -> str:
    chars = string.ascii_uppercase + string.digits
    suffix = "".join(random.choices(chars, k=6))
    return f"UMLR-{suffix}"


def _generate_pin() -> str:
    return "".join(random.choices(string.digits, k=4))


def _assignment_row_to_dict(row: List[str]) -> Dict[str, Any]:
    fields = row + [""] * (10 - len(row))
    return {
        "assignment_id": fields[0],
        "pin": fields[1],
        "diagram_type": fields[2],
        "title": fields[3],
        "answer_image_file_id": fields[4],
        "additional_instructions": fields[5],
        "scoring_criteria": fields[6],
        "enable_logging": fields[7] == "True",
        "simulator_type": fields[8] or "none",
        "created_at": fields[9],
    }


def _log_row_to_dict(row: List[str]) -> Dict[str, Any]:
    fields = row + [""] * (5 - len(row))
    return {
        "assignment_id": fields[0],
        "student_id": fields[1],
        "timestamp": fields[2],
        "score": int(fields[3]) if fields[3] else None,
        "feedback": fields[4],
    }


def load_assignments() -> List[Dict[str, Any]]:
    rows = drive_svc.read_all_rows(ASSIGNMENTS_SHEET)
    return [_assignment_row_to_dict(row) for row in rows]


def create_assignment(
    diagram_type: str,
    title: str,
    answer_image_bytes: bytes,
    answer_image_ext: str,
    additional_instructions: str = "",
    scoring_criteria: str = "",
    enable_logging: bool = False,
    simulator_type: str = "none",
) -> Dict[str, Any]:
    assignment_id = _generate_assignment_id()
    pin = _generate_pin()

    folder_id = drive_svc.create_folder(assignment_id, drive_svc.get_drive_folder_id())
    mimetype = f"image/{answer_image_ext.lower()}" if answer_image_ext.lower() != "jpg" else "image/jpeg"
    answer_image_file_id = drive_svc.upload_file(
        f"answer_image.{answer_image_ext}",
        answer_image_bytes,
        mimetype,
        folder_id,
    )
    created_at = datetime.now().isoformat()

    assignment: Dict[str, Any] = {
        "assignment_id": assignment_id,
        "pin": pin,
        "diagram_type": diagram_type,
        "title": title,
        "answer_image_file_id": answer_image_file_id,
        "additional_instructions": additional_instructions,
        "scoring_criteria": scoring_criteria,
        "enable_logging": enable_logging,
        "simulator_type": simulator_type,
        "created_at": created_at,
    }

    drive_svc.append_row(ASSIGNMENTS_SHEET, [
        assignment_id, pin, diagram_type, title, answer_image_file_id,
        additional_instructions, scoring_criteria, str(enable_logging),
        simulator_type, created_at,
    ])
    return assignment


def get_assignment(assignment_id: str, pin: str) -> Optional[Dict[str, Any]]:
    for a in load_assignments():
        if a["assignment_id"] == assignment_id and a["pin"] == pin:
            return a
    return None


def load_logs(assignment_id: str) -> List[Dict[str, Any]]:
    rows = drive_svc.read_all_rows(LOGS_SHEET)
    logs = [_log_row_to_dict(row) for row in rows if row and row[0] == assignment_id]
    logs.sort(key=lambda e: e["timestamp"])

    # submission_numberは保存時ではなく表示時に計算する(書き込み時の全件読み込みを避けるため)
    counts: Dict[str, int] = {}
    for entry in logs:
        counts[entry["student_id"]] = counts.get(entry["student_id"], 0) + 1
        entry["submission_number"] = counts[entry["student_id"]]
    return logs


def save_log_entry(
    assignment_id: str,
    student_id: str,
    feedback: str,
    score: Optional[int] = None,
) -> None:
    timestamp = datetime.now().isoformat()
    drive_svc.append_row(LOGS_SHEET, [
        assignment_id, student_id, timestamp,
        str(score) if score is not None else "", feedback,
    ])


def delete_assignment(assignment_id: str) -> None:
    assignments = load_assignments()
    target = next((a for a in assignments if a["assignment_id"] == assignment_id), None)
    if target is None:
        return

    # answer_image_file_idの親フォルダ({assignment_id}フォルダ)を削除すれば画像も道連れになる
    parent_id = drive_svc.get_parent_folder_id(target["answer_image_file_id"])
    if parent_id:
        drive_svc.delete_file(parent_id)

    drive_svc.delete_rows(
        ASSIGNMENTS_SHEET,
        lambda row: len(row) > 0 and row[0] == assignment_id,
    )
    drive_svc.delete_rows(
        LOGS_SHEET,
        lambda row: len(row) > 0 and row[0] == assignment_id,
    )
