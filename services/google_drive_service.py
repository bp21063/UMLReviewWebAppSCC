from typing import Callable, List, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaInMemoryUpload

from .config_loader import get_api_key

_SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/spreadsheets",
]

_drive_service = None
_sheets_service = None


def _get_credentials() -> Credentials:
    creds = Credentials(
        token=None,
        refresh_token=get_api_key("GOOGLE_OAUTH_REFRESH_TOKEN"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=get_api_key("GOOGLE_OAUTH_CLIENT_ID"),
        client_secret=get_api_key("GOOGLE_OAUTH_CLIENT_SECRET"),
        scopes=_SCOPES,
    )
    creds.refresh(Request())
    return creds


def _drive():
    global _drive_service
    if _drive_service is None:
        _drive_service = build("drive", "v3", credentials=_get_credentials())
    return _drive_service


def _sheets():
    global _sheets_service
    if _sheets_service is None:
        _sheets_service = build("sheets", "v4", credentials=_get_credentials())
    return _sheets_service


def get_drive_folder_id() -> str:
    return get_api_key("GOOGLE_DRIVE_FOLDER_ID")


def get_spreadsheet_id() -> str:
    return get_api_key("GOOGLE_SHEETS_ID")


def create_folder(name: str, parent_id: str) -> str:
    file = _drive().files().create(
        body={
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        },
        fields="id",
    ).execute()
    return file["id"]


def upload_file(name: str, content: bytes, mimetype: str, parent_id: str) -> str:
    media = MediaInMemoryUpload(content, mimetype=mimetype)
    file = _drive().files().create(
        body={"name": name, "parents": [parent_id]},
        media_body=media,
        fields="id",
    ).execute()
    return file["id"]


def delete_file(file_id: str) -> None:
    _drive().files().delete(fileId=file_id).execute()


def get_parent_folder_id(file_id: str) -> Optional[str]:
    file = _drive().files().get(fileId=file_id, fields="parents").execute()
    parents = file.get("parents", [])
    return parents[0] if parents else None


def _get_sheet_id_by_title(spreadsheet_id: str, sheet_name: str) -> int:
    spreadsheet = _sheets().spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    for sheet in spreadsheet["sheets"]:
        if sheet["properties"]["title"] == sheet_name:
            return sheet["properties"]["sheetId"]
    raise ValueError(f"シート '{sheet_name}' が見つかりません")


def append_row(sheet_name: str, values: List[str]) -> None:
    spreadsheet_id = get_spreadsheet_id()
    _sheets().spreadsheets().values().append(
        spreadsheetId=spreadsheet_id,
        range=f"{sheet_name}!A1",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body={"values": [values]},
    ).execute()


def read_all_rows(sheet_name: str) -> List[List[str]]:
    spreadsheet_id = get_spreadsheet_id()
    result = _sheets().spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range=f"{sheet_name}!A2:Z",
    ).execute()
    return result.get("values", [])


def delete_rows(sheet_name: str, matcher: Callable[[List[str]], bool]) -> int:
    """matcherに一致する行(ヘッダーを除く)を削除する。削除した行数を返す。"""
    spreadsheet_id = get_spreadsheet_id()
    rows = read_all_rows(sheet_name)
    matching_indices = [i for i, row in enumerate(rows) if matcher(row)]
    if not matching_indices:
        return 0

    sheet_id = _get_sheet_id_by_title(spreadsheet_id, sheet_name)
    requests = []
    # ヘッダー行(index 0)がシート上の1行目なので、データ行indexには+1のオフセットが必要
    for i in sorted(matching_indices, reverse=True):
        row_index = i + 1
        requests.append({
            "deleteDimension": {
                "range": {
                    "sheetId": sheet_id,
                    "dimension": "ROWS",
                    "startIndex": row_index,
                    "endIndex": row_index + 1,
                }
            }
        })
    _sheets().spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={"requests": requests},
    ).execute()
    return len(matching_indices)
