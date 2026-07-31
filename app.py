import io
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from html import escape
from typing import Dict, Optional, Tuple

import qrcode
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from services.qrcode_scanner import qrcode_scanner

from services import (
    LLMConfigurationError,
    LLMGenerationError,
    get_api_key,
    get_password,
    get_teacher_password,
    generate_python_code,
    generate_class_instances_and_verify,
    create_assignment,
    get_assignment,
    load_assignments,
    load_logs,
    save_log_entry,
    delete_assignment,
    list_simulator_options,
    get_simulator,
)
WAIT_INPUT_MARKER = "__UML_REVIEW_WAIT_INPUT__"
PROCESS_TIMEOUT_SECONDS = 300  # 5 分


@dataclass
class ExecutorState:
    status: str = "idle"
    output: str = ""
    waiting_input: bool = False
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    simulator_state: Optional[Dict[str, str]] = None


class ExecutorSession:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.process: Optional[subprocess.Popen[str]] = None
        self.stdout_thread: Optional[threading.Thread] = None
        self.stderr_thread: Optional[threading.Thread] = None
        self.timeout_timer: Optional[threading.Timer] = None
        self.temp_file: Optional[str] = None
        self.lock = threading.Lock()
        self.state = ExecutorState()
        self.simulator_module = None

    # --- lifecycle -----------------------------------------------------
    def start(self, code: str, simulator_key: Optional[str] = None) -> None:
        with self.lock:
            self._stop_internal(reason="stopped")
            self.simulator_module = get_simulator(simulator_key)
            self.state = ExecutorState(status="starting", started_at=time.time())
            self.state.output = ">>> 実行準備中...\n"

        script_path = self._write_script(code)
        python_executable = sys.executable or "python"
        try:
            process = subprocess.Popen(
                [python_executable, "-u", script_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except Exception as exc:  # pragma: no cover
            with self.lock:
                self.state.status = "error"
                self.state.error = f"プロセス起動に失敗しました: {exc}"
            self._cleanup_temp_file()
            return

        with self.lock:
            self.process = process
            self.state.status = "running"
            self.state.output += ">>> 実行を開始しました\n"

        self.stdout_thread = threading.Thread(target=self._consume_stdout, daemon=True)
        self.stderr_thread = threading.Thread(target=self._consume_stderr, daemon=True)
        self.stdout_thread.start()
        self.stderr_thread.start()

        self.timeout_timer = threading.Timer(PROCESS_TIMEOUT_SECONDS, self._handle_timeout)
        self.timeout_timer.start()

    def stop(self) -> None:
        with self.lock:
            self._stop_internal(reason="stopped")

    def _stop_internal(self, reason: str) -> None:
        if self.timeout_timer:
            self.timeout_timer.cancel()
            self.timeout_timer = None

        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            except Exception:
                self.process.kill()

        if self.process:
            if not self.process.stdout.closed:
                self.process.stdout.close()
            if not self.process.stderr.closed:
                self.process.stderr.close()
            if self.process.stdin and not self.process.stdin.closed:
                self.process.stdin.close()
            self.process = None

        if reason == "timeout":
            self.state.status = "timeout"
            self.state.output += ">>> 実行がタイムアウトしました (5 分上限)\n"
        elif reason == "stopped" and self.state.status not in {"error", "timeout"}:
            self.state.status = "stopped"
            if not self.state.output.endswith("\n"):
                self.state.output += "\n"
            self.state.output += ">>> 実行を停止しました\n"
        self.state.waiting_input = False
        self.state.finished_at = time.time()
        self._cleanup_temp_file()

    # --- stdout / stderr ------------------------------------------------
    def _consume_stdout(self) -> None:
        assert self.process and self.process.stdout
        for raw_line in iter(self.process.stdout.readline, ""):
            line = raw_line.rstrip("\n")
            if line == WAIT_INPUT_MARKER:
                with self.lock:
                    self.state.waiting_input = True
                continue
            if self.simulator_module and line.startswith(self.simulator_module.MARKER_PREFIX):
                payload = line[len(self.simulator_module.MARKER_PREFIX):]
                parsed = self.simulator_module.parse_state(payload)
                if parsed is not None:
                    with self.lock:
                        self.state.simulator_state = parsed
                continue
            self._append_output(line)
        self._finalize_process()

    def _consume_stderr(self) -> None:
        assert self.process and self.process.stderr
        for raw_line in iter(self.process.stderr.readline, ""):
            line = raw_line.rstrip("\n")
            self._append_output(f"[stderr] {line}")
        # stderr thread may end before stdout; no finalize here

    def _append_output(self, line: str) -> None:
        with self.lock:
            if self.state.output and not self.state.output.endswith("\n"):
                self.state.output += "\n"
            self.state.output += line

    def _finalize_process(self) -> None:
        proc = self.process
        if not proc:
            return
        proc.wait()
        exit_code = proc.returncode
        with self.lock:
            if self.timeout_timer:
                self.timeout_timer.cancel()
                self.timeout_timer = None
            if self.state.status not in {"timeout", "stopped"}:
                if exit_code == 0:
                    self.state.status = "completed"
                    if not self.state.output.endswith("\n"):
                        self.state.output += "\n"
                    self.state.output += ">>> 実行が正常に終了しました\n"
                else:
                    self.state.status = "error"
                    self.state.error = f"終了コード: {exit_code}"
                    if not self.state.output.endswith("\n"):
                        self.state.output += "\n"
                    self.state.output += ">>> 実行がエラーで終了しました\n"
            self.state.waiting_input = False
            self.state.finished_at = time.time()
        self._cleanup_temp_file()

    # --- timeout --------------------------------------------------------
    def _handle_timeout(self) -> None:
        with self.lock:
            self.state.output += ">>> タイムアウト検知: プロセスを終了します\n"
        self._stop_internal(reason="timeout")

    # --- script preparation ---------------------------------------------
    def _write_script(self, code: str) -> str:
        header = (
            "import sys, time\n"
            f"WAIT_INPUT_MARKER = {WAIT_INPUT_MARKER!r}\n"
            "sys.stdout.reconfigure(line_buffering=True)\n"
            "sys.stderr.reconfigure(line_buffering=True)\n"
            "def wait_input():\n"
            "    print(WAIT_INPUT_MARKER, flush=True)\n"
            "    line = sys.stdin.readline()\n"
            "    if not line:\n"
            "        raise RuntimeError('入力が終了しました')\n"
            "    return line.rstrip('\\n')\n"
        )
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w", encoding="utf-8")
        temp.write(header)
        temp.write("\n")
        temp.write(code)
        temp.flush()
        temp.close()
        self.temp_file = temp.name
        return temp.name

    def _cleanup_temp_file(self) -> None:
        if self.temp_file and os.path.exists(self.temp_file):
            try:
                os.remove(self.temp_file)
            except OSError:
                pass
        self.temp_file = None

    # --- public state ---------------------------------------------------
    def get_state(self) -> ExecutorState:
        with self.lock:
            return ExecutorState(
                status=self.state.status,
                output=self.state.output,
                waiting_input=self.state.waiting_input,
                error=self.state.error,
                started_at=self.state.started_at,
                finished_at=self.state.finished_at,
                simulator_state=self.state.simulator_state,
            )

    def send_input(self, value: str) -> bool:
        with self.lock:
            if not self.process or not self.process.stdin or self.process.poll() is not None:
                return False
            try:
                self.process.stdin.write(value + "\n")
                self.process.stdin.flush()
                self.state.waiting_input = False
                if not self.state.output.endswith("\n"):
                    self.state.output += "\n"
                self.state.output += f">>> 入力: {value}"
                return True
            except Exception as exc:
                self.state.status = "error"
                self.state.error = f"入力送信に失敗しました: {exc}"
                return False

    def clear_output(self) -> None:
        with self.lock:
            self.state.output = ""


class ExecutorManager:
    def __init__(self) -> None:
        self.sessions: Dict[str, ExecutorSession] = {}
        self.lock = threading.Lock()

    def get_session(self, session_id: str) -> ExecutorSession:
        with self.lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = ExecutorSession(session_id)
            return self.sessions[session_id]

    def start(self, session_id: str, code: str, simulator_key: Optional[str] = None) -> None:
        session = self.get_session(session_id)
        session.start(code, simulator_key=simulator_key)

    def stop(self, session_id: str) -> None:
        session = self.get_session(session_id)
        session.stop()

    def get_state(self, session_id: str) -> ExecutorState:
        session = self.get_session(session_id)
        return session.get_state()

    def send_input(self, session_id: str, value: str) -> bool:
        session = self.get_session(session_id)
        return session.send_input(value)

    def clear_output(self, session_id: str) -> None:
        session = self.get_session(session_id)
        session.clear_output()


def get_executor_manager() -> ExecutorManager:
    if "_executor_manager" not in st.session_state:
        st.session_state["_executor_manager"] = ExecutorManager()
    return st.session_state["_executor_manager"]


def ensure_session_defaults() -> None:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    if "page" not in st.session_state:
        st.session_state["page"] = "upload"
    if "generated_code" not in st.session_state:
        st.session_state["generated_code"] = ""
    if "code_buffer" not in st.session_state:
        st.session_state["code_buffer"] = ""
    if "diagram_type" not in st.session_state:
        st.session_state["diagram_type"] = ""
    if "generation_provider" not in st.session_state:
        st.session_state["generation_provider"] = ""
    if "password_input" not in st.session_state:
        st.session_state["password_input"] = ""
    if "class_diagram_analysis" not in st.session_state:
        st.session_state["class_diagram_analysis"] = None
    if "class_diagram_object_result" not in st.session_state:
        st.session_state["class_diagram_object_result"] = None
    if "class_diagram_instances_result" not in st.session_state:
        st.session_state["class_diagram_instances_result"] = None
    if "is_teacher" not in st.session_state:
        st.session_state["is_teacher"] = False
    if "teacher_password_input" not in st.session_state:
        st.session_state["teacher_password_input"] = ""
    if "session_assignments" not in st.session_state:
        st.session_state["session_assignments"] = []
    if "student_id" not in st.session_state:
        st.session_state["student_id"] = ""
    if "new_assignment" not in st.session_state:
        st.session_state["new_assignment"] = None
    if "selected_assignment" not in st.session_state:
        st.session_state["selected_assignment"] = None
    if "simulator_type" not in st.session_state:
        st.session_state["simulator_type"] = "none"
    if "qr_scanned_id" not in st.session_state:
        st.session_state["qr_scanned_id"] = ""
    if "qr_scanned_pin" not in st.session_state:
        st.session_state["qr_scanned_pin"] = ""
    if "qr_scanner_active" not in st.session_state:
        st.session_state["qr_scanner_active"] = False
    if "show_add_assignment_dialog" not in st.session_state:
        st.session_state["show_add_assignment_dialog"] = False


SUPPORTED_LLM_PROVIDER_LABELS = {
    "gemini": "Google Gemini",
    "openai": "OpenAI",
}


def get_active_llm_provider() -> str:
    return os.getenv("LLM_PROVIDER", "gemini").strip().lower()


def get_llm_provider_label(provider: Optional[str] = None) -> str:
    provider = (provider or get_active_llm_provider()).strip().lower()
    return SUPPORTED_LLM_PROVIDER_LABELS.get(provider, provider or "未設定")


def get_llm_configuration_error(provider: Optional[str] = None) -> Optional[str]:
    provider = (provider or get_active_llm_provider()).strip().lower()
    if provider == "gemini":
        if not get_api_key("GOOGLE_API_KEY"):
            return "GOOGLE_API_KEY が設定されていないため Gemini を利用できません。"
    elif provider == "openai":
        if not get_api_key("OPENAI_API_KEY"):
            return "OPENAI_API_KEY が設定されていないため OpenAI を利用できません。"
    else:
        return f"サポートされていない LLM_PROVIDER が設定されています: {provider}"
    return None


def get_sample_code(diagram_type: str) -> str:
    if diagram_type == "ステートマシン図":
        return """import time

def main():
    state = "縦:青"
    cycle = [
        ("縦:青", "歩行者ボタン(A)で黄に遷移します"),
        ("縦:黄", "2 秒後に赤へ遷移します"),
        ("縦:赤", "横断中。5 秒後に横:青へ遷移します"),
        ("横:青", "歩行者ボタン(B)で黄に遷移します"),
        ("横:黄", "2 秒後に赤へ遷移します"),
        ("横:赤", "5 秒後に縦:青へ戻ります"),
    ]
    index = 0

    while True:
        name, description = cycle[index % len(cycle)]
        print(f\"状態: {name}\")
        print(f\"説明: {description}\")

        if name.endswith(\"青\"):
            print(\"ボタン入力を待機します (A/B)\")
            pressed = wait_input()
            print(f\"入力: {pressed}\")
            if (name.startswith(\"縦\") and pressed == \"A\") or (
                name.startswith(\"横\") and pressed == \"B\"
            ):
                index += 1
                continue
            print(\"入力が想定と異なるため状態を維持します\")
            continue

        if name.endswith(\"黄\"):
            time.sleep(2)
        else:
            time.sleep(5)
        index += 1


if __name__ == \"__main__\":
    main()
"""
    elif diagram_type == "フローチャート図":
        return """import time

def main():
    print(\"センサー値を監視します。A=リセット, B=終了\")
    total = 0
    while True:
        print(f\"現在値: {total}\")
        total += 1
        time.sleep(1)
        if total % 5 == 0:
            print(\"入力待ち: A でリセット / B で終了\")
            option = wait_input()
            if option == \"A\":
                print(\"値をリセットします\")
                total = 0
            elif option == \"B\":
                print(\"終了します\")
                break


if __name__ == \"__main__\":
    main()
"""
    return """import time

def main():
    print(\"信号シミュレーションを開始します\")
    while True:
        for color, duration in [(\"青\", 3), (\"黄\", 2), (\"赤\", 4)]:
            print(f\"信号が {color} になりました\")
            time.sleep(duration)


if __name__ == \"__main__\":
    main()
"""


def render_output_area(text: str, session_id: str) -> None:
    container_id = f"output_{session_id}"
    safe_text = escape(text)
    html = (
        f'<div id="{container_id}" style="'
        "height: 260px;"
        "overflow: auto;"
        "padding: 0.75rem;"
        "border: 1px solid #444;"
        "border-radius: 0.5rem;"
        "background-color: #111;"
        "color: #f0f0f0;"
        "font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;"
        "font-size: 0.9rem;"
        "white-space: pre-wrap;"
        "line-height: 1.4;"
        f'">{safe_text}</div>'
        f"<script>const el=document.getElementById('{container_id}');"
        "if(el){el.scrollTop=el.scrollHeight;}</script>"
    )
    components.html(html, height=300)


# ---------------------------------------------------------------------------
# QRコード（課題ID・PINのエンコード/デコード）
# ---------------------------------------------------------------------------

def _make_qr_payload(assignment_id: str, pin: str) -> str:
    return f"{assignment_id}-{pin}"


def _parse_qr_payload(payload: str) -> Optional[Tuple[str, str]]:
    assignment_id, sep, pin = payload.strip().rpartition("-")
    if not sep or not assignment_id.startswith("UMLR-") or not pin:
        return None
    return assignment_id, pin


def _generate_qr_png_bytes(assignment_id: str, pin: str) -> bytes:
    image = qrcode.make(_make_qr_payload(assignment_id, pin))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 課題追加ダイアログ（学生用）
# ---------------------------------------------------------------------------

@st.dialog("QRコードを読み取る", width="large", dismissible=False)
def _qr_scanner_dialog():
    if st.button("キャンセル", use_container_width=True):
        st.session_state["qr_scanner_active"] = False
        st.rerun()
    error_placeholder = st.empty()
    st.caption("QRコードをカメラに写してください")
    scanned = qrcode_scanner(key="assignment_qr_scanner")
    if scanned:
        parsed = _parse_qr_payload(scanned)
        if parsed:
            st.session_state["qr_scanned_id"], st.session_state["qr_scanned_pin"] = parsed
            st.session_state["qr_scanner_active"] = False
            st.rerun()
        else:
            error_placeholder.error("有効なQRコードではありません")


@st.dialog("課題を追加")
def _add_assignment_dialog():
    if st.button("📷 QRコードを読み取る", use_container_width=True):
        st.session_state["qr_scanner_active"] = True
        st.rerun()

    assignment_id_input = st.text_input(
        "課題ID（例: UMLR-ABC123）",
        value=st.session_state.get("qr_scanned_id", ""),
    )
    pin_input = st.text_input(
        "PIN",
        type="password",
        value=st.session_state.get("qr_scanned_pin", ""),
    )
    student_id_input = st.text_input(
        "教員から指示されたIDを入力してください",
        value=st.session_state.get("student_id", ""),
    )
    if st.button("追加", type="primary", use_container_width=True):
        if not assignment_id_input.strip() or not pin_input.strip():
            st.error("課題IDとPINを入力してください")
            return
        assignment = get_assignment(assignment_id_input.strip(), pin_input.strip())
        if assignment is None:
            st.error("課題IDまたはPINが正しくありません")
            return
        if assignment.get("enable_logging") and not student_id_input.strip() and not st.session_state.get("student_id"):
            st.error("この課題ではIDの入力が必要です")
            return
        if student_id_input.strip():
            st.session_state["student_id"] = student_id_input.strip()
        existing_ids = [a["assignment_id"] for a in st.session_state.get("session_assignments", [])]
        if assignment["assignment_id"] not in existing_ids:
            st.session_state["session_assignments"].append(assignment)
        st.session_state["qr_scanned_id"] = ""
        st.session_state["qr_scanned_pin"] = ""
        st.session_state["show_add_assignment_dialog"] = False
        st.rerun()


# ---------------------------------------------------------------------------
# 教員ページ
# ---------------------------------------------------------------------------

def show_teacher_create_page():
    st.markdown("### 課題を作成")
    if st.button("← 戻る", type="secondary"):
        st.session_state["page"] = "upload"
        st.rerun()

    # フォーム外に置き、選択変更を即座に画面へ反映させる（st.form内は送信まで再実行されないため）
    diagram_type = st.selectbox(
        "図の種類 *",
        options=["ステートマシン図", "フローチャート図", "クラス図"],
        key="teacher_create_diagram_type",
    )
    simulator_type = "none"
    if diagram_type == "ステートマシン図":
        simulator_options = list_simulator_options()
        simulator_labels = dict(simulator_options)
        simulator_type = st.selectbox(
            "シミュレータ（任意）",
            options=[key for key, _ in simulator_options],
            format_func=lambda key: simulator_labels[key],
            help="選択すると、実行画面に図の状態遷移をビジュアル表示するシミュレータが追加されます",
            key="teacher_create_simulator_type",
        )

    with st.form("teacher_create_form"):
        title = st.text_input("課題タイトル *")
        answer_image_file = st.file_uploader(
            "模範解答画像 *",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
        )
        additional_instructions = st.text_area(
            "追加指示（任意）",
            help="採点基準・差分指摘の有無・ヒントの範囲など",
            height=100,
        )
        scoring_criteria = st.text_area(
            "採点基準（任意）",
            help="採点基準を記述すると、LLMがスコアを含むフィードバックを生成します",
            height=100,
        )
        enable_logging = st.toggle("提出ログを記録する")
        if enable_logging:
            st.caption("学生に事前にIDを伝えておいてください。IDはアプリ側では管理しません。")

        submitted = st.form_submit_button("課題を登録", type="primary", use_container_width=True)

    if submitted:
        if not title.strip():
            st.error("課題タイトルを入力してください")
        elif answer_image_file is None:
            st.error("模範解答画像をアップロードしてください")
        else:
            try:
                ext = answer_image_file.name.rsplit(".", 1)[-1].lower()
                with st.spinner("課題を登録中..."):
                    assignment = create_assignment(
                        diagram_type=diagram_type,
                        title=title.strip(),
                        answer_image_bytes=answer_image_file.getvalue(),
                        answer_image_ext=ext,
                        additional_instructions=additional_instructions.strip(),
                        scoring_criteria=scoring_criteria.strip(),
                        enable_logging=enable_logging,
                        simulator_type=simulator_type,
                    )
                st.session_state["new_assignment"] = assignment
                st.session_state["page"] = "teacher_result"
                st.rerun()
            except Exception as exc:
                st.error(f"❌ 課題の登録に失敗しました: {exc}")

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
    UML Review Web App <br> https://github.com/bp21063/UMLReviewWebApp
    </div>
    """, unsafe_allow_html=True)


def show_teacher_result_page():
    st.markdown("### 課題登録完了")

    assignment = st.session_state.get("new_assignment")
    if assignment is None:
        st.error("課題情報が見つかりません。")
        if st.button("課題作成に戻る"):
            st.session_state["page"] = "teacher_create"
            st.rerun()
        return

    st.success("課題が正常に登録されました。")
    st.markdown(f"**タイトル:** {assignment['title']}")
    st.markdown(f"**図の種類:** {assignment['diagram_type']}")
    st.markdown(f"**ログ記録:** {'有効' if assignment['enable_logging'] else '無効'}")

    st.markdown("---")
    st.markdown("#### 学生に配布する情報")
    col_id, col_pin = st.columns(2)
    with col_id:
        st.markdown("**課題ID**")
        st.code(assignment["assignment_id"], language=None)
    with col_pin:
        st.markdown("**PIN**")
        st.code(assignment["pin"], language=None)

    qr_bytes = _generate_qr_png_bytes(assignment["assignment_id"], assignment["pin"])
    _qr_left, _qr_mid, _qr_right = st.columns([1, 1, 1])
    with _qr_mid:
        st.markdown("**QRコード**")
        st.image(qr_bytes, width=320)
        st.download_button(
            "QRコードをダウンロード",
            data=qr_bytes,
            file_name=f"{assignment['assignment_id']}_qr.png",
            mime="image/png",
            key="teacher_result_qr_download",
            use_container_width=True,
        )

    st.markdown("---")
    col_create, col_dashboard = st.columns(2)
    with col_create:
        if st.button("別の課題を作成", use_container_width=True):
            st.session_state["new_assignment"] = None
            st.session_state["page"] = "teacher_create"
            st.rerun()
    with col_dashboard:
        if st.button("ダッシュボードへ", use_container_width=True):
            st.session_state["page"] = "teacher_dashboard"
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
    UML Review Web App <br> https://github.com/bp21063/UMLReviewWebApp
    </div>
    """, unsafe_allow_html=True)


def show_teacher_dashboard_page():
    st.markdown("### 教員ダッシュボード")
    if st.button("← 戻る", type="secondary"):
        st.session_state["page"] = "upload"
        st.rerun()

    assignments = load_assignments()
    if not assignments:
        st.info("登録された課題はありません。")
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.8em;'>
        UML Review Web App <br> https://github.com/bp21063/UMLReviewWebApp
        </div>
        """, unsafe_allow_html=True)
        return

    options = {f"{a['title']} ({a['assignment_id']})": a for a in assignments}
    selected_label = st.selectbox("課題を選択", options=list(options.keys()))
    selected = options[selected_label]

    st.markdown(f"**図の種類:** {selected['diagram_type']}")
    st.markdown(f"**ログ記録:** {'有効' if selected['enable_logging'] else '無効'}")
    st.markdown(f"**作成日時:** {selected['created_at'][:10]}")

    st.markdown("---")
    st.markdown("#### 学生に配布する情報")
    col_id, col_pin = st.columns(2)
    with col_id:
        st.markdown("**課題ID**")
        st.code(selected["assignment_id"], language=None)
    with col_pin:
        st.markdown("**PIN**")
        st.code(selected["pin"], language=None)

    dashboard_qr_bytes = _generate_qr_png_bytes(selected["assignment_id"], selected["pin"])
    _dash_qr_left, _dash_qr_mid, _dash_qr_right = st.columns([1, 1, 1])
    with _dash_qr_mid:
        st.image(dashboard_qr_bytes, width=320)
        st.download_button(
            "QRコードをダウンロード",
            data=dashboard_qr_bytes,
            file_name=f"{selected['assignment_id']}_qr.png",
            mime="image/png",
            key="dashboard_qr_download",
            use_container_width=True,
        )

    st.markdown("---")

    # 課題削除
    if "confirm_delete" not in st.session_state:
        st.session_state["confirm_delete"] = False
    if "confirm_delete2" not in st.session_state:
        st.session_state["confirm_delete2"] = False

    if not st.session_state["confirm_delete"]:
        if st.button("課題を削除", type="secondary"):
            st.session_state["confirm_delete"] = True
            st.rerun()
    elif not st.session_state["confirm_delete2"]:
        st.warning("本当に課題を削除しますか？")
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("はい", type="primary", use_container_width=True):
                st.session_state["confirm_delete2"] = True
                st.rerun()
        with col_no:
            if st.button("いいえ", use_container_width=True):
                st.session_state["confirm_delete"] = False
                st.rerun()
    else:
        st.error("削除した課題に関する回答や点数は完全に削除され二度と閲覧できません。本当に削除しますか？")
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("完全に削除する", type="primary", use_container_width=True):
                delete_assignment(selected["assignment_id"])
                st.session_state["confirm_delete"] = False
                st.session_state["confirm_delete2"] = False
                st.success("課題を削除しました。")
                st.rerun()
        with col_no:
            if st.button("キャンセル", use_container_width=True):
                st.session_state["confirm_delete"] = False
                st.session_state["confirm_delete2"] = False
                st.rerun()

    # ログ表示
    if selected["enable_logging"]:
        st.markdown("---")
        st.markdown("**提出ログ:**")
        logs = load_logs(selected["assignment_id"])
        if not logs:
            st.info("まだ提出がありません。")
        else:
            import pandas as pd
            rows = []
            for entry in logs:
                rows.append({
                    "ID": entry["student_id"],
                    "提出番号": entry["submission_number"],
                    "日時": entry["timestamp"][:16].replace("T", " "),
                    "スコア": entry["score"] if entry["score"] is not None else "-",
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # スコア推移グラフ（スコアありの場合）
            scored_logs = [e for e in logs if e["score"] is not None]
            if scored_logs:
                st.markdown("**スコア推移:**")
                score_data = {}
                for entry in scored_logs:
                    sid = entry["student_id"]
                    score_data.setdefault(sid, []).append(entry["score"])
                chart_data = pd.DataFrame(
                    {sid: pd.Series(scores) for sid, scores in score_data.items()}
                )
                st.line_chart(chart_data)
    else:
        st.info("この課題はログ記録が無効です。")

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
    UML Review Web App <br> https://github.com/bp21063/UMLReviewWebApp
    </div>
    """, unsafe_allow_html=True)


def main():
    # ページ設定
    st.set_page_config(
        page_title="UML Review Web App",
        page_icon="📊",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # セッション状態の初期化
    ensure_session_defaults()

    # サイドバー
    with st.sidebar:
        # 認証
        st.markdown("### 認証")
        st.text_input(
            "パスワード",
            type="password",
            key="password_input",
            help="コード生成機能を使用するにはパスワードが必要です。",
        )
        correct_password = get_password()
        if st.session_state["password_input"] == correct_password and correct_password:
            st.success("認証済み")
        elif st.session_state["password_input"]:
            st.error("パスワードが正しくありません")

        # 教員認証
        st.markdown("---")
        teacher_pw = st.text_input(
            "教員パスワード",
            type="password",
            key="teacher_password_input",
        )
        teacher_correct_pw = get_teacher_password()
        if teacher_correct_pw and teacher_pw == teacher_correct_pw:
            st.session_state["is_teacher"] = True
            st.success("教員として認証済み")
        elif teacher_pw:
            st.session_state["is_teacher"] = False
            st.error("教員パスワードが正しくありません")

        # 教員メニュー
        if st.session_state.get("is_teacher"):
            st.markdown("### 課題管理")
            if st.button("課題を作成", use_container_width=True):
                st.session_state["page"] = "teacher_create"
                st.rerun()
            if st.button("ダッシュボード", use_container_width=True):
                st.session_state["confirm_delete"] = False
                st.session_state["confirm_delete2"] = False
                st.session_state["page"] = "teacher_dashboard"
                st.rerun()

        # 課題（学生用）
        st.markdown("---")
        st.markdown("### 課題")
        if st.button("+ 課題を追加", use_container_width=True):
            st.session_state["show_add_assignment_dialog"] = True
            st.session_state["qr_scanner_active"] = False
        if st.session_state.get("qr_scanner_active"):
            _qr_scanner_dialog()
        elif st.session_state.get("show_add_assignment_dialog"):
            _add_assignment_dialog()
        for a in st.session_state.get("session_assignments", []):
            st.caption(f"📋 {a['title']} ({a['assignment_id']})")

    # カスタムCSS（モバイル対応）
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    .stFileUploader > div > div > div > div {
        text-align: center;
    }
    
    /* モバイル対応のスタイル */
    @media (max-width: 768px) {
        .main > div {
            padding: 1rem 0.5rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # ページ切り替え
    if st.session_state['page'] == 'upload':
        show_upload_page()
    elif st.session_state['page'] == 'execution':
        show_execution_page()
    elif st.session_state['page'] == 'class_diagram':
        show_class_diagram_page()
    elif st.session_state['page'] == 'teacher_create':
        show_teacher_create_page()
    elif st.session_state['page'] == 'teacher_result':
        show_teacher_result_page()
    elif st.session_state['page'] == 'teacher_dashboard':
        show_teacher_dashboard_page()

def show_upload_page():
    # タイトル表示
    st.markdown("### UML図の機能を検証")
    
    # 説明文
    st.markdown("""
    UML図の画像をアップロードして、Pythonコードを生成し、リアルタイムで実行することで機能の検証を行えます。
    """)

    # 空のスペースで中央上部に配置
    st.markdown("<br>", unsafe_allow_html=True)

    # 中央配置のレイアウト
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # ファイルアップローダー
        uploaded_file = st.file_uploader(
            "画像を選択",
            type=['png', 'jpg', 'jpeg', 'svg', 'bmp', 'tiff'],
            help="UML図の画像ファイルを選択してください。スマートフォンではカメラ撮影も可能です。"
        )

        # アップロードされたファイルの処理
        if uploaded_file is not None:
            try:
                # 画像の表示
                if uploaded_file.type.startswith('image/'):
                    st.session_state['uploaded_file'] = uploaded_file
                    if uploaded_file.type == 'image/svg+xml':
                        # SVGファイルの場合
                        st.markdown("**アップロードされた画像:**")
                        st.image(uploaded_file)
                    else:
                        # その他の画像ファイルの場合
                        image = Image.open(uploaded_file)
                        st.markdown("**アップロードされた画像:**")
                        st.image(image)
                        uploaded_file.seek(0)
                    
                    # 図の種類選択プルダウン（追加済み課題を含む）
                    _session_assignments = st.session_state.get("session_assignments", [])
                    _assignment_option_map = {
                        f"{a['title']} ({a['assignment_id']})": a
                        for a in _session_assignments
                    }
                    _standard_options = ["未選択", "ステートマシン図", "フローチャート図", "クラス図"]
                    diagram_type = st.selectbox(
                        "図の種類を選択してください:",
                        options=_standard_options + list(_assignment_option_map.keys()),
                        index=0,
                        key="diagram_type_select"
                    )
                    selected_assignment = _assignment_option_map.get(diagram_type)
                    actual_diagram_type = selected_assignment["diagram_type"] if selected_assignment else diagram_type

                    if selected_assignment:
                        st.info(f"課題: **{selected_assignment['title']}**　図の種類: {actual_diagram_type}")

                    llm_provider = get_active_llm_provider()
                    provider_label = get_llm_provider_label(llm_provider)
                    llm_config_error = get_llm_configuration_error(llm_provider)
                    st.caption(f"使用中のLLM: {provider_label}")
                    if llm_config_error:
                        st.warning(llm_config_error)

                    # パスワード認証チェック
                    correct_password = get_password()
                    is_authenticated = (
                        st.session_state["password_input"] == correct_password
                        and correct_password
                    )

                    # 生成ボタン
                    button_disabled = (
                        diagram_type == "未選択"
                        or llm_config_error is not None
                        or not is_authenticated
                    )
                    if diagram_type == "未選択":
                        st.info("図の種類を選択してからコード生成を行ってください")
                    if not is_authenticated:
                        st.info("左側のメニューからパスワードを入力してください")
                    button_label = "クラス図を解析" if actual_diagram_type == "クラス図" else "コード生成・実行"
                    if st.button(button_label, type="primary", use_container_width=True, disabled=button_disabled):
                        session_id = st.session_state['session_id']
                        st.session_state['diagram_type'] = actual_diagram_type
                        st.session_state['selected_assignment'] = selected_assignment
                        simulator_key = (
                            selected_assignment.get("simulator_type", "none") if selected_assignment else "none"
                        )
                        st.session_state['simulator_type'] = simulator_key
                        image_bytes = uploaded_file.getvalue()
                        uploaded_file.seek(0)

                        # 課題の追加指示をプロンプトに組み込む
                        prompt_overrides: dict = {}
                        if selected_assignment:
                            parts = []
                            if selected_assignment.get("additional_instructions"):
                                parts.append(selected_assignment["additional_instructions"])
                            if selected_assignment.get("scoring_criteria"):
                                parts.append("採点基準: " + selected_assignment["scoring_criteria"])
                            if parts:
                                prompt_overrides["additional_instructions"] = "\n".join(parts)

                        simulator_module = get_simulator(simulator_key)
                        if simulator_module:
                            prompt_overrides.update(
                                simulator_module.build_prompt_overrides(
                                    prompt_overrides.get("additional_instructions", "")
                                )
                            )

                        if actual_diagram_type == "クラス図":
                            try:
                                with st.spinner("クラス図を解析中..."):
                                    instances_result = generate_class_instances_and_verify(
                                        image_bytes=image_bytes,
                                        session_id=session_id,
                                    )
                                st.session_state['class_diagram_instances_result'] = instances_result
                                if selected_assignment and selected_assignment.get("enable_logging"):
                                    if st.session_state.get("student_id"):
                                        feedback_summary = "\n".join([
                                            f"{v.verdict}: {v.from_class}→{v.to_class} ({v.original_multiplicity}): {v.explanation}"
                                            for v in instances_result.multiplicity_verifications
                                        ])
                                        save_log_entry(
                                            assignment_id=selected_assignment["assignment_id"],
                                            student_id=st.session_state["student_id"],
                                            feedback=feedback_summary,
                                        )
                            except (LLMConfigurationError, LLMGenerationError) as exc:
                                st.error(f"❌ クラス図の解析に失敗しました: {exc}")
                            except Exception as exc:
                                st.error(f"❌ 予期せぬエラーが発生しました: {exc}")
                            else:
                                st.session_state['page'] = 'class_diagram'
                                st.rerun()
                        else:
                            manager = get_executor_manager()
                            try:
                                with st.spinner("コード生成中..."):
                                    generated_code = generate_python_code(
                                        diagram_type=actual_diagram_type,
                                        image_bytes=image_bytes,
                                        session_id=session_id,
                                        prompt_overrides=prompt_overrides if prompt_overrides else None,
                                    )
                                if selected_assignment and selected_assignment.get("enable_logging"):
                                    if st.session_state.get("student_id"):
                                        save_log_entry(
                                            assignment_id=selected_assignment["assignment_id"],
                                            student_id=st.session_state["student_id"],
                                            feedback=generated_code,
                                        )
                            except (LLMConfigurationError, LLMGenerationError) as exc:
                                st.session_state['generation_provider'] = ""
                                st.error(f"❌ コード生成に失敗しました: {exc}")
                            except Exception as exc:
                                st.session_state['generation_provider'] = ""
                                st.error(f"❌ 予期せぬエラーが発生しました: {exc}")
                            else:
                                st.session_state['generated_code'] = generated_code
                                st.session_state['code_buffer'] = generated_code
                                st.session_state['generation_provider'] = llm_provider
                                manager.stop(session_id)
                                manager.clear_output(session_id)
                                st.session_state['page'] = 'execution'
                                st.rerun()

                    if actual_diagram_type != "未選択" and actual_diagram_type != "クラス図" and not selected_assignment:
                        with st.expander("デモ用サンプルコードを試す"):
                            st.caption("実際のUML図がなくても、サンプルコードで動作を確認できます。")
                            if st.button(
                                "サンプルコードを読み込む",
                                key="load_sample_code_button",
                                use_container_width=True,
                            ):
                                session_id = st.session_state['session_id']
                                sample_code = get_sample_code(diagram_type)
                                st.session_state['diagram_type'] = diagram_type
                                st.session_state['simulator_type'] = "none"
                                st.session_state['generated_code'] = sample_code
                                st.session_state['code_buffer'] = sample_code
                                st.session_state['generation_provider'] = "sample"
                                manager = get_executor_manager()
                                manager.stop(session_id)
                                manager.clear_output(session_id)
                                st.session_state['page'] = 'execution'
                                st.rerun()
                    
                else:
                    st.error("❌ サポートされていないファイル形式です。")
                    
            except Exception as e:
                st.error(f"❌ ファイル処理中にエラーが発生しました: {str(e)}")
        
        else:
            pass

    # フッター
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
    UML Review Web App <br> https://github.com/bp21063/UMLReviewWebApp
    </div>
    """, unsafe_allow_html=True)

def show_class_diagram_page():
    st.markdown("### クラス図 検証結果")

    if st.button("← 戻る", type="secondary"):
        st.session_state['page'] = 'upload'
        st.rerun()

    result = st.session_state.get('class_diagram_instances_result')
    if result is None:
        st.error("解析結果が見つかりません。画像をアップロードしてから再試行してください。")
        if st.button("アップロード画面に戻る"):
            st.session_state['page'] = 'upload'
            st.rerun()
        return

    if 'uploaded_file' in st.session_state:
        st.markdown("**入力したクラス図:**")
        st.image(st.session_state['uploaded_file'])

    st.markdown("---")
    st.markdown("**生成されたインスタンス例:**")
    if result.instance_explanation:
        st.caption(result.instance_explanation)

    if result.instances:
        # クラスごとにインスタンスをグループ化（出現順を保持）
        class_groups: dict = {}
        for inst in result.instances:
            class_groups.setdefault(inst.class_name, []).append(inst)

        cols = st.columns(len(class_groups))
        for col, (class_name, insts) in zip(cols, class_groups.items()):
            with col:
                st.markdown(f"**{class_name}**")
                for inst in insts:
                    with st.container(border=True):
                        st.markdown(f"<u>{inst.instance_name} : {class_name}</u>", unsafe_allow_html=True)
                        for attr_name, attr_val in inst.attributes.items():
                            st.caption(f"{attr_name} = {attr_val}")

    if result.connections:
        st.markdown("**接続関係:**")
        conn_map: dict = {}
        for conn in result.connections:
            conn_map.setdefault(conn.from_instance, []).append(conn.to_instance)
        for src, targets in conn_map.items():
            st.markdown(f"- **{src}** → {', '.join(targets)}")

    st.markdown("---")
    st.markdown("**多重度の検証結果:**")
    if result.multiplicity_verifications:
        for v in result.multiplicity_verifications:
            if v.verdict == "ok":
                icon = "✅"
            elif v.verdict == "error":
                icon = "❌"
            else:
                icon = "⚠️"
            st.markdown(f"{icon} **{v.from_class} → {v.to_class}**（{v.original_multiplicity}）：{v.explanation}")
    else:
        st.info("検証する関係が見つかりませんでした。")

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
    UML Review Web App <br> https://github.com/bp21063/UMLReviewWebApp
    </div>
    """, unsafe_allow_html=True)


def show_execution_page():
    # タイトル表示
    st.markdown("### Pythonコード実行")

    session_id = st.session_state["session_id"]
    manager = get_executor_manager()
    executor_state = manager.get_state(session_id)

    # 戻るボタン
    if st.button("← 戻る", type="secondary"):
        manager.stop(session_id)
        manager.clear_output(session_id)
        st.session_state['page'] = 'upload'
        st.rerun()
    
    # アップロードされたファイル情報の表示
    if 'uploaded_file' in st.session_state and 'diagram_type' in st.session_state:
        diagram_type = st.session_state.get('diagram_type', '未選択')
        st.markdown(f"**図の種類:** {diagram_type}")
        generation_source = st.session_state.get("generation_provider", "")
        if generation_source:
            if generation_source == "sample":
                st.caption("生成元: サンプルコード")
            else:
                provider_label = get_llm_provider_label(generation_source)
                st.caption(f"生成元: {provider_label}")
        st.markdown("**アップロードされた画像:**")
        st.image(st.session_state['uploaded_file'])

        # 実行コントロール
        st.markdown("---")
        st.markdown("**実行コントロール:**")
        status_labels = {
            "idle": "待機中",
            "starting": "起動中",
            "running": "実行中",
            "completed": "正常終了",
            "error": "エラー",
            "timeout": "タイムアウト",
            "stopped": "停止済み",
        }
        st.markdown(f"現在の状態: `{status_labels.get(executor_state.status, executor_state.status)}`")
        if executor_state.error:
            st.error(f"エラー: {executor_state.error}")

        is_running = executor_state.status in {"starting", "running"}
        col_start, col_gap, col_stop, col_spacer = st.columns([1, 0.1, 1, 4])
        with col_start:
            if st.button("実行開始", type="primary", use_container_width=True, disabled=is_running):
                code_to_run = st.session_state.get("code_buffer", "")
                if code_to_run.strip():
                    manager.start(session_id, code_to_run, simulator_key=st.session_state.get("simulator_type"))
                    st.rerun()
                else:
                    st.warning("実行するコードが空です。")
        with col_gap:
            st.write("")
        with col_stop:
            if st.button("停止", use_container_width=True, disabled=not is_running):
                manager.stop(session_id)
                st.rerun()

        # 入力ボタン
        st.markdown("**入力ボタン:**")
        col_left, col_a, col_b, col_c, col_right = st.columns([0.5, 1, 1, 1, 0.5])
        waiting = executor_state.waiting_input
        with col_a:
            if st.button("A", use_container_width=True, disabled=not waiting):
                if manager.send_input(session_id, "A"):
                    st.rerun()
        with col_b:
            if st.button("B", use_container_width=True, disabled=not waiting):
                if manager.send_input(session_id, "B"):
                    st.rerun()
        with col_c:
            if st.button("C", use_container_width=True, disabled=not waiting):
                if manager.send_input(session_id, "C"):
                    st.rerun()
        with col_left:
            st.write("")
        with col_right:
            st.write("")
        if waiting:
            st.info("コードが入力待ち状態です。A/B/C ボタンで入力を送信できます。")

        # シミュレータ表示
        simulator_module = get_simulator(st.session_state.get("simulator_type"))
        if simulator_module:
            st.markdown("---")
            st.markdown(f"**シミュレータ（{simulator_module.SIMULATOR_LABEL}）:**")
            components.html(simulator_module.render(executor_state.simulator_state), height=340)

        # 出力表示
        header_col, clear_col = st.columns([4, 1])
        with header_col:
            st.markdown("**出力:**")
        with clear_col:
            if st.button("出力をクリア", key="clear_output_button", use_container_width=True):
                manager.clear_output(session_id)
                st.rerun()
        output_value = executor_state.output or "出力はここに表示されます"
        render_output_area(output_value, session_id)

        # コードエリア
        st.markdown("---")
        st.markdown("**生成されたPythonコード:**")
        if st.session_state.get("generated_code") and st.session_state.get("code_buffer") == "":
            st.session_state["code_buffer"] = st.session_state["generated_code"]
        st.text_area(
            "コードエディタ",
            key="code_buffer",
            height=280,
            help="必要に応じてコードを編集し、実行開始ボタンを押してください。",
        )
    else:
        st.error("ファイル情報が見つかりません。画像をアップロードしてから再試行してください。")
        if st.button("アップロード画面に戻る"):
            st.session_state['page'] = 'upload'
            st.rerun()

    # フッター
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
    UML Review Web App <br> https://github.com/bp21063/UMLReviewWebApp
    </div>
    """, unsafe_allow_html=True)

    if executor_state.status in {"running", "starting"}:
        time.sleep(0.5)
        st.rerun()

if __name__ == "__main__":
    main()
