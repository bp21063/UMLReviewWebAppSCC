import base64
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests
from dotenv import load_dotenv

from .config_loader import get_api_key

load_dotenv()


def _detect_mime_type(image_bytes: bytes) -> str:
    """画像バイトからMIMEタイプを検出する"""
    if len(image_bytes) >= 8 and image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    elif len(image_bytes) >= 2 and image_bytes[:2] == b'\xff\xd8':
        return "image/jpeg"
    elif len(image_bytes) >= 6 and image_bytes[:6] in (b'GIF87a', b'GIF89a'):
        return "image/gif"
    elif len(image_bytes) >= 12 and image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
        return "image/webp"
    return "image/png"  # デフォルト


class LLMConfigurationError(Exception):
    """Raised when the LLM provider is misconfigured."""


class LLMGenerationError(Exception):
    """Raised when the LLM provider fails to generate valid code."""


@dataclass
class ProviderResponse:
    text: str
    model: str
    usage: Dict[str, Any]
    raw: Dict[str, Any]


class LLMProvider:
    """Base provider interface."""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        image_base64: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> ProviderResponse:
        raise NotImplementedError


class GeminiProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        timeout: int = 600,
    ) -> None:
        if not api_key:
            raise LLMConfigurationError("Gemini API キー (GOOGLE_API_KEY) が設定されていません。")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        image_base64: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> ProviderResponse:
        parts: list[Dict[str, Any]] = [{"text": user_prompt}]
        if image_base64 and mime_type:
            parts.append({
                "inline_data": {
                    "mime_type": mime_type,
                    "data": image_base64,
                }
            })
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8192},
        }
        response = requests.post(
            self._url,
            params={"key": self.api_key},
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise LLMGenerationError(
                f"Gemini API 呼び出しに失敗しました: {response.status_code} {response.text.strip()}"
            )
        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise LLMGenerationError("Gemini からの応答に候補が含まれていません。")
        parts = candidates[0].get("content", {}).get("parts") or []
        texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
        text = "\n".join(filter(None, texts)).strip()
        if not text:
            # デバッグ用途：一時的に raw JSON をファイルに書くなど
            with open("logs/gemini_raw_debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
            raise LLMGenerationError("Gemini からコードを取得できませんでした。")
        usage = data.get("usageMetadata", {})
        return ProviderResponse(text=text, model=self.model, usage=usage, raw=data)


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        timeout: int = 60,
    ) -> None:
        if not api_key:
            raise LLMConfigurationError("OpenAI API キー (OPENAI_API_KEY) が設定されていません。")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._url = "https://api.openai.com/v1/chat/completions"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        image_base64: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> ProviderResponse:
        if image_base64 and mime_type:
            user_content: Any = [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_base64}"},
                },
            ]
        else:
            user_content = user_prompt
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.2,
            "max_tokens": 2048,
        }
        response = requests.post(
            self._url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            data=json.dumps(payload),
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise LLMGenerationError(
                f"OpenAI API 呼び出しに失敗しました: {response.status_code} {response.text.strip()}"
            )
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise LLMGenerationError("OpenAI からの応答に選択肢が含まれていません。")
        message = choices[0].get("message", {})
        text = (message.get("content") or "").strip()
        if not text:
            raise LLMGenerationError("OpenAI からコードを取得できませんでした。")
        usage = data.get("usage", {})
        return ProviderResponse(text=text, model=self.model, usage=usage, raw=data)


_provider_lock = threading.Lock()
_provider_cache: Dict[str, LLMProvider] = {}


def _get_provider(name: Optional[str]) -> Tuple[str, LLMProvider]:
    provider_name = (name or os.getenv("LLM_PROVIDER", "gemini")).strip().lower()
    with _provider_lock:
        if provider_name in _provider_cache:
            return provider_name, _provider_cache[provider_name]

        if provider_name == "gemini":
            provider = GeminiProvider(
                api_key=get_api_key("GOOGLE_API_KEY"),
                model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            )
        elif provider_name == "openai":
            provider = OpenAIProvider(
                api_key=get_api_key("OPENAI_API_KEY"),
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            )
        else:
            raise LLMConfigurationError(
                f"サポートされていない LLM_PROVIDER が設定されています: {provider_name}"
            )

        _provider_cache[provider_name] = provider
        return provider_name, provider


def _build_prompts(
    diagram_type: str,
    session_id: str,
    prompt_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    prompt_overrides = prompt_overrides or {}
    system_prompt = prompt_overrides.get(
        "system_prompt",
        (
            "You are an assistant that converts UML diagrams into executable Python code. "
            "Generate fresh, original code based solely on the user's UML diagram. "
            "Do NOT auto-complete or fix errors in the diagram—produce code exactly as depicted. "
            "Keep comments minimal (one short line max per function/class). "
            "Do NOT include any comments explaining wait_input() or other host-provided functions. "
            "All print() messages must be in Japanese. "
            "Use wait_input() ONLY when the UML diagram explicitly shows user input events; "
            "for time-based transitions, use time.sleep() without wait_input()."
        ),
    )
    instructions = prompt_overrides.get("additional_instructions", "")
    user_prompt = (
        "Analyze the UML diagram and generate runnable Python code.\n"
        f"- Session ID: {session_id}\n"
        f"- Diagram type: {diagram_type}\n"
        "- Requirements:\n"
        "  1. Single file with main() entry point and if __name__ == '__main__' guard.\n"
        "  2. Use only Python standard library.\n"
        "  3. wait_input() usage:\n"
        "     - ONLY use wait_input() if the UML diagram explicitly shows user input/button events.\n"
        "     - If the diagram shows only time-based transitions (e.g., traffic light cycling automatically), "
        "do NOT use wait_input(). Use time.sleep() instead.\n"
        "     - wait_input() returns only 'A' or 'B'. Map multiple actions to these two buttons creatively.\n"
        "     - Do NOT define wait_input() in the code.\n"
        "     - NEVER include comments explaining wait_input() (e.g., '# wait for user input', "
        "'# ユーザー入力を待つ', '# ホスト提供関数'). This is strictly forbidden.\n"
        "     - BEFORE calling wait_input(), ALWAYS print a guidance message explaining what A and B do "
        "in the current state. Example: print('A: 電源ON / B: チャイルドロック切替'). "
        "This helps users understand button actions.\n"
        "  4. Include time.sleep() to prevent busy loops.\n"
        "  5. Add brief print() statements for state transitions (in Japanese).\n"
        "  6. Keep comments minimal—at most one short line per function. No verbose explanations.\n"
        "- Output: Return ONLY Python code in a single ```python``` block. No markdown explanations.\n"
    )
    if instructions:
        user_prompt += f"\nAdditional requirements:\n{instructions}\n"
    return system_prompt, user_prompt


def _extract_code(text: str) -> str:
    """Extract Python code from an LLM response."""
    pattern = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(text)
    if matches:
        return matches[0].strip()
    return text.strip()


def _log_event(payload: Dict[str, Any]) -> None:
    try:
        os.makedirs("logs", exist_ok=True)
        with open(
            os.path.join("logs", "llm.log"),
            "a",
            encoding="utf-8",
        ) as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # ログが失敗してもアプリ全体には影響させない
        pass


def generate_python_code(
    diagram_type: str,
    image_bytes: bytes,
    session_id: str,
    prompt_overrides: Optional[Dict[str, Any]] = None,
) -> str:
    if not image_bytes:
        raise ValueError("画像データが空です。")

    mime_type = _detect_mime_type(image_bytes)
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    provider_name, provider = _get_provider(prompt_overrides.get("provider") if prompt_overrides else None)
    system_prompt, user_prompt = _build_prompts(
        diagram_type=diagram_type,
        session_id=session_id,
        prompt_overrides=prompt_overrides,
    )

    start_time = time.time()
    log_context: Dict[str, Any] = {
        "timestamp": time.time(),
        "session_id": session_id,
        "provider": provider_name,
        "diagram_type": diagram_type,
        "image_bytes": len(image_bytes),
    }

    response = None     #一旦ここで初期化

    try:
        response = provider.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_base64=image_base64,
            mime_type=mime_type,
        )
        duration = time.time() - start_time
        raw_code = _extract_code(response.text)
        if not raw_code:
            raise LLMGenerationError("LLM 応答からコードを抽出できませんでした。")
        try:
            compile(raw_code, "<generated_code>", "exec")
        except SyntaxError as exc:
            raise LLMGenerationError(f"生成されたコードに構文エラーがあります: {exc}") from exc

        log_context.update(
            {
                "status": "success",
                "model": response.model,
                "duration_ms": int(duration * 1000),
                "usage": response.usage,
                "response_preview": raw_code[:200],
                "raw_response": response.raw,
            }
        )
        return raw_code
    except (LLMConfigurationError, LLMGenerationError) as exc:
        log_context.update(
            {
                "status": "error",
                "error": str(exc),
                "raw_response": getattr(locals().get("response"), "raw", None),
                "raw_text": getattr(locals().get("response"), "text", None),
            }
        )
        raise
    except requests.RequestException as exc:
        log_context.update(
            {
                "status": "error",
                "error": f"network_error: {exc}",
            }
        )
        raise LLMGenerationError(f"LLM API 呼び出し中にネットワークエラーが発生しました: {exc}") from exc
    finally:
        log_context.setdefault("duration_ms", int((time.time() - start_time) * 1000))
        _log_event(log_context)
