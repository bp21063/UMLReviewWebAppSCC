"""十字路信号機シミュレータ。

縦方向・横方向の2軸それぞれが赤/黄/青のいずれかを取る十字路信号を、
実行画面の出力パネルの隣にビジュアル表示する。

状態はマーカー行方式（wait_input()と同じ考え方）で受け取る。
生成されたPythonコードが状態遷移のたびに
    print("__UML_REVIEW_SIM_TRAFFIC_LIGHT__:vertical=<color>,horizontal=<color>")
を出力し、ExecutorSession側でこの行だけを検知して可視出力からは除外、
構造化した状態として保持する。
"""
from html import escape
from typing import Dict, Optional

SIMULATOR_KEY = "traffic_light"
SIMULATOR_LABEL = "信号機（十字路）"
MARKER_PREFIX = "__UML_REVIEW_SIM_TRAFFIC_LIGHT__:"

_VALID_COLORS = {"red", "yellow", "green"}
_COLOR_LABELS = {"red": "赤", "yellow": "黄", "green": "青"}
_COLOR_HEX = {
    "red": "#e53935",
    "yellow": "#fdd835",
    "green": "#43a047",
}
_OFF_HEX = "#333"

_PROMPT_INSTRUCTIONS = (
    "  - Simulator output (Traffic Light):\n"
    "    * This code drives a visual 4-way intersection traffic light simulator with two axes: "
    "'vertical' (縦方向) and 'horizontal' (横方向).\n"
    "    * Each axis is always exactly one of: red, yellow, green.\n"
    "    * Immediately after EVERY state transition that changes either axis's light color, "
    "print exactly one line in this format (in addition to the normal Japanese print() messages):\n"
    f'      print("{MARKER_PREFIX}vertical=<color>,horizontal=<color>")\n'
    "    * <color> must be exactly one of: red, yellow, green (lowercase English, no spaces).\n"
    "    * Print this marker line once at the very start of main() to set the initial state, "
    "and again every time either axis changes color.\n"
    "    * NEVER include comments explaining this marker line.\n"
)


def build_prompt_overrides(base_additional_instructions: str = "") -> Dict[str, str]:
    """LLMプロンプトへ追加するシミュレータ専用指示を返す。"""
    parts = [base_additional_instructions.strip()] if base_additional_instructions.strip() else []
    parts.append(_PROMPT_INSTRUCTIONS)
    return {"additional_instructions": "\n".join(parts)}


def parse_state(payload: str) -> Optional[Dict[str, str]]:
    """マーカー行のペイロード（prefix除去後の文字列）を状態dictにパースする。"""
    state: Dict[str, str] = {}
    for pair in payload.split(","):
        if "=" not in pair:
            continue
        key, _, value = pair.partition("=")
        key = key.strip()
        value = value.strip().lower()
        if key in ("vertical", "horizontal") and value in _VALID_COLORS:
            state[key] = value
    if "vertical" not in state or "horizontal" not in state:
        return None
    return state


def _render_light_box(
    color: Optional[str],
    position_style: str,
    vertical_bar: bool,
    reversed_order: bool,
) -> str:
    """信号機を1つ描画する。

    実際の車両用信号機は、それを見る運転手から見て左から青(緑)・黄・赤の順に並ぶ。
    交差点の4方向それぞれで「運転手からの左」が指す画面上の向きが異なるため、
    vertical_bar（信号機を縦棒/横棒どちらで描くか）と reversed_order
    （円の並び順を反転するか＝180度回転）を呼び出し側で計算して渡す。
    """
    order = ("red", "yellow", "green") if reversed_order else ("green", "yellow", "red")
    circles = "".join(
        f'<div style="width:16px;height:16px;border-radius:50%;margin:3px;'
        f'background-color:{_COLOR_HEX[c] if color == c else _OFF_HEX};"></div>'
        for c in order
    )
    flex_direction = "column" if vertical_bar else "row"
    return (
        f'<div style="position:absolute;{position_style}background:#111;'
        'border:1px solid #444;border-radius:6px;padding:5px;z-index:2;'
        f'display:flex;flex-direction:{flex_direction};align-items:center;">'
        f"{circles}"
        "</div>"
    )


def render(state: Optional[Dict[str, str]]) -> str:
    """十字路の交差点をビジュアル表示するHTMLを返す（components.htmlでの表示を想定）。

    縦方向の信号機を交差点の上下に、横方向の信号機を交差点の左右に配置し、
    道路っぽい線（十字＋中央の破線）を描く。
    """
    vertical = state.get("vertical") if state else None
    horizontal = state.get("horizontal") if state else None
    vertical_label = _COLOR_LABELS.get(vertical, "-") if vertical else "-"
    horizontal_label = _COLOR_LABELS.get(horizontal, "-") if horizontal else "-"

    road_v = (
        '<div style="position:absolute;left:50%;top:0;bottom:0;width:64px;'
        'transform:translateX(-50%);background:#3a3a3a;">'
        '<div style="position:absolute;left:50%;top:0;bottom:0;width:0;'
        'transform:translateX(-50%);border-left:3px dashed #ddd;"></div>'
        "</div>"
    )
    road_h = (
        '<div style="position:absolute;top:50%;left:0;right:0;height:64px;'
        'transform:translateY(-50%);background:#3a3a3a;">'
        '<div style="position:absolute;top:50%;left:0;right:0;height:0;'
        'transform:translateY(-50%);border-top:3px dashed #ddd;"></div>'
        "</div>"
    )

    # top: 縦の道路を北へ進む車（下から上）が見る信号。横棒、左から青黄赤
    light_top = _render_light_box(
        vertical, "top:8px;left:50%;transform:translateX(-50%);",
        vertical_bar=False, reversed_order=False,
    )
    # bottom: 縦の道路を南へ進む車（上から下）が見る信号。横棒だが運転手の左右は上とは逆なので180度回転
    light_bottom = _render_light_box(
        vertical, "bottom:8px;left:50%;transform:translateX(-50%);",
        vertical_bar=False, reversed_order=True,
    )
    # left: 横の道路を西へ進む車（右から左）が見る信号。縦棒、運転手の左は画面下向き
    light_left = _render_light_box(
        horizontal, "left:8px;top:50%;transform:translateY(-50%);",
        vertical_bar=True, reversed_order=True,
    )
    # right: 横の道路を東へ進む車（左から右）が見る信号。縦棒、運転手の左は画面上向き
    light_right = _render_light_box(
        horizontal, "right:8px;top:50%;transform:translateY(-50%);",
        vertical_bar=True, reversed_order=False,
    )

    return (
        '<div style="position:relative;width:280px;height:280px;margin:0 auto;">'
        f"{road_v}{road_h}"
        f"{light_top}{light_bottom}{light_left}{light_right}"
        "</div>"
        '<div style="text-align:center;color:#ccc;font-size:0.85rem;margin-top:8px;">'
        f"縦方向: {escape(vertical_label)}　横方向: {escape(horizontal_label)}"
        "</div>"
    )
