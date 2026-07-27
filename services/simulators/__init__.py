"""シミュレータのレジストリ。

新しいシミュレータを追加する場合:
1. 同じディレクトリに新しいモジュール（例: railroad_crossing_simulator.py）を作成し、
   SIMULATOR_KEY / SIMULATOR_LABEL / MARKER_PREFIX / build_prompt_overrides() /
   parse_state() / render() を定義する
2. 下の import と _MODULES に1行追加する
それ以外（app.py・ExecutorSession等）の変更は不要。
"""
from typing import List, Optional, Tuple

from . import traffic_light_simulator

_MODULES = [traffic_light_simulator]

SIMULATOR_REGISTRY = {module.SIMULATOR_KEY: module for module in _MODULES}

NONE_KEY = "none"
NONE_LABEL = "なし"


def list_simulator_options() -> List[Tuple[str, str]]:
    """(key, label) のリストを返す。プルダウン表示用。先頭は必ず「なし」。"""
    options = [(NONE_KEY, NONE_LABEL)]
    options += [(module.SIMULATOR_KEY, module.SIMULATOR_LABEL) for module in _MODULES]
    return options


def get_simulator(key: Optional[str]):
    """キーからシミュレータモジュールを返す。未選択/不明キーはNone。"""
    if not key or key == NONE_KEY:
        return None
    return SIMULATOR_REGISTRY.get(key)
