from pathlib import Path
from typing import Optional

import streamlit.components.v1 as components

_frontend_dir = (Path(__file__).parent / "frontend").absolute()
_component_func = components.declare_component(
    "qrcode_scanner", path=str(_frontend_dir)
)


def qrcode_scanner(key: Optional[str] = None) -> Optional[str]:
    return _component_func(key=key)
