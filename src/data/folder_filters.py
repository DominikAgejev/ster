# src/engine/folder_filters.py
from __future__ import annotations
from typing import Any, List

def normalize_folder_filters(v: Any) -> List[str]:
    """
    Normalize included/excluded folder filters to a clean list[str].
    - None -> []
    - "focus/iphone" -> ["focus/iphone"]
    - ["focus", "", None] -> ["focus"]
    """
    if v is None:
        return []
    if isinstance(v, str):
        s = v.strip()
        return [s] if s else []
    if isinstance(v, (list, tuple, set)):
        out: List[str] = []
        for x in v:
            if x is None:
                continue
            s = str(x).strip()
            if s:
                out.append(s)
        return out
    s = str(v).strip()
    return [s] if s else []
