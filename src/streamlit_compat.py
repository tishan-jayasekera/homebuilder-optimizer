"""Streamlit width compatibility helpers."""
from __future__ import annotations


def patch_streamlit_width(st_module) -> None:
    """Patch Streamlit functions to accept width='stretch' on older versions.

    On older Streamlit releases, width expects an int and use_container_width is required.
    This wrapper retries with use_container_width when width is a string.
    """
    def _wrap(fn):
        def _inner(*args, **kwargs):
            width = kwargs.get("width")
            if isinstance(width, str):
                try:
                    return fn(*args, **kwargs)
                except TypeError as exc:
                    msg = str(exc).lower()
                    if "integer" in msg or "int" in msg:
                        kwargs.pop("width", None)
                        kwargs["use_container_width"] = (width == "stretch")
                        return fn(*args, **kwargs)
                    raise
            return fn(*args, **kwargs)
        return _inner

    for name in ("dataframe", "plotly_chart", "button"):
        if hasattr(st_module, name):
            setattr(st_module, name, _wrap(getattr(st_module, name)))
