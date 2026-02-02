"""Streamlit width compatibility helpers."""
from __future__ import annotations


def patch_streamlit_width(st_module) -> None:
    """Patch Streamlit functions to accept width='stretch' on older versions.

    On older Streamlit releases, width expects an int and use_container_width is required.
    This wrapper retries with use_container_width when width is a string.
    """
    def _wrap(fn, name: str):
        original = getattr(fn, "_codex_width_original", fn)
        if getattr(fn, "_codex_width_patched", False):
            return fn

        def _inner(*args, **kwargs):
            width = kwargs.get("width")
            # Button doesn't accept width at all on older versions.
            if width is not None and name == "button":
                kwargs.pop("width", None)
                kwargs["use_container_width"] = (width == "stretch")
                return original(*args, **kwargs)

            if isinstance(width, str):
                try:
                    return original(*args, **kwargs)
                except TypeError as exc:
                    msg = str(exc).lower()
                    if "integer" in msg or "int" in msg or "unexpected keyword" in msg:
                        kwargs.pop("width", None)
                        kwargs["use_container_width"] = (width == "stretch")
                        return original(*args, **kwargs)
                    raise
            return original(*args, **kwargs)
        _inner._codex_width_patched = True
        _inner._codex_width_original = original
        return _inner

    targets = ("dataframe", "plotly_chart", "button")

    # Patch module-level functions
    for name in targets:
        if hasattr(st_module, name):
            setattr(st_module, name, _wrap(getattr(st_module, name), name))

    # Patch DeltaGenerator methods (covers st.sidebar, st.container, etc.)
    try:
        from streamlit.delta_generator import DeltaGenerator
        for name in targets:
            if hasattr(DeltaGenerator, name):
                setattr(DeltaGenerator, name, _wrap(getattr(DeltaGenerator, name), name))
    except Exception:
        # Best-effort patching; ignore if internal API changes
        pass

    # Patch ButtonMixin directly as a last-resort guard
    try:
        from streamlit.elements.widgets.button import ButtonMixin
        if hasattr(ButtonMixin, "button"):
            ButtonMixin.button = _wrap(ButtonMixin.button, "button")
    except Exception:
        pass
