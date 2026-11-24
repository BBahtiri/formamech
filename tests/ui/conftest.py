from functools import wraps

import pytest


@pytest.fixture
def workbench_apptest(monkeypatch):
    st = pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from piginot.ui import job_history
    from piginot.ui.workbench import navigation, picker_component

    # Streamlit 1.51's Python-only runner cannot serve registered JS asset files.
    # The installed CLI browser tests cover the actual component mount.
    @wraps(st.components.v2.component)
    def register_placeholder(*_args, **_kwargs):
        return lambda **_: None

    registrations = (
        picker_component._component,
        navigation._focus_component,
        job_history._notice_component,
    )
    for register in registrations:
        register.cache_clear()
    monkeypatch.setattr(st.components.v2, "component", register_placeholder)
    yield AppTest
    for register in registrations:
        register.cache_clear()
