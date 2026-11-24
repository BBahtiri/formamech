"""Streamlit controls and readouts for an already evaluated Inspector snapshot."""

from __future__ import annotations

from math import hypot, isfinite
from typing import Any

from piginot.ui.inspector import (
    InspectorSnapshot,
    build_plotly_figures,
    finite_vertex_values,
    flagged_vertices,
)


def _coordinates(snapshot: InspectorSnapshot, vertex: int) -> str:
    x, y = snapshot.nodes[vertex]
    return f"({x:.6g}, {y:.6g}) mm"


def _render_admissibility(st: Any, snapshot: InspectorSnapshot) -> None:
    with st.container(key="wb-determinant-summary"):
        st.subheader("Admissibility")
        verdict = snapshot.tier0_admissibility
        recorded = (
            "Unavailable"
            if verdict is None
            else "Passed"
            if verdict["passed"]
            else "Failed"
        )
        st.markdown(f"**Recorded verdict** {recorded}")
        if verdict is not None:
            for check in verdict.get("failed_checks", ()):
                st.markdown(f"- {check.replace('_', ' ').capitalize()}")
        st.markdown(
            f"**Determinant floor** {snapshot.determinant_floor:.6g} (dimensionless)"
        )
        finite = finite_vertex_values(snapshot.det_f_2d, len(snapshot.nodes))
        if not finite:
            st.write(
                "Determinant data missing"
                if not snapshot.det_f_2d
                else "No finite determinant samples"
            )
            return
        vertex, value = min(finite, key=lambda item: item[1])
        st.markdown(
            f"**Minimum surrogate determinant** {value:.6g} at Vertex {vertex} · {_coordinates(snapshot, vertex)}"
        )
        count = len(flagged_vertices(snapshot))
        st.write(
            f"{count} flagged {'vertex' if count == 1 else 'vertices'} at or below the floor"
        )
        missing = len(snapshot.nodes) - len(finite)
        if missing:
            st.write(f"{missing} vertices have missing or non-finite determinant data.")


def _choices(snapshot: InspectorSnapshot) -> dict[str, int]:
    count = len(snapshot.nodes)
    choices = {
        f"QoI · {qoi['name']}": qoi["vertex_index"]
        for qoi in snapshot.qois
        if 0 <= qoi["vertex_index"] < count
    }
    for name, values, extremum in (
        ("Minimum determinant", snapshot.det_f_2d, min),
        ("Maximum displacement", snapshot.u_magnitude_mm, max),
        ("Maximum von Mises stress", snapshot.von_mises_mpa, max),
    ):
        if finite := finite_vertex_values(values, count):
            choices[name] = extremum(finite, key=lambda item: item[1])[0]
    choices.update({f"Vertex {i}": i for i in range(count)})
    return choices


def _render_values(
    st: Any, snapshot: InspectorSnapshot, selection: str, vertex: int
) -> None:
    exact_u = () if snapshot.u_fem is None else tuple(hypot(*u) for u in snapshot.u_fem)
    fields = (
        ("Displacement magnitude", snapshot.u_magnitude_mm, "mm", "Surrogate"),
        ("Von Mises stress", snapshot.von_mises_mpa, "MPa", "Surrogate"),
        ("Determinant", snapshot.det_f_2d, "1", "Surrogate"),
        (
            "Correction magnitude",
            snapshot.corrected_diff_mm or (),
            "mm",
            "Corrected − surrogate",
        ),
        ("Displacement magnitude", exact_u, "mm", "Exact FEM"),
        (
            "Displacement difference",
            snapshot.fem_diff_mm or (),
            "mm",
            "Exact FEM − surrogate",
        ),
    )
    rows = []
    for name, values, unit, source in fields:
        value = "Missing"
        if vertex < len(values):
            measured = values[vertex]
            value = f"{measured:.6g}" if isfinite(measured) else "Non-finite"
        rows.append({"Field": name, "Value": value, "Unit": unit, "Source": source})
    with st.container(key="wb-probe-detail"):
        st.markdown(f"**Vertex {vertex}** · {_coordinates(snapshot, vertex)}")
        if selection.startswith("QoI · "):
            name = selection.removeprefix("QoI · ")
            qoi = next(q for q in snapshot.qois if q["name"] == name)
            measured = qoi["value"]
            value = f"{measured:.6g}" if isfinite(measured) else "Non-finite"
            st.markdown(f"**{name}** {value} {qoi['unit']} · Surrogate QoI")
        st.table(rows)
        if snapshot.u_fem is None:
            st.write("Exact FEM reference unavailable")


def render_field_maps(
    st: Any, snapshot: InspectorSnapshot, scope: str
) -> dict[str, Any]:
    _render_admissibility(st, snapshot)
    choices = _choices(snapshot)
    if not choices:
        st.info("Vertex data missing. Field probing is unavailable.")
        return {}

    choice_key, chart_key = f"wb-probe::{scope}", f"wb-probe-chart::{scope}"
    selection = st.selectbox(
        "Probe location",
        tuple(choices),
        index=None,
        key=choice_key,
        placeholder="Choose a vertex, extremum, or quantity of interest",
    )
    vertex = choices.get(selection)

    def select_from_map() -> None:
        event = st.session_state[chart_key]
        points = event["selection"]["points"]
        if points:
            choice = f"Vertex {points[-1].get('customdata')}"
            if choice in choices:
                st.session_state[choice_key] = choice

    figures = build_plotly_figures(snapshot, probe_vertex=vertex)
    with st.container(key="wb-probe-map"):
        st.write(
            "Pick a point on the map or use the probe menu. × marks values at or below the floor."
        )
        if len(finite_vertex_values(snapshot.det_f_2d, len(snapshot.nodes))) < len(
            snapshot.nodes
        ):
            st.write("◇ marks missing or non-finite values.")
        st.plotly_chart(
            figures["det_f_2d"],
            key=chart_key,
            on_select=select_from_map,
            selection_mode="points",
            width="stretch",
            config={"scrollZoom": False},
        )
    if vertex is not None:
        _render_values(st, snapshot, selection, vertex)
    displacement, stress = st.columns(2)
    if "displacement_mm" in figures:
        displacement.plotly_chart(figures["displacement_mm"], width="stretch")
    if "von_mises_mpa" in figures:
        stress.plotly_chart(figures["von_mises_mpa"], width="stretch")
    return figures
