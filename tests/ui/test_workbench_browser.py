"""Browser regressions against the installed FormaMech CLI and Streamlit host."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import sysconfig
import time
from dataclasses import replace
from pathlib import Path
from statistics import median
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import ZipFile

import pytest

from tests.ui._evidence import save_qualification

playwright = pytest.importorskip("playwright.sync_api")
pytest.importorskip("streamlit")

PLATE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "cad" / "plate-with-hole.step"
)
BUNDLE = Path(__file__).resolve().parents[2] / "bundles" / "notched-v1-cp-freeze"


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


@pytest.fixture(scope="module")
def browser():
    with playwright.sync_playwright() as runtime:
        chromium = runtime.chromium.launch()
        yield chromium
        chromium.close()


def _probe_inspector(tmp_path, *, known_fields=False):
    script = """
from pathlib import Path
from piginot.ui import inspector
from piginot.ui.workbench.job_runner import WorkbenchJobRunner

def count(kind, wrapped):
    def measured(*args, **kwargs):
        with (Path(__file__).parent / 'probe.log').open('a', encoding='utf-8') as log:
            log.write(kind + '\\n')
        return wrapped(*args, **kwargs)
    return measured

WorkbenchJobRunner.list_jobs = count('jobs', WorkbenchJobRunner.list_jobs)
"""
    if known_fields:
        script += """
def initial_fields(*args, mode, **kwargs):
    on_floor = mode == 'exact'
    return inspector.InspectorSnapshot(
        bundle_name='probe-fixture', mode=mode, serving_path='test',
        verification_tier='test', claim_level='test', verdict='test',
        nodes=((0.0, 0.0), (4.0, 0.0), (0.0, 4.0)), cells=((0, 1, 2),),
        u_surrogate=((0.0, 0.0), (0.1, 0.0), (0.2, 0.0)),
        u_magnitude_mm=(0.0, 0.1, 0.2), det_f_2d=(0.0001 if on_floor else 0.9, 1.1, 1.2),
        j_3d=(1.0, 1.1, 1.2), von_mises_mpa=(float('nan'), 0.3, 0.8),
        corrected_diff_mm=None, fem_diff_mm=None, u_fem=None,
        qois=({'name': 'peak_u', 'value': 0.2, 'unit': 'mm', 'vertex_index': 2,
              'location_mm': [0.0, 4.0]},),
        tier0_admissibility={'passed': False,
                             'failed_checks': ['equilibrium_residual_exceeded']},
        certificate=None, bounds=None, trust_report=None, trust_receipt=None,
        trust_receipt_sha256=None, receipt_verified=False, receipt_errors=(),
        limitation_paragraph='Fixture',
    )
inspector.build_inspector_snapshot = initial_fields
"""
    script += "inspector.build_inspector_snapshot = count('inspector', inspector.build_inspector_snapshot)\n"
    (tmp_path / "sitecustomize.py").write_text(script, encoding="utf-8")


@pytest.fixture
def workbench_server(tmp_path, request):
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    workspace = tmp_path / "study"
    cli = Path(sysconfig.get_path("scripts")) / (
        "formamech.exe" if os.name == "nt" else "formamech"
    )
    log = tmp_path / "streamlit.log"
    environment = {
        **os.environ,
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        "STREAMLIT_SERVER_ADDRESS": "127.0.0.1",
    }
    bundle_args = []
    server_mode = getattr(request, "param", None)
    if server_mode in ("probe", "known_fields"):
        _probe_inspector(tmp_path, known_fields=server_mode == "known_fields")
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(tmp_path), os.environ.get("PYTHONPATH")))
        )
        bundle_args = ["--bundle", str(BUNDLE)]
    with log.open("wb") as output:
        server = subprocess.Popen(
            [
                str(cli),
                "ui",
                "--headless",
                "--port",
                str(port),
                "--workspace",
                str(workspace),
                *bundle_args,
            ],
            cwd=tmp_path,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
        )
    try:
        deadline = time.monotonic() + 45
        while server.poll() is None and time.monotonic() < deadline:
            try:
                with urlopen(f"{url}/_stcore/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except (URLError, TimeoutError):
                time.sleep(0.1)
        else:
            pytest.fail(
                f"Workbench server did not start:\n{log.read_text(errors='replace')}"
            )
        yield url, workspace
    finally:
        if server.poll() is None:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(server.pid)],
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
            else:
                os.killpg(server.pid, signal.SIGTERM)
            server.wait(timeout=10)


@pytest.fixture
def workbench_page(browser, workbench_server, request):
    url, _ = workbench_server
    page = browser.new_page(service_workers="block", **getattr(request, "param", {}))
    page.route(
        "**/*",
        lambda route: (
            route.continue_()
            if route.request.url.startswith(f"{url}/")
            else route.abort()
        ),
    )
    try:
        page.goto(url)
        playwright.expect(
            page.get_by_role("combobox", name="Workbench stage")
        ).to_be_visible(timeout=45_000)
        yield page
    finally:
        page.close()


def _go_stage(page, name):
    stage = page.get_by_role("combobox", name="Workbench stage")
    stage.press("ArrowDown")
    stage.fill(name)
    stage.press("Enter")
    playwright.expect(page.get_by_role("heading", name=name, exact=True)).to_be_visible(
        timeout=30_000
    )


@pytest.mark.parametrize(
    "workbench_page", [{"viewport": {"width": 320, "height": 900}}], indirect=True
)
def test_export_decision_follows_selected_bundle_and_rechecks_evidence(
    workbench_server, workbench_page
):
    page = workbench_page
    _, root = workbench_server
    candidate = root / "run" / "candidate-bundle"
    shutil.copytree(BUNDLE, candidate)
    manifest = root / "qualification" / "qualification-manifest.json"
    save_qualification(candidate, manifest)
    page.reload()
    _go_stage(page, "Trust Inspector")
    page.get_by_role("combobox", name="Verification Mode").click()
    page.get_by_role("option", name="fast", exact=True).click()

    panel = page.locator(".st-key-wb-deployment-evidence")
    qualified = page.get_by_role("button", name="Export qualified FMU")
    playwright.expect(qualified).to_be_enabled()
    playwright.expect(
        panel.get_by_text("Qualified for this policy", exact=True)
    ).to_be_visible()
    playwright.expect(panel).to_contain_text("Granted claim DEVELOPMENT_VERIFIED")
    playwright.expect(panel).to_contain_text(
        "Evaluation policy formamech-dev-standard-v1"
    )
    playwright.expect(page.get_by_text("Receipt integrity", exact=True)).to_be_visible(
        timeout=45_000
    )
    qualified.click()
    playwright.expect(
        panel.get_by_text(re.compile("^Exported FMI 3.0 FMU"))
    ).to_be_visible(timeout=30_000)
    (exported,) = root.glob("*.fmu")
    with ZipFile(exported) as archive:
        assert (
            json.loads(archive.read("resources/fmu_manifest.json"))["bundle_name"]
            == BUNDLE.name
        )
    exported.unlink()

    bundle_field = page.get_by_role("textbox", name="Bundle Directory")
    bundle_field.fill(str(BUNDLE.with_name("dogbone-v1-s13c-cp-freeze")))
    bundle_field.press("Enter")
    playwright.expect(
        page.get_by_text("Evidence belongs to another bundle", exact=True)
    ).to_be_visible()
    playwright.expect(
        page.get_by_role("button", name="Export FMU", exact=True)
    ).to_be_disabled()
    sidebar = page.get_by_test_id("stSidebar")
    playwright.expect(
        sidebar.get_by_text("Model Qualified · Complete", exact=True)
    ).to_be_attached()

    evidence_field = page.get_by_role(
        "textbox", name="Qualification Manifest (optional)"
    )
    evidence_field.fill("")
    evidence_field.press("Enter")
    playwright.expect(
        page.get_by_text("Qualification evidence missing", exact=True)
    ).to_be_visible()
    playwright.expect(
        page.get_by_role("button", name="Export unchecked FMU")
    ).to_be_enabled()

    external_evidence = root / "external-evidence.json"
    external_evidence.write_text("{}", encoding="utf-8")
    evidence_field.fill(str(external_evidence))
    evidence_field.press("Enter")
    unreadable = page.get_by_text("Qualification evidence unreadable", exact=True)
    playwright.expect(unreadable).to_be_visible()
    external_evidence.write_text("{malformed", encoding="utf-8")
    page.get_by_role("combobox", name="Verification Mode").click()
    page.get_by_role("option", name="certified", exact=True).click()
    playwright.expect(unreadable).to_be_visible()

    evidence_field.fill(str(manifest))
    evidence_field.press("Enter")
    bundle_field.fill(str(candidate))
    bundle_field.press("Enter")
    playwright.expect(qualified).to_be_enabled()
    save_qualification(candidate, manifest, qualified=False)
    qualified.click()
    playwright.expect(
        page.get_by_text("Qualification refused", exact=True)
    ).to_be_visible()
    playwright.expect(
        page.get_by_role("button", name="Export FMU", exact=True)
    ).to_be_disabled()
    assert list(root.glob("*.fmu")) == []


@pytest.mark.parametrize("workbench_server", ["known_fields"], indirect=True)
def test_field_probe_uses_cached_result_and_applies_inclusive_floor(
    workbench_server, workbench_page
):
    page = workbench_page
    _, root = workbench_server
    _go_stage(page, "Trust Inspector")
    limit = page.locator(".st-key-wb-determinant-summary")
    playwright.expect(limit).to_contain_text("0 flagged vertices")
    playwright.expect(limit).to_contain_text("Equilibrium residual exceeded")
    receipt = page.locator(".st-key-wb-receipt-integrity")
    playwright.expect(receipt).to_have_count(1)
    playwright.expect(receipt.get_by_text("Absent", exact=True)).to_be_visible()

    page.get_by_role("button", name="Run Warm-Start FEM & Compare").click()
    playwright.expect(limit).to_contain_text("1 flagged vertex")
    mode = page.get_by_role("combobox", name="Verification Mode")
    playwright.expect(mode).to_have_count(1)
    assert mode.evaluate(
        "el => el.value === 'exact' || el.getAttribute('aria-label').startsWith('Selected exact.')"
    )
    playwright.expect(limit).to_contain_text("0.0001")
    playwright.expect(limit).to_contain_text("Vertex 0")
    probe_log = root.parent / "probe.log"
    evaluations = probe_log.read_text(encoding="utf-8").splitlines().count("inspector")

    choice = page.get_by_role("combobox", name="Probe location")
    choice.focus()
    choice.press("ArrowDown")
    choice.fill("Minimum determinant")
    choice.press("Enter")
    detail = page.locator(".st-key-wb-probe-detail")
    playwright.expect(detail).to_contain_text("Vertex 0")
    playwright.expect(detail).to_contain_text("(0, 0) mm")
    playwright.expect(detail).to_contain_text("Surrogate")
    playwright.expect(detail).to_contain_text("Non-finite")
    playwright.expect(detail).to_contain_text("Exact FEM reference unavailable")

    choice.press("ArrowDown")
    page.get_by_role("option", name="QoI · peak_u").click()
    playwright.expect(detail).to_contain_text("Vertex 2")
    playwright.expect(detail).to_contain_text("peak_u 0.2 mm")

    plot = page.locator(".st-key-wb-probe-map .js-plotly-plot")
    playwright.expect(plot).to_have_count(1)
    points = plot.locator(".scatterlayer .trace").nth(1).locator(".point")
    target = points.nth(1)
    target.scroll_into_view_if_needed()
    box = target.bounding_box()
    assert box is not None
    page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    playwright.expect(detail).to_contain_text("Vertex 1")
    playwright.expect(detail).to_contain_text("(4, 0) mm")
    playwright.expect(plot).to_have_count(1)
    assert plot.evaluate(
        "el => el.data.find(trace => trace.name === 'Probe').customdata"
    ) == [1]
    assert (
        probe_log.read_text(encoding="utf-8").splitlines().count("inspector")
        == evaluations
    )


def _assert_mesh_and_boundaries(page):
    metrics = page.get_by_test_id("stMetric")
    nodes = metrics.filter(has_text="Nodes").get_by_test_id("stMetricValue")
    error = page.get_by_text(re.compile(r"^(CAD import|Sample generation) failed"))
    playwright.expect(nodes.or_(error)).to_be_visible(timeout=45_000)
    assert not error.is_visible(), error.inner_text()
    for name in ("Nodes", "Triangles", "Boundaries"):
        value = metrics.filter(has_text=name).get_by_test_id("stMetricValue")
        playwright.expect(value).to_have_text(re.compile(r"^\s*[1-9][\d,]*\s*$"))

    _go_stage(page, "Boundary Picker")
    selector = page.get_by_role("combobox", name="Selected CAD Boundary Group")
    selector.select_option("edge-000002")
    playwright.expect(page.get_by_role("textbox", name="Condition Name")).to_have_value(
        "right_load"
    )


def test_import_uploaded_cad_in_fresh_workbench(workbench_page):
    workbench_page.locator('input[type="file"]').set_input_files(PLATE)
    workbench_page.get_by_role("button", name="Import Uploaded CAD").click()
    _assert_mesh_and_boundaries(workbench_page)


def test_generate_sample_in_fresh_workbench(workbench_page):
    workbench_page.get_by_role("button", name="Generate Sample 2D Bracket").click()
    _assert_mesh_and_boundaries(workbench_page)


def test_retries_invalid_upload_in_same_session(workbench_page):
    page = workbench_page
    uploader = page.locator('input[type="file"]')
    uploader.set_input_files(
        {
            "name": PLATE.name,
            "mimeType": "application/octet-stream",
            "buffer": b"bad CAD",
        }
    )
    size = page.get_by_role("spinbutton", name="Target Mesh Size (mm)")
    size.fill("4")
    size.press("Tab")
    upload = page.get_by_role("button", name="Import Uploaded CAD")
    upload.click()
    playwright.expect(
        page.get_by_text(re.compile("^CAD import failed"))
    ).to_be_visible()
    playwright.expect(page.get_by_text("Technical details")).to_be_visible()
    playwright.expect(page.get_by_text(PLATE.name, exact=True)).to_be_visible()
    playwright.expect(size).to_have_value(re.compile(r"^4(?:\.0+)?$"))
    remove = page.get_by_role("button", name=f"Remove {PLATE.name}")
    remove.click()
    playwright.expect(remove).to_be_hidden()
    playwright.expect(page.get_by_text(re.compile("^CAD import failed"))).to_be_hidden(
        timeout=20_000
    )
    uploader.set_input_files(PLATE)
    playwright.expect(page.get_by_text(PLATE.name, exact=True)).to_be_visible()
    upload.click()
    _assert_mesh_and_boundaries(page)


@pytest.fixture
def populated_study(workbench_server):
    from formamech.ui import (
        WorkbenchWorkspace,
        default_draft_from_preview,
        prepare_and_plan_workspace,
    )

    _, root = workbench_server
    study = WorkbenchWorkspace(root)
    preview = study.import_cad_path(PLATE)
    prepare_and_plan_workspace(
        study,
        replace(default_draft_from_preview(preview), optimizer_steps=40),
        execution_profile=False,
    )
    study.run_dir.mkdir()
    study.qualification_dir.mkdir()
    (study.run_dir / "report.json").write_text("{}", encoding="utf-8")
    (study.qualification_dir / "qualification-manifest.json").write_text(
        "{}", encoding="utf-8"
    )
    return study


def test_sample_and_replacement_preserve_a_populated_study(
    populated_study, workbench_page
):
    study, page = populated_study, workbench_page
    original = {
        p: p.read_bytes()
        for p in (
            study.source_dir / PLATE.name,
            study.geometry_dir / "geometry.json",
            study.physics_path,
            study.preview_dir / "plan.json",
            study.run_dir / "report.json",
            study.qualification_manifest_path,
        )
    }
    page.get_by_text("Workspace details", exact=True).click()
    path = page.get_by_role("textbox", name="Workspace Directory")
    page.get_by_role("button", name="Open Sample in New Workspace").click()
    playwright.expect(path).not_to_have_value(str(study.root), timeout=45_000)
    playwright.expect(
        page.get_by_test_id("stMetric").filter(has_text="Triangles")
    ).to_be_visible()
    assert all(p.read_bytes() == payload for p, payload in original.items())
    path.fill(str(study.root))
    path.press("Enter")
    playwright.expect(
        page.get_by_role("heading", name="cad-workbench-model", exact=True)
    ).to_be_visible()

    uploader = page.locator('input[type="file"]')
    uploader.set_input_files(
        {
            "name": PLATE.name,
            "mimeType": "application/octet-stream",
            "buffer": b"bad CAD",
        }
    )
    playwright.expect(page.get_by_text(PLATE.name, exact=True)).to_be_visible()
    upload = page.get_by_role("button", name="Import Uploaded CAD")
    upload.click()
    review = page.get_by_text(re.compile(r"Replace .*plate-with-hole.step"))
    playwright.expect(review).to_be_visible()
    playwright.expect(
        page.get_by_text("Qualification evidence", exact=True)
    ).to_be_visible()
    page.get_by_role("button", name="Keep current study").click()
    playwright.expect(review).to_be_hidden(timeout=20_000)
    assert all(p.read_bytes() == payload for p, payload in original.items())

    upload.click()
    replace = page.get_by_role("button", name="Replace geometry", exact=True)
    replace.click()
    playwright.expect(
        page.get_by_text(re.compile("^CAD import failed"))
    ).to_be_visible()
    assert all(p.read_bytes() == payload for p, payload in original.items())
    page.get_by_role("button", name=f"Remove {PLATE.name}").click()
    uploader.set_input_files(PLATE)
    playwright.expect(page.get_by_text(PLATE.name, exact=True)).to_be_visible()
    size = page.get_by_role("spinbutton", name="Target Mesh Size (mm)")
    size.fill("4")
    size.press("Enter")
    playwright.expect(size).to_have_value("4.00")
    upload.click()
    replace.click()
    playwright.expect(replace).to_be_hidden(timeout=45_000)
    playwright.expect(page.get_by_text(re.compile("^CAD import failed"))).to_be_hidden()
    playwright.expect(
        page.get_by_text("Imported and meshed plate-with-hole.step.")
    ).to_be_visible()
    assert (study.geometry_dir / "geometry.json").read_bytes() != original[
        study.geometry_dir / "geometry.json"
    ]
    assert all(
        not p.exists()
        for p in original
        if p.parent not in (study.source_dir, study.geometry_dir)
    )


def _picker(page):
    _go_stage(page, "Boundary Picker")
    selector = page.get_by_role("combobox", name="Selected CAD Boundary Group")
    error = page.get_by_text("BidiComponent Error:", exact=True)
    playwright.expect(selector.or_(error)).to_be_visible(timeout=30_000)
    assert not error.is_visible(), error.locator("..").inner_text()
    return selector


def _assert_picker_selection(page, selector, region):
    playwright.expect(selector).to_have_value(region)
    playwright.expect(page.get_by_test_id("boundary-plot")).to_have_attribute(
        "data-highlighted-region", region, timeout=20_000
    )
    playwright.expect(page.get_by_test_id("boundary-editor")).to_have_attribute(
        "data-region", region
    )
    playwright.expect(page.locator('tr[aria-current="true"]')).to_contain_text(region)


def test_material_plan_reviews_exact_pending_changes(populated_study, workbench_page):
    page, study = workbench_page, populated_study
    from piginot.ui.workbench.physics_forms import load_physics_draft

    draft = load_physics_draft(study, study.load_preview())
    edge = next(b for b in draft.boundaries if b.kind == "traction_free").region
    _picker(page).select_option(edge)
    page.get_by_label("Boundary Condition Type").select_option("symmetry")
    page.get_by_label("Constrain y").check()
    page.get_by_role("button", name="Apply boundary condition", exact=True).click()
    playwright.expect(
        page.get_by_text(f"Applied boundary condition to {edge}.")
    ).to_be_visible()
    artifact_paths = (
        study.state_path,
        study.physics_path,
        study.preview_dir / "plan.json",
    )
    before = {p: p.read_bytes() for p in artifact_paths}

    _go_stage(page, "Material & Plan")
    steps = page.get_by_role("spinbutton", name="Optimizer steps")
    steps.fill("127")
    steps.press("Enter")
    review = page.locator(".st-key-wb-plan-review")
    playwright.expect(
        review.get_by_text("Changes since last prepared plan", exact=True)
    ).to_be_visible()
    rows = review.locator("tbody tr")
    playwright.expect(rows).to_have_count(2)
    playwright.expect(rows.filter(has_text=edge)).to_contain_text("traction-free")
    playwright.expect(rows.filter(has_text=edge)).to_contain_text("symmetry · y")
    budget = rows.filter(has_text="Optimizer steps").get_by_role("cell")
    playwright.expect(budget.nth(2)).to_have_text("40")
    playwright.expect(budget.nth(3)).to_have_text("127")
    assert {p: p.read_bytes() for p in artifact_paths} == before

    page.get_by_role("button", name="Update plan", exact=True).click()
    playwright.expect(
        review.get_by_text("No settings changes.", exact=True)
    ).to_be_visible(timeout=30_000)
    assert study.physics_path.read_bytes() != before[study.physics_path]
    _go_stage(page, "Train & Qualify")
    playwright.expect(page.get_by_text("Prepared revision", exact=True)).to_be_visible()
    playwright.expect(
        page.get_by_role("button", name="Start Compile Job")
    ).to_be_enabled()


def test_picker_keyboard_pointer_and_apply_through_streamlit(
    populated_study, workbench_page
):
    page = workbench_page
    first, second = "edge-000001", "edge-000002"
    selector = _picker(page)
    _assert_picker_selection(page, selector, first)
    selector.focus()
    selector.press("ArrowDown")
    _assert_picker_selection(page, selector, second)

    plot = page.get_by_test_id("boundary-plot")
    handle = plot.locator(".scatterlayer .trace:last-child .points .point").first
    handle.scroll_into_view_if_needed()
    bounds = handle.bounding_box()
    assert bounds is not None
    page.mouse.click(
        bounds["x"] + bounds["width"] / 2, bounds["y"] + bounds["height"] / 2
    )
    _assert_picker_selection(page, selector, first)

    selector.press("ArrowDown")
    expression = page.get_by_role("textbox", name="u_x expression (mm)")
    expression.fill("ux / 2")
    selector.press("ArrowUp")
    _assert_picker_selection(page, selector, first)
    selector.press("ArrowDown")
    _assert_picker_selection(page, selector, second)
    playwright.expect(expression).to_have_value("ux / 2")
    playwright.expect(page.locator('tr[aria-current="true"]')).not_to_contain_text(
        "ux / 2"
    )
    page.get_by_role("button", name="Apply boundary condition", exact=True).press(
        "Enter"
    )
    playwright.expect(
        page.get_by_text(f"Applied boundary condition to {second}.")
    ).to_be_visible(timeout=30_000)
    page.reload()
    selector = _picker(page)
    selector.press("ArrowDown")
    playwright.expect(
        page.get_by_role("textbox", name="u_x expression (mm)")
    ).to_have_value("ux / 2")
    saved = populated_study.load_state()["physics_draft"]["boundary_conditions"]
    assert next(b for b in saved if b["region"] == second)["values"] == [
        {"expression": "ux / 2", "unit": "mm"}
    ]

    _go_stage(page, "Material & Plan")
    prepare = page.locator(".st-key-wb-prepare-control")
    playwright.expect(prepare.get_by_role("button", name="Update plan")).to_be_enabled()
    _go_stage(page, "Train & Qualify")
    playwright.expect(
        page.get_by_role("button", name="Start Compile Job")
    ).to_be_disabled()
    playwright.expect(
        page.get_by_role("button", name=re.compile("Resume Compile"))
    ).to_be_disabled()


@pytest.mark.parametrize(
    "workbench_page",
    [
        {"viewport": {"width": 320, "height": 900}, "color_scheme": theme}
        for theme in ("light", "dark")
    ],
    indirect=True,
    ids=["light", "dark"],
)
def test_picker_at_320_css_pixels(populated_study, workbench_page):
    page = workbench_page
    theme = (
        "dark"
        if page.evaluate("matchMedia('(prefers-color-scheme: dark)').matches")
        else "light"
    )
    selector = _picker(page)
    selector.focus()
    selector.press("ArrowDown")
    _assert_picker_selection(page, selector, "edge-000002")
    drawing = page.get_by_test_id("boundary-plot").bounding_box()
    field = page.get_by_role("textbox", name="Condition Name").bounding_box()
    select = selector.bounding_box()
    assert select and drawing and field
    assert select["y"] < drawing["y"] < field["y"]
    assert all(
        box["x"] >= 0 and box["x"] + box["width"] <= 321
        for box in (select, drawing, field)
    )
    apply = page.get_by_role("button", name="Apply boundary condition", exact=True)
    apply.scroll_into_view_if_needed()
    playwright.expect(apply).to_be_in_viewport()

    ratio = selector.evaluate("""element => {
      const canvas = document.createElement('canvas');
      canvas.width = canvas.height = 1;
      const ctx = canvas.getContext('2d', {willReadFrequently: true});
      const parents = [];
      for (let node = element; node; node = node.parentElement) parents.unshift(node);
      ctx.fillStyle = 'white';
      ctx.fillRect(0, 0, 1, 1);
      for (const node of parents) {
        ctx.fillStyle = getComputedStyle(node).backgroundColor;
        ctx.fillRect(0, 0, 1, 1);
      }
      const luminance = () => [...ctx.getImageData(0, 0, 1, 1).data].slice(0, 3)
        .map(value => value / 255)
        .map(value => value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4)
        .reduce((total, value, i) => total + value * [0.2126, 0.7152, 0.0722][i], 0);
      const background = luminance();
      const style = getComputedStyle(element);
      const contrast = color => {
        const before = ctx.getImageData(0, 0, 1, 1);
        ctx.fillStyle = color;
        ctx.fillRect(0, 0, 1, 1);
        const foreground = luminance();
        ctx.putImageData(before, 0, 0);
        return (Math.max(foreground, background) + 0.05) / (Math.min(foreground, background) + 0.05);
      };
      return {background, text: contrast(style.color), outline: contrast(style.outlineColor),
              visibleOutline: style.outlineStyle !== 'none' && parseFloat(style.outlineWidth) >= 2};
    }""")
    assert (ratio["background"] > 0.5) == (theme == "light"), ratio
    assert ratio["text"] >= 4.5, ratio
    assert ratio["visibleOutline"] and ratio["outline"] >= 3.0, ratio
    print(
        f"{theme} theme at 320 CSS px. Text {ratio['text']:.2f}:1, focus {ratio['outline']:.2f}:1"
    )


@pytest.mark.parametrize("workbench_server", ["known_fields"], indirect=True)
@pytest.mark.parametrize(
    "workbench_page", [{"viewport": {"width": 320, "height": 900}}], indirect=True
)
def test_reopen_and_navigate_by_keyboard_preserves_picker(
    populated_study, workbench_page
):
    from formamech.ui import BoundaryAssignment

    from piginot.ui.workbench.physics_forms import apply_boundary

    page, study = workbench_page, populated_study
    assert apply_boundary(
        study,
        study.load_preview(),
        BoundaryAssignment("edge-000001", "free_1", "symmetry", ("y",)),
    )
    page.reload()
    next_action = page.locator(".st-key-wb-next-action").get_by_role(
        "button", name="Update plan"
    )
    playwright.expect(next_action).to_be_visible(timeout=30_000)
    next_action.press("Enter")
    stage = page.get_by_role("combobox", name="Workbench stage")
    playwright.expect(
        page.get_by_role("heading", name="Material & Plan", exact=True)
    ).to_be_visible()
    update = page.locator(".st-key-wb-prepare-control").get_by_role(
        "button", name="Update plan"
    )
    playwright.expect(update).to_be_focused()

    names = [
        "Import CAD",
        "Boundary Picker",
        "Material & Plan",
        "Train & Qualify",
        "Trust Inspector",
    ]
    stage.focus()
    stage.press("ArrowDown")
    playwright.expect(page.get_by_role("option")).to_have_text(names)
    stage.fill("Boundary Picker")
    stage.press("Enter")
    heading = page.get_by_role("heading", name="Boundary Picker", exact=True)
    playwright.expect(heading).to_be_focused()
    selector = page.get_by_role("combobox", name="Selected CAD Boundary Group")
    selector.press("ArrowDown")
    _assert_picker_selection(page, selector, "edge-000002")
    expression = page.get_by_role("textbox", name="u_x expression (mm)")
    expression.fill("ux / 3")
    plot = page.get_by_test_id("boundary-plot")
    read_zoom = "plot => plot._fullLayout.xaxis.range.slice()"
    initial_zoom = plot.evaluate(read_zoom)
    plot.locator('[data-title="Zoom in"]').click()
    zoom = plot.evaluate(read_zoom)
    assert zoom != initial_zoom
    probe = study.root.parent / "probe.log"
    inspector_calls = probe.read_text(encoding="utf-8").splitlines().count("inspector")

    for name in ("Material & Plan", "Trust Inspector", "Boundary Picker"):
        stage.focus()
        stage.press("ArrowDown")
        stage.fill(name)
        stage.press("Enter")
        playwright.expect(
            page.get_by_role("heading", name=name, exact=True)
        ).to_be_focused()
    _assert_picker_selection(page, selector, "edge-000002")
    playwright.expect(expression).to_have_value("ux / 3")
    assert plot.evaluate(read_zoom) == pytest.approx(zoom)
    assert page.get_by_test_id("stMain").evaluate(
        "el => el.scrollWidth <= el.clientWidth"
    )
    assert (
        probe.read_text(encoding="utf-8").splitlines().count("inspector")
        == inspector_calls
    )


@pytest.fixture(params=["fixture", "refined_bracket"])
def short_job(workbench_server, request):
    from formamech.ui import WorkbenchJobRunner, WorkbenchWorkspace

    from piginot.ui.workbench.job_runner import JobRecord

    _, root = workbench_server
    workspace = WorkbenchWorkspace(root)
    if request.param == "refined_bracket":
        preview = workspace.import_sample(mesh_size_mm=0.35)
        assert preview["mesh"]["triangle_count"] > 10_000
    else:
        preview = workspace.import_cad_path(PLATE)
    workspace.run_dir.mkdir()
    finished = root / "finish-test-job"
    response = workspace.run_dir / "report.json"
    command = (
        sys.executable,
        "-c",
        "import pathlib, time, sys\n"
        "while not pathlib.Path(sys.argv[1]).exists(): time.sleep(0.05)\n"
        "pathlib.Path(sys.argv[2]).write_text('{\"exit_code\":0}', encoding='utf-8')",
        str(finished),
        str(response),
    )
    runner = WorkbenchJobRunner(workspace.jobs_dir, cwd=root)
    with subprocess.Popen(command, cwd=root, stdout=subprocess.DEVNULL) as process:
        try:
            runner._save(
                JobRecord(
                    job_id="picker-test-job",
                    kind="compile",
                    command=command,
                    pid=process.pid,
                    state="RUNNING",
                    exit_code=None,
                    started_utc="2026-11-23T00:00:00Z",
                    finished_utc=None,
                    log_path=str(workspace.jobs_dir / "picker-test-job.log"),
                    output_dir=str(workspace.run_dir),
                )
            )
            yield runner, workspace, finished, process, preview["mesh"]
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)


@pytest.fixture
def job_history(populated_study):
    from formamech.ui import WorkbenchJobRunner

    from piginot.ui.workbench.job_runner import JobRecord

    study = populated_study
    shutil.copytree(BUNDLE, study.run_dir / "candidate-bundle")
    runner = WorkbenchJobRunner(study.jobs_dir, cwd=study.root)
    now = "2026-11-23T00:00:0"
    for i in range(7):
        job_id = f"compile-{i}"
        log = study.jobs_dir / f"{job_id}.log"
        if i != 1:
            log.write_text(f"log owned by {job_id}", encoding="utf-8")
        runner._save(
            JobRecord(
                job_id=job_id,
                kind="compile",
                command=(),
                pid=None,
                state="FAILED" if i < 2 else "SUCCEEDED",
                exit_code=1 if i < 2 else 0,
                started_utc=f"{now}{i}Z",
                finished_utc=f"{now}{i}Z",
                log_path=str(log),
                output_dir=str(study.run_dir),
                latest_checkpoint_step=10,
            )
        )

    (study.run_dir / "report.md").write_text(
        "Newer report from a shared output path", encoding="utf-8"
    )
    (study.run_dir / "report.json").write_text('{"exit_code": 0}', encoding="utf-8")
    flag = study.root / "finish-new-job"
    token = "7" * 32
    evidence = study.jobs_dir / "evidence" / token
    evidence.mkdir(parents=True)
    (evidence / "owner.json").write_text(
        json.dumps(
            {
                "job_id": "compile-7",
                "kind": "compile",
                "output_dir": str(study.run_dir),
            }
        ),
        encoding="utf-8",
    )
    command = (
        sys.executable,
        "-c",
        "import pathlib, time, sys\n"
        "while not pathlib.Path(sys.argv[1]).exists(): time.sleep(.05)\n"
        "pathlib.Path(sys.argv[2]).write_text('0', encoding='ascii')",
        str(flag),
        str(evidence / "exit.txt"),
    )
    with subprocess.Popen(
        command, cwd=study.root, stdout=subprocess.DEVNULL
    ) as process:
        try:
            runner._save(
                JobRecord(
                    job_id="compile-7",
                    kind="compile",
                    command=command,
                    pid=process.pid,
                    state="RUNNING",
                    exit_code=None,
                    started_utc=f"{now}7Z",
                    finished_utc=None,
                    log_path=str(study.jobs_dir / "compile-7.log"),
                    output_dir=str(study.run_dir),
                    evidence_id=token,
                )
            )
            yield flag, process, study
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)


@pytest.mark.parametrize("workbench_server", ["known_fields"], indirect=True)
def test_old_job_owns_its_log_during_new_job_polling(job_history, workbench_page):
    page = workbench_page
    flag, process, study = job_history
    _go_stage(page, "Train & Qualify")
    selector = page.get_by_role("combobox", name="Recorded job")
    selector.press("ArrowDown")
    playwright.expect(page.get_by_role("option")).to_have_count(8)
    selector.fill("compile-0")
    selector.press("Enter")
    details = page.locator(".st-key-wb-job-detail")
    detail = details.filter(has_text="compile-0").last
    playwright.expect(detail).to_contain_text("Execution failure")
    playwright.expect(detail).to_contain_text("log owned by compile-0")
    playwright.expect(detail).to_contain_text("Recorded checkpoint 10")
    playwright.expect(detail).to_contain_text("Report unavailable for this run")
    playwright.expect(
        detail.get_by_role("button", name="Resume compile")
    ).to_have_count(0)
    playwright.expect(detail).not_to_contain_text(
        "Newer report from a shared output path"
    )

    probe = study.root.parent / "probe.log"
    evaluations = probe.read_text(encoding="utf-8").splitlines().count("inspector")
    notice = page.get_by_test_id("job-completion-notice")
    notice.evaluate("el => el.dataset.initialNode = 'yes'")
    candidate = study.run_dir / "candidate-bundle"
    candidate.rename(study.run_dir / "previous-candidate")
    shutil.copytree(BUNDLE.with_name("dogbone-v1-s13c-cp-freeze"), candidate)
    flag.touch()
    process.wait(timeout=5)
    playwright.expect(notice).to_have_text(
        "compile-7 finished. Succeeded.", timeout=20_000
    )
    playwright.expect(notice).to_have_attribute("data-initial-node", "yes")
    playwright.expect(detail).to_contain_text("log owned by compile-0")
    playwright.expect(detail).to_contain_text("Report unavailable for this run")

    outcome = page.get_by_role("combobox", name="Job outcome")
    outcome.press("ArrowDown")
    outcome.fill("Execution failure")
    outcome.press("Enter")
    playwright.expect(
        page.get_by_text("2 of 8 recorded jobs", exact=True)
    ).to_be_visible()
    selector.press("ArrowDown")
    playwright.expect(page.get_by_role("option")).to_have_count(2)
    selector.fill("compile-1")
    selector.press("Enter")
    playwright.expect(details.filter(has_text="compile-1").last).to_contain_text(
        "Log unavailable for this run"
    )
    _go_stage(page, "Trust Inspector")
    refresh = page.get_by_role("button", name="Evaluate selected bundle")
    playwright.expect(refresh).to_be_enabled()
    assert (
        probe.read_text(encoding="utf-8").splitlines().count("inspector") == evaluations
    )
    refresh.click()
    playwright.expect(page.get_by_text("Verification Tier", exact=True)).to_be_visible()
    assert (
        probe.read_text(encoding="utf-8").splitlines().count("inspector")
        == evaluations + 1
    )


def _wait_for_job_record(runner, page, predicate):
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        if predicate(runner.load("picker-test-job")):
            return
        page.wait_for_timeout(100)
    pytest.fail("The job monitor did not publish the expected update.")


def _selection_latencies(page, selector):
    selector.evaluate("""select => {
      const drawing = document.querySelector('[data-testid="boundary-plot"]');
      const samples = [];
      let started = null;
      select.measurements = samples;
      select.addEventListener('change', () => { started = performance.now(); }, true);
      new MutationObserver(() => {
        if (started !== null && drawing.dataset.highlightedRegion === select.value) {
          samples.push(performance.now() - started);
          started = null;
        }
      }).observe(drawing, {attributes: true, attributeFilter: ['data-highlighted-region']});
    }""")
    for edge in (2, 3, 4, 5, 1) * 3 + (2,):
        region = f"edge-{edge:06d}"
        selector.select_option(region)
        _assert_picker_selection(page, selector, region)
    samples = selector.evaluate("select => select.measurements")
    assert len(samples) == 16
    return samples


@pytest.mark.parametrize("workbench_server", ["probe"], indirect=True)
def test_job_poll_preserves_picker_and_does_not_evaluate_inspector(
    short_job,
    workbench_page,
    browser,
):
    page = workbench_page
    runner, study, finished, process, mesh = short_job
    probe = study.root.parent / "probe.log"
    _go_stage(page, "Trust Inspector")
    playwright.expect(
        page.get_by_test_id("stMetric").filter(has_text="Verification Tier")
    ).to_be_visible(timeout=45_000)
    inspector_calls = probe.read_text(encoding="utf-8").splitlines().count("inspector")
    assert inspector_calls > 0
    selector = _picker(page)
    _assert_picker_selection(page, selector, "edge-000001")
    latencies = _selection_latencies(page, selector)
    plot = page.get_by_test_id("boundary-plot")
    read_zoom = "plot => plot._fullLayout.xaxis.range.slice()"
    initial_zoom = plot.evaluate(read_zoom)
    plot.locator('[data-title="Zoom in"]').click()
    zoom = plot.evaluate(read_zoom)
    assert zoom != initial_zoom
    expression = page.get_by_role("textbox", name="u_x expression (mm)")
    expression.fill("ux / 4")
    (study.run_dir / "checkpoints" / "step-000000010").mkdir(parents=True)
    _wait_for_job_record(runner, page, lambda job: job.latest_checkpoint_step == 10)
    playwright.expect(expression).to_be_focused()
    playwright.expect(expression).to_have_value("ux / 4")
    assert plot.evaluate(read_zoom) == pytest.approx(zoom)

    finished.touch()
    process.wait(timeout=5)
    _wait_for_job_record(runner, page, lambda job: job.state == "SUCCEEDED")
    playwright.expect(
        page.locator(".st-key-wb-job-detail").get_by_text(
            re.compile("picker-test-job.*Succeeded")
        )
    ).to_be_attached(timeout=15_000)
    playwright.expect(expression).to_be_focused()
    playwright.expect(expression).to_have_value("ux / 4")
    assert plot.evaluate(read_zoom) == pytest.approx(zoom)
    started = probe.read_text(encoding="utf-8").splitlines()
    page.wait_for_timeout(2200)
    assert probe.read_text(encoding="utf-8").splitlines() == started
    assert started.count("inspector") == inspector_calls
    page.get_by_role("button", name="Apply boundary condition", exact=True).click()
    playwright.expect(
        page.get_by_text("Applied boundary condition to edge-000002.")
    ).to_be_visible(timeout=30_000)
    assignments = study.load_state()["physics_draft"]["boundary_conditions"]
    assert assignments[1]["values"][0]["expression"] == "ux / 4"
    _go_stage(page, "Train & Qualify")
    job_panel = page.locator(".st-key-wb-panel-train-and-qualify")
    playwright.expect(
        job_panel.get_by_role("button", name="Refresh Job Status")
    ).to_be_visible()
    print(
        f"Chromium {browser.version}, {mesh['node_count']} nodes and {mesh['triangle_count']} triangles. "
        f"{len(latencies)} selections, {median(latencies):.1f} ms median and {max(latencies):.1f} ms max. "
        "Zero selection mismatches, draft or zoom losses, or extra Inspector evaluations."
    )
