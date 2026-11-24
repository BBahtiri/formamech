"""Per-job details and recovery in the workbench polling fragment."""

from __future__ import annotations

from functools import cache
from typing import Any

from piginot.ui.workbench.job_runner import JobRecord, WorkbenchJobRunner
from piginot.ui.workbench.state import WorkbenchWorkspace

OUTCOMES = {
    "PENDING": "Pending",
    "RUNNING": "Running",
    "SUCCEEDED": "Succeeded",
    "REFUSED_SCIENTIFIC": "Scientific refusal",
    "FAILED": "Execution failure",
    "CANCELLED": "Cancelled",
}
KINDS = {"compile": "Compile", "qualify": "Qualification"}


def _job_label(job: JobRecord) -> str:
    return f"{job.job_id} · {KINDS[job.kind]} · {OUTCOMES[job.state]}"


def _select_job(st: Any, jobs: list[JobRecord], scope: str) -> JobRecord | None:
    left, right = st.columns(2)
    kind = left.selectbox("Job kind", ("All", *KINDS.values()), key=f"{scope}-kind")
    outcome = right.selectbox(
        "Job outcome", ("All", *OUTCOMES.values()), key=f"{scope}-outcome"
    )
    records = {
        job.job_id: job
        for job in jobs
        if (kind == "All" or kind == KINDS[job.kind])
        and (outcome == "All" or outcome == OUTCOMES[job.state])
    }
    if not records:
        st.info("No recorded jobs match these filters.")
        return None
    if kind != "All" or outcome != "All":
        st.write(f"{len(records)} of {len(jobs)} recorded jobs")

    selected_key = f"{scope}-selected"
    widget_key = f"{scope}-selector"
    options_key = f"{scope}-options"
    labels = {job_id: _job_label(job) for job_id, job in records.items()}
    options = tuple(labels.items())
    previous = st.session_state.get(selected_key)
    target = previous if previous in records else next(iter(records))
    if previous is None:
        st.session_state[selected_key] = target
    if (
        st.session_state.get(options_key) != options
        or st.session_state.get(widget_key) != target
    ):
        st.session_state[widget_key] = target
        st.session_state[options_key] = options

    def remember_selection() -> None:
        st.session_state[selected_key] = st.session_state[widget_key]

    job_id = st.selectbox(
        "Recorded job",
        tuple(records),
        format_func=labels.__getitem__,
        index=None,
        key=widget_key,
        on_change=remember_selection,
    )
    return records.get(job_id)


def _render_job_detail(
    st: Any,
    runner: WorkbenchJobRunner,
    workspace: WorkbenchWorkspace,
    job: JobRecord,
    *,
    compile_active: bool,
) -> None:
    with st.container(key="wb-job-detail"):
        st.markdown(f"**{job.job_id}** · {OUTCOMES[job.state]}")
        st.write(f"Started (UTC) {job.started_utc}")
        if job.finished_utc is not None:
            st.write(f"Finished (UTC) {job.finished_utc}")
        if job.exit_code is not None:
            st.write(f"Exit code {job.exit_code}")
        elif job.state == "FAILED":
            st.write("Exit code unavailable for this run.")
        checkpoint = runner.job_checkpoint(job)
        if job.kind == "compile":
            st.write(
                f"Recorded checkpoint {checkpoint}"
                if checkpoint is not None
                else "No checkpoint recorded for this run."
            )

        report = runner.report_for(job)
        if report is not None:
            with st.expander("Report for this run"):
                st.markdown(report.decode("utf-8", errors="replace"))
                st.download_button(
                    "Download report for this run",
                    data=report,
                    file_name=f"{job.job_id}-report.md",
                    mime="text/markdown",
                    key=f"wb-report::{job.job_id}",
                )
        elif job.is_terminal:
            st.write("Report unavailable for this run")

        if (
            not compile_active
            and runner.can_resume(job, workspace)
            and st.button(
                "Resume compile",
                key=f"wb-resume::{job.job_id}",
            )
        ):
            try:
                runner.start_compile(
                    spec_path=workspace.inputs_dir / "problem.json",
                    config_path=workspace.inputs_dir / "config.json",
                    output_dir=job.output_dir,
                    resume=True,
                    resume_job_id=job.job_id,
                )
            except (OSError, ValueError) as exc:
                st.error(str(exc))
            else:
                st.rerun()
        st.markdown("**Log tail**")
        try:
            log = runner.read_log_tail(job.job_id, max_lines=80)
        except OSError:
            log = ""
        if log:
            st.code(log, language="text")
        else:
            st.write("Log unavailable for this run")


@cache
def _notice_component() -> Any:
    from streamlit.components.v2 import component

    return component(
        "formamech.job_status",
        html='<div role="status" aria-live="polite" aria-atomic="true" data-testid="job-completion-notice"></div>',
        js="""
export default function({parentElement, data}) {
  const region = parentElement.querySelector('[role="status"]');
  const message = data?.message ?? '';
  if (region && region.textContent !== message) region.textContent = message;
}
""",
    )


def render_job_completion(st: Any, workspace: WorkbenchWorkspace) -> None:
    key = f"wb-history::{workspace.root}-notice"
    message = st.session_state.get(key, "")
    _notice_component()(data={"message": message}, key="wb-job-completion")


def render_job_monitor(
    st: Any, runner: WorkbenchJobRunner, workspace: WorkbenchWorkspace
) -> None:
    from streamlit import fragment

    jobs = runner.list_jobs()
    if not jobs:
        return
    watched = {job.job_id for job in jobs if not job.is_terminal}
    scope = f"wb-history::{workspace.root}"
    seen_key = f"{scope}-active"
    notice_key = f"{scope}-notice"

    @fragment(run_every=1.0 if watched else None)
    def monitor() -> None:
        st.divider()
        st.subheader("Background jobs")
        refresh = st.button("Refresh Job Status", key=f"{scope}-refresh")
        current = runner.list_jobs() if watched or refresh else jobs
        refreshed = [
            runner.poll(job.job_id) if not job.is_terminal else job for job in current
        ]
        running = [job for job in refreshed if not job.is_terminal]
        running_ids = {job.job_id for job in running}
        previous = st.session_state.get(seen_key, set()) | watched
        completed = tuple(
            f"{job.job_id} finished. {OUTCOMES[job.state]}."
            for job in refreshed
            if job.is_terminal and job.job_id in previous
        )
        st.session_state[seen_key] = running_ids
        if completed:
            st.session_state[notice_key] = " ".join(completed)
        if completed or watched != running_ids:
            # Register the new polling interval without resetting the selected job.
            st.rerun()

        if running:
            st.markdown("**Active jobs**")
            for job in running:
                st.write(_job_label(job))
                if st.button(f"Cancel {job.job_id}", key=f"cancel_job::{job.job_id}"):
                    try:
                        runner.cancel(job.job_id)
                    except (OSError, RuntimeError, ValueError) as exc:
                        st.error(str(exc))
                    else:
                        st.rerun()
        st.markdown("**Job history**")
        if selected := _select_job(st, refreshed, scope):
            _render_job_detail(
                st,
                runner,
                workspace,
                selected,
                compile_active=any(job.kind == "compile" for job in running),
            )

    monitor()
