"""Run FormaMech with ``python -m formamech``."""

from formamech.cli import main
from piginot._job_evidence import bind_job_process, record_job_exit

bind_job_process()
try:
    code = main()
except SystemExit as exc:
    record_job_exit(
        exc.code if isinstance(exc.code, int) else int(exc.code is not None)
    )
    raise
except KeyboardInterrupt:
    record_job_exit(130)
    raise
except Exception:
    record_job_exit(1)
    raise
else:
    record_job_exit(code)
    raise SystemExit(code)
