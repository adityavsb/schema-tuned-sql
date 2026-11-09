"""MLflow experiment tracking wrapper with strict secret exclusion."""

from __future__ import annotations

import json
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

FORBIDDEN_PARAM_KEYS = {
    "api_key",
    "openrouter_api_key",
    "secret",
    "token",
    "password",
    "authorization",
}


def _inside_git_work_tree(start: Path | None = None) -> bool:
    """True when a .git directory exists at or above `start`, without invoking git.

    Calling the git binary outside a checkout in containerized or cloud environments
    can print filesystem boundary errors when git refuses to cross isolated mount points.
    """
    path = (start or Path.cwd()).resolve()
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return True
        if candidate.as_posix() in {"/kaggle", "/kaggle/working"}:
            break
    return False


def _skip_git_metadata_outside_repo() -> None:
    """Stop mlflow from shelling out to git when this process is not in a clone."""
    if _inside_git_work_tree():
        return
    try:
        import mlflow.utils.git_utils as git_utils
    except ImportError:
        return

    def _missing(*_args: Any, **_kwargs: Any) -> None:
        return None

    for name in ("get_git_commit", "get_git_branch", "get_git_repo_url"):
        if hasattr(git_utils, name):
            setattr(git_utils, name, _missing)


def sanitize_params(params: dict[str, Any]) -> dict[str, Any]:
    """Filters out any parameter key that could contain credentials."""
    clean: dict[str, Any] = {}
    for k, v in params.items():
        if any(bad in k.lower() for bad in FORBIDDEN_PARAM_KEYS):
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = json.dumps(v)
    return clean


class MLflowTracker:
    """Logs training/evaluation runs to local SQLite MLflow without exposing credentials."""

    def __init__(
        self,
        experiment_name: str = "schema-tuned-sql",
        tracking_uri: str = "sqlite:///mlflow.db",
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self._mlflow: Any = None
        self._active_run: Any = None

    def _get_mlflow(self) -> Any:
        if self._mlflow is None:
            try:
                import mlflow

                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.experiment_name)
                self._mlflow = mlflow
            except ImportError:
                self._mlflow = False
        return self._mlflow

    @contextmanager
    def start_run(
        self, run_name: str, params: dict[str, Any] | None = None
    ) -> Generator[MLflowTracker, None, None]:
        mlf = self._get_mlflow()
        safe_params = sanitize_params(params or {})
        if mlf:
            _skip_git_metadata_outside_repo()
            with mlf.start_run(run_name=run_name) as run:
                self._active_run = run
                if safe_params:
                    mlf.log_params(safe_params)
                try:
                    yield self
                finally:
                    self._active_run = None
        else:
            try:
                yield self
            finally:
                pass

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        mlf = self._get_mlflow()
        scalars = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        if mlf and self._active_run:
            mlf.log_metrics(scalars, step=step)

    def log_dict_artifact(
        self, payload: dict[str, Any], filename: str, out_dir: str = ".artifacts"
    ) -> Path:
        p = Path(out_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        mlf = self._get_mlflow()
        if mlf and self._active_run:
            mlf.log_artifact(str(p))
        return p
