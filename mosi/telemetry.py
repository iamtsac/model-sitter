import os
import re
import logging
from pathlib import Path
from typing import Optional, List, Dict, Union, Any, Sequence

class WandbManager:
    def __init__(
        self,
        fpath: Path,
        overwrite: bool,
        resume: bool,
        logger: logging.Logger,
        *,
        project: Optional[str] = None,
        group: Optional[str] = None,
        name: Optional[str] = None,
        job_type: Optional[str] = None,
        tags: Sequence[str] = (),
        notes: Optional[str] = None,
        id_prefix: Optional[str] = None,
    ):
        try:
            import wandb
        except ImportError as e:
            raise ImportError("WandB logging needs wandb: pip install 'model-sitter[wandb]'") from e
        self.wandb = wandb
        self.fpath = fpath
        self.logger = logger
        self.defined_metrics = set()
        self.run = None
        self._meta = dict(project=project, group=group, name=name, job_type=job_type,
                          tags=list(tags) or None, notes=notes, id_prefix=id_prefix or fpath.name)

        self._init_run(overwrite, resume)

    @staticmethod
    def _safe_id(text: str) -> str:
        """W&B ids may not contain characters like '/', so group/run names are sanitised."""
        return re.sub(r"[^A-Za-z0-9_.-]", "_", text)

    def _init_run(self, overwrite: bool, resume: bool):
        from wandb.sdk.lib.runid import generate_id
        m = self._meta
        project = m["project"] or os.environ.get("WANDB_PROJECT")
        assert project is not None, "Pass project= or set the WANDB_PROJECT env variable"

        id_file = self.fpath / "wandb_id.txt"
        lineage_ids = [line.strip() for line in id_file.read_text().split('\n') if line.strip()] if id_file.exists() else []

        # Only this run's own lineage is deleted, siblings in the same group are never touched.
        if overwrite and lineage_ids:
            self._delete_cloud_lineage(project, lineage_ids)
            lineage_ids = []

        # A fresh id every time, resumes included: W&B refuses to reuse the id of a deleted run, and a
        # resumed run branches (history up to the checkpoint is replayed) so it never collides with
        # steps the server already holds past the checkpoint. `wandb_id.txt` keeps the whole lineage.
        run_id = f"{self._safe_id(m['id_prefix'])}_{generate_id()}"

        keep_lineage = resume and not overwrite and bool(lineage_ids)
        with open(id_file, "a" if keep_lineage else "w") as f:
            f.write(f"\n{run_id}" if keep_lineage else run_id)

        self._base_name = m["name"] or os.environ.get("WANDB_RUN") or run_id
        self.run = self.wandb.init(
            project=project, id=run_id, dir=str(self.fpath),
            name=self._base_name,
            group=m["group"], job_type=m["job_type"], tags=m["tags"], notes=m["notes"],
            settings=self.wandb.Settings(_disable_stats=True, _disable_meta=True)
        )
        self.run.define_metric("step", hidden=True)

    def _delete_cloud_lineage(self, project: str, lineage_ids: List[str]):
        self.logger.info(f"🗑️ Deleting {len(lineage_ids)} old WandB run(s)...")
        api = self.wandb.Api()
        for old_id in lineage_ids:
            try:
                api.run(f"{project}/{old_id}").delete()
            except Exception as e:
                self.logger.warning(f"Could not delete {old_id}: {e}")

    def define_metrics(self, flat_stats: dict):
        for k in flat_stats.keys():
            if k not in self.defined_metrics and k != "step":
                self.run.define_metric(k, step_metric="step")
                self.defined_metrics.add(k)

    def log_metrics(self, entry: dict, step: int):
        self.run.log(entry, step=step, commit=True)

    def update_config(self, config: dict):
        self.run.config.update(config, allow_val_change=True)

    def replay_history(self, history_rows: List[dict]):
        for i, row in enumerate(history_rows):
            step = row.get("step", i + 1)
            self.run.log(row, step=step)
        self.logger.info(f"✅ Replayed {len(history_rows)} steps of history into WandB.")

    def append_resume_step(self, step: int):
        """Marks the branched run: name `<name>_resume_from_step_<step>` (always from the original name, so it
        never accumulates over several resumes) and a `resumed` tag to filter on."""
        if self.run is None:
            return

        new_name = f"{self._base_name}_resume_from_step_{step}"
        self.run.name = new_name
        self.run.tags = tuple(self.run.tags or ()) + ("resumed",)

        if getattr(getattr(self.run, "settings", None), "mode", "online") != "online":
            return  # nothing on the server to rename (offline / disabled)
        try:
            server_run = self.wandb.Api().run(self.run.path)
            server_run.name = new_name
            server_run.update()
        except Exception as e:
            self.logger.warning(f"Could not push name change to WandB server: {e}")

    def finish(self):
        if self.run is not None:
            self.run.finish()
            self.run = None

    # Artifact specific logs
    def log_plot(self, name: str, plot: Any, step: int): self.run.log({f"plot/{name}": plot}, step=step)
    def log_image(self, name: str, path: Path, step: int): self.run.log({f"images/{name}": self.wandb.Image(str(path))}, step=step)
    def log_video(self, name: str, path: Path, step: int): self.run.log({f"videos/{name}": self.wandb.Video(str(path), format="mp4")}, step=step)
    def log_table(self, name: str, data: dict, step: int):
        keys, rows = list(data.keys()), list(zip(*list(data.values())))
        self.run.log({f"analysis/{name}": self.wandb.Table(columns=keys, data=rows)}, step=step)
