''

from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class SamplingContext:
    ''

    def __init__(self, progress: Progress | None, task: TaskID | None, num_prompts: int, num_steps: int):
        self._progress = progress
        self._task = task
        self._num_prompts = num_prompts
        self._num_steps = num_steps

    def start_video(self, video_idx: int) -> None:
        ''
        if self._progress is None or self._task is None:
            return
                                                                
        self._progress.reset(self._task, total=self._num_steps)
        self._progress.update(
            self._task,
            completed=0,
            video=f"{video_idx + 1}/{self._num_prompts}",
            info=f"step 0/{self._num_steps}",
        )

    def advance_step(self) -> None:
        ''
        if self._progress is None or self._task is None:
            return
        self._progress.advance(self._task)
        completed = int(self._progress.tasks[self._task].completed)
        self._progress.update(self._task, info=f"step {completed}/{self._num_steps}")

    def cleanup(self) -> None:
        ''
        if self._progress is None or self._task is None:
            return
        self._progress.update(self._task, visible=False)


class StandaloneSamplingProgress:
    ''

    def __init__(self, num_steps: int, description: str = "Generating"):
        ''
        self._num_steps = num_steps
        self._description = description
        self._progress: Progress | None = None
        self._task: TaskID | None = None

    def __enter__(self) -> "StandaloneSamplingProgress":
        ''
        self._progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40, style="blue"),
            TextColumn("{task.fields[info]}", style="cyan"),
            TimeElapsedColumn(),
            TextColumn("ETA:"),
            TimeRemainingColumn(compact=True),
        )
        self._progress.__enter__()
        self._task = self._progress.add_task(
            self._description,
            total=self._num_steps,
            info=f"step 0/{self._num_steps}",
        )
        return self

    def __exit__(self, *args) -> None:
        ''
        if self._progress is not None:
            self._progress.__exit__(*args)

    def advance_step(self) -> None:
        ''
        if self._progress is None or self._task is None:
            return
        self._progress.advance(self._task)
        completed = int(self._progress.tasks[self._task].completed)
        self._progress.update(self._task, info=f"step {completed}/{self._num_steps}")


class TrainingProgress:
    ''

    def __init__(self, enabled: bool, total_steps: int):
        ''
        self._enabled = enabled
        self._total_steps = total_steps
        self._train_task: TaskID | None = None

        if not enabled:
            self._progress = None
            return

                                                        
        self._progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            TextColumn("{task.fields[video]}", style="magenta"),
            BarColumn(bar_width=40, style="blue"),
            TextColumn("{task.fields[info]}", style="cyan"),
            TimeElapsedColumn(),
            TextColumn("ETA:"),
            TimeRemainingColumn(compact=True),
        )

    def __enter__(self) -> "TrainingProgress":
        ''
        if self._progress is not None:
            self._progress.__enter__()
            self._train_task = self._progress.add_task(
                "Training",
                total=self._total_steps,
                video=f"0/{self._total_steps}",
                info="Starting...",
            )
        return self

    def __exit__(self, *args) -> None:
        ''
        if self._progress is not None:
            self._progress.__exit__(*args)

    @property
    def enabled(self) -> bool:
        ''
        return self._enabled

    def update_training(
        self,
        *,
        loss: float,
        lr: float,
        step_time: float,
        advance: bool = True,
    ) -> None:
        ''
        if self._progress is None or self._train_task is None:
            return

        info = f"Loss: {loss:.4f} | LR: {lr:.2e} | {step_time:.2f}s/step"
        self._progress.update(
            self._train_task,
            advance=1 if advance else 0,
            info=info,
        )
                                           
        completed = int(self._progress.tasks[self._train_task].completed)
        self._progress.update(self._train_task, video=f"{completed}/{self._total_steps}")

    def start_sampling(self, num_prompts: int, num_steps: int) -> SamplingContext:
        ''
        if self._progress is None:
                                                              
            return SamplingContext(
                progress=None,
                task=None,
                num_prompts=num_prompts,
                num_steps=num_steps,
            )

        task = self._progress.add_task(
            "Sampling",
            total=num_steps,
            completed=0,
            video=f"0/{num_prompts}",
            info=f"step 0/{num_steps}",
        )

        return SamplingContext(
            progress=self._progress,
            task=task,
            num_prompts=num_prompts,
            num_steps=num_steps,
        )
