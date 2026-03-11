from __future__ import annotations

import sys
from typing import TextIO

from tqdm import tqdm


class ProgressBar:
    def __init__(
        self,
        total: int | None,
        description: str = "清洗进度",
        stream: TextIO | None = None,
        width: int = 30,
    ) -> None:
        self.total = total
        self.description = description
        self.stream = stream or sys.stderr
        self._progress = tqdm(
            total=total,
            desc=description,
            file=self.stream,
            ncols=width + 40,
            dynamic_ncols=True,
            unit="row",
            disable=False,
        )

    def advance(self, amount: int) -> None:
        if amount <= 0:
            return
        self._progress.update(amount)

    def set_summary(
        self,
        total_before: int,
        total_removed: int,
        total_after: int,
        removed_blank_rows: int = 0,
        removed_symbol_rows: int = 0,
        removed_emoji_rows: int = 0,
        removed_garbled_rows: int = 0,
    ) -> None:
        self._progress.set_postfix(
            {
                "总数": total_before,
                "删除": total_removed,
                "保留": total_after,
                "空行": removed_blank_rows,
                "符号": removed_symbol_rows,
                "表情": removed_emoji_rows,
                "乱码": removed_garbled_rows,
            }
        )

    def close(self) -> None:
        self._progress.close()


def run_stage(
    description: str,
    stream: TextIO | None = None,
    total: int | None = 1,
) -> None:
    target_stream = stream or sys.stderr
    stage = tqdm(
        total=total,
        desc=description,
        file=target_stream,
        dynamic_ncols=True,
        unit="step",
        leave=False,
    )
    stage.update(total or 1)
    stage.close()
