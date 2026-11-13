"""
추론 과정 전체를 하나의 로그 파일로 추적하는 InferenceLogger.

사용법:
    logger = InferenceLogger(log_dir="/path/to/inference_debug", tag="20260211_121500_speaker0")
    logger.section("REQUEST")
    logger.kv("run", "results_pretrained")
    logger.kv("checkpoint", "best.pt")
    logger.log("추론 시작")
    ...
    logger.finish()  # 소요시간 요약 및 파일 기록 완료
"""
import os
import time
from datetime import datetime
from typing import Optional, Any


class InferenceLogger:
    """추론 1회의 모든 중간과정을 단일 로그 파일에 기록."""

    def __init__(self, log_dir: str, tag: str, compact: bool = True):
        """
        Args:
            log_dir: 로그 파일 저장 디렉토리 (inference_debug/)
            tag: 로그 파일 식별자 (예: "20260211_121500_speaker0")
            compact: True면 섹션명 + summary 한 줄만 (타입/이름 위주)
        """
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"{tag}_inference.log")
        self.tag = tag
        self.compact = compact
        self.start_time = time.time()
        self._section_times: list[tuple[str, float]] = []
        self._current_section: Optional[str] = None
        self._buffer: list[str] = []

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write_raw(f"INFERENCE LOG  {now}  tag={tag}")
        self._write_raw("")

    # ── public API ──────────────────────────────────────────

    def section(self, name: str):
        """섹션 시작. compact면 제목 한 줄만."""
        now = time.time()
        if self._current_section is not None and self._section_times and not self.compact:
            elapsed = now - self._section_times[-1][1]
            self._write_raw(f"  (elapsed {elapsed:.3f}s)")
        self._current_section = name
        self._section_times.append((name, now))
        if self.compact:
            self._write_raw(f"[{self._elapsed(now)}] --- {name} ---")
        else:
            bar = f"  {name}  ".center(72, "\u2500")
            self._write_raw(f"  [{self._elapsed(now)}] {bar}")

    def summary(self, msg: str):
        """compact용: 해당 섹션의 타입/이름 한 줄 (in: ... out: ...)."""
        self._write_raw(f"  {msg}")

    def log(self, message: str):
        if self.compact:
            return
        self._write_raw(f"  [{self._elapsed()}]   {message}")

    def kv(self, key: str, value: Any, width: int = 28):
        if self.compact:
            return
        self._write_raw(f"  [{self._elapsed()}]   {key:<{width}}: {value}")

    def log_dict(self, title: str, d: dict):
        if self.compact:
            return
        self.log(f"{title}:")
        for k, v in d.items():
            self.kv(f"  {k}", v)

    def warning(self, message: str):
        self._write_raw(f"  [WARNING] {message}")

    def error(self, message: str):
        self._write_raw(f"  [ERROR] {message}")

    def separator(self):
        if self.compact:
            return
        self._write_raw(f"  {'- ' * 36}")

    def finish(self):
        now = time.time()
        total = now - self.start_time
        self._write_raw("")
        self._write_raw(f"SUMMARY  total={total:.3f}s")
        self._flush_to_file()
        print(f"[InferenceLogger] Log saved: {self.log_path}")

    # ── internal ────────────────────────────────────────────

    def _elapsed(self, now: Optional[float] = None) -> str:
        if now is None:
            now = time.time()
        return f"{now - self.start_time:8.3f}s"

    def _write_raw(self, line: str):
        """버퍼에 추가 + stdout 출력."""
        self._buffer.append(line)
        print(line)

    def _flush_to_file(self):
        """버퍼 내용을 로그 파일에 기록."""
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self._buffer) + "\n")
