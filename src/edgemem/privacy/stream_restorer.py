from __future__ import annotations

import re

from edgemem.privacy.redactor import RedactionSession

_PARTIAL_PREFIX_PATTERN = re.compile(
    r"""
    (?:
        __EDGEMEM_[A-Z]+_\d+_?
      | __EDGEMEM_[A-Z]+_?
      | __EDGEMEM_?
      | __EDGEME
      | __EDGEM
      | __EDGE
      | __EDG
      | __ED
      | __E
      | __
      | _
    )$
    """,
    re.VERBOSE,
)


class StreamRestorer:
    def __init__(self, session: RedactionSession) -> None:
        self.session = session
        self._buffer: str = ""
        self._placeholder_pattern = re.compile(r"__EDGEMEM_[A-Z]+_\d+__")
        self._partial_pattern = _PARTIAL_PREFIX_PATTERN

    def feed(self, chunk: str) -> str:
        if not self.session.has_redactions:
            return chunk
        if not chunk:
            return ""

        self._buffer += chunk

        if self._placeholder_pattern.search(self._buffer):
            mapping = self.session.placeholder_to_secret
            self._buffer = self._placeholder_pattern.sub(
                lambda m: mapping.get(m.group(0), m.group(0)),
                self._buffer,
            )

        match = self._partial_pattern.search(self._buffer)
        if match:
            suffix_len = min(len(match.group(0)), 64)
            emit = self._buffer[:-suffix_len]
            self._buffer = self._buffer[-suffix_len:]
            return emit

        emit = self._buffer
        self._buffer = ""
        return emit

    def flush(self) -> str:
        if not self.session.has_redactions or not self._buffer:
            return ""

        if self._placeholder_pattern.search(self._buffer):
            mapping = self.session.placeholder_to_secret
            self._buffer = self._placeholder_pattern.sub(
                lambda m: mapping.get(m.group(0), m.group(0)),
                self._buffer,
            )

        remaining = self._buffer
        self._buffer = ""
        return remaining
