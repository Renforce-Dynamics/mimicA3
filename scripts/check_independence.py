"""Reject runtime coupling to sibling source trees or Git/path dependencies."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_SUFFIXES = {".py", ".toml", ".yaml", ".yml"}
FORBIDDEN = (
    re.compile(r"metabot-workspace/"),
    re.compile(r"from\s+(active_adaptation|any4hdmi|coordina)\b"),
    re.compile(r"import\s+(active_adaptation|any4hdmi|coordina)\b"),
    re.compile(r"git\+https?://"),
)


def main() -> None:
    violations: list[str] = []
    for base in (ROOT / "src", ROOT / "scripts", ROOT / "tests"):
        for path in base.rglob("*"):
            if path.suffix not in SCAN_SUFFIXES or "__pycache__" in path.parts:
                continue
            if path == Path(__file__).resolve():
                continue
            text = path.read_text(encoding="utf-8")
            for pattern in FORBIDDEN:
                if pattern.search(text):
                    violations.append(f"{path.relative_to(ROOT)}: {pattern.pattern}")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    if re.search(r"\b(path|git)\s*=", pyproject):
        violations.append("pyproject.toml: path/git dependency")
    if violations:
        raise SystemExit("repository independence check failed:\n" + "\n".join(violations))
    print("repository independence check passed")


if __name__ == "__main__":
    main()
