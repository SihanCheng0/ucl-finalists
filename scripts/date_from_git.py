"""Give each tracked file under the given paths the time of the last commit that changed it.

A fresh checkout stamps every file with the time of the checkout, but UCL Lab dates its data by file times (when the
data was built, when a cached UEFA copy was fetched). The nightly workflow runs this before anything reads the data:

    python scripts/date_from_git.py data out
"""
from __future__ import annotations

import os
import subprocess
import sys


def git(*args: str) -> str:
    return subprocess.run(["git", "-c", "core.quotePath=false", *args], capture_output=True, text=True,
                          check=True).stdout


def main(paths: list[str]) -> int:
    edited = set(git("diff", "--name-only", "HEAD", "--", *paths).splitlines())  # changed since: their times are right
    when: int | None = None
    seen: set[str] = set()
    dated = 0
    for line in git("log", "--format=@%ct", "--name-only", "--", *paths).splitlines():
        if line.startswith("@"):
            when = int(line[1:])
        elif line and when is not None and line not in seen:  # newest commit first, so the first time counts
            seen.add(line)
            if line not in edited and os.path.isfile(line):
                os.utime(line, (when, when))
                dated += 1
    print(f"dated {dated:,} files by their last commits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["."]))
