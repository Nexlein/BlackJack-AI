#!/bin/sh
# Enforces Conventional Commits on the first line: type(scope)!: subject. Plain sh, no dependency.
first=$(head -n1 "$1")
case "$first" in
  "Merge "* | "Revert \""* | "fixup! "* | "squash! "*) exit 0 ;;
esac
if ! printf '%s\n' "$first" | grep -qE '^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\([a-z0-9._-]+\))?!?: .+'; then
  echo "commit-msg: first line must follow Conventional Commits, e.g. \"fix(checks): handle empty CV\"" >&2
  echo "  got: $first" >&2
  exit 1
fi
