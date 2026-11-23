#!/usr/bin/env bash
# Create the git tags of the weekly plan after checking that the working tree is clean.
#
#   bash scripts/tag_week.sh 1            # tags week1 on HEAD
#   bash scripts/tag_week.sh eval-freeze  # the evaluation freeze of Week 11 (make_figures.py --require-freeze looks for it)
#   bash scripts/tag_week.sh v1.0         # the final commit
#   git push origin --tags                # publish them (not done by this script)
set -euo pipefail
cd "$(dirname "$0")/.."
name=${1:?usage: tag_week.sh <1..11 | eval-freeze | v1.0>}
case "$name" in
  [0-9]|1[0-1]) tag="week$name" ;;
  eval-freeze|v1.0) tag="$name" ;;
  *) echo "unknown tag '$name' (use 1..11, eval-freeze or v1.0)"; exit 2 ;;
esac
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "working tree has uncommitted changes: commit them first so the tag marks exactly what you ran"; exit 1
fi
if git rev-parse -q --verify "refs/tags/$tag" >/dev/null; then echo "tag $tag already exists"; exit 1; fi
git tag -a "$tag" -m "$tag"
echo "tagged $(git rev-parse --short HEAD) as $tag (publish with: git push origin $tag)"
