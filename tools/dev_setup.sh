#!/usr/bin/env bash
set -e

# 1. 初始化并更新 submodule 到 .gitmodules 指定 branch 的最新 commit
git submodule update --init --recursive --remote

# 2. 在每个 submodule 中切到 bench_patch 分支
git submodule foreach '
  echo ">>> Entering $name at $path"
  if git show-ref --verify --quiet refs/heads/bench_patch; then
    echo "    Switching to existing branch bench_patch"
    git switch bench_patch
  else
    echo "    Creating and tracking branch bench_patch from origin/bench_patch"
    git switch -c bench_patch --track origin/bench_patch
  fi
'
