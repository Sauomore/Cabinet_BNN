# -*- coding: utf-8 -*-
"""
抓取 GitHub 仓库信息，为综合介绍网站准备内容。

用法：
    python scripts/39_fetch_repos.py
输出：
    site/data/repos.json
"""

from __future__ import annotations

import base64
import io
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

USER = "Sauomore"
# 原本排除 Cabinet_BNN，后按用户要求纳入（该仓库是公开的）。
# 留空表示全部抓取。
EXCLUDE: set[str] = set()
API = "https://api.github.com"
HDR = {"User-Agent": "Mozilla/5.0", "Accept": "application/vnd.github+json"}


def get(url: str):
    req = urllib.request.Request(url, headers=HDR)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 403:
                time.sleep(5)
                continue
            return None
        except Exception:
            time.sleep(3)
    return None


def get_text(url: str) -> str | None:
    req = urllib.request.Request(url, headers=HDR)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:
        return None


def main() -> int:
    print("=" * 74)
    print(f"抓取 github.com/{USER} 的仓库")
    print("=" * 74)

    repos = get(f"{API}/users/{USER}/repos?per_page=100&sort=updated")
    if not repos:
        print("  拉取仓库列表失败（可能是网络限流）")
        return 1

    out = []
    for p in repos:
        name = p["name"]
        if name in EXCLUDE:
            print(f"  [跳过] {name}（按要求排除）")
            continue
        print(f"  [{name}] 读取详情 …")

        # README
        readme = None
        for fn in ("README.md", "readme.md", "README.MD", "README.rst",
                   "README.txt", "README"):
            t = get_text(f"{API}/repos/{USER}/{name}/contents/{fn}")
            if t:
                try:
                    readme = json.loads(t).get("content", "")
                    readme = base64.b64decode(readme).decode("utf-8", "replace")
                except Exception:
                    readme = t
                if readme:
                    break

        # 目录结构（顶层）
        tree = get(f"{API}/repos/{USER}/{name}/contents/")
        files = []
        if isinstance(tree, list):
            files = [{"name": x["name"], "type": x["type"],
                      "size": x.get("size", 0)} for x in tree]

        # 语言构成
        langs = get(f"{API}/repos/{USER}/{name}/languages") or {}

        # 提交数（用 commits 接口的 per_page=1 读 Link header 太麻烦，
        # 改为取最近提交做展示）
        commits = get(f"{API}/repos/{USER}/{name}/commits?per_page=5")
        recent = []
        if isinstance(commits, list):
            for c in commits:
                recent.append({
                    "sha": c["sha"][:7],
                    "date": c["commit"]["author"]["date"][:10],
                    "msg": c["commit"]["message"].splitlines()[0][:100],
                })

        out.append({
            "name": name,
            "description": p.get("description") or "",
            "url": p["html_url"],
            "homepage": p.get("homepage") or "",
            "language": p.get("language") or "",
            "languages": langs,
            "stars": p["stargazers_count"],
            "forks": p["forks_count"],
            "watchers": p.get("watchers_count", 0),
            "open_issues": p.get("open_issues_count", 0),
            "created": p["created_at"][:10],
            "updated": p["updated_at"][:10],
            "pushed": p["pushed_at"][:10],
            "size_kb": p.get("size", 0),
            "topics": p.get("topics", []),
            "license": (p.get("license") or {}).get("spdx_id", ""),
            "files": files,
            "recent_commits": recent,
            "readme": readme or "",
        })
        time.sleep(0.5)          # 温和限流

    outdir = ROOT / "site/data"
    outdir.mkdir(parents=True, exist_ok=True)
    dst = outdir / "repos.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    print("\n" + "=" * 74)
    print(f"完成，共 {len(out)} 个仓库 -> {dst}")
    print("=" * 74)
    print(f"  {'名称':<28}{'语言':<12}{'★':>4}{'文件数':>7}{'README':>8}{'创建':>12}")
    print("  " + "-" * 68)
    for r in out:
        print(f"  {r['name']:<28}{r['language']:<12}{r['stars']:>4}"
              f"{len(r['files']):>7}{len(r['readme']):>8}{r['created']:>12}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
