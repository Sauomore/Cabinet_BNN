# -*- coding: utf-8 -*-
"""
把网站打包成可直接上传的静态文件。

产出：
    dist/sauomore-site.zip          上传这个到服务器
    dist/sauomore-site/             解压后的内容（与 zip 一致，便于检查）

设计取舍：
    · 部署目录只放【公开】内容。repos.json 是构建用的中间数据，
      虽然只含公开 API 信息，但没必要暴露在网站上，所以放到 build/ 里。
    · 打包前做四项校验，任何一项失败就中止 —— 避免把坏包传上去。
    · 同时生成部署说明，包含 nginx / Apache / GitHub Pages 三种配置。

用法：
    python scripts/41_package_site.py
"""

from __future__ import annotations

import html
import io
import json
import re
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SITE = ROOT / "site"
DIST = ROOT / "dist"
STAGE = DIST / "sauomore-site"
ZIP = DIST / "sauomore-site.zip"


def validate(index_html: str) -> tuple[bool, list[str]]:
    """打包前的硬性校验。返回 (是否通过, 消息列表)。"""
    msgs: list[str] = []
    ok = True

    # 1. 外链完整性：所有 href 必须可解析
    hrefs = set(re.findall(r'href="([^"]+)"', index_html))
    bad = [h for h in hrefs
           if not (h.startswith("http://") or h.startswith("https://")
                   or h.startswith("#"))]
    if bad:
        ok = False
        msgs.append(f"❌ 存在无法解析的链接: {bad}")
    else:
        msgs.append(f"✅ 链接全部为绝对 URL（{len(hrefs)} 个）")

    # 2. 自包含：不能有外部资源引用（img/script/link 指向外部）
    ext = re.findall(r'(?:src|href)="(https?://(?!github\.com|doi\.org)[^"]+)"',
                     index_html)
    ext = [e for e in ext if not e.startswith("https://github.com")
           and not e.startswith("https://doi.org")]
    if ext:
        ok = False
        msgs.append(f"❌ 引用了外部资源（离线会挂）: {ext}")
    else:
        msgs.append("✅ 无外部资源依赖（离线可用）")

    # 3. 本地文件引用必须存在
    local_refs = re.findall(r'(?:src|href)="(?!https?://|#|mailto:)([^"]+)"',
                            index_html)
    missing = [r for r in set(local_refs) if not (SITE / r).exists()]
    if missing:
        ok = False
        msgs.append(f"❌ 引用的本地文件缺失: {missing}")
    elif local_refs:
        msgs.append(f"✅ 本地引用 {len(set(local_refs))} 个均存在")
    else:
        msgs.append("✅ 无本地文件引用（单文件，天然自包含）")

    # 4. 基本结构
    for tag, name in [("<html", "html 标签"), ("<title", "title"),
                      ("</body>", "body 闭合"), ("</html>", "html 闭合")]:
        if tag not in index_html:
            ok = False
            msgs.append(f"❌ 缺少 {name}")
    if ok:
        msgs.append("✅ HTML 结构完整")

    # 5. 项目数量
    n = len(re.findall(r'class="card"', index_html))
    if n < 1:
        ok = False
        msgs.append("❌ 没有项目卡片")
    else:
        msgs.append(f"✅ 项目卡片 {n} 个")

    return ok, msgs


def write_readme(stage: Path, card_count: int, size_kb: float) -> None:
    txt = f"""# sauomore.cn 部署说明

打包时间：{datetime.now().strftime("%Y-%m-%d %H:%M")}
项目数量：{card_count}
文件大小：{size_kb:.1f} KB

## 包含什么

    index.html      整个网站（单文件，HTML + CSS + JS 全内联）

**就这一个文件。** 没有图片、没有字体、没有 JS 库、没有 CDN ——
因为页面里所有的样式、脚本和数据都写进了 index.html。
所以它离线可用，也不会因为某个 CDN 挂掉而白屏。

## 三种部署方式

### 1. Nginx

把 index.html 放到网站根目录：

    server {{
        listen 80;
        server_name sauomore.cn www.sauomore.cn;
        root /var/www/sauomore;
        index index.html;
        location / {{
            try_files $uri $uri/ /index.html;
        }}
    }}

上传：

    scp index.html user@your-server:/var/www/sauomore/

### 2. Apache

放到 DocumentRoot 下即可。若用了 HTTPS，建议加一条重定向：

    RewriteEngine On
    RewriteCond %{{HTTPS}} !=on
    RewriteRule ^ https://%{{SERVER_NAME}}%{{REQUEST_URI}} [R=301,L]

### 3. GitHub Pages

新建仓库 `Sauomore.github.io`，把 index.html 推到根目录，
然后在仓库 Settings → Pages 里选 main 分支。

若要用自己的域名 sauomore.cn，在仓库根目录加一个 `CNAME` 文件，
内容一行：

    sauomore.cn

再到域名服务商处添加解析：
    A 记录     @    ->  185.199.108.153 / .109.154 / .110.153 / .111.153
    CNAME      www  ->  sauomoore.github.io

## 建议

· 开启 HTTPS（Let's Encrypt 免费）
· 页面无外部依赖，不需要配缓存策略，但可以给 index.html 加上
  `Cache-Control: public, max-age=300` 让 CDN 友好一些

## 本地查看

直接双击 index.html 即可，不需要起服务器。
"""
    (stage / "README-部署.md").write_text(txt, encoding="utf-8")


def main() -> int:
    src = SITE / "index.html"
    if not src.exists():
        print(f"找不到 {src}，请先运行 scripts/40_build_site.py")
        return 1

    index_html = src.read_text(encoding="utf-8")

    print("=" * 70)
    print("打包静态站点")
    print("=" * 70)
    print(f"  源 {src}  ({len(index_html):,} 字符)")

    # ---- 校验 ----
    print("\n  打包前校验:")
    ok, msgs = validate(index_html)
    for m in msgs:
        print(f"    {m}")
    if not ok:
        print("\n  ✗ 校验未通过，中止打包")
        return 1

    # ---- 准备目录 ----
    if DIST.exists():
        shutil.rmtree(DIST)
    STAGE.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src, STAGE / "index.html")

    # 构建数据放到 build/（不随站点发布）
    build_dir = DIST / "build-reference"
    build_dir.mkdir(parents=True, exist_ok=True)
    for f in ("data/repos.json",):
        p = SITE / f
        if p.exists():
            shutil.copy2(p, build_dir / Path(f).name)

    card_count = len(re.findall(r'class="card"', index_html))
    size_kb = (STAGE / "index.html").stat().st_size / 1024
    write_readme(STAGE, card_count, size_kb)

    # ---- 打 zip ----
    files = sorted(STAGE.rglob("*"))
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in files:
            if f.is_file():
                z.write(f, f.relative_to(STAGE))

    print("\n  产物:")
    for f in files:
        if f.is_file():
            rel = f.relative_to(STAGE)
            print(f"    {str(rel):<24} {f.stat().st_size:>8,} bytes")
    print(f"    {'(zip)':<24} {ZIP.stat().st_size:>8,} bytes")
    print(f"\n  ✅ {ZIP}")
    print(f"     {card_count} 个项目 · 单文件 · 零依赖")
    print(f"\n  上传: 解压后把 index.html 放到网站根目录即可")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
