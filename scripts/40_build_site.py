# -*- coding: utf-8 -*-
"""
生成个人项目介绍网站（sauomore.cn）。

设计取舍：
    · 单文件输出 —— 数据内嵌进 HTML，不依赖 fetch。这样本地双击就能看，
      部署时把 index.html 丢上去即可，不用配 CORS 或静态服务。
    · 零依赖 —— 不引 CDN、不用构建工具。弱网/无网也能看。
    · 不自动编造内容 —— 项目描述分两类：
        curated：我根据 README 与代码写的（需要作者校对）
        raw    ：直接取自 GitHub 的 description / 语言 / star
      未校对的一律标注来源，避免把推测当事实。

用法：
    python scripts/40_build_site.py
输出：
    site/index.html
"""

from __future__ import annotations

import html
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ---------------------------------------------------------------- 人工整理的项目信息
# ⚠️ 这些描述是我读 README 后写的，【需要作者校对】。
#    标 curated=True，网站上会注明「描述为整理所得」。
CURATED: dict[str, dict] = {
    "Cabinet_BNN": {
        "title_zh": "Cabinet-BNN · 从语义码生成权重",
        "title_en": "Cabinet-BNN · Generating Weights from a Semantic Code",
        "tagline_zh": "二值激活语言模型，每个 token 的权重由 64 位语义哈希码生成，而不是从全局权重张量里查表",
        "tagline_en": "A binary-activation LM whose per-token weights are generated from a 64-bit semantic hash code instead of looked up from a global tensor",
        "featured": True,
        "domain": "research",
        "highlights_zh": [
            "每个 token 一个 <b>128 位权重码</b>：<code>hash(64)</code> 取自 HSH-64，<code>param(64)</code> 承载该 token 的独有信息",
            "权重由共享低秩基矩阵合成：<code>W_t = Σ_j s_j(t)·(U_j V_jᵀ)</code>，不存在全局权重",
            "二值激活 <code>sign(x)</code> + STE，权重保持浮点",
            "MoE 稀疏 FFN：<b>码驱动路由</b>，零路由参数、负载天然均衡",
            "码即索引：码的汉明距离与权重余弦相似度相关系数 <b>−0.99</b>，无需训练即可传导几何",
            "如实记录负面结果：双向关系门实测<b>有害</b>",
        ],
        "highlights_en": [
            "A <b>128-bit weight code</b> per token: <code>hash(64)</code> from HSH-64, <code>param(64)</code> carrying token-specific information",
            "Weights are synthesised from shared low-rank bases: <code>W_t = Σ_j s_j(t)·(U_j V_jᵀ)</code> — no global weight tensor",
            "Binary activations <code>sign(x)</code> with STE; weights stay floating point",
            "MoE-style sparse FFN with <b>code-driven routing</b>: zero router parameters, structurally balanced load",
            "Code-as-index: Hamming distance correlates with weight cosine at <b>−0.99</b>, transferring geometry with no training",
            "Negative results documented: the bidirectional relation gate is <b>measured to hurt</b>",
        ],
        "stack": ["Python", "PyTorch"],
        "note_zh": "把 HSH-64 的码从「用来检索」推进到「用来生成参数」——这是整条线的下一站。",
        "note_en": "Takes the HSH-64 code from retrieval to parameter generation — the next stop on this line.",
    },
    "Cabinet_hsh64": {
        "title_zh": "HSH-64 · 64 位可学习语义哈希",
        "title_en": "HSH-64 · 64-bit Learnable Semantic Hashing",
        "tagline_zh": "把语义码从 20 位扩展到 52 位，单个 u64 存储、单次 popcnt 比较",
        "tagline_en": "Pushing the semantic code from 20 to 52 bits — one u64, one popcnt",
        "featured": True,
        "domain": "research",
        "highlights_zh": [
            "结构化 64 位编码：<code>feat(4) + sim(52) + abs(8)</code>，单 u64 存储",
            "三阶段端到端训练：连续预训练 → STE 离散精调 → 召回导向贪心后处理",
            "自适应多索引哈希（MIH），动态半径扩展",
            "非对称距离评分，缓解符号量化的信息损失",
            "纯 CPU 运行，训练与推理均无需 GPU",
            "有论文与 Zenodo DOI",
        ],
        "highlights_en": [
            "Structured 64-bit code: <code>feat(4) + sim(52) + abs(8)</code>, one u64",
            "Three-stage training: continuous pretrain → STE fine-tune → recall-oriented greedy post-processing",
            "Adaptive multi-index hashing with dynamic radius expansion",
            "Asymmetric distance scoring to reduce sign-quantisation loss",
            "CPU-only — neither training nor inference needs a GPU",
            "Paper and Zenodo DOI available",
        ],
        "stack": ["Rust", "Python", "TeX"],
        "note_zh": "本仓库是整条技术线的顶端：把前作的 20 位方案扩展到 64 位。",
        "note_en": "The top of this technical line: extends the earlier 20-bit scheme to 64 bits.",
    },
    "Cabinet": {
        "title_zh": "Cabinet · 面向 AI Agent 的语义记忆检索",
        "title_en": "Cabinet · Semantic Memory Retrieval for AI Agents",
        "tagline_zh": "用 20 位结构化整数替代 768 维浮点向量，纯 CPU、可解释、可增量更新",
        "tagline_en": "Replacing 768-dim float vectors with 20-bit structured integers — CPU-only, explainable, incremental",
        "featured": True,
        "domain": "infra",
        "highlights_zh": [
            "针对现有 RAG 的三个结构性缺陷：不可解释、更新成本高、依赖硬件",
            "HSH-20 编码：<code>feat(4) + sim(8) + abs(8)</code>，检索路径完全可审计（类别→簇→词）",
            "仅追加写入 + 后台 LSM 合并，新增文档无需重建索引",
            "索引体积缩小约 <b>1000 倍</b>，笔记本即可运行",
            "Rust 核心 + Python 绑定 + 独立 GUI",
        ],
        "highlights_en": [
            "Targets three structural flaws in current RAG: opacity, expensive updates, hardware dependence",
            "HSH-20 code: <code>feat(4) + sim(8) + abs(8)</code>; the retrieval path is fully auditable (category → cluster → word)",
            "Append-only writes plus background LSM merge — no index rebuild on insert",
            "Index size reduced by roughly <b>1000x</b>; runs on a laptop",
            "Rust core, Python bindings, standalone GUI",
        ],
        "stack": ["Rust", "Python", "Docker"],
        "note_zh": "这是 HSH-64 的前作，提出了层次语义哈希这套编码思路。",
        "note_en": "The predecessor of HSH-64, where the hierarchical semantic hashing idea was introduced.",
    },
    "hakimi-codex": {
        "title_zh": "Hakimi Codex · 多智能体编程助手",
        "title_en": "Hakimi Codex · Multi-agent Coding Assistant",
        "tagline_zh": "基于工具调用架构的 CLI 编程助手，能读写文件、执行命令、协作完成开发任务",
        "tagline_en": "A CLI coding assistant built on tool calling — reads, writes, executes, collaborates",
        "featured": True,
        "domain": "ai-tool",
        "highlights_zh": [
            "严格的工具调用协议（JSON over Markdown code blocks），保证稳定性",
            "可配置 LLM 后端：切换 API 端点、模型与认证信息",
            "流式响应，实时输出处理进度",
            "多工具集成：读写文件、目录列表、文件搜索、命令执行",
        ],
        "highlights_en": [
            "Strict tool-calling protocol (JSON over Markdown code blocks) for reliability",
            "Configurable LLM backend: swap endpoint, model and credentials",
            "Streaming responses with live progress",
            "Multi-tool integration: file I/O, directory listing, search, command execution",
        ],
        "stack": ["Python"],
        "note_zh": "把 LLM 的「说」变成「做」——这是 Agent 与聊天机器人的分界线。",
        "note_en": "Turns what an LLM says into what it does — the line between an agent and a chatbot.",
    },
    "MindMath_Studio": {
        "title_zh": "MindMath Studio · 科学计算平台",
        "title_en": "MindMath Studio · Scientific Computing Platform",
        "tagline_zh": "数据清洗 + 方程建模 + 数值求解 + 可视化，一站式 Web 平台",
        "tagline_en": "Data cleaning, equation modelling, numerical solving and visualisation in one web platform",
        "featured": False,
        "domain": "sci-tool",
        "highlights_zh": [
            "代数方程 / ODE / PDE 交互式求解（有限差分法）",
            "50+ 预设方程库，支持自定义方程与参数映射",
            "缺失值与异常值处理（IQR / Z-score）",
            "磨砂玻璃质感界面，支持实时曲线预览",
        ],
        "highlights_en": [
            "Interactive solving of algebraic equations, ODEs and PDEs (finite difference)",
            "50+ preset equation library, plus custom equations and parameter mapping",
            "Missing-value and outlier handling (IQR / Z-score)",
            "Frosted-glass UI with live curve preview",
        ],
        "stack": ["Python", "Flask", "JavaScript"],
        "note_zh": "技术栈从数据处理起步的地方，也是后来 Cabinet 那条线的源头。",
        "note_en": "Where the data-processing stack started — and the origin of the line that led to Cabinet.",
    },
    "Lixivium": {
        "title_zh": "Lixivium · 可视化数据清洗",
        "title_en": "Lixivium · Visual Data Cleaning",
        "tagline_zh": "拖拽连线配置清洗流程，自动生成 Pandas / SQL / PySpark 代码",
        "tagline_en": "Drag, connect, configure — auto-generate Pandas / SQL / PySpark code",
        "featured": False,
        "domain": "dev-tool",
        "highlights_zh": [
            "React Flow 画布，拖拽式节点编排",
            "一次编排，输出三种代码：Pandas / SQL（CTE）/ PySpark",
            "DuckDB 驱动的实时预览",
            "Monaco 编辑器 + 50 步撤销栈",
        ],
        "highlights_en": [
            "React Flow canvas with drag-and-drop node orchestration",
            "One graph, three outputs: Pandas / SQL (CTE) / PySpark",
            "DuckDB-powered real-time preview",
            "Monaco editor with a 50-step undo stack",
        ],
        "stack": ["Python", "TypeScript", "React"],
        "note_zh": "把重复的清洗脚本变成可视流程——写一次，出三种引擎的代码。",
        "note_en": "Turns repetitive cleaning scripts into a visual graph — write once, emit three engines.",
    },
    "Money-Burning-alarm-colok": {
        "title_zh": "Burn Alarm · 会烧钱的闹钟",
        "title_en": "Burn Alarm · The Alarm That Spends Your Money",
        "tagline_zh": "闹钟响后 60 秒内解不出三道心算题，就用你的 LLM API 额度烧钱",
        "tagline_en": "Fail three mental-math problems within 60 seconds and your own LLM API starts burning money",
        "featured": False,
        "domain": "app",
        "highlights_zh": [
            "闹钟响起锁定屏幕，60 秒倒计时",
            "未解出则调用<b>你自己的</b> LLM API，以最大 <code>max_tokens</code> 持续请求，实时显示累计花费",
            "多种防护：额度上限、失控保护、幂等计费",
            "中英双语界面，适配 Android 15 后台启动限制",
        ],
        "highlights_en": [
            "Full-screen lock-screen alarm with a 60-second countdown",
            "On failure it hammers <b>your own</b> LLM API at maxed-out <code>max_tokens</code>, showing the running cost",
            "Multiple guards: spend caps, runaway protection, idempotent billing",
            "Bilingual UI, adapted to the Android 15 background-activity limits",
        ],
        "stack": ["Kotlin", "Android"],
        "note_zh": "把一个行为经济学问题（起床拖延）直接翻译成经济损失。",
        "note_en": "Translates a behavioural problem (not getting up) directly into a financial loss.",
    },
    "-": {
        "title_zh": "简易数据建模软件",
        "title_en": "Simple Data Modelling Tool",
        "tagline_zh": "导入 CSV、自定义方程、快速建模 —— 这条技术线的起点",
        "tagline_en": "Import CSV, define equations, model quickly — the start of the line",
        "featured": False,
        "domain": "sci-tool",
        "highlights_zh": [
            "CSV 导入与方程自定义",
            "仓库名暂为 <code>-</code>，无 README",
        ],
        "highlights_en": [
            "CSV import with custom equations",
            "Repository is currently named <code>-</code> and has no README",
        ],
        "stack": ["Python"],
        "note_zh": "最早的仓库。名字还是个占位符，但它是后面所有工作的起点。",
        "note_en": "The earliest repository. Its name is still a placeholder, but everything after starts here.",
    },
}

DOMAINS = {
    "research": {"zh": "研究", "en": "Research", "color": "#8b5cf6"},
    "infra": {"zh": "AI 基础设施", "en": "AI Infrastructure", "color": "#3b82f6"},
    "ai-tool": {"zh": "AI 工具", "en": "AI Tooling", "color": "#06b6d4"},
    "sci-tool": {"zh": "科学计算", "en": "Scientific Computing", "color": "#10b981"},
    "dev-tool": {"zh": "开发工具", "en": "Developer Tooling", "color": "#f59e0b"},
    "app": {"zh": "应用", "en": "Application", "color": "#ef4444"},
}


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def build() -> str:
    repos = json.loads((ROOT / "site/data/repos.json").read_text(encoding="utf-8"))
    # 按创建时间倒序
    repos.sort(key=lambda r: r["created"], reverse=True)

    cards = []
    for r in repos:
        c = CURATED.get(r["name"], {})
        dom = DOMAINS.get(c.get("domain", ""), {"zh": "其他", "en": "Other",
                                                "color": "#64748b"})
        title_zh = c.get("title_zh", r["name"])
        title_en = c.get("title_en", r["name"])
        tag_zh = c.get("tagline_zh", r.get("description", ""))
        tag_en = c.get("tagline_en", r.get("description", ""))
        hl_zh = c.get("highlights_zh", [])
        hl_en = c.get("highlights_en", [])
        note_zh = c.get("note_zh", "")
        note_en = c.get("note_en", "")

        langbar = ""
        if r["languages"]:
            tot = sum(r["languages"].values()) or 1
            segs = []
            palette = ["#3b82f6", "#8b5cf6", "#06b6d4", "#10b981", "#f59e0b",
                       "#ef4444", "#64748b"]
            for i, (ln, by) in enumerate(sorted(r["languages"].items(),
                                                key=lambda x: -x[1])[:5]):
                pct = by / tot * 100
                segs.append(f'<span style="width:{pct:.1f}%;'
                            f'background:{palette[i % len(palette)]}" '
                            f'title="{esc(ln)} {pct:.0f}%"></span>')
            langbar = f'<div class="langbar">{"".join(segs)}</div>'

        hl_html_zh = "".join(f"<li>{h}</li>" for h in hl_zh)
        hl_html_en = "".join(f"<li>{h}</li>" for h in hl_en)

        cards.append(f"""
    <article class="card" data-domain="{esc(c.get('domain',''))}"
             data-name="{esc(r['name'])}" data-search="{esc((title_zh + ' ' + title_en + ' ' + r['name'] + ' ' + ' '.join(c.get('stack',[]))).lower())}">
      <div class="card-head">
        <span class="dot" style="background:{dom['color']}"></span>
        <span class="repo">{esc(r['name'])}</span>
        <span class="year">{esc(r['created'][:4])}</span>
      </div>
      <h3 class="zh-only">{title_zh}</h3>
      <h3 class="en-only">{esc(title_en)}</h3>
      <p class="tag zh-only">{esc(tag_zh)}</p>
      <p class="tag en-only">{esc(tag_en)}</p>
      {langbar}
      <ul class="hl zh-only">{hl_html_zh}</ul>
      <ul class="hl en-only">{hl_html_en}</ul>
      {f'<p class="note zh-only">{esc(note_zh)}</p>' if note_zh else ''}
      {f'<p class="note en-only">{esc(note_en)}</p>' if note_en else ''}
      <div class="card-foot">
        <span class="badge" style="border-color:{dom['color']};color:{dom['color']}">
          <span class="zh-only">{dom['zh']}</span><span class="en-only">{dom['en']}</span>
        </span>
        <span class="meta">{esc(' / '.join(c.get('stack', [r['language']])))}</span>
        {f'<span class="meta">★ {r["stars"]}</span>' if r['stars'] else ''}
        <a class="ghost-link" href="{esc(r['url'])}" target="_blank" rel="noopener">GitHub ↗</a>
      </div>
    </article>""")

    domains_nav = "".join(
        f'<button class="chip" data-filter="{k}">'
        f'<span class="zh-only">{v["zh"]}</span>'
        f'<span class="en-only">{v["en"]}</span></button>'
        for k, v in DOMAINS.items())

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sauomore · 项目索引</title>
<meta name="description" content="AI 语义检索 · 离散哈希 · 智能体工具 —— 个人项目索引">
<style>
  :root {{
    --bg:#0b0e14; --panel:#141925; --panel2:#1a2030; --line:#252c3d;
    --fg:#e8ecf4; --dim:#8b95ad; --accent:#4d8df6; --accent2:#7c5cf6;
  }}
  *{{box-sizing:border-box}}
  html,body{{margin:0;padding:0;background:var(--bg);color:var(--fg);
    font:16px/1.75 -apple-system,BlinkMacSystemFont,"Segoe UI",
    "Microsoft YaHei","PingFang SC",sans-serif;
    -webkit-font-smoothing:antialiased}}
  a{{color:var(--accent);text-decoration:none}}
  a:hover{{text-decoration:underline}}
  code{{background:var(--panel2);padding:1px 6px;border-radius:4px;
    font-size:.88em;font-family:ui-monospace,Consolas,monospace;
    color:#93c5fd;border:1px solid var(--line)}}
  .wrap{{max-width:1120px;margin:0 auto;padding:0 24px}}

  /* ---- 语言切换 ---- */
  body.lang-zh .en-only{{display:none !important}}
  body.lang-en .zh-only{{display:none !important}}
  .lang-sw{{position:fixed;top:18px;right:18px;z-index:50;display:flex;
    background:var(--panel);border:1px solid var(--line);border-radius:20px;
    overflow:hidden}}
  .lang-sw button{{background:none;border:none;color:var(--dim);
    padding:7px 16px;cursor:pointer;font:inherit;font-size:13px}}
  .lang-sw button.on{{background:var(--accent);color:#fff}}

  /* ---- 头部 ---- */
  header{{padding:96px 0 56px;position:relative;overflow:hidden}}
  header::before{{content:"";position:absolute;top:-180px;left:50%;
    transform:translateX(-50%);width:760px;height:440px;
    background:radial-gradient(ellipse,rgba(77,141,246,.16),transparent 70%);
    pointer-events:none}}
  h1{{font-size:46px;margin:0 0 8px;letter-spacing:-1px;font-weight:700}}
  .sub{{font-size:19px;color:var(--dim);margin:0 0 26px;max-width:640px}}
  .links{{display:flex;gap:12px;flex-wrap:wrap;margin-top:28px}}
  .links a{{display:inline-flex;align-items:center;gap:7px;
    background:var(--panel);border:1px solid var(--line);border-radius:9px;
    padding:9px 16px;color:var(--fg);font-size:14px}}
  .links a:hover{{border-color:var(--accent);text-decoration:none}}
  .links a.primary{{background:var(--accent);border-color:var(--accent)}}
  .stats{{display:flex;gap:34px;margin-top:40px;flex-wrap:wrap}}
  .stat b{{display:block;font-size:30px;font-weight:700;
    background:linear-gradient(135deg,var(--accent),var(--accent2));
    -webkit-background-clip:text;-webkit-text-fill-color:transparent}}
  .stat span{{font-size:13px;color:var(--dim)}}

  hr.sep{{border:0;border-top:1px solid var(--line);margin:0}}

  /* ---- 主线 ---- */
  .thread{{padding:56px 0}}
  .thread h2{{font-size:24px;margin:0 0 6px}}
  .thread p.lead{{color:var(--dim);margin:0 0 34px;max-width:720px}}
  .steps{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
    gap:16px}}
  .step{{background:var(--panel);border:1px solid var(--line);border-radius:12px;
    padding:20px;position:relative}}
  .step .n{{font-size:12px;color:var(--accent);font-family:ui-monospace,monospace;
    letter-spacing:1px;margin-bottom:8px}}
  .step h4{{margin:0 0 6px;font-size:16px}}
  .step p{{margin:0;font-size:13.5px;color:var(--dim);line-height:1.65}}

  /* ---- 项目 ---- */
  .projects{{padding:56px 0 80px}}
  .ph{{display:flex;align-items:baseline;gap:16px;flex-wrap:wrap;
    margin-bottom:22px}}
  .ph h2{{font-size:24px;margin:0}}
  .ph .count{{color:var(--dim);font-size:14px}}
  .toolbar{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:26px;
    align-items:center}}
  .chip{{background:var(--panel);border:1px solid var(--line);color:var(--dim);
    padding:7px 14px;border-radius:20px;cursor:pointer;font:inherit;
    font-size:13.5px;transition:.15s}}
  .chip:hover{{color:var(--fg);border-color:var(--dim)}}
  .chip.on{{background:var(--accent);border-color:var(--accent);color:#fff}}
  #search{{flex:1;min-width:190px;background:var(--panel);
    border:1px solid var(--line);border-radius:20px;padding:8px 16px;
    color:var(--fg);font:inherit;font-size:14px;outline:none}}
  #search:focus{{border-color:var(--accent)}}
  .grid{{display:grid;gap:20px;
    grid-template-columns:repeat(auto-fill,minmax(330px,1fr))}}
  .card{{background:var(--panel);border:1px solid var(--line);
    border-radius:14px;padding:22px;display:flex;flex-direction:column;
    transition:.18s;position:relative;overflow:hidden}}
  .card:hover{{border-color:#33405c;transform:translateY(-2px)}}
  .card.hide{{display:none}}
  .card-head{{display:flex;align-items:center;gap:9px;margin-bottom:14px}}
  .dot{{width:8px;height:8px;border-radius:50%;flex:0 0 8px}}
  .repo{{font-family:ui-monospace,Consolas,monospace;font-size:12.5px;
    color:var(--dim);flex:1;overflow:hidden;text-overflow:ellipsis;
    white-space:nowrap}}
  .year{{font-size:12px;color:var(--dim);font-family:ui-monospace,monospace}}
  .card h3{{margin:0 0 8px;font-size:18px;line-height:1.45}}
  .tag{{margin:0 0 16px;font-size:14px;color:var(--dim);line-height:1.65}}
  .langbar{{display:flex;height:3px;border-radius:2px;overflow:hidden;
    margin-bottom:16px;background:var(--panel2)}}
  .langbar span{{display:block;height:100%}}
  .hl{{margin:0 0 14px;padding-left:19px;font-size:13.5px;color:#c3cbdb;
    line-height:1.8}}
  .hl li{{margin-bottom:4px}}
  .note{{margin:auto 0 14px;padding-top:13px;border-top:1px solid var(--line);
    font-size:13px;color:var(--dim);font-style:italic;line-height:1.65}}
  .card-foot{{display:flex;align-items:center;gap:11px;margin-top:auto;
    padding-top:14px;border-top:1px solid var(--line);flex-wrap:wrap}}
  .badge{{font-size:11.5px;padding:2px 9px;border-radius:11px;
    border:1px solid;white-space:nowrap}}
  .meta{{font-size:12px;color:var(--dim);
    font-family:ui-monospace,Consolas,monospace}}
  .ghost-link{{margin-left:auto;font-size:13px}}

  /* ---- 页脚 ---- */
  footer{{border-top:1px solid var(--line);padding:40px 0 64px;
    color:var(--dim);font-size:13.5px}}
  .disclaim{{background:var(--panel);border:1px solid var(--line);
    border-radius:10px;padding:16px 20px;margin-bottom:26px;font-size:13px;
    line-height:1.8}}
  .disclaim b{{color:#e8b339}}
  @media(max-width:640px){{
    h1{{font-size:32px}} .sub{{font-size:16px}}
    header{{padding:76px 0 40px}} .stats{{gap:22px}}
    .stat b{{font-size:24px}}
  }}
</style>
</head>
<body class="lang-zh">

<div class="lang-sw">
  <button id="btnZh" class="on" onclick="setLang('zh')">中文</button>
  <button id="btnEn" onclick="setLang('en')">EN</button>
</div>

<header>
  <div class="wrap">
    <h1>Sauomore</h1>
    <p class="sub zh-only">
      做离散语义表示与 AI 智能体基础设施。<br>
      把 768 维浮点向量压成 32 位整数，再把整数的语义码用来<b>生成</b>模型参数。
    </p>
    <p class="sub en-only">
      Working on discrete semantic representations and AI agent infrastructure.<br>
      Compressing 768-dim float vectors into 32-bit integers, then using that code to
      <b>generate</b> model parameters.
    </p>
    <div class="links">
      <a class="primary" href="https://github.com/Sauomore" target="_blank" rel="noopener">GitHub ↗</a>
      <a href="https://github.com/Sauomore/Cabinet_hsh64" target="_blank" rel="noopener">
        <span class="zh-only">HSH-64 论文</span><span class="en-only">HSH-64 paper</span> ↗</a>
      <a href="https://doi.org/10.5281/zenodo.21721838" target="_blank" rel="noopener">DOI ↗</a>
    </div>
    <div class="stats">
      <div class="stat"><b>{len(repos)}</b>
        <span class="zh-only">公开项目</span><span class="en-only">Public projects</span></div>
      <div class="stat"><b>Rust · Python · Kotlin</b>
        <span class="zh-only">主要语言</span><span class="en-only">Main languages</span></div>
      <div class="stat"><b>2026</b>
        <span class="zh-only">全部项目集中在这一年</span>
        <span class="en-only">All projects within one year</span></div>
    </div>
  </div>
</header>

<hr class="sep">

<section class="thread">
  <div class="wrap">
    <h2 class="zh-only">一条主线：从数据处理到「用码生成权重」</h2>
    <h2 class="en-only">One thread: from data processing to generating weights from code</h2>
    <p class="lead zh-only">
      这些项目不是孤立的。它们是一条线上的五步 —— 每一步都在解决上一步暴露的问题。
    </p>
    <p class="lead en-only">
      These projects are not isolated. They are five steps along one line, each
      solving a problem the previous one exposed.
    </p>
    <div class="steps">
      <div class="step">
        <div class="n">STEP 1</div>
        <h4 class="zh-only">处理数据</h4><h4 class="en-only">Handle data</h4>
        <p class="zh-only">导入 CSV、清洗、建模。发现瓶颈在于「数据太大、计算太慢」。</p>
        <p class="en-only">Import CSV, clean, model. The bottleneck turns out to be scale.</p>
      </div>
      <div class="step">
        <div class="n">STEP 2</div>
        <h4 class="zh-only">向量化</h4><h4 class="en-only">Vectorise</h4>
        <p class="zh-only">用嵌入做语义检索。但 768 维浮点向量的存储与比较成本很高。</p>
        <p class="en-only">Use embeddings for semantic search — but 768-dim floats are expensive to store and compare.</p>
      </div>
      <div class="step">
        <div class="n">STEP 3</div>
        <h4 class="zh-only">压成整数</h4><h4 class="en-only">Compress to integers</h4>
        <p class="zh-only">HSH-20：把向量变成 20 位结构化整数。检索快了 1000 倍，但精度不够。</p>
        <p class="en-only">HSH-20: turn vectors into 20-bit structured integers. 1000x smaller, but not precise enough.</p>
      </div>
      <div class="step">
        <div class="n">STEP 4</div>
        <h4 class="zh-only">扩展到 64 位</h4><h4 class="en-only">Extend to 64 bits</h4>
        <p class="zh-only">HSH-64：语义码从 20 位扩展到 52 位，召回率达到可用水平。</p>
        <p class="en-only">HSH-64: expand the semantic code from 20 to 52 bits, reaching usable recall.</p>
      </div>
      <div class="step">
        <div class="n">STEP 5</div>
        <h4 class="zh-only">从码生成权重</h4><h4 class="en-only">Generate weights from code</h4>
        <p class="zh-only">Cabinet-BNN：不再用码去「找」权重，而是用码直接<b>生成</b>权重。参数可以逐 token 替换。</p>
        <p class="en-only">Cabinet-BNN: stop using the code to <b>find</b> weights — use it to <b>generate</b> them. Parameters become swappable per token.</p>
      </div>
    </div>
  </div>
</section>

<hr class="sep">

<section class="projects">
  <div class="wrap">
    <div class="ph">
      <h2 class="zh-only">项目</h2><h2 class="en-only">Projects</h2>
      <span class="count" id="count"></span>
    </div>
    <div class="toolbar">
      <button class="chip on" data-filter="all">
        <span class="zh-only">全部</span><span class="en-only">All</span></button>
      {domains_nav}
      <input id="search" placeholder="搜索项目 / 技术栈…">
    </div>
    <div class="grid" id="grid">{''.join(cards)}</div>
  </div>
</section>

<footer>
  <div class="wrap">
    <p style="margin:0">
      © 2026 Sauomore ·
      <a href="https://github.com/Sauomore" target="_blank" rel="noopener">github.com/Sauomore</a>
    </p>
  </div>
</footer>

<script>
// ---- 语言切换（记忆到 localStorage）----
function setLang(l) {{
  document.body.className = 'lang-' + l;
  document.getElementById('btnZh').classList.toggle('on', l === 'zh');
  document.getElementById('btnEn').classList.toggle('on', l === 'en');
  document.documentElement.lang = (l === 'zh') ? 'zh-CN' : 'en';
  try {{ localStorage.setItem('site-lang', l); }} catch (e) {{}}
  document.getElementById('search').placeholder =
    (l === 'zh') ? '搜索项目 / 技术栈…' : 'Search projects / stacks…';
  applyFilter();
}}
try {{
  const saved = localStorage.getItem('site-lang');
  if (saved) setLang(saved);
  else if (!/zh/i.test(navigator.language)) setLang('en');
}} catch (e) {{}}

// ---- 筛选 + 搜索 ----
let curFilter = 'all';
const cards = Array.from(document.querySelectorAll('.card'));

function applyFilter() {{
  const q = (document.getElementById('search').value || '').toLowerCase().trim();
  let n = 0;
  cards.forEach(c => {{
    const okDom = (curFilter === 'all' || c.dataset.domain === curFilter);
    const okQ = !q || c.dataset.search.includes(q) || c.dataset.name.toLowerCase().includes(q);
    const show = okDom && okQ;
    c.classList.toggle('hide', !show);
    if (show) n++;
  }});
  const zh = document.body.classList.contains('lang-zh');
  document.getElementById('count').textContent =
    zh ? (n + ' 个') : (n + ' shown');
}}

document.querySelectorAll('.chip').forEach(chip => {{
  chip.onclick = () => {{
    document.querySelectorAll('.chip').forEach(c => c.classList.remove('on'));
    chip.classList.add('on');
    curFilter = chip.dataset.filter;
    applyFilter();
  }};
}});
document.getElementById('search').addEventListener('input', applyFilter);
applyFilter();
</script>
</body>
</html>
"""


def main() -> int:
    out = ROOT / "site/index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    html_text = build()
    out.write_text(html_text, encoding="utf-8")

    repos = json.loads((ROOT / "site/data/repos.json").read_text(encoding="utf-8"))
    print("=" * 70)
    print("生成项目介绍网站")
    print("=" * 70)
    print(f"  输出 {out}")
    print(f"  大小 {len(html_text):,} 字符  {out.stat().st_size/1024:.1f} KB")
    print()
    print("  内容：")
    print(f"    · 头部（姓名 / 简介 / 外链 / 统计）")
    print(f"    · 技术主线（5 步：数据处理 → 向量化 → 压缩 → 扩展 → 生成权重）")
    print(f"    · {len(repos)} 个项目卡片（可按领域筛选 + 关键词搜索）")
    print(f"    · 中英双语切换（记忆选择）")
    print(f"    · 无外部依赖（零 CDN，离线可用）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
