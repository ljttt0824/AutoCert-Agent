"""知识库存储层：负责认证法规与历史案例库（RAG）的文档加载与语义检索。

说明：
- 使用 coze-coding-dev-sdk 的 KnowledgeClient 进行文本导入与语义搜索。
- 文档来自 assets/docs/ 下的 6 份 Markdown。
- ensure_loaded 具备幂等性：通过特征检索判断文档是否已入库，避免重复导入产生脏数据。
"""
import logging
import os
import glob
import re
import time

from coze_coding_dev_sdk import (
    KnowledgeClient,
    Config,
    KnowledgeDocument,
    DataSourceType,
    ChunkConfig,
)

logger = logging.getLogger(__name__)

# 知识库数据集名称
DATASET = "autocert_agent_demo_knowledge"

# 用于幂等判断的特征串（取自 06 号文档，避免与其它数据集冲突）
_MARKER_QUERY = "AutoCert-Agent synthetic knowledge corpus marker"

# 知识文档相对于项目根目录的路径
_DOC_DIR = os.path.join("assets", "docs")
_DOC_PATTERN = os.path.join(_DOC_DIR, "*.md")

# 模块级加载标志（进程内避免重复导入）
_loaded = False


def _read_docs() -> list:
    """读取 assets/docs 下的所有 .md 知识文档，返回 [(文件名, 文本), ...]。

    自动剥离开头的 AIGC YAML front-matter，并在内容前注入"来源文档"标记，
    便于检索结果溯源到具体文件。
    """
    workspace = os.getenv("COZE_WORKSPACE_PATH", ".")
    pattern = os.path.join(workspace, _DOC_PATTERN) if workspace != "." else _DOC_PATTERN
    docs = []
    for path in sorted(glob.glob(pattern)):
        fname = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        # 剥离 AIGC YAML front-matter（--- ... ---）
        if text.startswith("---"):
            end = text.find("\n---", 3)
            if end != -1:
                text = text[end + 4:]
        text = f"【来源文档：{fname}】\n{text.strip()}"
        docs.append((fname, text))
    return docs


def ensure_loaded(ctx=None) -> bool:
    """确保知识库文档已入库（幂等）。返回是否已就绪。"""
    global _loaded
    if _loaded:
        return True
    try:
        cfg = Config()
        client = KnowledgeClient(config=cfg, ctx=ctx)
        # 特征检索：若已存在则跳过导入（带 ctx 失败时回退默认凭据探测）
        probe_res = None
        try:
            probe_res = client.search(query=_MARKER_QUERY, top_k=1, min_score=0.0)
        except Exception:  # noqa: BLE001
            probe_res = None
        if not (probe_res and getattr(probe_res, "code", None) == 0 and getattr(probe_res, "chunks", None)):
            try:
                client2 = KnowledgeClient(config=cfg, ctx=None)
                probe_res = client2.search(query=_MARKER_QUERY, top_k=1, min_score=0.0)
            except Exception:  # noqa: BLE001
                probe_res = None
        if probe_res and getattr(probe_res, "code", None) == 0 and getattr(probe_res, "chunks", None):
            _loaded = True
            logger.info("知识库文档已存在，跳过导入")
            return True

        docs = _read_docs()
        if not docs:
            logger.warning("未在 %s 目录找到知识文档", _DOC_DIR)
            return False

        documents = [
            KnowledgeDocument(source=DataSourceType.TEXT, raw_data=text)
            for _, text in docs
        ]
        chunk_config = ChunkConfig(
            separator="\n",
            max_tokens=1000,
            remove_extra_spaces=True,
            remove_urls_emails=True,
        )
        resp = client.add_documents(
            documents=documents,
            table_name=DATASET,
            chunk_config=chunk_config,
        )
        if getattr(resp, "code", None) != 0:
            logger.error("知识库导入失败: %s", getattr(resp, "msg", resp))
            return False
        logger.info("知识库已导入 %d 份文档: %s", len(docs), [n for n, _ in docs])
        _loaded = True
        return True
    except Exception as e:  # noqa: BLE001
        logger.error("知识库 ensure_loaded 异常: %s", e)
        return False


# 本地知识语料（内存缓存），用于确定性兜底召回
_local_corpus = None


def _get_local_corpus() -> list:
    """返回 [(doc_name, block), ...]，block 已注入【来源文档：xx】前缀。"""
    global _local_corpus
    if _local_corpus is not None:
        return _local_corpus
    blocks = []
    for fname, text in _read_docs():
        # 按空行/标题切块
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if len(para) < 8:
                continue
            blocks.append((fname, para))
    _local_corpus = blocks
    return blocks


def _query_terms(query: str) -> set:
    """从查询中提取检索词（中文连续子串、字母数字、通用同义词展开）。"""
    q = query
    terms = set()
    # 字母数字
    for m in re.findall(r"[A-Za-z0-9]+", q):
        if len(m) >= 2:
            terms.add(m.lower())
    # 中文 2~4 字连续子串（仅过滤真·停用词，保留业务关键词如"变更/历史/来源/参数"）
    stop = {"什么", "怎么", "为什么", "请问", "一下", "哪些", "一个", "关于", "这个",
            "那个", "时候", "相关", "信息", "是不是", "可不可以", "能不能", "给你",
            "帮我", "我们", "你们", "这里", "那里", "就是", "这样", "那样"}
    zh = re.findall(r"[\u4e00-\u9fff]+", q)
    for word in zh:
        for n in (4, 3, 2):
            for i in range(0, len(word) - n + 1):
                sub = word[i:i + n]
                if sub not in stop:
                    terms.add(sub)
    for word in zh:
        if len(word) >= 2 and word not in stop:
            terms.add(word)
    return terms or {q.lower()}


def _local_search(query: str, k: int) -> list:
    """基于本地知识文档原文的关键词召回（确定性兜底）。"""
    terms = _query_terms(query)
    scored = []
    for fname, para in _get_local_corpus():
        low = para.lower()
        hit = sum(1 for t in terms if t in low)
        if hit == 0:
            continue
        pos = low.find(next((t for t in terms if t in low), ""))
        # 加分：含年份/日期（20XX年/月）更可能是"历史变更/来源"答案
        bonus = 0
        if re.search(r"20\d{2}\s*年", para):
            bonus += 2
        if re.search(r"(由|调整为|新增|改为|提升|不变|变更为)", para):
            bonus += 1
        if re.search(r"来源|第\d+批|P\d+", para):
            bonus += 1
        scored.append((hit + bonus, pos, para))
    scored.sort(key=lambda x: (-x[0], x[1]))
    out = []
    seen = set()
    for cnt, _pos, para in scored[:max(k * 3, 12)]:
        key = para[:40]
        if key in seen:
            continue
        seen.add(key)
        out.append({"content": para, "score": 0.5 + min(cnt * 0.06, 0.35), "doc_id": ""})
        if len(out) >= k:
            break
    return out


def search_kb(query: str, ctx=None, top_k: int = 10, min_score: float = 0.0) -> list:
    """混合检索：优先语义召回，语义结果不足时用本地原文关键词召回补齐。

    返回 chunk 列表 [{content, score, doc_id}, ...]，保证关键内容稳定命中。
    """
    ensure_loaded(ctx)
    cfg = Config()

    def _do(c: object) -> list:
        client = KnowledgeClient(config=cfg, ctx=c)
        resp = client.search(query=query, top_k=top_k, min_score=min_score)
        out = []
        if getattr(resp, "code", None) == 0 and getattr(resp, "chunks", None):
            for ch in resp.chunks:
                out.append(
                    {
                        "content": getattr(ch, "content", ""),
                        "score": getattr(ch, "score", 0.0),
                        "doc_id": getattr(ch, "doc_id", ""),
                    }
                )
        return out

    semantic = _do(ctx)
    if not semantic:
        # ctx 空结果时用默认凭据重试一次（索引/鉴权兜底）
        time.sleep(1.0)
        semantic = _do(None)

    # 用本地原文补齐，保证关键内容命中（先合并完整候选池，不提前截断，
    # 否则当平台语义召回数量 ≥ top_k 时，本地"历史变更表"块会被挤出候选池）
    local = _local_search(query, max(top_k * 2, 20))
    merged, seen = [], set()
    for it in semantic + local:
        key = it["content"][:50]
        if key in seen:
            continue
        seen.add(key)
        merged.append(it)

    # 变更/历史/来源类查询的定向重排：把同时含"年份+变更语义+来源页码"的块浮到最前，
    # 避免平台语义召回把"自测题问题/文档标题/车型锚点"误排到标准答案之前导致模型拒答。
    if re.search(r"变更|历史|调整|改版|更新过|来源|变化", query):
        ranked = []
        for idx, it in enumerate(merged):
            ch = 0
            if re.search(r"20\d{2}\s*年", it["content"]):
                ch += 2
            if re.search(r"(由\s*[\w/.]+\s*(?:调整|改|提升|降|变为)|调整为|新增长续航|新增|变更为)", it["content"]):
                ch += 2
            if re.search(r"(P\s*\d+|第\d+批|来源)", it["content"]):
                ch += 1
            ranked.append((idx, ch, it))
        ranked.sort(key=lambda x: (-x[1], x[0]))
        merged = [it for _, _, it in ranked][:top_k]
    else:
        merged = merged[:top_k]
    return merged
