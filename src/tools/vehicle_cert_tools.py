"""车型认证智能协同平台 —— Agent 工具集。

包含三个核心工具（对应方案中的"1 个知识库 + 2 条工作流"能力）：
1. knowledge_base_query    : 认证法规与历史案例库（RAG）语义检索
2. parameter_semantic_check: 参数语义核对工作流（研发侧 vs 检测侧）
3. progress_risk_scan      : 申报进度风险扫描工作流

设计原则（遵循产品方案"能不用大模型就不用"）：
- 单位换算、数值比对、风险分类等确定性任务用规则引擎实现；
- 语义同义词对齐用固定的同步词表 + 规则；RAG 语义检索使用知识库向量检索。
"""
import json
import re
import datetime
import logging
from typing import List, Dict, Any, Optional

from langchain.tools import tool
from coze_coding_utils.log.write_log import request_context
from coze_coding_utils.runtime_ctx.context import new_context

from storage.knowledge.kb import search_kb

_logger = logging.getLogger(__name__)

# ============================================================
# 1. 认证知识库问答（RAG）
# ============================================================
@tool
def knowledge_base_query(query: str) -> str:
    """在「认证法规与历史案例库」中进行语义检索。

    适用于查规则、查政策、查流程、查历史申报案例、问参数口径与法规依据等知识类问题。
    返回检索到的相关片段；若没有相关知识，返回空提示。
    """
    ctx = new_context(method="knowledge_base_query")
    _logger.info("knowledge_base_query invoked, query_length=%d", len(query or ""))
    try:
        chunks = search_kb(query, ctx=ctx, top_k=10, min_score=0.0)
    except Exception:  # noqa: BLE001
        _logger.exception("knowledge_base_query failed")
        return "知识库检索暂不可用，未生成事实结论；请检查工作区知识库配置。"
    if not chunks:
        return "知识库中未找到相关信息，请补充经授权的资料或咨询领域负责人。"
    lines = []
    for i, ch in enumerate(chunks, 1):
        content = ch.get("content", "").strip()
        score = ch.get("score", 0.0)
        lines.append(f"[片段{i}｜相关度{score:.2f}]\n{content}")
    return "\n\n".join(lines)


# ============================================================
# 2. 参数语义核对工作流
# ============================================================
# 参数语义同义词组（对应 05 号文档"跨系统参数同义词对照"）
_SYNONYM_GROUPS = [
    {"key": "等速续航", "unit": "km", "keywords": ["等速"], "exclude": []},
    {"key": "CLTC综合工况续航里程", "unit": "km",
     "keywords": ["综合续航", "cltc", "续航里程", "续航"], "exclude": ["等速"]},
    {"key": "动力电池容量", "unit": "kWh",
     "keywords": ["电池容量", "电量", "额定能量", "电池包", "电池系统能量"], "exclude": ["能量密度"]},
    {"key": "标称电压", "unit": "V", "keywords": ["电压"], "exclude": ["能量密度"]},
    {"key": "驱动电机峰值功率", "unit": "kW",
     "keywords": ["驱动电机峰值功率", "电机峰值功率", "峰值功率", "最大功率", "马力"], "exclude": []},
    {"key": "整备质量", "unit": "kg", "keywords": ["整备质量", "空车质量", "自重"], "exclude": []},
    {"key": "轴距", "unit": "mm", "keywords": ["轴距"], "exclude": []},
    {"key": "外廓尺寸", "unit": "mm", "keywords": ["外廓尺寸", "长宽高"], "exclude": []},
    {"key": "最高设计车速", "unit": "km/h", "keywords": ["最高车速", "极速", "最高设计车速"], "exclude": []},
    {"key": "电池系统能量密度", "unit": "Wh/kg", "keywords": ["能量密度"], "exclude": []},
    {"key": "充电时间", "unit": "min", "keywords": ["充电时间", "快充", "慢充"], "exclude": []},
    {"key": "轮胎规格", "unit": "", "keywords": ["轮胎规格", "轮胎"], "exclude": []},
]


def _clean(name: str) -> str:
    return (name or "").replace(" ", "").replace("　", "").strip()


def _canonicalize(name: str) -> Optional[Dict[str, str]]:
    n = _clean(name).lower()
    for g in _SYNONYM_GROUPS:
        if any(ex in n for ex in g["exclude"]):
            continue
        if any(kw in n for kw in g["keywords"]):
            return {"key": g["key"], "unit": g["unit"]}
    return None


def _parse_number(value: str) -> Optional[float]:
    m = re.match(r"^[+-]?(\d+(?:\.\d+)?)$", (value or "").strip())
    if m:
        return float(m.group(0))
    return None


def _unit_sig(unit: str) -> str:
    normalized = (unit or "").replace(" ", "").replace("\u3000", "").lower()
    aliases = {"度": "kwh", "千瓦时": "kwh", "瓦时": "wh", "分钟": "min", "小时": "h"}
    return aliases.get(normalized, re.sub(r"[^a-z]", "", normalized))


def _to_canonical(num: float, unit: str, canon_unit: str) -> Optional[float]:
    u = _unit_sig(unit)
    cu = _unit_sig(canon_unit)
    if not cu or u == cu:
        return num
    if cu == "kwh":
        if u == "wh":
            return num / 1000.0
    if cu == "min" and u in ("h", "hr"):
        return num * 60
    if cu == "v" and u == "kv":
        return num * 1000
    return None


def _split_records(text: str) -> List[str]:
    """把输入切成一条条参数记录：兼容换行、分号、顿号连接的整表粘贴。"""
    # 顿号"、"常见于"名称|值|单位、名称|值|单位"这类一行粘多条的粘贴场景
    normalized = (text or "").replace("\r", "")
    return [r.strip() for r in re.split(r"[\n;、]+", normalized) if r.strip()]


def _is_header(line: str) -> bool:
    low = _clean(line).lower()
    # 明显的表头特征：含"参数名称/参数名"，或同时含"数值/值"与"单位"（或"备注"）
    if "参数名称" in low or "参数名" in low:
        return True
    has_value_kw = ("数值" in low or "value" in low or _clean("值") and "值" in low)
    has_unit_kw = ("单位" in low or "unit" in low or "备注" in low or "remark" in low)
    if has_value_kw and has_unit_kw:
        return True
    return False


def _parse_param_lines(text: str) -> List[Dict[str, str]]:
    params = []
    for line in _split_records(text):
        line = line.replace("\ufeff", "").strip()
        if not line or _is_header(line):
            continue

        name = value = unit = ""
        # 情形A：带分隔符（| 或 , 或 ，），多数为 "名称,数值,单位[,备注]"
        delim = "|" if "|" in line else ("," if ("," in line or "，" in line) else None)
        if delim is not None:
            parts = [p.strip() for p in line.split(delim)]
            parts = [p for p in parts if p != ""]
            if len(parts) >= 2:
                name, value, unit = parts[0], parts[1], (parts[2] if len(parts) > 2 else "")
        # 情形B：无分隔符，尝试 "名称：值 单位"、"名称:1750kg" 或 "名称 1750 kg"
        if not name or not value:
            m = re.match(r"^\s*(?:[\d]+[\.\、])?\s*(.+?)[:：]\s*([\d.]+)\s*([a-zA-Z%×/]*)\s*$", line)
            if m:
                name, value, unit = m.group(1).strip(), m.group(2), m.group(3)
        if not name or not value:
            tokens = line.split()
            if tokens:
                name, value, unit = tokens[0], "", ""
                for tk in tokens[1:]:
                    if _parse_number(tk) is not None:
                        if not value:
                            value = tk
                        else:
                            unit = tk
                    elif not value:
                        m = re.match(r"^(\d+(?:\.\d+)?)([a-zA-Z%×/]+)$", tk)
                        if m:
                            value, unit = m.group(1), m.group(2)
                        else:
                            name = name + " " + tk
                    else:
                        unit = tk
        # 剥离编号前缀（如 "1."、"1、"、"1："）
        name = re.sub(r"^\s*[\d]+[\.\、\:]\s*", "", name or "")
        name = _clean(name)
        # 值带单位（如 "1750kg"、"510km"）
        if value and not unit:
            m = re.match(r"^(\d+(?:\.\d+)?)\s*([a-zA-Z%×/]+)$", value)
            if m:
                value, unit = m.group(1), m.group(2)
        value = (value or "").replace(",", "").replace("，", "").strip()
        unit = (unit or "").strip()
        if name and value:
            params.append({"name": name, "value": value, "unit": unit})
    return params


def _pair_params(rd: List[Dict], test: List[Dict]) -> List[Dict]:
    rd_used, test_used = set(), set()
    pairs = []
    for i, r in enumerate(rd):
        rc = _canonicalize(r["name"])
        if not rc:
            continue
        for j, t in enumerate(test):
            if j in test_used:
                continue
            tc = _canonicalize(t["name"])
            if tc and tc["key"] == rc["key"]:
                pairs.append(
                    {
                        "rd": r, "test": t,
                        "canon": rc["key"], "canon_unit": rc["unit"],
                    }
                )
                rd_used.add(i)
                test_used.add(j)
                break
    unmatched_rd = [r for i, r in enumerate(rd) if i not in rd_used]
    unmatched_test = [t for j, t in enumerate(test) if j not in test_used]
    return pairs, unmatched_rd, unmatched_test


def _name_same(a: str, b: str) -> bool:
    return _clean(a).lower() == _clean(b).lower()


def _classify(pair: Dict) -> Dict:
    r, t = pair["rd"], pair["test"]
    cv_num = _parse_number(r["value"])
    tv_num = _parse_number(t["value"])
    name_same = _name_same(r["name"], t["name"])
    unit_same = _unit_sig(r["unit"]) == _unit_sig(t["unit"])

    if cv_num is not None and tv_num is not None:
        rv = _to_canonical(cv_num, r["unit"], pair["canon_unit"])
        tv = _to_canonical(tv_num, t["unit"], pair["canon_unit"])
        if rv is None or tv is None:
            level, advice = "🟡", "单位缺少可靠换算依据，请补充必要参数后人工确认"
        elif abs(rv - tv) >= 1e-6:
            level, advice = "🔴", "数值实质不一致，不得直接申报，必须更正后重新核对"
        elif name_same and unit_same:
            level, advice = "🟢", "两侧语义一致"
        else:
            level, advice = "🟡", "仅参数命名/单位口径不同，数值等效，待人工确认"
    else:
        value_equal = (r["value"].strip() == t["value"].strip())
        if value_equal and name_same:
            level, advice = "🟢", "两侧语义一致"
        elif value_equal:
            level, advice = "🟡", "仅参数命名/单位口径不同，数值相同，待人工确认"
        else:
            level, advice = "🔴", "数值实质不一致，不得直接申报，必须更正后重新核对"

    return {
        "name": f"{r['name']} / {t['name']}",
        "rd_value": f"{r['value']} {r['unit']}".strip(),
        "test_value": f"{t['value']} {t['unit']}".strip(),
        "level": level,
        "advice": advice,
    }


@tool
def parameter_semantic_check(rd_params: str, test_params: str, vehicle_config: str = "") -> str:
    """对研发侧参数与检测侧参数进行语义级核对。

    能识别同义参数名（如标称电压/额定电压）与单位差异（如 kWh/Wh、min/h），
    逐项标注差异等级：🔴 严重差异（不得直接申报）/🟡 待确认/🟢 一致。
    入参：rd_params=研发侧参数文本，test_params=检测侧参数文本，vehicle_config=车型配置（可空）。
    """
    ctx = request_context.get() or new_context(method="parameter_semantic_check")
    _ = ctx
    rd = _parse_param_lines(rd_params)
    test = _parse_param_lines(test_params)
    if not rd or not test:
        return "参数数据不完整：请分别提供研发侧与检测侧参数（名称、数值、单位），格式支持逗号分隔或空格分隔。"

    pairs, unm_rd, unm_test = _pair_params(rd, test)
    if not pairs:
        return "未匹配到可比对的参数对，请检查两侧参数名称，或补充研发侧/检测侧完整参数表。"

    rows = []
    counts = {"🔴": 0, "🟡": 0, "🟢": 0}
    for p in pairs:
        c = _classify(p)
        rows.append(c)
        counts[c["level"]] += 1
    # 汇总排序：🔴 置顶
    order = {"🔴": 0, "🟡": 1, "🟢": 2}
    rows.sort(key=lambda x: order[x["level"]])

    lines = ["## 参数语义核对结果", f"车型配置：{vehicle_config or '未注明'}", ""]
    lines.append("| 参数名称（研发侧 / 检测侧） | 研发侧数值 | 检测侧数值 | 差异等级 | 处理建议 |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['name']} | {r['rd_value']} | {r['test_value']} | {r['level']} | {r['advice']} |")

    if unm_rd:
        lines.append("")
        lines.append("「仅研发侧存在（未比对）」：" + "、".join(f"{u['name']}({u['value']}{u['unit']})" for u in unm_rd))
    if unm_test:
        lines.append("「仅检测侧存在（未比对）」：" + "、".join(f"{u['name']}({u['value']}{u['unit']})" for u in unm_test))

    lines.append("")
    lines.append("## 汇总结论")
    lines.append(
        f"- 🔴 共 {counts['🔴']} 项、🟡 共 {counts['🟡']} 项、🟢 共 {counts['🟢']} 项"
    )
    if counts["🔴"]:
        lines.append("- ⚠️ 必须优先处理全部 🔴 项：红色项数值实质不一致，**未更正前不得直接申报**，请核对最新研发冻结表与检测报告后重新核对。")
    lines.append("- 本结果仅供参考，正式申报以研发盖章文件、检测机构正式报告及公告系统校验为准，请进行人工复核。")
    return "\n".join(lines)


# ============================================================
# 3. 申报进度风险扫描工作流
# ============================================================
_NEAR_DAYS = 7


def _parse_date(s: str) -> Optional[datetime.date]:
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
        try:
            return datetime.datetime.strptime(s.strip(), fmt).date()
        except (ValueError, AttributeError):
            continue
    return None


def _escalate(cat: str, role: str, task_id: str, name: str) -> str:
    if cat == "已延迟":
        return f"{task_id} {name} 已超计划时间，请 {role} 今天内推进，并升级至项目负责人协调。"
    if cat == "阻塞":
        return f"{task_id} {name} 存在阻塞，请 {role} 定位原因；外部阻塞超 2 天时由项目负责人协调责任方。"
    if cat == "阻塞+临期":
        return f"{task_id} {name} 阻塞且临近截止，请 {role} 当天联系责任方并同步项目负责人。"
    if cat == "临期":
        return f"{task_id} {name} 临近截止，请 {role} 按计划节点推进并及时催办。"
    if cat == "已延迟+阻塞":
        return f"{task_id} {name} 已逾期且阻塞，请 {role} 当天升级至项目负责人协调资源。"
    return f"{task_id} {name}，请 {role} 按计划推进。"


def _derive_critical_path(tasks: list) -> list:
    by_id = {}
    for task in tasks:
        task_id = str(task.get("id", "")).strip()
        if task_id and task_id not in by_id:
            by_id[task_id] = task

    task_ids = list(by_id)
    indegree = {task_id: 0 for task_id in task_ids}
    children = {task_id: [] for task_id in task_ids}
    for task_id, task in by_id.items():
        for dependency in task.get("dependencies", []) or []:
            dependency = str(dependency)
            if dependency in by_id:
                indegree[task_id] += 1
                children[dependency].append(task_id)

    queue = [task_id for task_id in task_ids if indegree[task_id] == 0]
    distance = {task_id: 1 for task_id in task_ids}
    predecessor = {}
    visited = 0
    while queue:
        current = queue.pop(0)
        visited += 1
        for child in children[current]:
            if distance[current] + 1 > distance[child]:
                distance[child] = distance[current] + 1
                predecessor[child] = current
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    if visited != len(task_ids) or not task_ids:
        return []

    last = max(task_ids, key=lambda task_id: distance[task_id])
    path = [last]
    while last in predecessor:
        last = predecessor[last]
        path.append(last)
    path.reverse()
    return path if len(path) > 1 else []


@tool
def progress_risk_scan(tasks_json: str, batch_deadline: str = "") -> str:
    """扫描申报任务清单 JSON，识别已延迟、临期、阻塞与正常任务并给出升级建议。

    入参：tasks_json=申报任务清单 JSON（含任务名称、责任岗位、计划完成时间、当前状态、前置依赖），
          batch_deadline=可选截止日期；也可在 JSON 中提供 submission_deadline 和 today 以便复现。
    """
    ctx = request_context.get() or new_context(method="progress_risk_scan")
    _ = ctx
    try:
        data = json.loads(tasks_json)
    except Exception as e:  # noqa: BLE001
        return f"任务清单 JSON 无法解析：{e}。请提供符合格式的 JSON 数据。"

    tasks = data.get("tasks", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    if isinstance(tasks, dict):
        tasks = [tasks]
    if isinstance(tasks, list) and tasks and isinstance(tasks[0], dict):
        pass
    else:
        tasks = []
    if not tasks:
        return "任务清单中没有有效任务，请提供包含 id、name、planned_date 和 status 的任务列表。"

    today = _parse_date(str(data.get("today", ""))) if isinstance(data, dict) else None
    today = today or datetime.date.today()
    deadline_value = data.get("submission_deadline", "") if isinstance(data, dict) else ""
    deadline = _parse_date(str(deadline_value)) or _parse_date(batch_deadline)
    if deadline is None:
        return "缺少有效截止日期，请在任务 JSON 中提供 submission_deadline，或传入 batch_deadline。"
    days_to_deadline = (deadline - today).days

    categories = {"已延迟": [], "阻塞+临期": [], "临期": [], "阻塞": [], "已延迟+阻塞": [], "正常": []}
    for t in tasks:
        tid = str(t.get("id", ""))
        name = str(t.get("name", ""))
        role = str(t.get("role", "未指定"))
        pdate = _parse_date(str(t.get("planned_date", "")))
        status = str(t.get("status", ""))
        block_reason = str(t.get("block_reason", "") or "")
        done = "已完成" in status or status.lower() in ("done", "completed", "完成")

        overdue = (not done) and pdate is not None and pdate < today
        near = (not done) and (not overdue) and pdate is not None and 0 <= (pdate - today).days <= _NEAR_DAYS
        blocked = ("阻塞" in status) or status.lower() in {"blocked", "blocked_by_dependency"} or bool(block_reason)

        if overdue and blocked:
            cat = "已延迟+阻塞"
        elif overdue:
            cat = "已延迟"
        elif blocked and near:
            cat = "阻塞+临期"
        elif near:
            cat = "临期"
        elif blocked:
            cat = "阻塞"
        else:
            cat = "正常"
        categories[cat].append(
            {"id": tid, "name": name, "role": role, "cat": cat,
             "planned": str(t.get("planned_date", "")), "status": status,
             "deps": t.get("dependencies", [])}
        )

    deadline_status = (f"截止已过 {abs(days_to_deadline)} 天" if days_to_deadline < 0
                       else f"距截止 {days_to_deadline} 天")
    lines = ["## 进度风险扫描结果",
             f"时间锚点：{today}；任务截止：{deadline}；{deadline_status}。", ""]
    critical_path = _derive_critical_path(tasks)

    def _risk_block(cat):
        if cat == "正常":
            lines.append(f"### 🟢 正常（{len(categories[cat])} 项）")
            if categories[cat]:
                names = "、".join(x["id"] for x in categories[cat])
                lines.append(f"按计划推进：{names}")
            lines.append("")
            return
        lines.append(f"### {('🔴' if '延迟' in cat else '🟠' if '阻塞' in cat else '🟡')} {cat}（{len(categories[cat])} 项）")
        for x in categories[cat]:
            lines.append(f"- **{x['id']}｜{x['name']}**（{x['role']}）：计划完成 {x['planned']}，状态「{x['status']}」")
            lines.append(f"  - 升级建议：{_escalate(cat, x['role'], x['id'], x['name'])}")
        lines.append("")

    # 输出顺序：已延迟 / 阻塞类 / 临期 / 正常
    for cat in ["已延迟", "已延迟+阻塞", "阻塞+临期", "阻塞", "临期", "正常"]:
        if categories[cat]:
            _risk_block(cat)

    if critical_path:
        task_names = {str(t.get("id", "")): str(t.get("name", "")) for t in tasks}
        path_labels = [f"{task_id} {task_names.get(task_id, '')}".strip() for task_id in critical_path]
        lines.append("### ⚡ 最长依赖链")
        lines.append(" → ".join(path_labels))
        lines.append("路径上的上游任务延期可能推动后续节点延迟。")
        lines.append("")

    # 今日行动清单（按 id 去重）
    todo = []  # [(tid, text)]
    added = set()

    def _push(tid, text):
        if tid not in added:
            added.add(tid)
            todo.append((tid, text))

    for x in categories["已延迟"]:
        _push(x["id"], f"🔴 {x['id']} {x['name']}：已逾期，{x['role']} 今天内协调完成，并升级至项目负责人。")
    for x in categories["已延迟+阻塞"]:
        _push(x["id"], f"🔴 {x['id']} {x['name']}：已逾期且阻塞，{x['role']} 当天升级项目负责人协调资源。")
    for x in categories["阻塞+临期"]:
        _push(x["id"], f"🟠 {x['id']} {x['name']}：阻塞+临期，{x['role']} 当天联系责任方并同步项目负责人。")
    for x in categories["阻塞"]:
        _push(x["id"], f"🟠 {x['id']} {x['name']}：阻塞，{x['role']} 当天定位原因并联动对应责任方。")
    for x in categories["临期"]:
        _push(x["id"], f"🟡 {x['id']} {x['name']}：临期，{x['role']} 于 {x['planned']} 前完成。")
    all_cat = (categories["已延迟"] + categories["已延迟+阻塞"] + categories["阻塞+临期"]
               + categories["阻塞"] + categories["临期"] + categories["正常"])
    by_id = {x["id"]: x for x in all_cat}
    for cid in critical_path:
        c = by_id.get(cid)
        if c:
            _push(cid, f"🔗 最长依赖链 {cid} {c['name']}：{c['role']} 按计划推进，避免延误下游任务。")

    lines.append("### ✅ 今日优先行动（{} 条）".format(len(todo)))
    for i, item in enumerate(todo, 1):
        lines.append(f"{i}. {item[1]}")
    lines.append("")
    lines.append("## 整体风险结论")
    deadline_summary = (f"截止已过 {abs(days_to_deadline)} 天" if days_to_deadline < 0
                        else f"距截止还有 {days_to_deadline} 天")
    lines.append(
        f"{deadline_summary}；共 {len(categories['已延迟']) + len(categories['已延迟+阻塞'])} 项延迟、"
        f"{len(categories['临期']) + len(categories['阻塞+临期'])} 项临期、"
        f"{len(categories['阻塞']) + len(categories['阻塞+临期']) + len(categories['已延迟+阻塞'])} 项阻塞。"
    )
    lines.append("优先处理顺序：已延迟 → 阻塞 → 临期。请根据项目流程协调责任人，并由领域人员复核扫描结果。")
    lines.append("*进度结论为辅助管理参考，不替代项目例会确认。*")
    return "\n".join(lines)
