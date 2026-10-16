"""车型认证智能协同平台 —— 主 Agent。

将 Coze「多 Agent 模式」（1 主 Bot 路由 + 4 子 Agent + 1 知识库 + 2 条工作流）落地为
单一 Agent：通过系统提示词完成意图路由与 4 类专业助手角色的切换，通过工具承载
知识库 RAG 检索、参数语义核对、申报进度风险扫描三类能力。
"""
import os
import json
from typing import Annotated

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from coze_coding_utils.runtime_ctx.context import default_headers

from storage.memory.memory_saver import get_memory_saver
from tools.vehicle_cert_tools import (
    knowledge_base_query,
    parameter_semantic_check,
    progress_risk_scan,
)

LLM_CONFIG = "config/agent_llm_config.json"

# 默认保留最近 20 轮对话（40 条消息）
MAX_MESSAGES = 40


def _windowed_messages(old, new):
    """滑动窗口：只保留最近 MAX_MESSAGES 条消息"""
    return add_messages(old, new)[-MAX_MESSAGES:]  # type: ignore


class AgentState(MessagesState):
    messages: Annotated[list[AnyMessage], _windowed_messages]


def build_agent(ctx=None):
    workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
    config_path = os.path.join(workspace_path, LLM_CONFIG)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    api_key = os.getenv("COZE_WORKLOAD_IDENTITY_API_KEY")
    base_url = os.getenv("COZE_INTEGRATION_MODEL_BASE_URL")

    llm = ChatOpenAI(
        model=cfg["config"].get("model"),
        api_key=api_key,
        base_url=base_url,
        temperature=cfg["config"].get("temperature", 0.4),
        streaming=True,
        timeout=cfg["config"].get("timeout", 600),
        extra_body={
            "thinking": {
                "type": cfg["config"].get("thinking", "disabled")
            }
        },
        default_headers=default_headers(ctx) if ctx else {},
    )

    tools = [
        knowledge_base_query,
        parameter_semantic_check,
        progress_risk_scan,
    ]

    return create_agent(
        model=llm,
        system_prompt=cfg.get("sp"),
        tools=tools,
        checkpointer=get_memory_saver(),
        state_schema=AgentState,
    )