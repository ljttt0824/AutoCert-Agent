# AutoCert-Agent：车型认证智能协同多 Agent 系统

AutoCert-Agent 是一个基于 Coze Coding、LangChain 和 LangGraph 实现的车型认证准备助手。它把知识检索、参数语义核对和项目进度风险扫描接入同一对话入口，并支持按用户提供的信息起草待审核材料。

项目代码使用一个 LangGraph Agent，通过提示词路由和工具调用承担多个专业角色；它不是由多个独立 Agent 实例组成的分布式系统。仓库中的知识文档和任务清单均为合成演示材料，不代表真实车型、企业流程或官方法规。

## 能力范围

| 能力 | 实现方式 | 输出 |
| --- | --- | --- |
| 认证知识问答 | Coze 知识库语义检索 + 本地关键词兜底 | 带文件来源的候选回答；无证据时拒答 |
| 参数语义核对 | 参数同义词归一、部分明确单位换算、差异分级 | 严重差异、待确认、一致三类结果 |
| 进度风险扫描 | 截止日期、计划日期、任务状态和依赖关系规则计算 | 延迟、阻塞、临期分类及最长依赖链 |
| 材料初稿 | Agent 提示词生成，无独立材料生成工具 | 缺失值使用占位符的待审核草稿 |

## 架构

```text
用户请求
   ↓
LangGraph Agent（意图识别、追问、工具结果整理）
   ├─ knowledge_base_query → Coze KnowledgeClient + 本地语料检索
   ├─ parameter_semantic_check → 参数名匹配 + 可验证的单位换算
   └─ progress_risk_scan → 日期风险分类 + 任务依赖链分析
   ↓
带来源、差异等级或风险项的候选答复
```

短期会话检查点优先使用环境配置的 PostgreSQL，缺少数据库配置时回退到进程内存。FastAPI 运行入口提供 Coze 工作区使用的 HTTP 接口；该服务默认监听本机地址，不带访问认证，不应直接暴露到公网。

## 仓库结构

```text
config/agent_llm_config.json   模型参数与脱敏后的系统提示词
src/                           Agent、工具、RAG、存储和 FastAPI 入口
scripts/                       Coze 工作区安装、启动和环境变量脚本
assets/docs/                   六份合成知识库演示材料
assets/data/demo_task_list.json  合成任务依赖与风险扫描样例
docs/architecture.md           组件、数据流和已知限制
docs/coze-setup.md             Coze Coding 配置步骤
.env.example                   环境变量名称示例，不含凭据
uv.lock                        Python 依赖锁定文件
```

## 使用方式

项目面向 Linux Coze Coding 工作区，要求 Python 3.12、可用模型和 Coze 工作负载身份。普通本地 Python 环境缺少 Coze KnowledgeClient、工作负载身份和托管存储能力，不能仅靠安装依赖复现端到端流程。

1. 在有权使用的 Coze Coding 工作区克隆仓库，并确认工作区使用 Python 3.12。
2. 按 `docs/coze-setup.md` 配置模型、工作负载身份和可选 PostgreSQL。
3. 将 `config/agent_llm_config.json` 中的模型名称改成当前工作区可用的模型。
4. 在工作区中执行：

```bash
uv sync --frozen
export COZE_WORKSPACE_PATH="$PWD"
bash scripts/local_run.sh -m agent
```

部署 HTTP 服务时，在受控的 Coze 服务网络中设置 `COZE_BIND_HOST=0.0.0.0`，再运行 `bash scripts/http_run.sh -p 5000`。本地测试保留默认 `127.0.0.1`。服务没有内建身份认证；不要把它直接映射到公网。

确定性演示输入见 `assets/data/demo_task_list.json` 和下文示例。进度扫描读取 JSON 的 `today`、`submission_deadline` 和 `tasks` 字段；在不提供 `today` 时使用运行机器的当前日期。

## 已实现示例

以下内容使用合成数据，可在 Coze 知识库中检索或作为 Agent 对话输入：

- 知识检索：询问“DEMO-CAR-ALPHA 的演示参数有哪些？”；回答应引用 `02_demo_vehicle_profile.md`，并说明这些值只是合成数据。
- 参数核对：研发侧提供续航 `123 km`、电池能量 `45.6 kWh`、电机功率 `92 kW`、质量 `1540 kg`；检测侧提供续航 `123 km`、能量 `45600 Wh`、功率 `88 kW`、质量 `1540 kg`。预期结果分别体现命名待确认、Wh/kWh 换算后待确认、功率严重差异和质量一致。
- 进度扫描：将 `assets/data/demo_task_list.json` 内容交给 Agent；工具根据依赖关系推导 `D-01 → D-02 → D-03` 最长任务链，并标出阻塞或临期任务。
- 证据不足：询问未包含在演示知识库中的车型或法规细节；系统应说明没有检索到依据，不用模型记忆补写。
- 材料初稿：要求起草演示材料但不提供完整参数；Agent 应先追问或用 `【待补充】` 标记，不应猜数值。

这些示例证明流程和工具行为，不代表正式车型申报结果、法规适用结论或生产环境效果。仓库没有提供可用于复算真实业务准确率的金标准数据。

## 使用边界与数据安全

- 仓库中的知识语料、车型参数和任务清单均为人工构造的演示数据；知识库不是官方法规数据库，也不提供政策更新服务。
- 模型输出仅用于辅助整理。法规适用、技术参数、申报材料和最终合规判断必须依据最新官方文件及正式检测材料，由有资质的人员复核。
- 不要把真实 VIN、个人信息、未公开车型参数、企业内部流程、客户资料或其他受限数据发送给模型、Coze 知识库或搜索服务，除非数据所有者明确授权且服务配置符合要求。
- 模型与 Coze 服务可能接收用户输入并产生费用；使用前确认数据处理范围、保留策略、地区和权限设置。
- HTTP 服务默认不带认证。仅在有访问控制的隔离网络内运行；公网部署前需增加认证、限流、审计、TLS 和输入访问控制。
- 本仓库不含 API 密钥、数据库密码、Coze 工作区标识或真实业务材料。凭据应配置在 Coze 工作负载身份或受控的本地环境中，不要提交 `.env`。
- 仓库未声明开源许可证。依赖、模型和第三方材料仍受各自许可约束；复用代码前请先与作者确认授权。

## 复现限制

`uv.lock` 固定 Python 依赖版本，但安装与运行还依赖 Coze Coding SDK、工作负载身份、模型权限和可用知识库服务。搜索与模型输出会随平台配置变化。进度分类和参数比较是规则逻辑，演示任务固定了日期以便复现；日期、依赖或输入字段变化会改变结果。

详见 [`docs/architecture.md`](docs/architecture.md) 与 [`docs/coze-setup.md`](docs/coze-setup.md)。
