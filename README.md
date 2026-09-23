# AI Short Video

基于 FastAPI + LangGraph 的短视频辅助智能体：对用户输入做**意图路由**，按 domain 加载 `instruction` 说明，再做**下一步规划**。

## 功能概览

- 意图路由：`audio` / `video` / `subtitle` / `short` / `synthetical`
- 说明加载：从 Postgres `instruction` 表按 `domain` 拉取有效说明
- 规划决策：返回结构化 `action`（补充用户信息 / 继续加载说明）

## 快速开始

### 环境要求

- Python 3.11+
- PostgreSQL（库名建议：`short-video`）

### 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env：填写 OPENAI_* 与 POSTGRES_DSN
```

### 初始化数据库

```bash
psql "$POSTGRES_DSN" -f sql/instruction.sql
```

需至少有一条 `domain=outline` 且 `status=1` 的记录，作为系统提示词；否则图无法启动。

### 启动服务

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- 健康检查：`GET /health`
- 接口文档：`http://localhost:8000/docs`

## 配置说明

| 来源 | 作用 |
| --- | --- |
| `.env` / 环境变量 | LLM 地址、密钥、模型名、`POSTGRES_DSN`（见 `config.py`） |
| `config.json` | 业务参数：`num`（路由返回意图个数）、`degree_threshold`（进入规划的置信度阈值） |

优先级：环境变量 > `.env` > `config.py` 默认值。

`config.json` 示例：

```json
{
  "num": 2,
  "degree_threshold": 0.72
}
```

## HTTP 接口

### `GET /agent/start`

启动一轮：`intent → instruction → plan`。

| 参数 | 说明 |
| --- | --- |
| `user_input` | 用户输入（必填） |
| `id` | 框架元信息 id（可选，默认可空） |

响应：

```json
{
  "action": {
    "action": "user_input",
    "answer": "用户输入信息不全，无法进行规划，请补充"
  }
}
```

或：

```json
{
  "action": {
    "action": "instruction",
    "list": ["subtitle", "short"]
  }
}
```

`action` 取值：

| action | 含义 |
| --- | --- |
| `user_input` | 需用户补充信息，`answer` 为指导话术 |
| `instruction` | 需继续加载说明，`list` 为 domain 列表 |

## LangGraph 流程

```text
START
  → intent
  → instruction ──(instruction_route)──→ plan
       ↑                                  │
       └──────── plan_route ──────────────┤
                                          │（非继续加载说明）
                                          v
                                   before_execute → execute → END
```

### 节点职责

| 节点 | 职责 |
| --- | --- |
| `intent` | 调用 LLM 做意图路由；写入 `intents`、`domains`（`degree > degree_threshold`）；`prev_node=intent` |
| `instruction` | 只读 `domains` 查表；将说明以 `HumanMessage(input_type=agent)` 写入 `messages`；**不改** `prev_node` |
| `plan` | 首次用 `PLAN_PROMPT_TEMPLATE` 请求；之后用「请继续尝试规划」；写入 `action`；若需继续加载则更新 `domains` |
| `before_execute` | 执行前处理（当前为占位） |
| `execute` | 执行（当前为占位） |

### 路由规则

- **instruction_route**：`prev_node == intent` → `plan`；否则 → `prev_node`（一般为 `plan`）
- **plan_route**：`action.action == instruction` 且 `instructions`、`domains` 非空 → 回流 `instruction`；否则 → `before_execute` → `execute` → `END`

### GraphState

| 字段 | 说明 |
| --- | --- |
| `messages` | 对话消息；用户消息 `additional_kwargs` 仅含 `id`，agent 侧可带 `input_type` |
| `intents` | 路由原始结果 `[{intent, degree}, ...]` |
| `domains` | 待加载 domain，由 `intent` 或 `plan` 写入 |
| `instructions` | 当前查到的说明行 |
| `action` | 规划结果 |
| `prev_node` | 上一节点名，供 instruction 回流 |
| `plan_prompt_used` | 是否已用过完整规划提示词 |

启动时 `messages`：

```text
[SystemMessage(outline), HumanMessage(用户正文, {id})]
```

`outline` 来自 `instruction` 表 `domain=outline`。

## 目录结构

```text
.
├── main.py                 # FastAPI 入口
├── config.py               # 环境变量 / .env 配置
├── config.json             # 业务参数
├── requirements.txt
├── agent/
│   ├── graph.py            # LangGraph 图与节点
│   ├── prompts.py          # 路由 / 规划提示词
│   ├── parse.py            # 结果解析
│   ├── instruction_store.py# instruction 表访问
│   └── config.py           # 加载 config.json
├── api/
│   └── agent_routes.py     # /agent/start
├── sql/
│   └── instruction.sql     # 表结构
└── intent/                 # 早期独立路由模块（遗留，当前 HTTP 未使用）
```

## instruction 表

见 `sql/instruction.sql`。主要字段：

| 字段 | 说明 |
| --- | --- |
| `title` | 标题 |
| `text` | 正文 |
| `domain` | 域：`outline` / `audio` / `video` / `subtitle` / `short` / `shot` / `synthetical` 等 |
| `type` | 类型 |
| `en_name` | 英文名 |
| `status` | `1` 有效，`0` 无效 |

## 开发提示

- 改 `config.json` 后需重启服务（启动时加载一次）
- 路由 / 规划提示词在 `agent/prompts.py`
- 图路径与节点约定以 `agent/graph.py` 模块注释为准
