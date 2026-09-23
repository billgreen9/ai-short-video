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

会话按 `thread_id` 持久化到 PostgreSQL（LangGraph PostgresSaver，服务重启不丢）。
所有接口均为 GET。

### `GET /agent/start`

启动一轮：`intent → instruction → plan`。

| 参数 | 说明 |
| --- | --- |
| `user_input` | 用户输入（必填） |
| `id` | 会话/框架 id（可选；为空时服务端生成 UUID，并在响应 `thread_id` 中返回） |

当 plan 判定需要用户补充信息 / 参数时，图在 `human_input` 节点通过 interrupt **真正挂起**，
响应 `status=paused`：

```json
{
  "thread_id": "e2e-001",
  "status": "paused",
  "interrupt": {"type": "param", "msg": "缺少待剪辑音频的audio_id参数，请提供对应的音频id"},
  "action": null
}
```

图走到终点（can_execute / 强制 end 等）时 `status=done`：

```json
{
  "thread_id": "e2e-001",
  "status": "done",
  "interrupt": null,
  "action": {"action": "can_execute", "msg": "...", "plans": [{"en_name": "auto_generate_subtitle", "depends_on": []}]}
}
```

### `GET /agent/resume`

对处于 `paused` 的会话提交用户补充，图从挂起点继续（可能再次暂停或走到 done）。

| 参数 | 说明 |
| --- | --- |
| `thread_id` | start 响应中返回的会话 ID（必填） |
| `user_input` | 用户补充的信息 / 参数（必填） |

```
GET /agent/resume?thread_id=e2e-001&user_input=audio_id=10086，去掉开头5秒
```

`action` 取值：

| action | 含义 | 图上行为 |
| --- | --- | --- |
| `user_input` | 需用户补充，`answner` 为指导话术 | → `human_input` 挂起，resume 后回 `plan` |
| `param` | 缺少执行参数，`msg` 说明缺参 | → `human_input` 挂起，resume 后回 `plan` |
| `instruction` | 继续加载说明，`list` 为 domain，`help` 默认 false | → `instruction` |
| `plan` | 详细执行计划 `plans[{en_name,depends_on}]` | 追加确认话术后自环 `plan` |
| `can_execute` | 已拆成原子方法，可执行 | → `before_execute` → `execute` → END |
| `end` | 循环次数超 `plan_enter_max` 的强制结束（仅守卫产生，LLM 不返回） | END |

## LangGraph 流程

```text
START
  → intent
  → instruction ──(instruction_route)──→ plan ←──(action=plan 自环)
       ↑                                  │
       └──── action=instruction ──────────┤
                                          ├─ user_input / param → human_input（interrupt 挂起）
                                          │                         │  GET /agent/resume
                                          │                         ▼
                                          │                       plan（用户补充入 messages）
                                          ├─ end（循环超限）→ END
                                          └─ can_execute → before_execute → execute → END
```

### 节点职责

| 节点 | 职责 |
| --- | --- |
| `intent` | 意图路由；写入 `intents`、`domains`；`prev_node=intent` |
| `instruction` | 按 `domains` 查表，说明写入 `HumanMessage(input_type=agent)`；不改 `prev_node` |
| `plan` | 首次进入向全量 messages 末尾注入一次 `PLAN_PROMPT_TEMPLATE`（仅 action 协议，操作说明不重复携带）；回流直接基于全量 messages 请求 LLM；入口做循环计数与超限保护；按返回 action 分支 |
| `human_input` | 调 `interrupt()` 挂起并下发指导话术/缺参载荷；resume 后把用户补充作为新 HumanMessage 写入 messages，回 `plan`。interrupt 独立成节点，恢复时只重放本节点，不重复触发 plan 的 LLM 调用 |
| `before_execute` | 执行前校验 / 收集参数（占位） |
| `execute` | 执行（占位） |

### 路由规则

- **instruction_route**：`prev_node == intent` → `plan`；否则 → `prev_node`
- **plan_route**：见上表「图上行为」

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
│   ├── graph.py            # LangGraph 图与节点（含 human_input 中断节点）
│   ├── prompts.py          # 路由 / 规划提示词
│   ├── parse.py            # 结果解析
│   ├── instruction_store.py# instruction 表访问
│   ├── checkpoint.py       # PostgresSaver 检查点（interrupt 挂起/恢复的会话持久化）
│   └── config.py           # 加载 config.json
├── api/
│   └── agent_routes.py     # /agent/start、/agent/resume
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
