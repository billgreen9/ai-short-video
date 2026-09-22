-- 用户匹配到一级意图后，按 domain 加载规划说明。
-- status: 1 有效，0 无效
CREATE TABLE IF NOT EXISTS instruction (
    id bigserial PRIMARY KEY,
    title varchar NOT NULL,
    "text" text NOT NULL DEFAULT ''::text,
    domain varchar NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    status int4 NOT NULL DEFAULT 1
);

COMMENT ON TABLE instruction IS '一级意图命中后，按 domain 加载的操作说明';
COMMENT ON COLUMN instruction.status IS '1:有效 0:无效';

CREATE INDEX IF NOT EXISTS idx_instruction_domain_status
    ON instruction (domain, status);
