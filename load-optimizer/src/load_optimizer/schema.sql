-- Idempotent schema for LLM session memory + logging

CREATE TABLE IF NOT EXISTS sessions (
    id          bigserial PRIMARY KEY,
    session_id  text UNIQUE NOT NULL,
    user_id     text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    last_active_at timestamptz NOT NULL DEFAULT now(),
    metadata    jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS messages (
    id          bigserial PRIMARY KEY,
    session_id  text NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    role        text NOT NULL,
    content     text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    tokens      int,
    metadata    jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS session_state (
    id          bigserial PRIMARY KEY,
    session_id  text NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    key         text NOT NULL,
    value       jsonb NOT NULL,
    updated_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (session_id, key)
);

CREATE INDEX IF NOT EXISTS idx_messages_session_id_created_at ON messages (session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_session_state_session_id ON session_state (session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions (user_id);