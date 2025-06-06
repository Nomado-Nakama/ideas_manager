CREATE TABLE IF NOT EXISTS telegram_users (
    telegram_id BIGINT PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    role TEXT DEFAULT 'noone',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS user_urls (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    url TEXT NOT NULL,
    telegram_id  BIGINT NOT NULL,
    status TEXT DEFAULT 'undigested', -- 'undigested', 'ingested', 'unsupported'
    added_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT unique_url UNIQUE(url)
);

CREATE TABLE IF NOT EXISTS raw_docs (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	doc_id TEXT NOT NULL,
	url TEXT NOT NULL,
	content TEXT NOT NULL
)