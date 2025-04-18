CREATE TABLE IF NOT EXISTS telegram_users (
    telegram_id BIGINT PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    role TEXT DEFAULT 'noone',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS content_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    platform TEXT NOT NULL,
    original_link TEXT NOT NULL UNIQUE,
    content_id TEXT NOT NULL,
    content_type TEXT NOT NULL,
    metadata JSONB, -- Flexible storage for varying metadata
    date_added TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS telegram_files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_item_id UUID REFERENCES content_items(id) ON DELETE CASCADE,
    telegram_file_id TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_path TEXT,
    uploaded_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS transcriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_item_id UUID REFERENCES content_items(id) ON DELETE CASCADE,
    transcription_text TEXT,
    transcription_file_id TEXT,
    audio_file_id TEXT,
    transcribed_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS frames (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_item_id UUID REFERENCES content_items(id) ON DELETE CASCADE,
    frame_timestamp TIME,
    image_file_id TEXT,
    ocr_text TEXT
);

CREATE TABLE IF NOT EXISTS tags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS content_tags (
    content_item_id UUID REFERENCES content_items(id) ON DELETE CASCADE,
    tag_id UUID REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (content_item_id, tag_id)
);

CREATE TABLE IF NOT EXISTS embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_item_id UUID REFERENCES content_items(id) ON DELETE CASCADE,
    embedding_vector JSONB NOT NULL,
    model_used TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS user_links (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    link TEXT NOT NULL,
    status TEXT DEFAULT 'unprocessed', -- 'unprocessed', 'processed', 'unsupported'
    content_item_id UUID REFERENCES content_items(id) ON DELETE SET NULL,
    added_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS user_links_users (
    user_link_id UUID REFERENCES user_links(id) ON DELETE CASCADE,
    telegram_user_id BIGINT REFERENCES telegram_users(telegram_id) ON DELETE CASCADE,
    PRIMARY KEY (user_link_id, telegram_user_id)
);
