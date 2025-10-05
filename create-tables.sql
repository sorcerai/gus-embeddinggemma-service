-- Enable pgvector extension for vector operations
CREATE EXTENSION IF NOT EXISTS vector;

-- Table for conversation memories with vector embeddings
CREATE TABLE conversation_memories (
    id SERIAL PRIMARY KEY,
    conversation_date DATE NOT NULL,
    content TEXT NOT NULL,
    summary TEXT,
    importance_score INTEGER CHECK (importance_score >= 1 AND importance_score <= 10),
    emotional_tone TEXT,
    topics TEXT[],
    embedding VECTOR(384), -- MiniLM-L6-v2 produces 384-dimensional vectors
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for daily schedules
CREATE TABLE daily_schedules (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    time TIME NOT NULL,
    event_type TEXT NOT NULL,
    description TEXT NOT NULL,
    importance_level INTEGER CHECK (importance_level >= 1 AND importance_level <= 10),
    completed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for medical information
CREATE TABLE medical_info (
    id SERIAL PRIMARY KEY,
    medications TEXT[],
    conditions TEXT[],
    allergies TEXT[],
    special_instructions TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX idx_conversation_memories_date ON conversation_memories(conversation_date);
CREATE INDEX idx_conversation_memories_importance ON conversation_memories(importance_score);
CREATE INDEX idx_daily_schedules_date ON daily_schedules(date);
CREATE INDEX idx_daily_schedules_time ON daily_schedules(date, time);

-- Create vector similarity index for fast embedding searches
CREATE INDEX idx_conversation_memories_embedding ON conversation_memories 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);