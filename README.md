# ElevenLabs MCP Connector

An elderly care assistant MCP (Model Context Protocol) server designed to work with ElevenLabs 11.ai platform.

## Features

- **MCP Tools**: 
  - `load_todays_context` - Load complete context including schedule, memories, and medical information
  - `search_memory_kb` - Search conversation memories using vector similarity

- **Vector Search**: Semantic memory search using MiniLM-L6-v2 embeddings with PostgreSQL pgvector
- **SSE Transport**: Server-Sent Events for real-time communication
- **ElevenLabs Integration**: HMAC webhook verification for secure communication
- **N8N Integration**: Embedding generation endpoint for workflow automation

## Architecture

- **Redis**: Session management and pub/sub messaging
- **PostgreSQL**: Vector embeddings storage with pgvector extension
- **Node.js**: Async MCP message processing with health monitoring

## Database Schema

The system uses three main tables:
- `conversation_memories` - Stores conversation history with vector embeddings
- `daily_schedules` - Manages daily scheduling information
- `medical_info` - Stores medical information and care instructions

## Setup

1. Install dependencies:
```bash
npm install
```

2. Set environment variables:
```bash
export REDIS_URL="your_redis_connection_string"
export DATABASE_URL="your_neon_postgres_connection_string"
```

3. Initialize database:
```bash
psql -f create-tables.sql "your_database_connection_string"
```

4. Start the server:
```bash
npm start
```

## API Endpoints

- `GET /` - Health check and server info
- `GET /sse` - SSE connection for MCP communication
- `POST /messages/` - MCP message processing
- `POST /generate-embedding` - Generate text embeddings for N8N
- `POST /verify-hmac` - Verify ElevenLabs webhook signatures
- `GET /health` - Server health status

## License

MIT