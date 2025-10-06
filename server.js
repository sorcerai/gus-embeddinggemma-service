#!/usr/bin/env node

/**
 * Proper MCP Server for ElevenLabs 11.ai with Redis for state management
 * Implements Model Context Protocol over SSE transport with asynchronous responses.
 */

import http from 'http';
import https from 'https';
import redis from 'redis';
import pg from 'pg';
import { pipeline } from '@xenova/transformers';

const PORT = process.env.PORT || 3000;
const REDIS_URL = process.env.REDIS_URL;

if (!REDIS_URL) {
  console.error('FATAL: REDIS_URL environment variable is not set.');
  process.exit(1);
}

// Create Redis clients
const redisClient = redis.createClient({ url: REDIS_URL });
const subscriber = redisClient.duplicate();

// Note: DATABASE_URL not needed - this service only generates embeddings
// n8n workflows handle all database operations

// Load EmbeddingGemma-300M model for embeddings
let embeddingModel = null;
async function initializeModel() {
  try {
    console.log('Loading EmbeddingGemma-300M model...');
    embeddingModel = await pipeline('feature-extraction', 'google/embeddinggemma-300m');
    console.log('EmbeddingGemma-300M model loaded successfully (768 dimensions)');
  } catch (error) {
    console.error('Failed to load EmbeddingGemma model:', error);
    process.exit(1);
  }
}

// Generate embedding for text
async function generateEmbedding(text) {
  if (!embeddingModel) {
    throw new Error('Embedding model not loaded');
  }
  
  const output = await embeddingModel(text, { pooling: 'mean', normalize: true });
  return Array.from(output.data);
}

// Map to hold active SSE response objects for this instance
const activeConnections = new Map();
const instanceId = `mcp-instance-${process.pid}`;

(async () => {
  try {
    // Initialize Redis connection
    await redisClient.connect();
    await subscriber.connect();
    console.log('Connected to Redis successfully.');

    // Initialize MiniLM model
    await initializeModel();

    // Subscribe to a Redis channel for pushing messages to clients on this instance
    await subscriber.subscribe(instanceId, (message) => {
      try {
        const { sessionId, event, data } = JSON.parse(message);
        const res = activeConnections.get(sessionId);
        if (res && !res.destroyed) {
          res.write(`event: ${event}\n`);
          res.write(`data: ${JSON.stringify(data)}\n\n`);
          console.log(`Pushed event '${event}' to session ${sessionId} on this instance.`);
        }
      } catch (e) {
        console.error('Error processing Redis message:', e);
      }
    });
    console.log(`Subscribed to Redis channel: ${instanceId}`);

  } catch (err) {
    console.error('Failed to initialize server:', err);
    process.exit(1);
  }
})();

const server = http.createServer((req, res) => {
  console.log(`\n=== ${req.method} ${req.url} ===`);

  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', '*');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  // SSE connection endpoint
  if (req.method === 'GET' && (req.url === '/sse' || req.url === '/')) {
    if (req.headers.accept && req.headers.accept.includes('text/event-stream')) {
      handleSseConnection(req, res);
    } else {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ name: 'elderly-care-mcp', version: '3.0.0-async', status: 'ready' }));
    }
    return;
  }

  // MCP message endpoint (now fully async)
  if (req.method === 'POST' && req.url.startsWith('/messages/')) {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', async () => {
      console.log('Received POST body:', body);
      // Acknowledge the request immediately
      res.writeHead(202, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ status: "accepted" }));
      
      // Process the message asynchronously
      try {
        const message = JSON.parse(body);
        await processMcpMessage(message, req.url);
      } catch (error) {
        console.error('Error processing MCP message:', error);
      }
    });
    return;
  }

  // Health check
  if (req.method === 'GET' && req.url === '/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'healthy', redis_connected: redisClient.isOpen, active_connections: activeConnections.size }));
    return;
  }

  // Embedding generation endpoint for n8n
  if (req.method === 'POST' && req.url === '/generate-embedding') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', async () => {
      try {
        const { text } = JSON.parse(body);
        if (!text) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Text is required' }));
          return;
        }
        
        const embedding = await generateEmbedding(text);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ embedding }));
      } catch (error) {
        console.error('Embedding generation error:', error);
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Failed to generate embedding' }));
      }
    });
    return;
  }

  // HMAC verification endpoint for n8n
  if (req.method === 'POST' && req.url === '/verify-hmac') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', async () => {
      try {
        const crypto = require('crypto');
        const webhookData = JSON.parse(body);
        
        const signature = webhookData.headers['elevenlabs-signature'];
        if (!signature) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ verified: false, error: 'No signature header' }));
          return;
        }
        
        const sigParts = signature.split(',');
        const timestamp = sigParts[0].split('=')[1];
        const receivedSig = sigParts[1].split('=')[1];
        
        const secret = 'wsec_bfda3fc973732f97745762fa88621012a7b3f2c737cabbed624515477498c163';
        const payload = timestamp + '.' + JSON.stringify(webhookData.body);
        const calculatedSig = crypto.createHmac('sha256', secret).update(payload).digest('hex');
        
        const verified = calculatedSig === receivedSig;
        
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ 
          verified, 
          calculated: calculatedSig,
          received: receivedSig
        }));
      } catch (error) {
        console.error('HMAC verification error:', error);
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Failed to verify HMAC' }));
      }
    });
    return;
  }

  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: 'Not found' }));
});

async function handleSseConnection(req, res) {
  const sessionId = Date.now().toString();
  
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'X-Accel-Buffering': 'no'
  });

  await redisClient.set(`session:elevenlabs:${sessionId}`, JSON.stringify({ instanceId, lastSeen: Date.now() }), { EX: 3600 });
  activeConnections.set(sessionId, res);
  console.log(`SSE connection ${sessionId} established on instance ${instanceId}`);

  res.write(`event: endpoint\n`);
  res.write(`data: /messages/?session_id=${sessionId}\n\n`);

  const keepAlive = setInterval(() => {
    if (res.destroyed) {
      clearInterval(keepAlive);
      return;
    }
    res.write(`: keepalive - ${new Date().toISOString()}\n\n`);
  }, 25000);

  req.on('close', async () => {
    console.log(`SSE client ${sessionId} disconnected.`);
    clearInterval(keepAlive);
    activeConnections.delete(sessionId);
    await redisClient.del(`session:elevenlabs:${sessionId}`);
  });
}

async function processMcpMessage(message, url) {
  const { method, params, id } = message;
  const sessionId = url.split('=')[1];
  console.log(`Processing MCP method '${method}' for session ${sessionId}`);

  let response;

  switch (method) {
    case 'initialize':
      response = { jsonrpc: "2.0", id, result: { protocolVersion: "2025-03-26", capabilities: { tools: {}, resources: {}, prompts: {} }, serverInfo: { name: "elderly-care-mcp", version: "3.1.0-vector", description: "Elderly care assistant with semantic memory search" } } };
      break;

    case 'tools/list':
      response = { 
        jsonrpc: "2.0", 
        id, 
        result: { 
          tools: [
            { 
              name: "load_todays_context", 
              description: "Load complete context: today's schedule, recent memories, and medical information",
              inputSchema: { 
                type: "object", 
                properties: {}, 
                required: [] 
              } 
            },
            { 
              name: "search_memory_kb", 
              description: "Search conversation memories for specific information",
              inputSchema: { 
                type: "object", 
                properties: { 
                  query: { type: "string", description: "Search query to find relevant memories" }
                }, 
                required: ["query"] 
              } 
            }
          ] 
        } 
      };
      break;

    case 'tools/call':
      try {
        let result;
        const toolName = params.name;
        
        if (toolName === 'load_todays_context') {
          result = await loadTodaysContext();
        } else if (toolName === 'search_memory_kb') {
          const query = params.arguments?.query || '';
          result = await searchMemoryKB(query);
        } else {
          throw new Error(`Unknown tool: ${toolName}`);
        }
        
        response = { 
          jsonrpc: "2.0", 
          id, 
          result: { 
            content: [{ 
              type: "text", 
              text: JSON.stringify(result, null, 2) 
            }] 
          } 
        };
      } catch (error) {
        console.error(`Tool ${params.name} failed:`, error);
        response = { 
          jsonrpc: "2.0", 
          id, 
          error: { 
            code: -32000, 
            message: `Tool call failed: ${error.message}` 
          } 
        };
      }
      break;

    default:
      response = { jsonrpc: "2.0", id, error: { code: -32601, message: `Method '${method}' not found` } };
      break;
  }

  if (response) {
    await pushMessageToClient(sessionId, 'message', response);
  }
}

async function pushMessageToClient(sessionId, event, data) {
  try {
    const sessionInfo = await redisClient.get(`session:elevenlabs:${sessionId}`);
    if (!sessionInfo) {
      console.log(`Session ${sessionId} not found in Redis. Cannot push message.`);
      return;
    }
    const { instanceId } = JSON.parse(sessionInfo);
    await redisClient.publish(instanceId, JSON.stringify({ sessionId, event, data }));
    console.log(`Published message for session ${sessionId} to instance ${instanceId}`);
  } catch (error) {
    console.error(`Failed to push message to client ${sessionId}:`, error);
  }
}

// Load today's complete context: schedule, memories, and medical info
// NOTE: Disabled - this service only provides embedding generation
// n8n handles all database operations
async function loadTodaysContext() {
  throw new Error('DATABASE_URL not configured - MCP tools disabled');
  try {
    // Get recent conversation memories (last 7 days)
    const memoryQuery = `
      SELECT 
        conversation_date,
        content,
        summary,
        importance_score,
        emotional_tone,
        topics
      FROM conversation_memories 
      WHERE conversation_date >= CURRENT_DATE - INTERVAL '7 days'
      ORDER BY conversation_date DESC, importance_score DESC
      LIMIT 20
    `;
    
    // Get today's schedule from database
    const scheduleQuery = `
      SELECT time, event_type, description, importance_level, completed
      FROM daily_schedules 
      WHERE date = CURRENT_DATE
      ORDER BY time
    `;
    
    // Get medical info from database
    const medicalQuery = `
      SELECT medications, conditions, allergies, special_instructions
      FROM medical_info 
      ORDER BY updated_at DESC
      LIMIT 1
    `;
    
    const [memoryResult, scheduleResult, medicalResult] = await Promise.all([
      client.query(memoryQuery),
      client.query(scheduleQuery), 
      client.query(medicalQuery)
    ]);
    
    // Format schedule with emojis based on importance
    const formatSchedule = (time, event_type, description, importance, completed) => {
      const emoji = completed ? "✅" : (importance >= 8 ? "🔴" : importance >= 6 ? "🟡" : "🟢");
      return `${emoji} ${time.slice(0,5)}: ${description}`;
    };
    
    const todaysContext = {
      person_name: "Mary Johnson",
      date: new Date().toISOString().split('T')[0],
      
      todays_schedule: scheduleResult.rows.map(row => 
        formatSchedule(row.time, row.event_type, row.description, row.importance_level, row.completed)
      ),
      
      medical_info: {
        medications: medicalResult.rows[0]?.medications || [],
        conditions: medicalResult.rows[0]?.conditions || [],
        allergies: medicalResult.rows[0]?.allergies || [],
        special_instructions: medicalResult.rows[0]?.special_instructions || "No special instructions"
      },
      
      recent_memories: memoryResult.rows.map(row => ({
        date: row.conversation_date,
        content: row.content,
        summary: row.summary,
        importance: row.importance_score,
        emotional_tone: row.emotional_tone,
        topics: row.topics
      })),
      
      system_status: `Loaded ${memoryResult.rows.length} recent memories, ${scheduleResult.rows.length} schedule items, medical info from database`
    };
    
    return todaysContext;
    
  } finally {
    client.release();
  }
}

// Search memory knowledge base using vector similarity
// NOTE: Disabled - this service only provides embedding generation
// n8n handles all database operations
async function searchMemoryKB(query) {
  throw new Error('DATABASE_URL not configured - MCP tools disabled');
  try {
    // Generate embedding for the search query
    console.log(`Generating embedding for query: "${query}"`);
    const queryEmbedding = await generateEmbedding(query);
    
    // Vector similarity search using pgvector
    const searchQuery = `
      SELECT 
        conversation_date,
        content,
        summary,
        importance_score,
        emotional_tone,
        topics,
        (embedding <-> $1::vector) as similarity_distance
      FROM conversation_memories 
      ORDER BY embedding <-> $1::vector
      LIMIT 10
    `;
    
    const result = await client.query(searchQuery, [JSON.stringify(queryEmbedding)]);
    
    return {
      query: query,
      search_method: "vector_similarity",
      results: result.rows.map(row => ({
        date: row.conversation_date,
        content: row.content,
        summary: row.summary,
        importance: row.importance_score,
        emotional_tone: row.emotional_tone,
        topics: row.topics,
        similarity_score: (1 - row.similarity_distance).toFixed(4) // Convert distance to similarity score
      })),
      total_found: result.rows.length
    };
    
  } finally {
    client.release();
  }
}

server.listen(PORT, () => {
  console.log(`MCP Server v3.1 (Vector Search) on port ${PORT} - Updated DB`);
  console.log('Features: Redis sessions, EmbeddingGemma-300M (768d) semantic search, direct DB access');
});