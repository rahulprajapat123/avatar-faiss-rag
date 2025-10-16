import dotenv from 'dotenv';
import express from 'express';
import cors from 'cors';
import OpenAI from 'openai';
import { toFile } from 'openai/uploads';
import fetch from 'node-fetch';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import fs from 'fs';

import https from 'https';
import http from 'http';

// ⚡ OPTIMIZATION: Connection pooling
const httpsAgent = new https.Agent({
  keepAlive: true,
  keepAliveMsecs: 30000,
  maxSockets: 50,
  maxFreeSockets: 10
});

const httpAgent = new http.Agent({
  keepAlive: true,
  keepAliveMsecs: 30000,
  maxSockets: 50,
  maxFreeSockets: 10
});


// ============================================
// ⚡ NEW: Import Optimized RAG Engine and Smart Router
// ============================================
import OptimizedRAGEngine from './kb/optimized-rag-engine.js';
import SmartRouter from './kb/smart-router.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.static(__dirname));

const CONFIG = {
  HEYGEN_API_KEY: process.env.HEYGEN_API_KEY,
  HEYGEN_AVATAR_ID: process.env.HEYGEN_AVATAR_ID,
  HEYGEN_VOICE_ID: process.env.HEYGEN_VOICE_ID,
  OPENAI_API_KEY: process.env.OPENAI_API_KEY,
  PORT: process.env.PORT || 3000
};

const HEYGEN_API = {
  BASE_URL: 'https://api.heygen.com',
  CREATE_SESSION: '/v1/streaming.new',
  START_SESSION: '/v1/streaming.start',
  SPEAK: '/v1/streaming.task',
  STOP: '/v1/streaming.stop'
};

const openai = new OpenAI({ 
  apiKey: CONFIG.OPENAI_API_KEY,
  timeout: 8000,  // ⚡ SPEED: Reduced for faster responses
  maxRetries: 0   // ⚡ No retries for speed
});

const TRANSCRIPTION_MODEL = process.env.TRANSCRIPTION_MODEL || 'gpt-4o-mini-transcribe';

const activeSessions = new Map();
const responseCache = new Map();
const CACHE_TTL = 300000;

// 💾 Path to persistent cache file
const PERSISTENT_CACHE_FILE = join(__dirname, 'kb', 'persistent-cache.json');
const FALLBACK_PREFIXES = [
  "i don't have specific information about that",
  "i don't have enough information to answer that",
  "i apologize, but i'm having trouble processing"
];

const normalizeForComparison = (text) => {
  if (!text || typeof text !== 'string') return '';
  return text.trim().toLowerCase().replace(/[’]/g, "'");
};

const isCacheEntryValuable = (entry) => {
  if (!entry || typeof entry !== 'object') return false;

  const normalized = normalizeForComparison(entry.responseText);
  if (!normalized || FALLBACK_PREFIXES.some(prefix => normalized.startsWith(prefix))) {
    return false;
  }

  if (entry.usedRAG) {
    const hasSources = Array.isArray(entry.sources) && entry.sources.length > 0;
    if (!hasSources) return false;
    if (typeof entry.confidence === 'number' && entry.confidence <= 0) return false;
  }

  return true;
};

// ============================================
// PERSISTENT CACHE LOADING
// ============================================
function loadPersistentCache() {
  try {
    if (fs.existsSync(PERSISTENT_CACHE_FILE)) {
      const cacheData = JSON.parse(fs.readFileSync(PERSISTENT_CACHE_FILE, 'utf8'));
      const entries = Object.entries(cacheData);
      
      let loadedCount = 0;
      let prunedCount = 0;
      const sanitizedCache = {};
      for (const [key, value] of entries) {
        if (!isCacheEntryValuable(value)) {
          prunedCount++;
          continue;
        }

        responseCache.set(key, {
          response: value,
          timestamp: Date.now() // Fresh timestamp for server start
        });
        sanitizedCache[key] = value;
        loadedCount++;
      }
      
      if (prunedCount > 0) {
        fs.writeFileSync(PERSISTENT_CACHE_FILE, JSON.stringify(sanitizedCache, null, 2));
        console.log(`🧹 Pruned ${prunedCount} stale cached responses`);
      }
      
      if (loadedCount === 0) {
        console.log('ℹ️  Persistent cache contained no reusable entries.');
        return 0;
      }
      
      console.log(`💾 Loaded ${loadedCount} cached responses from persistent cache`);
      return loadedCount;
    } else {
      console.log('ℹ️  No persistent cache file found.');
      return 0;
    }
  } catch (error) {
    console.error('⚠️  Failed to load persistent cache:', error.message);
    return 0;
  }
}

// ============================================
// AUTO PRE-WARM CACHE ON FIRST RUN
// ============================================
async function autoPrewarmCache() {
  console.log('\n🔥 Auto-generating cache for common queries...');
  console.log('⏳ This will take ~60-90 seconds (only needed once)\n');
  
  const COMMON_QUERIES = [
    'What are the specs of DL380 Gen12?',
    'DL360 Gen12 specifications',
    'Tell me about DL384 features',
    'What processors does DL380 support?',
    'DL380 memory capacity',
    'DL360 storage options',
    'DL380 performance benchmarks',
    'How fast is DL384?',
    'DL360 Gen12 benchmark results',
    'AI inference performance DL384',
    'Compare DL380 vs DL384',
    'Difference between DL360 and DL380',
    'Which is better DL380 or DL360?',
    'DL384 vs DL380 Gen12',
    'VMware support on DL380',
    'KVM virtualization DL384',
    'Best server for virtualization',
    'VMware alternative for HPE servers',
    'Hypervisor support DL360',
    'How to configure HPE iLO?',
    'iLO management features',
    'HPE OneView support',
    'Remote management DL380',
    'AI inference on DL384',
    'ML workloads DL380',
    'GPU support DL380',
    'AI acceleration features',
    'Customer case studies',
    'Who uses HPE ProLiant?',
    'Success stories DL380',
    'Tell me success stories of DL380',
    'DL380 customer success',
    'Latest HPE ProLiant updates',
    'Gen12 new features',
    'Recent announcements',
    'What is HPE ProLiant?',
    'Tell me about Gen12 servers',
    'HPE server portfolio',
    'ProLiant Gen12 overview',
    'DL360 customer stories',
    'Tell me about DL384 success'
  ];
  
  const persistentCache = {};
  let successCount = 0;
  let failCount = 0;
  
  for (let i = 0; i < COMMON_QUERIES.length; i++) {
    const query = COMMON_QUERIES[i];
    
    try {
      // Show progress
      if (i % 5 === 0) {
        console.log(`   Processing: [${i + 1}/${COMMON_QUERIES.length}]...`);
      }
      
      const classification = smartRouter.classifyQuestion(query);
      
      if (smartRouter.shouldUseRAG(classification)) {
        const filter = smartRouter.extractProductFilter(query);
        
        const result = await ragEngine.query(query, {
          topK: 2,
          filter: filter,
          minScore: 0.7
        });

        const hasSources = Array.isArray(result?.sources) && result.sources.length > 0;
        const meaningfulResponse = isCacheEntryValuable({
          responseText: result?.answer,
          sources: result?.sources,
          usedRAG: true,
          confidence: result?.confidence
        });

        if (!hasSources || result?.noResults || !meaningfulResponse) {
          console.log(`   ⚠️ Skipping cache entry for: ${query.substring(0, 40)}... (no confident RAG match)`);
          continue;
        }
        
        const cacheKey = query.toLowerCase().trim();
        persistentCache[cacheKey] = {
          responseText: result.answer,
          sources: result.sources,
          usedRAG: true,
          classification: classification.type,
          confidence: result.confidence,
          timestamp: Date.now()
        };
        
        // Also add to in-memory cache
        responseCache.set(cacheKey, {
          response: persistentCache[cacheKey],
          timestamp: Date.now()
        });
        
        successCount++;
      }
      
      // Small delay to avoid rate limits
      await new Promise(resolve => setTimeout(resolve, 100));
      
    } catch (error) {
      console.error(`   ⚠️ Failed query: ${query.substring(0, 30)}...`);
      failCount++;
    }
  }
  
  // Save to file
  try {
    fs.writeFileSync(PERSISTENT_CACHE_FILE, JSON.stringify(persistentCache, null, 2));
    console.log(`\n✅ Cache generated successfully!`);
    console.log(`   📦 ${successCount} responses cached`);
    console.log(`   💾 Saved to: kb/persistent-cache.json`);
    console.log(`   ⚡ Next startup will be instant!\n`);
  } catch (error) {
    console.error(`\n⚠️ Failed to save cache file: ${error.message}`);
  }
}

// ============================================
// RESPONSE CACHING FUNCTIONS WITH SEMANTIC MATCHING
// ============================================

/**
 * Calculate similarity between two strings
 * Returns a score from 0 (no match) to 1 (perfect match)
 */
function calculateSimilarity(str1, str2) {
  const s1 = str1.toLowerCase().trim();
  const s2 = str2.toLowerCase().trim();
  
  // Exact match
  if (s1 === s2) return 1.0;
  
  // Normalize: remove common words and extra spaces
  const removeWords = ['tell', 'me', 'about', 'the', 'a', 'an', 'of', 'for', 'what', 'are', 'is', 'can', 'you', 'please'];
  
  const normalize = (str) => {
    return str
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ') // Remove punctuation
      .split(/\s+/)
      .filter(word => word.length > 0 && !removeWords.includes(word))
      .sort() // Sort words to handle word order differences
      .join(' ');
  };
  
  const normalized1 = normalize(s1);
  const normalized2 = normalize(s2);
  
  // Check normalized match
  if (normalized1 === normalized2) return 0.95;
  
  // Calculate word overlap
  const words1 = new Set(normalized1.split(' '));
  const words2 = new Set(normalized2.split(' '));
  
  if (words1.size === 0 || words2.size === 0) return 0;
  
  const intersection = new Set([...words1].filter(x => words2.has(x)));
  const union = new Set([...words1, ...words2]);
  
  const jaccardSimilarity = intersection.size / union.size;
  
  // Boost score if key product names match
  const productNames = ['dl380', 'dl360', 'dl384', 'dl20', 'ml350', 'ml30', 'dl580', 'gen12', 'gen11'];
  let productBoost = 0;
  
  for (const product of productNames) {
    if (s1.includes(product) && s2.includes(product)) {
      productBoost += 0.1;
    }
  }
  
  return Math.min(jaccardSimilarity + productBoost, 1.0);
}

/**
 * Get cached response with semantic matching
 * Finds responses even if query is phrased differently
 */
function getCachedResponse(query) {
  const key = query.toLowerCase().trim();
  
  // Try exact match first (fastest)
  const exactMatch = responseCache.get(key);
  if (exactMatch && Date.now() - exactMatch.timestamp < CACHE_TTL) {
    console.log('⚡ Response cache hit (exact match)!');
    return exactMatch.response;
  }
  
  // Try semantic matching (find similar queries)
  let bestMatch = null;
  let bestScore = 0;
  const SIMILARITY_THRESHOLD = 0.75; // 75% similarity required
  
  for (const [cachedKey, cachedValue] of responseCache.entries()) {
    if (Date.now() - cachedValue.timestamp >= CACHE_TTL) continue;
    
    const similarity = calculateSimilarity(query, cachedKey);
    
    if (similarity > bestScore && similarity >= SIMILARITY_THRESHOLD) {
      bestScore = similarity;
      bestMatch = cachedValue.response;
    }
  }
  
  if (bestMatch) {
    console.log(`⚡ Response cache hit (semantic match: ${(bestScore * 100).toFixed(0)}% similar)!`);
    return bestMatch;
  }
  
  return null;
}

function cacheResponse(query, response) {
  const normalized = normalizeForComparison(response?.responseText);
  if (normalized && FALLBACK_PREFIXES.some(prefix => normalized.startsWith(prefix))) {
    console.log(`⏭️  Skipping cache for fallback response: "${query.substring(0, 50)}..."`);
    return;
  }

  if (response?.usedRAG && (!Array.isArray(response.sources) || response.sources.length === 0)) {
    console.log(`⏭️  Skipping cache (no RAG sources) for: "${query.substring(0, 50)}..."`);
    return;
  }

  const key = query.toLowerCase().trim();
  responseCache.set(key, {
    response,
    timestamp: Date.now()
  });
  console.log(`💾 Cached response for: "${query.substring(0, 50)}..."`);
}

// ============================================
// ⚡ NEW: Initialize Optimized RAG Engine and Smart Router
// ============================================
const ragEngine = new OptimizedRAGEngine();
const smartRouter = new SmartRouter();

console.log('✅ Optimized RAG Engine created (will initialize on first use)');
console.log('✅ Smart Router initialized');

// 💾 Load persistent cache from disk (or generate if missing)
async function initializeCache() {
  const cachedCount = loadPersistentCache();
  
  if (cachedCount > 0) {
    console.log(`⚡ ${cachedCount} responses ready for instant retrieval!`);
    return cachedCount;
  } else {
    console.log('🔥 First-time setup: Generating cache for optimal performance...');
    await autoPrewarmCache();
    return responseCache.size;
  }
}

// Initialize cache and then start cleanup interval
initializeCache().then(() => {
  console.log('✅ Cache initialization complete!\n');
}).catch(error => {
  console.error('⚠️ Cache initialization failed:', error.message);
  console.log('ℹ️  Server will continue with dynamic caching.\n');
});

setInterval(() => {
  const now = Date.now();
  
  // Clean up response cache
  for (const [key, value] of responseCache.entries()) {
    if (now - value.timestamp > CACHE_TTL) {
      responseCache.delete(key);
    }
  }
  
  if (responseCache.size > 0) {
    console.log(`🧹 Cache cleanup: ${responseCache.size} response cache entries`);
  }
}, 60000);

const fetchOptions = {
  timeout: 10000
};

class LatencyTracker {
  constructor(requestId) {
    this.requestId = requestId;
    this.times = {};
    this.mark('start');
  }
  
  mark(label) {
    this.times[label] = Date.now();
  }
  
  measure(startLabel, endLabel) {
    if (!this.times[startLabel] || !this.times[endLabel]) return null;
    return this.times[endLabel] - this.times[startLabel];
  }
  
  report() {
    const total = this.times.end ? this.times.end - this.times.start : 0;
    const steps = {
      transcription: this.measure('start', 'transcriptionEnd'),
      ttft: this.measure('responseStart', 'firstToken'),
      response: this.measure('responseStart', 'responseEnd'),
      avatar: this.measure('responseEnd', 'end')
    };
    const format = (value) => (typeof value === 'number' ? `${value}ms` : 'n/a');
    
    console.log(`\n⏱️  [${this.requestId}] LATENCY BREAKDOWN:`);
    console.log(`   Transcription: ${format(steps.transcription)}`);
    console.log(`   TTFT:          ${format(steps.ttft)}`);
    console.log(`   Response:      ${format(steps.response)}`);
    console.log(`   Avatar:        ${format(steps.avatar)}`);
    console.log(`   TOTAL:         ${format(total)}\n`);
    
    return { total, steps };
  }
}

function validateConfig() {
  const required = [
    'HEYGEN_API_KEY',
    'HEYGEN_AVATAR_ID',
    'HEYGEN_VOICE_ID',
    'OPENAI_API_KEY'
  ];
  
  const missing = required.filter(key => !CONFIG[key]);
  
  if (missing.length > 0) {
    console.error('❌ Missing required environment variables:', missing.join(', '));
    process.exit(1);
  }
  
  console.log('✅ All environment variables loaded');
}

validateConfig();

// ============================================
// HEYGEN API FUNCTIONS
// ============================================

async function heygenRequest(endpoint, method = 'POST', body = null) {
  const url = `${HEYGEN_API.BASE_URL}${endpoint}`;
  
  const options = {
    method,
    headers: {
      'Authorization': `Bearer ${CONFIG.HEYGEN_API_KEY}`,
      'Content-Type': 'application/json',
      'Accept': 'application/json'
    },
    ...fetchOptions
  };
  
  if (body) {
    options.body = JSON.stringify(body);
  }
  
  const response = await fetch(url, options);
  
  let data;
  try {
    data = await response.json();
  } catch (e) {
    const textResponse = await response.text();
    console.error('❌ Parse error:', textResponse);
    throw new Error('Invalid response from HeyGen API');
  }
  
  if (!response.ok) {
    console.error('❌ HeyGen Error:', response.status, data.message);
    throw new Error(data.message || `HeyGen API error: ${response.status}`);
  }
  
  return data;
}

async function createAvatarSession(userId) {
  try {
    console.log(`🎭 Creating session: ${userId}`);
    
    const sessionResponse = await heygenRequest(HEYGEN_API.CREATE_SESSION, 'POST', {
      version: 'v2',
      avatar_id: CONFIG.HEYGEN_AVATAR_ID,
      voice: {
        voice_id: CONFIG.HEYGEN_VOICE_ID,
        rate: 1.0
      },
      quality: 'low',
      video_encoding: 'H264'
    });
    
    const sessionData = sessionResponse.data;
    const sessionId = sessionData.session_id;
    
    console.log(`✅ Session created: ${sessionId.substring(0, 15)}...`);
    
    activeSessions.set(userId, {
      sessionId: sessionId,
      token: sessionData.access_token,
      url: sessionData.url,
      createdAt: Date.now(),
      isStarted: false
    });
    
    return {
      sessionId: sessionId,
      token: sessionData.access_token,
      url: sessionData.url
    };
    
  } catch (error) {
    console.error('❌ Session creation error:', error.message);
    throw error;
  }
}

async function startAvatarSession(userId) {
  const session = activeSessions.get(userId);
  if (!session) {
    throw new Error('No active avatar session found');
  }
  
  try {
    console.log(`🚀 Starting session: ${userId}`);
    
    await heygenRequest(HEYGEN_API.START_SESSION, 'POST', {
      session_id: session.sessionId
    });
    
    session.isStarted = true;
    activeSessions.set(userId, session);
    
    console.log('✅ Session started');
    
  } catch (error) {
    console.error('❌ Start error:', error.message);
    throw error;
  }
}

async function makeAvatarSpeak(userId, text, taskType = 'repeat') {
  const session = activeSessions.get(userId);
  if (!session) {
    console.warn('⚠️ No active avatar session found, skipping speech');
    return;
  }
  
  if (!session.isStarted) {
    console.warn('⚠️ Session not started yet, skipping speech');
    return;
  }

  if (!text || text.trim() === '') {
    console.warn('⚠️ Empty text provided, skipping speech');
    return;
  }
  
  // ⚡ CRITICAL FIX: HeyGen has ~500 char limit per request
  // Split long text into sentence-based chunks
  const MAX_CHUNK_SIZE = 450; // Safe limit
  const chunks = splitTextIntoChunks(text, MAX_CHUNK_SIZE);
  
  try {
    if (chunks.length === 1) {
      console.log(`🗣️ Speaking: "${text.substring(0, 50)}..."`);
      
      await heygenRequest(HEYGEN_API.SPEAK, 'POST', {
        session_id: session.sessionId,
        text: text,
        task_type: taskType,
        task_mode: 'async'
      });
      
      console.log('✅ Speech queued');
    } else {
      console.log(`🗣️ Speaking in ${chunks.length} chunks...`);
      
      // Send chunks sequentially with small delay
      for (let i = 0; i < chunks.length; i++) {
        console.log(`   📤 Chunk ${i + 1}/${chunks.length}: "${chunks[i].substring(0, 40)}..."`);
        
        await heygenRequest(HEYGEN_API.SPEAK, 'POST', {
          session_id: session.sessionId,
          text: chunks[i],
          task_type: taskType,
          task_mode: 'async'
        });
        
        // Small delay between chunks to prevent overlap
        if (i < chunks.length - 1) {
          await new Promise(resolve => setTimeout(resolve, 100));
        }
      }
      
      console.log('✅ All speech chunks queued');
    }
    
  } catch (error) {
    console.error('❌ Speak error:', error.message);
  }
}

// Helper function to split text into chunks at sentence boundaries
function splitTextIntoChunks(text, maxSize) {
  if (text.length <= maxSize) {
    return [text];
  }
  
  const chunks = [];
  const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
  let currentChunk = '';
  
  for (const sentence of sentences) {
    // If single sentence is too long, split it
    if (sentence.length > maxSize) {
      if (currentChunk) {
        chunks.push(currentChunk.trim());
        currentChunk = '';
      }
      
      // Split long sentence by commas or at word boundaries
      const parts = sentence.split(/,\s+/);
      for (const part of parts) {
        if (currentChunk.length + part.length + 2 <= maxSize) {
          currentChunk += (currentChunk ? ', ' : '') + part;
        } else {
          if (currentChunk) chunks.push(currentChunk.trim());
          currentChunk = part;
        }
      }
    } else {
      // Normal sentence handling
      if (currentChunk.length + sentence.length + 1 <= maxSize) {
        currentChunk += (currentChunk ? ' ' : '') + sentence;
      } else {
        if (currentChunk) chunks.push(currentChunk.trim());
        currentChunk = sentence;
      }
    }
  }
  
  if (currentChunk) {
    chunks.push(currentChunk.trim());
  }
  
  return chunks;
}

async function closeAvatarSession(userId) {
  const session = activeSessions.get(userId);
  
  if (session) {
    try {
      heygenRequest(HEYGEN_API.STOP, 'POST', {
        session_id: session.sessionId
      }).catch(err => console.error('❌ Stop error:', err.message));
      
      activeSessions.delete(userId);
      console.log(`✅ Session closed: ${userId}`);
    } catch (error) {
      console.error('❌ Close error:', error.message);
      activeSessions.delete(userId);
    }
  }
}

// ============================================
// INTELLIGENT CONVERSATION HANDLER 
// ============================================

/**
 * Handle conversations using RAG only (no Voiceflow fallback)
 * Eliminates external runtime latency and avoids extra OpenAI calls
 */
async function handleConversation(userQuery, userId, classification, tracker = null) {
  const startTime = Date.now();
  const filter = smartRouter.extractProductFilter(userQuery);
  const topK = classification.type === 'route' ? 4 : 3;
  const minScore = classification.type === 'route' ? 0.42 : 0.38;

  let buffer = '';
  let firstTokenTime = null;
  const spokenSegments = [];
  let speechChain = Promise.resolve();

  const enqueueSpeech = (segment) => {
    if (!segment || !segment.trim()) return;
    speechChain = speechChain
      .then(() => makeAvatarSpeak(userId, segment.trim(), 'repeat'))
      .catch(err => console.error('Avatar speak error:', err.message || err));
  };

  const flushBuffer = () => {
    const segment = buffer.trim();
    if (!segment) {
      buffer = '';
      return;
    }
    spokenSegments.push(segment);
    enqueueSpeech(segment);
    buffer = '';
  };

  const shouldFlush = () => /[.!?]\s$/.test(buffer) || buffer.length >= 220;

  const normalize = (text) => text.replace(/\s+/g, ' ').trim();

  try {
    console.log(`🧠 [${userId}] Intelligent conversation handling`);

    // Check if this is a discovery conversation that needs SDR approach
    const lowerQuery = userQuery.toLowerCase();
    const needsSDRDiscovery = (
      lowerQuery.includes('help') || 
      lowerQuery.includes('can you help') ||
      lowerQuery.includes('how can you') ||
      (lowerQuery.includes('bank') && lowerQuery.includes('cto')) ||
      (lowerQuery.includes('identify') && lowerQuery.includes('server')) ||
      (lowerQuery.includes('best') && lowerQuery.includes('need')) ||
      lowerQuery.includes('which server') ||
      lowerQuery.includes('recommend') ||
      lowerQuery.includes('not yet') ||
      lowerQuery.includes('can you not tell') ||
      (lowerQuery.includes('million') && lowerQuery.includes('users')) ||
      (lowerQuery.includes('ai') && lowerQuery.includes('development')) ||
      (lowerQuery.includes('microservices') || lowerQuery.includes('saas'))
    );

    let ragResult;

    if (needsSDRDiscovery) {
      console.log(`🎯 [${userId}] Using SDR discovery approach`);
      // Force use of SDR discovery by creating a result with no matches
      ragResult = {
        answer: ragEngine.generateSDRDiscoveryResponse(userQuery),
        sources: [],
        confidence: 0.8,
        latency: 50,
        usedSDRFallback: true,
        noResults: false
      };
    } else {
      ragResult = await ragEngine.query(userQuery, {
        topK,
        filter,
        minScore,
        stream: true,
        onToken: ({ token }) => {
          if (!token) return;
          buffer += token;
          if (!firstTokenTime) {
            firstTokenTime = Date.now() - startTime;
            if (tracker) {
              tracker.mark('firstToken');
            }
          }
          if (shouldFlush()) {
            flushBuffer();
          }
        }
      });
    }

    if (buffer.trim()) {
      flushBuffer();
    }

    const finalAnswer = ragResult?.answer && ragResult.answer.trim() ? ragResult.answer.trim() : null;
    if (!firstTokenTime && tracker && ragResult?.cached) {
      tracker.mark('firstToken');
      firstTokenTime = Date.now() - startTime;
    }
    let combinedSpoken = normalize(spokenSegments.join(' '));

    if (finalAnswer) {
      const normalizedAnswer = normalize(finalAnswer);
      if (!normalizedAnswer.startsWith(combinedSpoken) || normalizedAnswer.length > combinedSpoken.length) {
        const remainder = normalizedAnswer.slice(combinedSpoken.length).trim();
        if (remainder) {
          spokenSegments.push(remainder);
          enqueueSpeech(remainder);
          combinedSpoken = normalize(spokenSegments.join(' '));
        }
      }
    }

    if (!spokenSegments.length && finalAnswer) {
      spokenSegments.push(finalAnswer);
      enqueueSpeech(finalAnswer);
    }

    const success = ragResult && !ragResult.noResults && finalAnswer;
    const isSDRResponse = ragResult?.usedSDRFallback && finalAnswer;
    
    let text, method;
    
    if (success) {
      text = finalAnswer;
      method = 'rag';
    } else if (isSDRResponse) {
      text = finalAnswer;
      method = 'sdr_discovery';
    } else {
      text = "Thanks for your question! To give you the most accurate guidance, could you help me understand your specific needs? Are you looking at servers for virtualization, databases, analytics, or other workloads?";
      method = 'fallback';
      spokenSegments.push(text);
      enqueueSpeech(text);
    }

    return {
      text,
      sources: success ? ragResult.sources : [],
      method,
      latency: ragResult?.latency || Date.now() - startTime,
      ttft: firstTokenTime
    };

  } catch (error) {
    console.error(`❌ [${userId}] Conversation error:`, error.message);
    const fallback = "I'm experiencing a brief technical hiccup. While I sort that out, could you tell me what's driving your server evaluation? Are you looking at replacing existing infrastructure, expanding capacity, or addressing specific performance needs?";
    enqueueSpeech(fallback);
    return {
      text: fallback,
      sources: [],
      method: 'error_fallback',
      latency: Date.now() - startTime,
      ttft: firstTokenTime
    };
  }
}

// ============================================
// SPEECH-TO-TEXT FUNCTION
// ============================================

async function transcribeAudio(audioBase64) {
  try {
    console.log('🎤 Transcribing...');
    const startTime = Date.now();
    const audioBuffer = Buffer.from(audioBase64, 'base64');
    
    // ⚡ SPEED: Log audio size
    const audioSizeKB = (audioBuffer.length / 1024).toFixed(2);
    console.log(`📦 Audio size: ${audioSizeKB} KB`);
    
    const file = await toFile(audioBuffer, 'audio.webm', {
      type: 'audio/webm'
    });
    
    // ⚡ SPEED: Dynamic timeout based on file size (larger files need more time)
    const fileSizeKB = file.size / 1024;
    const timeoutDuration = fileSizeKB > 10 ? 4000 : 2500; // ⚡ Reduced: 4s for large, 2.5s for small
    console.log(`⏱️  Timeout set to ${timeoutDuration}ms for ${fileSizeKB.toFixed(2)} KB file`);
    
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutDuration);
    
    try {
      // ⚡ SPEED: Minimal config for maximum speed
      const transcription = await openai.audio.transcriptions.create({
        file: file,
        model: TRANSCRIPTION_MODEL,
        language: 'en',              // ⚡ Explicit language = 30% faster
        response_format: 'text',      // ⚡ Fastest format
        temperature: 0,               // ⚡ Deterministic = faster
        prompt: 'HPE ProLiant server' // ⚡ Shorter prompt for speed
      }, {
        signal: controller.signal,
        timeout: timeoutDuration
      });
      
      clearTimeout(timeout);
      const text = transcription.trim();
      const duration = Date.now() - startTime;
      console.log(`✅ Transcribed (${duration}ms): "${text}"`);
      return text;
      
    } catch (error) {
      clearTimeout(timeout);
      
      // Handle timeout with retry
      if (error.name === 'AbortError' || error.code === 'ETIMEDOUT') {
        console.warn(`⚠️ Timeout (>${timeoutDuration}ms), retrying without timeout...`);
        try {
          const transcription = await openai.audio.transcriptions.create({
            file: file,
            model: TRANSCRIPTION_MODEL,
            language: 'en',
            response_format: 'text'
          });
          const text = transcription.trim();
          const duration = Date.now() - startTime;
          console.log(`✅ Transcribed (retry, ${duration}ms): "${text}"`);
          return text;
        } catch (retryError) {
          console.error('❌ Retry also failed:', retryError.message);
          throw retryError;
        }
      }
      
      // Handle connection errors with retry
      if (error.message.includes('Connection error') || error.code === 'ECONNRESET' || error.code === 'ENOTFOUND') {
        console.warn('⚠️ Connection error, retrying once...');
        try {
          await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1s before retry
          const transcription = await openai.audio.transcriptions.create({
            file: file,
            model: TRANSCRIPTION_MODEL,
            language: 'en',
            response_format: 'text'
          });
          const text = transcription.trim();
          const duration = Date.now() - startTime;
          console.log(`✅ Transcribed (retry after connection error, ${duration}ms): "${text}"`);
          return text;
        } catch (retryError) {
          console.error('❌ Connection retry failed:', retryError.message);
          throw retryError;
        }
      }
      
      throw error;
    }
    
  } catch (error) {
    console.error('❌ Transcription error:', error.message);
    throw error;
  }
}

// ============================================
// API ENDPOINTS
// ============================================

app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    activeSessions: activeSessions.size,
    cacheSize: responseCache.size,
    ragEnabled: true
  });
});

app.get('/api/test-heygen', async (req, res) => {
  try {
    console.log('🧪 Testing HeyGen...');
    
    const response = await fetch(`${HEYGEN_API.BASE_URL}/v1/streaming.list`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${CONFIG.HEYGEN_API_KEY}`,
        'Content-Type': 'application/json'
      },
      ...fetchOptions
    });
    
    const data = await response.json();
    
    res.json({
      success: response.ok,
      status: response.status,
      data: data,
      message: response.ok ? '✅ API connection successful!' : '❌ API connection failed'
    });
    
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

app.post('/api/start-avatar', async (req, res) => {
  try {
    const { userId } = req.body;
    
    if (!userId) {
      return res.status(400).json({
        success: false,
        error: 'userId is required'
      });
    }
    
    if (activeSessions.has(userId)) {
      closeAvatarSession(userId);
    }
    
    const session = await createAvatarSession(userId);
    
    res.json({
      success: true,
      sessionId: session.sessionId,
      token: session.token,
      url: session.url,
      message: 'Avatar session created successfully'
    });
    
  } catch (error) {
    console.error('❌ Start avatar error:', error.message);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

app.post('/api/start-session', async (req, res) => {
  try {
    const { userId } = req.body;
    
    if (!userId) {
      return res.status(400).json({
        success: false,
        error: 'userId is required'
      });
    }
    
    await startAvatarSession(userId);
    
    res.json({
      success: true,
      message: 'Avatar session started successfully'
    });
    
  } catch (error) {
    console.error('❌ Start session error:', error.message);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

app.post('/api/session-ready', async (req, res) => {
  try {
    const { userId } = req.body;
    
    if (!userId) {
      return res.status(400).json({ success: false, error: 'userId is required' });
    }

    const session = activeSessions.get(userId);
    if (!session) {
      return res.status(404).json({ success: false, error: 'Session not found' });
    }

    // ⚡ FIX: Ensure session is started before making avatar speak
    if (!session.isStarted) {
      console.log('⚠️ Session not started yet, starting now...');
      try {
        await startAvatarSession(userId);
      } catch (startError) {
        console.error('❌ Failed to start session:', startError.message);
        return res.status(500).json({ 
          success: false, 
          error: 'Failed to start avatar session: ' + startError.message 
        });
      }
    }

    console.log(`✅ Session ready: ${userId}`);

    // ⚡ INSTANT GREETING: Pre-cached welcome message (no Voiceflow delay)
        const instantWelcome = "Hi, I'm the HPE team's virtual SDR for ProLiant. May I ask a few quick questions to match the right option for you?";
    
    // Return immediately with instant greeting
    res.json({ 
      success: true, 
      message: 'Session ready',
      welcomeText: instantWelcome
    });

    // ⚡ Make avatar speak instantly (fire-and-forget, no Voiceflow wait)
    makeAvatarSpeak(userId, instantWelcome, 'repeat').catch(err => 
      console.error('❌ Avatar speak error:', err.message)
    );
    console.log('✅ Instant welcome sent:', instantWelcome.substring(0, 50) + '...');

  } catch (error) {
    console.error('❌ Session ready error:', error.message);
    res.status(500).json({ success: false, error: error.message });
  }
});

// ============================================
// ⚡ HYBRID ENDPOINT - Streaming RAG
// This is where the magic happens!
// ============================================
app.post('/api/speak', async (req, res) => {
  const requestId = `speak-${Date.now()}`;
  const tracker = new LatencyTracker(requestId);
  
  try {
    const { userId, audioData } = req.body;
    
    if (!userId || !audioData) {
      return res.status(400).json({ 
        success: false, 
        error: 'userId and audioData are required' 
      });
    }
    
    console.log(`🎤 [${requestId}] Processing speech: ${userId}`);
    
    // STEP 1: Transcribe with error handling
    let transcribedText;
    try {
      transcribedText = await transcribeAudio(audioData);
      tracker.mark('transcriptionEnd');
      
      if (!transcribedText || transcribedText.trim() === '') {
        throw new Error('Could not transcribe audio');
      }
      
      console.log(`📝 [${requestId}] User: "${transcribedText}"`);
    } catch (transcriptionError) {
      console.error(`❌ [${requestId}] Transcription failed:`, transcriptionError.message);
      
      // Send user-friendly error response
      const errorMessage = "I'm having trouble hearing you clearly. Could you please try speaking again? Make sure you're in a quiet environment.";
      
      res.json({
        success: false,
        error: 'transcription_failed',
        message: errorMessage,
        technicalError: transcriptionError.message
      });
      
      // Try to make avatar speak the error message
      const session = activeSessions.get(userId);
      if (session) {
        makeAvatarSpeak(userId, errorMessage, 'repeat').catch(err => 
          console.error('Avatar speak error:', err)
        );
      }
      
      return;
    }
    // ⚡ CHECK CACHE FIRST
    const cachedResult = getCachedResponse(transcribedText);
    if (cachedResult) {
      console.log(`⚡ [${requestId}] Using cached response`);
      
      res.json({
        success: true,
        transcribedText: transcribedText,
        avatarResponse: cachedResult.responseText,
        sources: cachedResult.sources || [],
        usedRAG: cachedResult.usedRAG || false,
        classification: cachedResult.classification || 'cached',
        cached: true
      });
      
      // Avatar speaks cached response
      const session = activeSessions.get(userId);
      if (session) {
        makeAvatarSpeak(userId, cachedResult.responseText, 'repeat').catch(err => 
          console.error('Avatar speak error:', err)
        );
      }
      
      tracker.mark('end');
      tracker.report();
      return;
    }
    
    // ⚡ FAST CLASSIFICATION PATH
    const classification = smartRouter.classifyQuestion(transcribedText);
    
    console.log(`🧠 [${requestId}] Classification:`, classification);
    
    let responseText;
    let sources = [];
    let usedRAG = false;
    let responseMethod = 'unknown';
    
  // ⚡ OPTIMIZED: Use intelligent conversation handler (no Voiceflow latency)
  tracker.mark('responseStart');
  const conversationResult = await handleConversation(transcribedText, userId, classification, tracker);
    
    responseText = conversationResult.text;
    sources = conversationResult.sources;
    usedRAG = (conversationResult.method === 'rag');
    responseMethod = conversationResult.method;
    
    console.log(`✅ [${requestId}] ${conversationResult.method} success: ${conversationResult.latency}ms`);
    if (conversationResult.ttft !== null && conversationResult.ttft !== undefined) {
      console.log(`⏱️  [${requestId}] TTFT: ${conversationResult.ttft}ms`);
    }
    
    tracker.mark('responseEnd');
    
    if (!responseText) {
      throw new Error('No response generated');
    }
    
    // ⚡ CACHE THE RESPONSE
    cacheResponse(transcribedText, {
      responseText,
      sources,
      usedRAG,
      classification: classification.type,
      method: responseMethod,
      ttft: conversationResult.ttft ?? null
    });
    
    // ⚡ CRITICAL FIX: Send response IMMEDIATELY, avatar streaming already in progress
    res.json({
      success: true,
      transcribedText: transcribedText,
      avatarResponse: responseText,
      sources: sources,
      usedRAG: usedRAG,
      classification: classification.type,
      method: responseMethod,
      ttft: conversationResult.ttft ?? null
    });
    
    tracker.mark('end');
    tracker.report();
    
  } catch (error) {
    console.error(`❌ [${requestId}] Error:`, error.message);
    res.status(500).json({ 
      success: false, 
      error: error.message 
    });
  }
});
// ============================================
// ⚡ NEW: HYBRID CHAT ENDPOINT
// ============================================
app.post('/api/chat', async (req, res) => {
  const requestId = `chat-${Date.now()}`;
  const tracker = new LatencyTracker(requestId);
  
  try {
    const { userId, message } = req.body;
    
    if (!userId || !message) {
      return res.status(400).json({ 
        success: false, 
        error: 'userId and message are required' 
      });
    }

    console.log(`💬 [${requestId}] Chat: ${userId}`);
    console.log(`📝 [${requestId}] User: "${message}"`);
    
    tracker.mark('transcriptionEnd'); // No actual transcription for text
    
    // ⚡ CHECK CACHE FIRST
    const cachedResult = getCachedResponse(message);
    if (cachedResult) {
      console.log(`⚡ [${requestId}] Using cached response`);
      
      res.json({
        success: true,
        userMessage: message,
        avatarResponse: cachedResult.responseText,
        sources: cachedResult.sources || [],
        usedRAG: cachedResult.usedRAG || false,
        classification: cachedResult.classification || 'cached',
        cached: true
      });
      
      // Avatar speaks cached response
      const session = activeSessions.get(userId);
      if (session) {
        makeAvatarSpeak(userId, cachedResult.responseText, 'repeat').catch(err => 
          console.error('Avatar speak error:', err)
        );
      }
      
      tracker.mark('end');
      tracker.report();
      return;
    }
    
    // ⚡ RAG-ONLY ROUTING: Same logic as speech endpoint
    const classification = smartRouter.classifyQuestion(message);
    console.log(`🧠 [${requestId}] Classification:`, classification);
    
  tracker.mark('responseStart');
  const conversationResult = await handleConversation(message, userId, classification, tracker);
    const responseText = conversationResult.text;
    const sources = conversationResult.sources;
    const usedRAG = (conversationResult.method === 'rag');
    const responseMethod = conversationResult.method;
    
    console.log(`✅ [${requestId}] ${responseMethod} success: ${conversationResult.latency}ms`);
    if (conversationResult.ttft !== null && conversationResult.ttft !== undefined) {
      console.log(`⏱️  [${requestId}] TTFT: ${conversationResult.ttft}ms`);
    }
    
    tracker.mark('responseEnd');
    
    if (!responseText) {
      throw new Error('No response generated');
    }
    
    // ⚡ CACHE THE RESPONSE
    cacheResponse(message, {
      responseText,
      sources,
      usedRAG,
      classification: classification.type,
      method: responseMethod,
      ttft: conversationResult.ttft ?? null
    });
    
    // Send response immediately
    res.json({
      success: true,
      userMessage: message,
      avatarResponse: responseText,
      sources: sources,
      usedRAG: usedRAG,
      classification: classification.type,
      method: responseMethod,
      ttft: conversationResult.ttft ?? null
    });
    
    tracker.mark('end');
    const timing = tracker.report();
    
    console.log(`📊 [${requestId}] Method: ${responseMethod}`);
    console.log(`⏱️  [${requestId}] Total: ${timing.total}ms`);
    
  } catch (error) {
    console.error(`❌ [${requestId}] Error:`, error.message);
    res.status(500).json({ 
      success: false, 
      error: error.message 
    });
  }
});

app.post('/api/stop-avatar', async (req, res) => {
  try {
    const { userId } = req.body;
    
    if (!userId) {
      return res.status(400).json({
        success: false,
        error: 'userId is required'
      });
    }
    
    res.json({
      success: true,
      message: 'Avatar session stopped'
    });

    closeAvatarSession(userId);
    
  } catch (error) {
    console.error('❌ Stop error:', error.message);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// ============================================
// NEW: RAG Stats Endpoint (for monitoring)
// ============================================
app.get('/api/rag-stats', (req, res) => {
  const ragStats = typeof ragEngine.getCacheStats === 'function'
    ? ragEngine.getCacheStats()
    : {};
  res.json({
    cache: ragStats,
    router: 'smart-router',
    status: 'operational',
    routerStats: smartRouter.getCacheStats()
  });
});

// ============================================
// NEW: Debug Endpoints for Monitoring Routing
// ============================================
app.post('/api/debug/classify', (req, res) => {
  try {
    const { text } = req.body;
    
    if (!text) {
      return res.status(400).json({ error: 'text is required' });
    }
    
    const classification = smartRouter.classifyQuestion(text);
    const filter = smartRouter.extractProductFilter(text);
    const shouldUseRAG = smartRouter.shouldUseRAG(classification);
    const shouldUseIntelligent = smartRouter.shouldUseIntelligentResponse(classification);
    
    res.json({
      input: text,
      classification,
      filter,
      routing: {
        shouldUseRAG,
        shouldUseIntelligent,
        recommendedPath: shouldUseRAG ? 'RAG' : 'Fallback message'
      }
    });
    
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/debug/cache', (req, res) => {
  try {
    const cacheEntries = [];
    
    // Get a sample of cache entries (limit to 20 for performance)
    let count = 0;
    for (const [key, value] of responseCache.entries()) {
      if (count >= 20) break;
      cacheEntries.push({
        query: key.substring(0, 50) + (key.length > 50 ? '...' : ''),
        age: Math.floor((Date.now() - value.timestamp) / 1000) + 's',
        method: value.response.method || 'unknown',
        classification: value.response.classification || 'unknown',
        ttft: value.response.ttft ?? 'n/a'
      });
      count++;
    }
    
    const ragStats = typeof ragEngine.getCacheStats === 'function'
      ? ragEngine.getCacheStats()
      : {};

    res.json({
      totalCacheEntries: responseCache.size,
      sampleEntries: cacheEntries,
      ragEngineCache: ragStats,
      routerStats: smartRouter.getCacheStats()
    });
    
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.use(express.static(__dirname));

app.get('/', (req, res) => {
  res.sendFile(join(__dirname, 'index.html'));
});

app.get('*', (req, res) => {
  res.sendFile(join(__dirname, 'index.html'));
});

// ============================================
// START SERVER
// ============================================

process.on('SIGINT', async () => {
  console.log('\n🛑 Shutting down...');
  
  for (const userId of activeSessions.keys()) {
    await closeAvatarSession(userId);
  }
  
  process.exit(0);
});

app.listen(CONFIG.PORT, () => {
  console.log('\n' + '='.repeat(70));
  console.log('🚀 HeyGen + Streaming RAG Server');
  console.log('='.repeat(70));
  console.log(`\n✅ Server running on port ${CONFIG.PORT}`);
  console.log(`🔗 URL: http://localhost:${CONFIG.PORT}`);

});
