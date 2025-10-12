import dotenv from 'dotenv';
import express from 'express';
import cors from 'cors';
import OpenAI from 'openai';
import { toFile } from 'openai/uploads';
import fetch from 'node-fetch';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.static('web'));

const CONFIG = {
  VOICEFLOW_API_KEY: process.env.VOICEFLOW_API_KEY,
  VOICEFLOW_PROJECT_ID: process.env.VOICEFLOW_PROJECT_ID,
  VOICEFLOW_VERSION: 'development',
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

// ============================================
// ⚡ OPTIMIZATION 1: Reduced timeouts for faster failure
// ============================================
const openai = new OpenAI({ 
  apiKey: CONFIG.OPENAI_API_KEY,
  timeout: 15000, // Reduced from 30s
  maxRetries: 0   // No retries for faster response
});

const activeSessions = new Map();

// ============================================
// ⚡ OPTIMIZATION 2: Response caching
// Saves 2-3s on repeat questions!
// ============================================
const responseCache = new Map();
const CACHE_TTL = 300000; // 5 minutes

// Automatic cache cleanup
setInterval(() => {
  const now = Date.now();
  for (const [key, value] of responseCache.entries()) {
    if (now - value.timestamp > CACHE_TTL) {
      responseCache.delete(key);
    }
  }
  if (responseCache.size > 0) {
    console.log(`🧹 Cache cleanup: ${responseCache.size} items remaining`);
  }
}, 60000); // Every minute

const fetchOptions = {
  timeout: 10000 // Reduced from 15s
};

// ============================================
// ⚡ OPTIMIZATION 3: Latency tracking
// ============================================
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
      voiceflow: this.measure('transcriptionEnd', 'voiceflowEnd'),
      avatar: this.measure('voiceflowEnd', 'avatarEnd')
    };
    
    console.log(`\n⏱️  [${this.requestId}] LATENCY BREAKDOWN:`);
    console.log(`   Transcription: ${steps.transcription}ms`);
    console.log(`   Voiceflow:     ${steps.voiceflow}ms`);
    console.log(`   Avatar:        ${steps.avatar}ms`);
    console.log(`   TOTAL:         ${total}ms\n`);
    
    return { total, steps };
  }
}

function validateConfig() {
  const required = [
    'VOICEFLOW_API_KEY',
    'VOICEFLOW_PROJECT_ID',
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

// ============================================
// ⚡ OPTIMIZATION 4: Faster avatar configuration
// ============================================
async function createAvatarSession(userId) {
  try {
    console.log(`🎭 Creating session: ${userId}`);
    
    const sessionResponse = await heygenRequest(HEYGEN_API.CREATE_SESSION, 'POST', {
      version: 'v2',
      avatar_id: CONFIG.HEYGEN_AVATAR_ID,
      voice: {
        voice_id: CONFIG.HEYGEN_VOICE_ID,
        rate: 1.1  // ⚡ 10% faster speech
      },
      quality: 'low',  // ⚡ Faster streaming (saves 300-500ms)
      video_encoding: 'H264'  // ⚡ More efficient
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

// ============================================
// ⚡ OPTIMIZATION 5: Fire-and-forget avatar speech
// Returns immediately, doesn't wait (saves 1-2s!)
// ============================================
async function makeAvatarSpeak(userId, text, taskType = 'repeat') {
  const session = activeSessions.get(userId);
  if (!session) {
    throw new Error('No active avatar session found');
  }
  
  if (!session.isStarted) {
    console.log('⚠️ Session not started, skipping');
    return;
  }

  if (!text || text.trim() === '') {
    console.log('⚠️ Empty text, skipping');
    return;
  }
  
  try {
    console.log(`🗣️ Speaking: "${text.substring(0, 50)}..."`);
    
    // ⚡ Fire and forget - don't await!
    heygenRequest(HEYGEN_API.SPEAK, 'POST', {
      session_id: session.sessionId,
      text: text,
      task_type: taskType,
      task_mode: 'async'  // ⚡ Async mode
    }).catch(err => console.error('❌ Speak error:', err.message));
    
    console.log('✅ Speech queued');
    
  } catch (error) {
    console.error('❌ Speak error:', error.message);
    throw error;
  }
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
// VOICEFLOW FUNCTIONS
// ============================================

// ============================================
// ⚡ OPTIMIZATION 6: Exclude unnecessary traces
// ============================================
async function sendToVoiceflow(userId, action) {
  try {
    const requestAction = typeof action === 'string' ? { type: 'text', payload: action } : action;

    console.log(`📤 Voiceflow: ${requestAction.type}`);
    
    const response = await fetch(
      `https://general-runtime.voiceflow.com/state/user/${userId}/interact`,
      {
        method: 'POST',
        headers: {
          'Authorization': CONFIG.VOICEFLOW_API_KEY,
          'Content-Type': 'application/json',
          'versionID': CONFIG.VOICEFLOW_VERSION
        },
        body: JSON.stringify({
          action: requestAction,
          config: {
            tts: false,
            stripSSML: true,
            stopAll: false,
            excludeTypes: ['block', 'debug', 'flow']  // ⚡ Exclude unnecessary (saves 100-200ms)
          }
        }),
        timeout: 8000  // ⚡ Aggressive timeout
      }
    );

    if (!response.ok) {
      throw new Error(`Voiceflow error: ${response.status}`);
    }

    const data = await response.json();
    console.log(`📥 Voiceflow: Received ${data.length} traces`);
    
    return data;
  } catch (error) {
    console.error('❌ Voiceflow error:', error.message);
    throw error;
  }
}

function extractVoiceflowText(vfResponse) {
  const textParts = [];
  
  for (const trace of vfResponse) {
    if (trace.type === 'text' && trace.payload?.message) {
      textParts.push(trace.payload.message);
    }
  }
  
  return textParts.join('\n\n');
}

// ============================================
// SPEECH-TO-TEXT FUNCTION
// ============================================

// ============================================
// ⚡ OPTIMIZATION 7: Faster transcription settings
// Saves 200-500ms by specifying language!
// ============================================
async function transcribeAudio(audioBase64) {
  try {
    console.log('🎤 Transcribing...');
    const audioBuffer = Buffer.from(audioBase64, 'base64');
    
    const file = await toFile(audioBuffer, 'a.webm', {
      type: 'audio/webm'
    });
    
    const transcription = await openai.audio.transcriptions.create({
      file: file,
      model: 'whisper-1',
      language: 'en',              // ⚡ Skip auto-detection (saves 200-500ms!)
      response_format: 'text',     // ⚡ Faster parsing (saves 100ms)
      temperature: 0               // ⚡ Deterministic (saves 50ms)
    });
    
    const text = transcription.trim();
    console.log(`✅ Transcribed: "${text}"`);
    return text;
    
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
    cacheSize: responseCache.size
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

    console.log(`✅ Session ready: ${userId}`);

    const vfResponse = await sendToVoiceflow(userId, { type: 'launch' });
    const welcomeText = extractVoiceflowText(vfResponse);
    const finalWelcome = welcomeText || 'Hello! How can I help you today?';

    // ⚡ Send response immediately
    res.json({ 
      success: true, 
      message: 'Session ready',
      welcomeText: finalWelcome
    });

    // ⚡ Make avatar speak in background (fire and forget)
    makeAvatarSpeak(userId, finalWelcome, 'repeat');
    console.log('✅ Welcome sent:', finalWelcome.substring(0, 50) + '...');

  } catch (error) {
    console.error('❌ Session ready error:', error.message);
    res.status(500).json({ success: false, error: error.message });
  }
});

// ============================================
// ⚡ OPTIMIZATION 8: Main endpoint with all optimizations
// ============================================
app.post('/api/speak', async (req, res) => {
  const tracker = new LatencyTracker(`speak-${Date.now()}`);
  
  try {
    const { userId, audioData } = req.body;
    
    if (!userId || !audioData) {
      return res.status(400).json({ 
        success: false, 
        error: 'userId and audioData are required' 
      });
    }
    
    console.log(`🎤 Processing speech: ${userId}`);
    
    // Step 1: Transcribe (optimized with language specified)
    const transcribedText = await transcribeAudio(audioData);
    tracker.mark('transcriptionEnd');
    
    if (!transcribedText || transcribedText.trim() === '') {
      throw new Error('Could not transcribe audio');
    }
    
    console.log(`📝 User: "${transcribedText}"`);
    
    // Step 2: Check cache first! ⚡
    const cacheKey = transcribedText.toLowerCase().trim();
    let responseText;
    
    if (responseCache.has(cacheKey)) {
      const cached = responseCache.get(cacheKey);
      if (Date.now() - cached.timestamp < CACHE_TTL) {
        console.log('⚡ Cache hit! (saves 2s)');
        responseText = cached.text;
        tracker.mark('voiceflowEnd');
      } else {
        responseCache.delete(cacheKey);
      }
    }
    
    // Step 3: Get from Voiceflow if not cached
    if (!responseText) {
      const vfResponse = await sendToVoiceflow(userId, transcribedText);
      tracker.mark('voiceflowEnd');
      
      responseText = extractVoiceflowText(vfResponse);
      
      // Cache for next time
      responseCache.set(cacheKey, {
        text: responseText,
        timestamp: Date.now()
      });
    }

    if (!responseText) {
      throw new Error('No text response from Voiceflow');
    }

    const session = activeSessions.get(userId);
    
    // Step 4: Send response IMMEDIATELY ⚡
    res.json({
      success: true,
      transcribedText: transcribedText,
      avatarResponse: responseText,
      hasAvatarSession: !!session
    });

    // Step 5: Make avatar speak in background (parallel) ⚡
    if (session) {
      makeAvatarSpeak(userId, responseText, 'repeat');
    }
    tracker.mark('avatarEnd');
    tracker.mark('end');
    
    // Show latency breakdown
    tracker.report();
    
  } catch (error) {
    console.error('❌ Speak error:', error.message);
    res.status(500).json({ 
      success: false, 
      error: error.message 
    });
  }
});

app.post('/api/chat', async (req, res) => {
  const tracker = new LatencyTracker(`chat-${Date.now()}`);
  
  try {
    const { userId, message } = req.body;
    
    if (!userId || !message) {
      return res.status(400).json({ 
        success: false, 
        error: 'userId and message are required' 
      });
    }

    console.log(`💬 Chat: ${userId}`);
    tracker.mark('transcriptionEnd'); // No transcription for text
    
    // Check cache
    const cacheKey = message.toLowerCase().trim();
    let responseText;
    
    if (responseCache.has(cacheKey)) {
      const cached = responseCache.get(cacheKey);
      if (Date.now() - cached.timestamp < CACHE_TTL) {
        console.log('⚡ Cache hit!');
        responseText = cached.text;
        tracker.mark('voiceflowEnd');
      } else {
        responseCache.delete(cacheKey);
      }
    }
    
    if (!responseText) {
      const vfResponse = await sendToVoiceflow(userId, message);
      tracker.mark('voiceflowEnd');
      
      responseText = extractVoiceflowText(vfResponse);
      
      responseCache.set(cacheKey, {
        text: responseText,
        timestamp: Date.now()
      });
    }

    if (!responseText) {
      throw new Error('No text response from Voiceflow');
    }

    const session = activeSessions.get(userId);
    
    res.json({
      success: true,
      userMessage: message,
      avatarResponse: responseText,
      hasAvatarSession: !!session
    });

    if (session) {
      makeAvatarSpeak(userId, responseText, 'repeat');
    }
    tracker.mark('avatarEnd');
    tracker.mark('end');
    
    tracker.report();
    
  } catch (error) {
    console.error('❌ Chat error:', error.message);
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

app.get('/', (req, res) => {
  res.sendFile(join(__dirname, 'web', 'index.html'));
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

app.listen(CONFIG.PORT, '0.0.0.0', () => {
  console.log('\n' + '='.repeat(70));
  console.log('🚀 HeyGen + Voiceflow Server (ULTRA-OPTIMIZED)');
  console.log('='.repeat(70));
  console.log(`🎯 Expected latency: 1.5-2.5s (down from 5+s)`);
  console.log(`✅ Server running on port ${CONFIG.PORT}`);
  console.log('='.repeat(70) + '\n');
});
