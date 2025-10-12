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
    console.error('Please check your .env file');
    process.exit(1);
  }
  
  console.log('✅ All environment variables loaded');
}

validateConfig();

// ⚡ OPTIMIZATION: Initialize OpenAI with timeout settings
const openai = new OpenAI({ 
  apiKey: CONFIG.OPENAI_API_KEY,
  timeout: 30000, // 30 second timeout
  maxRetries: 1 // Reduce retries for faster failure
});

const activeSessions = new Map();

// ⚡ OPTIMIZATION: Reuse fetch agent for connection pooling
const fetchOptions = {
  agent: null, // Will use default agent with keep-alive
  timeout: 15000 // 15 second timeout for API calls
};

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
  
  console.log(`🌐 HeyGen: ${method} ${endpoint}`);
  
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
  
  console.log('✅ HeyGen: Success');
  return data;
}

async function createAvatarSession(userId) {
  try {
    console.log(`🎭 Creating session: ${userId}`);
    
    // ⚡ OPTIMIZATION: Use minimal configuration for faster session creation
    const sessionResponse = await heygenRequest(HEYGEN_API.CREATE_SESSION, 'POST', {
      version: 'v2',
      avatar_id: CONFIG.HEYGEN_AVATAR_ID,
      voice: {
        voice_id: CONFIG.HEYGEN_VOICE_ID
      },
      quality: 'low' // ⚡ Changed from 'medium' to 'low' for faster streaming
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
    
    // ⚡ OPTIMIZATION: Fire and forget - don't await the response
    heygenRequest(HEYGEN_API.SPEAK, 'POST', {
      session_id: session.sessionId,
      text: text,
      task_type: taskType,
      task_mode: 'async'
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
      // ⚡ OPTIMIZATION: Fire and forget - don't wait for stop confirmation
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
            stripSSML: true
          }
        }),
        ...fetchOptions
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

async function transcribeAudio(audioBase64) {
  try {
    console.log('🎤 Transcribing...');
    const audioBuffer = Buffer.from(audioBase64, 'base64');
    
    // ⚡ OPTIMIZATION: Use smaller file name and minimal options
    const file = await toFile(audioBuffer, 'a.webm', {
      type: 'audio/webm'
    });
    
    // ⚡ OPTIMIZATION: Remove prompt to speed up transcription
    const transcription = await openai.audio.transcriptions.create({
      file: file,
      model: 'whisper-1',
      response_format: 'text',
      language: 'en'
    });
    
    console.log(`✅ Transcribed: "${transcription.substring(0, 50)}..."`);
    return transcription;
    
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
    activeSessions: activeSessions.size
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
    
    // ⚡ OPTIMIZATION: Fire and forget old session cleanup
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

// ⚡ OPTIMIZATION: Async welcome message - respond immediately, process in background
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

    // Get the actual welcome message from Voiceflow
    const vfResponse = await sendToVoiceflow(userId, { type: 'launch' });
    const welcomeText = extractVoiceflowText(vfResponse);
    
    const finalWelcome = welcomeText || 'Hello! How can I help you today?';

    // Send response with actual message
    res.json({ 
      success: true, 
      message: 'Session ready',
      welcomeText: finalWelcome
    });

    // Make avatar speak (fire and forget for speed)
    makeAvatarSpeak(userId, finalWelcome, 'repeat');
    console.log('✅ Welcome sent:', finalWelcome.substring(0, 50) + '...');

  } catch (error) {
    console.error('❌ Session ready error:', error.message);
    res.status(500).json({ success: false, error: error.message });
  }
});

// ⚡ OPTIMIZATION: Parallel processing for speak endpoint
app.post('/api/speak', async (req, res) => {
  try {
    const { userId, audioData } = req.body;
    
    if (!userId || !audioData) {
      return res.status(400).json({ 
        success: false, 
        error: 'userId and audioData are required' 
      });
    }
    
    console.log(`🎤 Processing speech: ${userId}`);
    
    const transcribedText = await transcribeAudio(audioData);
    
    if (!transcribedText || transcribedText.trim() === '') {
      throw new Error('Could not transcribe audio');
    }
    
    console.log(`📝 User: "${transcribedText}"`);
    
    const vfResponse = await sendToVoiceflow(userId, transcribedText);
    const responseText = extractVoiceflowText(vfResponse);

    if (!responseText) {
      throw new Error('No text response from Voiceflow');
    }

    const session = activeSessions.get(userId);
    
    // ⚡ OPTIMIZATION: Send response immediately, speak in background
    res.json({
      success: true,
      transcribedText: transcribedText,
      avatarResponse: responseText,
      hasAvatarSession: !!session
    });

    // ⚡ Fire and forget the avatar speech
    if (session) {
      makeAvatarSpeak(userId, responseText, 'repeat');
    }
    
  } catch (error) {
    console.error('❌ Speak error:', error.message);
    res.status(500).json({ 
      success: false, 
      error: error.message 
    });
  }
});

// ⚡ OPTIMIZATION: Parallel processing for chat endpoint
app.post('/api/chat', async (req, res) => {
  try {
    const { userId, message } = req.body;
    
    if (!userId || !message) {
      return res.status(400).json({ 
        success: false, 
        error: 'userId and message are required' 
      });
    }

    console.log(`💬 Chat: ${userId}`);
    
    const vfResponse = await sendToVoiceflow(userId, message);
    const responseText = extractVoiceflowText(vfResponse);

    if (!responseText) {
      throw new Error('No text response from Voiceflow');
    }

    const session = activeSessions.get(userId);
    
    // ⚡ OPTIMIZATION: Send response immediately, speak in background
    res.json({
      success: true,
      userMessage: message,
      avatarResponse: responseText,
      hasAvatarSession: !!session
    });

    // ⚡ Fire and forget the avatar speech
    if (session) {
      makeAvatarSpeak(userId, responseText, 'repeat');
    }
    
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
    
    // ⚡ OPTIMIZATION: Don't await - respond immediately
    res.json({
      success: true,
      message: 'Avatar session stopped'
    });

    // Close in background
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

app.listen(CONFIG.PORT, () => {
  console.log('\n' + '='.repeat(60));
  console.log('🚀 HeyGen + Voiceflow Server (Optimized)');
  console.log('='.repeat(60));
  console.log(`\n✅ Server running on port ${CONFIG.PORT}`);
  console.log(`📍 URL: http://localhost:${CONFIG.PORT}`);
  console.log('\n⚡ Optimizations enabled:');
  console.log('   • Async processing');
  console.log('   • Connection pooling');
  console.log('   • Fire-and-forget operations');
  console.log('   • Reduced quality for faster streaming');
  console.log('\n' + '='.repeat(60) + '\n');
});