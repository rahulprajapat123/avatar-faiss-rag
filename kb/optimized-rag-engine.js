import faiss from 'faiss-node';
import { pipeline } from '@xenova/transformers';
import OpenAI from 'openai';
import dotenv from 'dotenv';
import fs from 'fs';
import path from 'path';

const { IndexFlatIP } = faiss;

dotenv.config();

const DEFAULT_DIMENSION = 384;
const DEFAULT_CACHE_TTL = 5 * 60 * 1000;

function ensureCompleteSentence(text) {
  if (!text || text.trim() === '') return text;
  const trimmed = text.trim();
  if (/[.!?]$/.test(trimmed)) return trimmed;
  const lastBreak = Math.max(trimmed.lastIndexOf('.'), trimmed.lastIndexOf('!'), trimmed.lastIndexOf('?'));
  if (lastBreak > 0 && lastBreak > trimmed.length * 0.6) {
    return trimmed.substring(0, lastBreak + 1).trim();
  }
  return `${trimmed}.`;
}

function semanticSimilarity(a, b) {
  if (!a || !b) return 0;
  const clean = (value) => value
    .toLowerCase()
    .replace(/[\p{P}\p{S}]/gu, ' ')
    .split(/\s+/)
    .filter(Boolean);

  const wordsA = new Set(clean(a));
  const wordsB = new Set(clean(b));
  if (wordsA.size === 0 || wordsB.size === 0) return 0;

  let intersection = 0;
  for (const word of wordsA) {
    if (wordsB.has(word)) {
      intersection += 1;
    }
  }

  const union = wordsA.size + wordsB.size - intersection;
  return union === 0 ? 0 : intersection / union;
}

class OptimizedRAGEngine {
  constructor() {
    this.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
      timeout: Number(process.env.OPENAI_TIMEOUT || 5000),
      maxRetries: 0
    });

    this.responseCache = new Map();
    this.retrievalCache = new Map();
    this.embeddingCache = new Map();
    this.semanticEmbeddingCache = [];
    this.cacheStats = {
      embeddingExactHits: 0,
      embeddingSemanticHits: 0,
      retrievalHits: 0,
      responseHits: 0
    };

    this.CACHE_TTL = Number(process.env.RAG_CACHE_TTL || DEFAULT_CACHE_TTL);
    this.RETRIEVAL_CACHE_TTL = Number(process.env.RAG_RETRIEVAL_CACHE_TTL || this.CACHE_TTL);
  this.TEXT_SIMILARITY_THRESHOLD = Number(process.env.RAG_TEXT_SIMILARITY || 0.82);

    this.faissIndex = null;
    this.vectorMetadata = [];
    this.embeddingModel = null;

    this.dataDir = path.join(process.cwd(), 'kb', 'faiss-data');
    this.indexPath = path.join(this.dataDir, 'vectors.index');
    this.metadataPath = path.join(this.dataDir, 'metadata.json');
    this.indexConfigPath = path.join(this.dataDir, 'index-config.json');

    this.indexConfig = {
      type: 'flat',
      dimension: DEFAULT_DIMENSION,
      nprobe: Number(process.env.FAISS_NPROBE || 4)
    };

    this.isInitialized = false;
  }

  async initialize() {
    if (this.isInitialized) return;

    const start = Date.now();
    console.log('🚀 Initializing FAISS RAG engine');

    if (!fs.existsSync(this.indexPath) || !fs.existsSync(this.metadataPath)) {
      throw new Error('FAISS index not found. Please run "npm run build:faiss" to precompute embeddings.');
    }

    if (fs.existsSync(this.indexConfigPath)) {
      try {
        const cfg = JSON.parse(fs.readFileSync(this.indexConfigPath, 'utf8'));
        this.indexConfig = { ...this.indexConfig, ...cfg };
      } catch (error) {
        console.warn('⚠️  Failed to read index-config.json, using defaults:', error.message);
      }
    }

    console.log(`📂 Loading FAISS index (${this.indexConfig.type})`);
    this.faissIndex = IndexFlatIP.read(this.indexPath);
    this.vectorMetadata = JSON.parse(fs.readFileSync(this.metadataPath, 'utf8'));

    if (typeof this.faissIndex.setNprobe === 'function') {
      this.faissIndex.setNprobe(this.indexConfig.nprobe || 4);
      console.log(`🎯 FAISS nprobe set to ${this.indexConfig.nprobe || 4}`);
    } else {
      console.log('ℹ️  FAISS index does not expose nprobe tuning (flat index).');
    }

    console.log(`✅ Loaded ${this.vectorMetadata.length} vectors (dimension ${this.indexConfig.dimension})`);

    await this.loadEmbeddingModel();

    this.isInitialized = true;
    console.log(`⏱️  RAG initialization completed in ${Date.now() - start}ms`);
  }

  async loadEmbeddingModel() {
    if (this.embeddingModel) return;
    console.log('📦 Loading local embedding model (Xenova/all-MiniLM-L6-v2)...');
    this.embeddingModel = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
      quantized: process.env.TRANSFORMERS_QUANTIZED === 'true',
      progress_callback: null
    });
    console.log('✅ Embedding model ready');
  }

  getCacheKey(query, filter, options = {}) {
    return JSON.stringify({ query, filter, options });
  }

  pruneCache(map, ttl) {
    const now = Date.now();
    for (const [key, entry] of map.entries()) {
      if (now - entry.timestamp > ttl) {
        map.delete(key);
      }
    }
  }

  async getQueryEmbedding(query) {
    const normalized = query.toLowerCase().trim();

    if (this.embeddingCache.has(normalized)) {
      this.cacheStats.embeddingExactHits += 1;
      return this.embeddingCache.get(normalized);
    }

    for (const item of this.semanticEmbeddingCache) {
      if (item.key === normalized) {
        this.cacheStats.embeddingSemanticHits += 1;
        this.embeddingCache.set(normalized, item.vector);
        return item.vector;
      }
      const similarity = semanticSimilarity(normalized, item.key);
      if (similarity >= this.TEXT_SIMILARITY_THRESHOLD) {
        this.cacheStats.embeddingSemanticHits += 1;
        this.embeddingCache.set(normalized, item.vector);
        return item.vector;
      }
    }

    const vector = await this.computeEmbedding(query);
    this.embeddingCache.set(normalized, vector);
    this.semanticEmbeddingCache.push({ key: normalized, vector });

    if (this.semanticEmbeddingCache.length > 256) {
      this.semanticEmbeddingCache.shift();
    }

    return vector;
  }

  async computeEmbedding(text) {
    await this.loadEmbeddingModel();
    const result = await this.embeddingModel(text, { pooling: 'mean', normalize: true });
    return Array.from(result.data);
  }

  matchesFilter(metadata, filter) {
    if (!filter) return true;
    if (filter.$or) {
      return filter.$or.some(sub => this.matchesFilter(metadata, sub));
    }
    for (const [key, value] of Object.entries(filter)) {
      if (value && typeof value === 'object' && value.$in) {
        const actual = metadata[key] || [];
        if (Array.isArray(actual)) {
          if (!value.$in.some(v => actual.includes(v))) return false;
        } else if (!value.$in.includes(actual)) {
          return false;
        }
        continue;
      }
      if (metadata[key] !== value) {
        return false;
      }
    }
    return true;
  }

  async search(query, options = {}) {
    await this.initialize();

    const { topK = 3, filter = null, minScore = 0.45, searchK = topK * 3 } = options;
    const retrievalKey = this.getCacheKey(query, filter, { topK, minScore, searchK });

    this.pruneCache(this.retrievalCache, this.RETRIEVAL_CACHE_TTL);

    if (this.retrievalCache.has(retrievalKey)) {
      this.cacheStats.retrievalHits += 1;
      return this.retrievalCache.get(retrievalKey).results;
    }

    const timingStart = Date.now();
    const embedding = await this.getQueryEmbedding(query);
    const embedTime = Date.now() - timingStart;

    const searchStart = Date.now();
  const result = this.faissIndex.search(Array.from(embedding), Math.min(searchK, this.vectorMetadata.length));
    const searchTime = Date.now() - searchStart;

    const matches = [];
    for (let i = 0; i < result.labels.length; i++) {
      const id = result.labels[i];
      const score = result.distances[i];
      if (id < 0 || id >= this.vectorMetadata.length) continue;
      if (score < minScore) continue;
      const metadata = this.vectorMetadata[id];
      if (!this.matchesFilter(metadata, filter)) continue;
      matches.push({ id: metadata.id, score, metadata });
      if (matches.length >= topK) break;
    }

    console.log(`⚡ FAISS search done in ${Date.now() - timingStart}ms (embed ${embedTime}ms, search ${searchTime}ms) → ${matches.length} hits`);

    this.retrievalCache.set(retrievalKey, {
      results: matches,
      timestamp: Date.now()
    });

    return matches;
  }

  async generateAnswer(query, searchResults, options = {}) {
    if (!searchResults || searchResults.length === 0) {
      // SDR conversational fallback - start discovery instead of giving up
      const sdrResponse = this.generateSDRDiscoveryResponse(query);
      return {
        answer: sdrResponse,
        sources: [],
        confidence: 0.7, // Higher confidence to indicate we're handling it conversationally
        usedSDRFallback: true
      };
    }

    const { stream = false, onToken } = options;
    const context = searchResults
      .slice(0, 2)
      .map(result => result.metadata.text.substring(0, 400))
      .join('\n\n');

    const systemPrompt = `Master Prompt: HPE ProLiant SDR (Conversational Edition)

═══════════════════════════════════════════════════════════════════

1) ROLE & MISSION

You are a Senior SDR for HPE ProLiant Compute in APAC.

Your goal: Have a natural, consultative conversation to:
• Qualify using conversational BANT (not interrogation)
• Recommend ProLiant family + 1-2 APAC Smart Choice SKUs with clear "why"
• Arrange HPE expert callback

Boundaries:
• DO NOT discuss: Pricing, discounts, SLAs, contracts, non-ProLiant products
• When off-scope: Ask one clarifier OR offer expert callback

═══════════════════════════════════════════════════════════════════

2) CONVERSATION STYLE (CRITICAL!)

Professional & Consultative:
• Warm, human tone - you're helping, not selling
• 80-110 words per turn (concise and punchy, not verbose)
• Use context bridges: "Got it...", "That makes sense...", "I hear you..."
• ALWAYS acknowledge what they just told you before asking next question
• Build on their answers - don't ignore what they shared

ONE Question Per Turn:
• Ask ONE thematic question (may have 2 tightly related sub-parts)
• Let them guide the conversation naturally
• If they volunteer info, don't re-ask it!

Natural Flow:
• Listen → Acknowledge → Build → Ask
• Brief mini-recap before shifting topics
• No robotic lists or surveys
• Sound like a senior SDR, not a chatbot

Industry/Vertical Context:
• If user mentions industry (banking, healthcare, retail), acknowledge it
• Offer relevant insights: "Many banking customers prioritize [X]..."
• Reference common use cases for their vertical
• Examples:
  - Banking: Security (iLO Silicon Root of Trust), compliance, HA
  - Healthcare: Data security, uptime, HIPAA considerations
  - Retail: Seasonal scaling, analytics, PCI compliance
• Keep it brief (1 sentence) and natural, not forced

═══════════════════════════════════════════════════════════════════

3) DISCOVERY FLOW

Start with High-Level Context:
"So I can point you well—what are your main workloads (VMs, databases, analytics), and what's driving your server evaluation?"

**RECOMMENDATION TRIGGER:**
• Need ONLY 3 key facts to recommend:
  1. Users/Scale (how many users or workload size)  
  2. Workloads (VMs, analytics, databases, etc.)
  3. ONE of: Virtualization OR Timeline OR Location OR HA needs

• If you have 3 facts, give recommendation in NEXT response
• DON'T over-qualify - 3 facts = enough!

MINI-RECAP before recommending:
"So based on what you've shared: [users], [workloads], [key need]..."

═══════════════════════════════════════════════════════════════════

Use ONLY the context below to provide accurate information about HPE ProLiant servers.

Context:\n${context}`;

    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: query }
    ];

    if (stream && typeof onToken === 'function') {
      const streamResponse = await this.openai.chat.completions.create({
        model: process.env.RAG_COMPLETION_MODEL || 'gpt-3.5-turbo',
        messages,
        temperature: 0.3,
        max_tokens: 160,
        stream: true
      });

      let answer = '';
      for await (const chunk of streamResponse) {
        const delta = chunk.choices?.[0]?.delta?.content || '';
        if (!delta) continue;
        answer += delta;
        onToken({ token: delta, text: answer });
      }

      answer = ensureCompleteSentence(answer);

      return {
        answer,
        sources: searchResults.map(r => ({ source: r.metadata.source, score: r.score.toFixed(3) })),
        confidence: searchResults[0]?.score || 0
      };
    }

    const completion = await this.openai.chat.completions.create({
      model: process.env.RAG_COMPLETION_MODEL || 'gpt-3.5-turbo',
      messages,
      temperature: 0.3,
      max_tokens: 160,
      stream: false
    });

    const choice = completion?.choices?.[0]?.message?.content || '';
    const answer = ensureCompleteSentence(choice || "I could not assemble a confident answer this time.");

    return {
      answer,
      sources: searchResults.map(r => ({ source: r.metadata.source, score: r.score.toFixed(3) })),
      confidence: searchResults[0]?.score || 0
    };
  }

  async query(userQuery, options = {}) {
    const start = Date.now();
    const filter = options.filter || null;
    const cacheKey = this.getCacheKey(userQuery, filter, { topK: options.topK, minScore: options.minScore });

    this.pruneCache(this.responseCache, this.CACHE_TTL);

    if (this.responseCache.has(cacheKey)) {
      this.cacheStats.responseHits += 1;
      const cached = this.responseCache.get(cacheKey);
      return { ...cached, latency: Date.now() - start, cached: true };
    }

    let searchResults = [];
    const originalFilter = options.filter || null;

    try {
      searchResults = await this.search(userQuery, options);
    } catch (error) {
      console.error('❌ Primary FAISS search failed:', error.message);
    }

    if (!searchResults.length && originalFilter) {
      try {
        console.log('🔁 No FAISS matches with filter; retrying without filter');
        searchResults = await this.search(userQuery, { ...options, filter: null });
      } catch (error) {
        console.error('❌ Unfiltered FAISS retry failed:', error.message);
      }
    }

    if (!searchResults.length) {
      return {
        answer: "I don’t have that information right now. For more details, please contact the HPE support team or your account representative directly.",
        sources: [],
        confidence: 0,
        latency: Date.now() - start,
        noResults: true
      };
    }

    const answer = await this.generateAnswer(userQuery, searchResults, {
      stream: options.stream,
      onToken: options.onToken
    });

    const result = {
      ...answer,
      latency: Date.now() - start
    };

    this.responseCache.set(cacheKey, {
      ...answer,
      timestamp: Date.now()
    });

    return result;
  }

  clearCache() {
    this.responseCache.clear();
    this.retrievalCache.clear();
    this.embeddingCache.clear();
    this.semanticEmbeddingCache = [];
    console.log('🧹 Cleared RAG caches');
  }

  getCacheStats() {
    return {
      ...this.cacheStats,
      responseCacheSize: this.responseCache.size,
      retrievalCacheSize: this.retrievalCache.size,
      embeddingCacheSize: this.embeddingCache.size
    };
  }

  generateSDRDiscoveryResponse(query) {
    const lowerQuery = query.toLowerCase();
    
    // RECOMMENDATION TRIGGER - Check if user is asking for specific server recommendations
    if ((lowerQuery.includes('which server') || lowerQuery.includes('tell me about') || lowerQuery.includes('recommend') || 
         lowerQuery.includes('what server') || lowerQuery.includes('best server')) &&
        (lowerQuery.includes('virtualization') || lowerQuery.includes('scalability') || lowerQuery.includes('availability'))) {
      return this.generateServerRecommendation(lowerQuery);
    }
    
    // RECOMMENDATION TRIGGER - When they have shared enough context (users + workloads + needs)
    if ((lowerQuery.includes('million') || lowerQuery.includes('100 users') || lowerQuery.includes('25 to 100') || 
         lowerQuery.includes('developers') || lowerQuery.includes('operations')) &&
        (lowerQuery.includes('ai') || lowerQuery.includes('application') || lowerQuery.includes('saas') || 
         lowerQuery.includes('microservices') || lowerQuery.includes('database'))) {
      return this.generateServerRecommendation(lowerQuery);
    }
    
    // FOLLOW-UP after gathering info - User says "not yet" or asks for recommendations
    if (lowerQuery.includes('not yet') || lowerQuery.includes('can you not tell') || 
        lowerQuery.includes('should i use') || lowerQuery.includes('which servers')) {
      return this.generateServerRecommendation(lowerQuery);
    }
    
    // SDR Discovery - Banking CTO context
    if (lowerQuery.includes('bank') || lowerQuery.includes('cto')) {
      return "Got it—you're a bank CTO. That's a great context! Banking customers often prioritize security with iLO Silicon Root of Trust, compliance features, and high availability. So I can point you in the right direction—what's driving your server evaluation right now? Are you looking at replacing aging infrastructure, expanding capacity, or addressing specific compliance requirements?";
    }
    
    // General help request
    if (lowerQuery.includes('help') || lowerQuery.includes('assist')) {
      return "Absolutely! I'd be happy to help you find the right ProLiant server solution. To get started, could you share a bit about your environment? For example, how many users you're supporting, what workloads you're running (VMs, databases, analytics), and whether you're looking at on-prem, hybrid, or modernizing current infrastructure?";
    }
    
    // Performance/Speed requests
    if (lowerQuery.includes('speed') || lowerQuery.includes('performance') || lowerQuery.includes('memory') || lowerQuery.includes('scalable')) {
      return "That makes sense—speed, memory, and scalability are key priorities. Many customers see significant performance gains with Gen12 processors and DDR5 memory. To recommend the right configuration, what's your current environment like? Are you running virtualized workloads, and roughly how many users or VMs are we talking about?";
    }
    
    // Server identification requests
    if (lowerQuery.includes('server') || lowerQuery.includes('identify') || lowerQuery.includes('best') || lowerQuery.includes('need')) {
      return "Perfect! I can definitely help identify the best ProLiant server for your needs. To point you in the right direction, could you tell me about your workloads—are you running VMs, databases, analytics, or file services? And what's your current scale in terms of users or data volume?";
    }
    
    // Default SDR opening
    return "Thanks for reaching out! To help match you with the right ProLiant solution, may I ask a couple of quick questions? What type of workloads are you looking to support, and what's driving your server evaluation—growth, modernization, or replacing aging hardware?";
  }

  generateServerRecommendation(queryContext) {
    const lowerContext = queryContext.toLowerCase();
    
    // Determine scale and workload from context
    const isLargeScale = lowerContext.includes('million') || lowerContext.includes('large');
    const isMidScale = lowerContext.includes('100 users') || lowerContext.includes('25 to 100') || lowerContext.includes('mid-sized');
    const hasAI = lowerContext.includes('ai') || lowerContext.includes('machine learning');
    const hasVirtualization = lowerContext.includes('virtualization') || lowerContext.includes('vm');
    const hasSaaS = lowerContext.includes('saas') || lowerContext.includes('database') || lowerContext.includes('application');
    
    let recommendation;
    
    if (isLargeScale && hasAI) {
      recommendation = "DL384 Gen11 would be a strong starting point for you. For example, APAC Smart Choice configs like P84646-375 or P84648-375. Why this fits: Large-scale AI workloads with million+ users need DL384's GPU acceleration and 4U expansion capacity; iLO+COM streamline management across your distributed infrastructure.";
    } else if (isMidScale && hasAI) {
      recommendation = "DL380 Gen11 would be a strong starting point for you. For example, APAC Smart Choice configs like P84627-375 or P84628-375. Why this fits: AI development with microservices and SaaS platforms benefits from DL380's 2U expansion headroom; iLO and COM streamline DevOps automation and container orchestration.";
    } else if (hasVirtualization && (isMidScale || isLargeScale)) {
      recommendation = "DL380 Gen11 would be a strong starting point for you. For example, APAC Smart Choice configs like P84626-375 or P84627-375. Why this fits: Virtualization with growth plans benefits from DL380's memory capacity and 2U expandability; iLO+COM enable centralized VM management and automated scaling.";
    } else {
      recommendation = "DL360 Gen11 would be a strong starting point for you. For example, APAC Smart Choice configs like P84654-375 or P84658-375. Why this fits: Mixed workloads with virtualization needs benefit from DL360's 1U density and performance; iLO+COM streamline policy management and updates.";
    }
    
    return `Perfect! Based on what you've shared—your scale, AI/virtualization needs, and growth plans—${recommendation} We'll right-size cores, RAM, and storage with a human expert to nail the exact config. I can arrange a callback with a local HPE expert who can dive deeper. May I capture your details for follow-up?`;
  }
}

export default OptimizedRAGEngine;
