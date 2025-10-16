class SmartRouter {
  constructor() {
    // ⚡ OPTIMIZATION 1: Pre-compile regex patterns for speed
    this.compiledPatterns = {
      simpleGreetings: /^(hi|hello|hey|thanks|thank you|yes|no|ok|okay|sure|tell me more|continue|go on|what else)$/i,
      questionStarters: /^(what|how|which|tell me|can|does|is|are)/i
    };
    
    // ⚡ OPTIMIZATION 2: Flatten routing rules for O(1) lookup
    this.keywordRouteMap = new Map();
    
    // Build keyword map from routing rules
    const routingRules = {
      "specs_queries": {
        match: ['spec', 'specification'],
        route: { document_type: 'family-guide', category: 'specs' }
      },
      "performance_queries": {
        match: ['performance', 'benchmark', 'speed'],
        route: { document_type: 'benchmark-results', category: 'performance' }
      },
      "how_to_queries": {
        match: ['how to', 'manage', 'management', 'solution'],
        route: { document_type: 'solution-brief', category: 'management' }
      },
      "customer_proof": {
        match: ['customer', 'case study', 'reference', 'success stories', 'success story', 'customer success'],
        route: { document_type: 'customer-case-study' }
      },
      "latest_info": {
        match: ['latest', 'recent', 'update', 'news'],
        route: { document_id: 'web-scraped-content' }
      },
      "virtualization": {
        match: ['vmware', 'kvm', 'virtualization', 'hypervisor', 'vmware alternative'],
        route: {
          $or: [
            { category: 'virtualization' },
            { topics: { $in: ['virtualization', 'vmware-alternative'] } }
          ]
        }
      },
      "ai_inference": {
        match: ['ai inference', 'inference', 'ml ai', 'model benchmark'],
        route: { topics: 'ai-inference', document_type: 'benchmark-results' }
      }
    };
    
    // ⚡ OPTIMIZATION 3: Create sorted keyword map (longest first for multi-word matching)
    for (const [ruleName, rule] of Object.entries(routingRules)) {
      for (const keyword of rule.match) {
        this.keywordRouteMap.set(keyword.toLowerCase(), {
          ...rule.route,
          ruleName
        });
      }
    }
    
    // ⚡ OPTIMIZATION 4: Pre-sorted keywords by length (longest first)
    this.sortedKeywords = Array.from(this.keywordRouteMap.keys())
      .sort((a, b) => b.length - a.length);
    
    // ⚡ OPTIMIZATION 5: Product mapping with regex for faster matching
    this.productPatterns = [
      { pattern: /\bdl\s?380a?\b/i, product: 'DL380' },
      { pattern: /\bdl\s?384\b/i, product: 'DL384' },
      { pattern: /\bdl\s?360\b/i, product: 'DL360' },
      { pattern: /\bdl\s?20\b/i, product: 'DL20' },
      { pattern: /\bml\s?350\b/i, product: 'ML350' },
      { pattern: /\bml\s?30\b/i, product: 'ML30' },
      { pattern: /\bdl\s?580\b/i, product: 'DL580' }
    ];
    
    // ⚡ OPTIMIZATION 6: Technical keywords as Set for O(1) lookup
    this.kbKeywordsSet = new Set([
      'processor', 'memory', 'storage', 'ilo', 'com', 'feature', 'price', 'cost',
      'compare', 'comparison', 'difference', 'generation', 'gen11', 'gen12',
      'sku', 'part number', 'model', 'configuration',
      'power', 'cooling', 'rack', 'dimensions',
      'warranty', 'support', 'service', 'hpe', 'proliant',
      'dl380', 'dl360', 'dl20', 'ml350', 'ml30', 'dl384', 'dl580',
      'success', 'stories', 'customer'
    ]);
    
    // ⚡ OPTIMIZATION 7: Category patterns pre-compiled
    this.categoryPatterns = [
      { pattern: /\b(price|cost|pricing)\b/i, category: 'pricing' },
      { pattern: /\b(spec|specification)\b/i, category: 'specs' },
      { pattern: /\b(performance|benchmark)\b/i, category: 'performance' },
      { pattern: /\b(virtualization|vmware|kvm|hypervisor)\b/i, category: 'virtualization' },
      { pattern: /\bmanagement\b/i, category: 'management' },
      { pattern: /\bcase study\b/i, category: 'customer-case-study' },
      { pattern: /\b(ai inference|model benchmark)\b/i, category: 'ai-inference' }
    ];

    this.categorySynonyms = {
      pricing: ['pricing', 'price', 'cost', 'tco', 'total cost', 'roi'],
      specs: ['specs', 'specifications', 'technical specs', 'datasheet', 'configuration'],
      performance: ['performance', 'benchmark', 'speed', 'throughput', 'latency'],
      virtualization: ['virtualization', 'vmware', 'vsphere', 'hypervisor', 'vmware alternative', 'kvm', 'virtual machine'],
      management: ['management', 'oneview', 'ilo', 'compute ops', 'ops management', 'remote management'],
      'customer-case-study': ['customer case', 'case study', 'success story', 'reference', 'customer proof'],
      'ai-inference': ['ai inference', 'ai', 'ml', 'machine learning', 'inference', 'gpu acceleration']
    };
    
    // ⚡ OPTIMIZATION 8: Question patterns pre-compiled
    this.questionPatterns = [
      /what (is|are|can|does|do)/i,
      /how (many|much|does|do|can)/i,
      /which (one|model|server)/i,
      /tell me about/i,
      /can (you|it|they)/i,
      /does (it|this|that)/i
    ];
    
    // ⚡ OPTIMIZATION 9: Cache for repeated queries
    this.classificationCache = new Map();
    this.filterCache = new Map();
    this.CACHE_SIZE = 100;
  }
  
  /**
   * ⚡ OPTIMIZED: Classify question with caching and fast path detection
   */
  classifyQuestion(text) {
    const startTime = Date.now();
    
    // Normalize input once
    const lowerText = text.toLowerCase().trim();
    const textLength = lowerText.length;
    
    // ⚡ Check cache first
    if (this.classificationCache.has(lowerText)) {
      console.log(`⚡ Classification cache hit! (0ms)`);
      return this.classificationCache.get(lowerText);
    }
    
    let result;
    
    // ⚡ FAST PATH 1: Very simple greetings/acknowledgments only (much more restrictive)
    if (textLength < 8 || /^(hi|hello|hey|yes|no|ok|thanks)$/i.test(lowerText)) {
      result = {
        type: 'simple_conversation',
        confidence: 0.95,
        reason: 'Very short greeting'
      };
      this._cacheResult(lowerText, result);
      console.log(`⚡ Classification: ${result.type} (${Date.now() - startTime}ms)`);
      return result;
    }
    
    // ⚡ FAST PATH 2: Rule-based routing (sorted by keyword length)
    for (const keyword of this.sortedKeywords) {
      if (lowerText.includes(keyword)) {
        const route = this.keywordRouteMap.get(keyword);
        result = {
          type: 'route',
          route: route,
          confidence: 0.96,
          reason: `Matched: ${route.ruleName}`
        };
        this._cacheResult(lowerText, result);
        console.log(`⚡ Classification: ${result.type} (${Date.now() - startTime}ms)`);
        return result;
      }
    }
    
    // ⚡ FAST PATH 3: Technical keywords (Set lookup is O(1))
    const words = lowerText.split(/\s+/);
    for (const word of words) {
      if (this.kbKeywordsSet.has(word)) {
        result = {
          type: 'knowledge_base',
          confidence: 0.9,
          reason: 'Contains technical/product keywords'
        };
        this._cacheResult(lowerText, result);
        console.log(`⚡ Classification: ${result.type} (${Date.now() - startTime}ms)`);
        return result;
      }
    }
    
    // ⚡ NEW: Question detection (more permissive - route questions through intelligent system)
    if (textLength > 8) {
      for (const pattern of this.questionPatterns) {
        if (pattern.test(lowerText)) {
          result = {
            type: 'intelligent_response',
            confidence: 0.8,
            reason: 'Question that needs intelligent response'
          };
          this._cacheResult(lowerText, result);
          console.log(`⚡ Classification: ${result.type} (${Date.now() - startTime}ms)`);
          return result;
        }
      }
      
      // If it contains "tell me about", "what about", etc. - also route intelligently
      if (/tell me|what about|explain|describe|i want|i need|help me|can you|problem with|issue with|sales in/i.test(lowerText)) {
        result = {
          type: 'intelligent_response',
          confidence: 0.75,
          reason: 'Requires intelligent response'
        };
        this._cacheResult(lowerText, result);
        console.log(`⚡ Classification: ${result.type} (${Date.now() - startTime}ms)`);
        return result;
      }
    }
    
    // ⚡ MORE PERMISSIVE: Any text longer than 15 characters gets intelligent handling
    if (textLength > 15) {
      result = {
        type: 'intelligent_response',
        confidence: 0.65,
        reason: 'Extended text needs intelligent handling'
      };
      this._cacheResult(lowerText, result);
      console.log(`⚡ Classification: ${result.type} (${Date.now() - startTime}ms)`);
      return result;
    }
    
    // Default: simple conversation (only for very short texts now)
    result = {
      type: 'simple_conversation',
      confidence: 0.6,
      reason: 'Simple conversation'
    };
    
    this._cacheResult(lowerText, result);
    console.log(`⚡ Classification: ${result.type} (${Date.now() - startTime}ms)`);
    return result;
  }
  
  /**
   * ⚡ OPTIMIZED: Extract product filter with caching
   */
  extractProductFilter(text) {
    const lowerText = text.toLowerCase();
    
    // ⚡ Check cache first
    if (this.filterCache.has(lowerText)) {
      console.log('⚡ Filter cache hit!');
      return this.filterCache.get(lowerText);
    }
    
    let filter = null;
    
    // 🚫 DISABLE PRODUCT FILTERS FOR CASE STUDIES/SUCCESS STORIES
    // These queries should search ALL documents, not filter by specific product
    // because case study documents have product: "all" in metadata
    const caseStudyKeywords = /case stud|success stor|customer success|customer stor|customer case|customer proof|who uses/i;
    if (caseStudyKeywords.test(lowerText)) {
      console.log('⚡ No filter (case study/success story query)');
      this._cacheFilter(lowerText, null);
      return null;
    }
    
    // ⚡ FAST PATH 1: Product detection (but use OR filter to include 'all' products)
    for (const { pattern, product } of this.productPatterns) {
      if (pattern.test(lowerText)) {
        filter = { 
          $or: [
            { product: product },
            { product: 'all' },
            { referenced_products: { $in: [product] } }
          ]
        };
        this._cacheFilter(lowerText, filter);
        return filter;
      }
    }
    
    // ⚡ FAST PATH 2: Category detection (pre-compiled patterns)
    for (const { pattern, category } of this.categoryPatterns) {
      if (pattern.test(lowerText)) {
        filter = this.buildFlexibleCategoryFilter(category);
        this._cacheFilter(lowerText, filter);
        return filter;
      }
    }

    for (const [category, synonyms] of Object.entries(this.categorySynonyms)) {
      if (synonyms.some(term => this.textContainsSynonym(lowerText, term))) {
        filter = this.buildFlexibleCategoryFilter(category);
        this._cacheFilter(lowerText, filter);
        return filter;
      }
    }
    
    // No filter found
    this._cacheFilter(lowerText, null);
    return null;
  }
  
  /**
   * ⚡ OPTIMIZED: Fast decision on RAG usage  
   */
  shouldUseRAG(classification) {
    // Direct property access (faster than multiple conditions)
    return (
      classification.type === 'route' || 
      (classification.type === 'knowledge_base' && classification.confidence > 0.65)
    );
  }

  /**
   * ⚡ NEW: Determine if question needs intelligent response (RAG or OpenAI direct)
   */
  shouldUseIntelligentResponse(classification) {
    return (
      classification.type === 'route' ||
      classification.type === 'knowledge_base' ||
      classification.type === 'intelligent_response'
    );
  }

  normalizeTerm(term) {
    return typeof term === 'string' ? term.trim().toLowerCase() : '';
  }

  textContainsSynonym(text, term) {
    const normalized = this.normalizeTerm(term);
    if (!normalized) return false;
    if (normalized.includes(' ')) {
      return text.includes(normalized);
    }
    const escaped = normalized.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const regex = new RegExp(`\\b${escaped}\\b`);
    return regex.test(text);
  }

  buildFlexibleCategoryFilter(categoryKeyword) {
    const normalizedCategory = this.normalizeTerm(categoryKeyword);
    if (!normalizedCategory) return null;

    const synonyms = new Set([normalizedCategory]);
    const extra = this.categorySynonyms[normalizedCategory] || [];
    for (const syn of extra) {
      const term = this.normalizeTerm(syn);
      if (term) synonyms.add(term);
    }

    const clauses = [];
    const added = new Set();
    const pushClause = (clause) => {
      const key = JSON.stringify(clause);
      if (!added.has(key)) {
        added.add(key);
        clauses.push(clause);
      }
    };

    for (const term of synonyms) {
      if (!term) continue;
      pushClause({ category: term });
      pushClause({ document_type: term });
      pushClause({ document_id: term });
      pushClause({ topics: { $in: [term] } });
      pushClause({ referenced_products: { $in: [term] } });
      pushClause({ key_features: { $in: [term] } });
      pushClause({ use_cases: { $in: [term] } });
      pushClause({ search_keywords: { $in: [term] } });
      pushClause({ tags: { $in: [term] } });
    }

    return clauses.length ? { $or: clauses } : null;
  }
  
  /**
   * ⚡ OPTIMIZATION: Cache helper with size limit
   */
  _cacheResult(key, result) {
    if (this.classificationCache.size >= this.CACHE_SIZE) {
      // Remove oldest entry (first key)
      const firstKey = this.classificationCache.keys().next().value;
      this.classificationCache.delete(firstKey);
    }
    this.classificationCache.set(key, result);
  }
  
  /**
   * ⚡ OPTIMIZATION: Cache helper for filters
   */
  _cacheFilter(key, filter) {
    if (this.filterCache.size >= this.CACHE_SIZE) {
      const firstKey = this.filterCache.keys().next().value;
      this.filterCache.delete(firstKey);
    }
    this.filterCache.set(key, filter);
  }
  
  /**
   * ⚡ NEW: Batch classification for multiple queries
   */
  batchClassify(texts) {
    return texts.map(text => this.classifyQuestion(text));
  }
  
  /**
   * ⚡ NEW: Clear caches (useful for testing or memory management)
   */
  clearCache() {
    this.classificationCache.clear();
    this.filterCache.clear();
    console.log('🧹 SmartRouter caches cleared');
  }
  
  /**
   * ⚡ NEW: Get cache statistics
   */
  getCacheStats() {
    return {
      classificationCacheSize: this.classificationCache.size,
      filterCacheSize: this.filterCache.size,
      totalKeywords: this.sortedKeywords.length,
      productPatterns: this.productPatterns.length,
      categoryPatterns: this.categoryPatterns.length
    };
  }
  
  /**
   * ⚡ NEW: Pre-warm cache with common queries
   */
  prewarmCache(commonQueries = []) {
    console.log('🔥 Pre-warming SmartRouter cache...');
    const defaultQueries = [
      'What are the specs of DL380 Gen12?',
      'Tell me about DL360 performance',
      'How to configure HPE iLO?',
      'Compare DL380 vs DL384',
      'VMware alternative virtualization',
      'AI inference benchmarks',
      'Customer case studies',
      'Latest updates',
      'Hello',
      'Thanks'
    ];
    
    const queriesToWarm = commonQueries.length > 0 ? commonQueries : defaultQueries;
    
    for (const query of queriesToWarm) {
      this.classifyQuestion(query);
      this.extractProductFilter(query);
    }
    
    console.log(`✅ Pre-warmed cache with ${queriesToWarm.length} queries`);
  }
}

export default SmartRouter;