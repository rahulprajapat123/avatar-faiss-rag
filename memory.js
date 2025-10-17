/**
 * Conversation Memory Manager
 * Stores and retrieves conversation history for each user
 * Can be upgraded to use Redis for production
 */

class ConversationManager {
  constructor() {
    // In-memory storage: userId -> conversation history
    this.conversations = new Map();
    this.MAX_HISTORY = 10; // Keep last 10 exchanges
    this.CONTEXT_TTL = 30 * 60 * 1000; // 30 minutes
  }

  /**
   * Add a message to conversation history
   */
  addMessage(userId, role, content) {
    if (!this.conversations.has(userId)) {
      this.conversations.set(userId, {
        messages: [],
        metadata: {},
        lastUpdated: Date.now()
      });
    }

    const conversation = this.conversations.get(userId);
    conversation.messages.push({
      role, // 'user' or 'assistant'
      content,
      timestamp: Date.now()
    });

    // Keep only recent messages
    if (conversation.messages.length > this.MAX_HISTORY * 2) {
      conversation.messages = conversation.messages.slice(-this.MAX_HISTORY * 2);
    }

    conversation.lastUpdated = Date.now();
    
    console.log(`💬 [${userId}] Added ${role} message to history (${conversation.messages.length} total)`);
  }

  /**
   * Get conversation history for a user
   */
  getHistory(userId) {
    const conversation = this.conversations.get(userId);
    
    if (!conversation) {
      return [];
    }

    // Check if conversation is still valid (not expired)
    if (Date.now() - conversation.lastUpdated > this.CONTEXT_TTL) {
      console.log(`⏰ [${userId}] Conversation expired, clearing history`);
      this.conversations.delete(userId);
      return [];
    }

    return conversation.messages;
  }

  /**
   * Get formatted conversation history for system prompt
   */
  getFormattedHistory(userId, maxMessages = 6) {
    const history = this.getHistory(userId);
    
    if (history.length === 0) {
      return '';
    }

    // Get recent messages
    const recentMessages = history.slice(-maxMessages);
    
    const formatted = recentMessages.map(msg => {
      const role = msg.role === 'user' ? 'User' : 'You (SDR)';
      return `${role}: ${msg.content}`;
    }).join('\n');

    return `\n\nPREVIOUS CONVERSATION:\n${formatted}\n\nIMPORTANT: Use the information from the conversation above. DO NOT re-ask questions that were already answered!`;
  }

  /**
   * Extract discovered facts from conversation
   */
  extractDiscoveredFacts(userId) {
    const history = this.getHistory(userId);
    const facts = {
      role: null,
      industry: null,
      users: null,
      workloads: [],
      needs: [],
      drivers: [],
      virtualization: null,
      timeline: null,
      ha: null,
      location: null
    };

    // Analyze conversation to extract facts
    const allText = history
      .filter(msg => msg.role === 'user')
      .map(msg => msg.content.toLowerCase())
      .join(' ');

    // Extract role/industry
    if (allText.includes('cto')) facts.role = 'CTO';
    if (allText.includes('technical lead')) facts.role = 'Technical Lead';
    if (allText.includes('bank')) facts.industry = 'banking';
    if (allText.includes('healthcare')) facts.industry = 'healthcare';
    if (allText.includes('retail')) facts.industry = 'retail';

    // Extract user scale
    const userMatch = allText.match(/(\d+)\s*(users|concurrent)/i);
    if (userMatch) facts.users = parseInt(userMatch[1]);
    
    if (allText.includes('million')) {
      const millionMatch = allText.match(/(\d+)\s*million/i);
      if (millionMatch) facts.users = parseInt(millionMatch[1]) * 1000000;
    }

    // Extract workloads
    if (allText.includes('analytics')) facts.workloads.push('analytics');
    if (allText.includes('virtualization') || allText.includes('vm')) facts.workloads.push('virtualization');
    if (allText.includes('database')) facts.workloads.push('database');
    if (allText.includes('ai') || allText.includes('ml')) facts.workloads.push('ai');
    if (allText.includes('security')) facts.workloads.push('security');
    if (allText.includes('personalization')) facts.workloads.push('personalization');

    // Extract needs
    if (allText.includes('performance') || allText.includes('speed')) facts.needs.push('performance');
    if (allText.includes('scalability') || allText.includes('scale')) facts.needs.push('scalability');
    if (allText.includes('availability') || allText.includes('ha')) facts.needs.push('high availability');
    if (allText.includes('security') || allText.includes('secure')) facts.needs.push('security');
    if (allText.includes('energy') || allText.includes('power')) facts.needs.push('energy efficiency');
    if (allText.includes('latency')) facts.needs.push('low latency');

    // Extract drivers
    if (allText.includes('aging') || allText.includes('old')) facts.drivers.push('aging infrastructure');
    if (allText.includes('expand') || allText.includes('growth')) facts.drivers.push('expansion');
    if (allText.includes('compliance')) facts.drivers.push('compliance');

    // Extract virtualization
    if (allText.includes('vmware')) facts.virtualization = 'VMware';
    if (allText.includes('kvm')) facts.virtualization = 'KVM';
    if (allText.includes('hyper-v')) facts.virtualization = 'Hyper-V';

    // Extract timeline
    if (allText.includes('now') || allText.includes('immediate')) facts.timeline = 'immediate';
    if (allText.includes('q1') || allText.includes('q2')) facts.timeline = allText.match(/q[1-4]/i)?.[0];

    return facts;
  }

  /**
   * Check if we have enough information to make a recommendation
   */
  canRecommend(userId) {
    const facts = this.extractDiscoveredFacts(userId);
    
    // Need at least: users + workloads + one driver/need
    const hasUsers = facts.users !== null;
    const hasWorkloads = facts.workloads.length > 0;
    const hasDriverOrNeed = facts.drivers.length > 0 || facts.needs.length > 0;
    
    const ready = hasUsers && hasWorkloads && hasDriverOrNeed;
    
    if (ready) {
      console.log(`✅ [${userId}] Ready to recommend! Facts:`, facts);
    } else {
      console.log(`⏳ [${userId}] Still discovering. Facts so far:`, facts);
    }
    
    return ready;
  }

  /**
   * Get recommendation prompt based on discovered facts
   */
  getRecommendationPrompt(userId) {
    const facts = this.extractDiscoveredFacts(userId);
    
    let prompt = `Based on the conversation, the customer needs:\n`;
    
    if (facts.role) prompt += `- Role: ${facts.role}\n`;
    if (facts.industry) prompt += `- Industry: ${facts.industry}\n`;
    if (facts.users) prompt += `- Users: ${facts.users}\n`;
    if (facts.workloads.length) prompt += `- Workloads: ${facts.workloads.join(', ')}\n`;
    if (facts.needs.length) prompt += `- Needs: ${facts.needs.join(', ')}\n`;
    if (facts.drivers.length) prompt += `- Drivers: ${facts.drivers.join(', ')}\n`;
    
    prompt += `\nProvide a SERVER RECOMMENDATION following the master prompt format:\n`;
    prompt += `1. Mini-recap (one sentence)\n`;
    prompt += `2. Recommended ProLiant family (DL380/DL360/ML350/etc) with WHY it fits\n`;
    prompt += `3. Mention 1-2 APAC Smart Choice SKUs\n`;
    prompt += `4. Keep response 80-110 words\n`;
    
    return prompt;
  }

  /**
   * Clear conversation for a user
   */
  clearConversation(userId) {
    this.conversations.delete(userId);
    console.log(`🧹 [${userId}] Conversation cleared`);
  }

  /**
   * Get all active conversations
   */
  getActiveConversations() {
    return Array.from(this.conversations.keys());
  }

  /**
   * Cleanup expired conversations (call periodically)
   */
  cleanup() {
    const now = Date.now();
    let cleaned = 0;
    
    for (const [userId, conversation] of this.conversations.entries()) {
      if (now - conversation.lastUpdated > this.CONTEXT_TTL) {
        this.conversations.delete(userId);
        cleaned++;
      }
    }
    
    if (cleaned > 0) {
      console.log(`🧹 Cleaned up ${cleaned} expired conversations`);
    }
  }
}

export default ConversationManager;
