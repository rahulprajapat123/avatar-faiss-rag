#!/usr/bin/env node
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { pipeline } from '@xenova/transformers';
import faiss from 'faiss-node';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.join(__dirname, '..');
const dataRoot = path.join(projectRoot, 'processed');
const outputDir = path.join(projectRoot, 'faiss-data');

const args = process.argv.slice(2);
const argMap = args.reduce((acc, arg) => {
  const [key, value] = arg.split('=');
  if (key && value) {
    acc[key.replace(/^--/, '')] = value;
  }
  return acc;
}, {});

const CHUNK_SIZE = Number(argMap.chunkSize || process.env.CHUNK_SIZE || 700);
const CHUNK_OVERLAP = Number(argMap.chunkOverlap || process.env.CHUNK_OVERLAP || 120);
const SOURCE_FILE = argMap.source || process.env.CHUNK_SOURCE || 'chunks.json';
const MODEL_ID = argMap.model || process.env.EMBEDDING_MODEL || 'Xenova/all-MiniLM-L6-v2';
const INDEX_TYPE = (argMap.index || process.env.FAISS_INDEX_TYPE || 'flat').toLowerCase();

const { IndexFlatIP } = faiss;

const toArray = (value) => {
  if (Array.isArray(value)) return value;
  if (typeof value === 'string' && value.trim().length > 0) {
    return [value.trim()];
  }
  return [];
};

function chunkText(text, size, overlap) {
  if (!text) return [];
  const clean = text.replace(/\s+/g, ' ').trim();
  if (clean.length <= size) return [clean];

  const chunks = [];
  let start = 0;
  while (start < clean.length) {
    const end = Math.min(start + size, clean.length);
    let slice = clean.slice(start, end);
    const lastPeriod = slice.lastIndexOf('.');
    if (lastPeriod !== -1 && end !== clean.length && lastPeriod > size * 0.4) {
      slice = slice.slice(0, lastPeriod + 1);
    }
    chunks.push(slice.trim());
    start += size - overlap;
  }
  return chunks;
}

async function main() {
  console.log('🔄 Building FAISS index (offline)...');

  if (!fs.existsSync(dataRoot)) {
    throw new Error(`Processed data folder not found: ${dataRoot}`);
  }

  const sourcePath = path.join(dataRoot, SOURCE_FILE);
  if (!fs.existsSync(sourcePath)) {
    throw new Error(`Source chunk file not found: ${sourcePath}`);
  }

  const raw = JSON.parse(fs.readFileSync(sourcePath, 'utf8'));
  const documents = Array.isArray(raw) ? raw : raw.documents || [];
  if (!documents.length) {
    throw new Error('No documents found in source dataset.');
  }

  console.log(`📄 Loaded ${documents.length} base chunks from ${SOURCE_FILE}`);

  const expanded = [];
  for (const doc of documents) {
    const baseMeta = (doc && typeof doc.metadata === 'object' && doc.metadata !== null)
      ? doc.metadata
      : {};

    const baseId = doc.id || baseMeta.id || baseMeta.chunk_id || doc.source || 'doc';
    const baseSource = doc.source
      || baseMeta.source
      || baseMeta.source_document
      || baseMeta.document_id
      || 'unknown';

    const baseCategory = doc.category
      || baseMeta.category
      || baseMeta.primary_category
      || 'general';

    const baseDocumentType = doc.document_type
      || baseMeta.document_type
      || baseMeta.content_type
      || 'general';

    const topicsArray = toArray(doc.topics).length
      ? toArray(doc.topics)
      : toArray(baseMeta.topics).length
        ? toArray(baseMeta.topics)
        : toArray(baseMeta.search_keywords);

    const referencedProducts = toArray(doc.referenced_products).length
      ? toArray(doc.referenced_products)
      : toArray(baseMeta.referenced_products);

    const keyFeatures = toArray(doc.key_features).length
      ? toArray(doc.key_features)
      : toArray(baseMeta.key_features);

    const useCases = toArray(doc.use_cases).length
      ? toArray(doc.use_cases)
      : toArray(baseMeta.use_cases);

    const searchKeywords = toArray(doc.search_keywords).length
      ? toArray(doc.search_keywords)
      : toArray(baseMeta.search_keywords);

    const tags = toArray(doc.tags).length
      ? toArray(doc.tags)
      : toArray(baseMeta.tags);

    const primaryProduct = doc.product
      || baseMeta.product
      || (referencedProducts.length > 0 ? referencedProducts[0] : 'all');

    const textChunks = chunkText(doc.text || '', CHUNK_SIZE, CHUNK_OVERLAP);

    textChunks.forEach((chunk, idx) => {
      expanded.push({
        id: `${baseId}_${idx}`,
        text: chunk,
        source: baseSource,
        product: primaryProduct,
        category: baseCategory,
        document_type: baseDocumentType,
        topics: topicsArray,
        referenced_products: referencedProducts,
        key_features: keyFeatures,
        use_cases: useCases,
        search_keywords: searchKeywords,
        tags,
        metadata: {
          ...baseMeta,
          original_id: doc.id || baseMeta.id || null,
          chunk_index: idx,
          total_chunks: textChunks.length
        }
      });
    });
  }

  console.log(`🪓 Re-chunked into ${expanded.length} embedding segments (size ${CHUNK_SIZE}, overlap ${CHUNK_OVERLAP})`);

  console.log(`📦 Loading embedding model: ${MODEL_ID}`);
  const embedder = await pipeline('feature-extraction', MODEL_ID, {
    quantized: process.env.TRANSFORMERS_QUANTIZED === 'true',
    progress_callback: null
  });

  const embeddings = [];
  const vectors = [];
  const batchSize = Number(argMap.batchSize || process.env.EMBED_BATCH || 32);

  for (let i = 0; i < expanded.length; i += batchSize) {
    const batch = expanded.slice(i, i + batchSize);
    console.log(`   ⏳ Embedding batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(expanded.length / batchSize)}`);
    for (const chunk of batch) {
      const result = await embedder(chunk.text, { pooling: 'mean', normalize: true });
      const vector = Array.from(result.data);
      if (!vector.length) {
        throw new Error('Empty embedding encountered.');
      }
      vectors.push(Float32Array.from(vector));
      embeddings.push({
        id: chunk.id,
        text: chunk.text,
        source: chunk.source,
        product: chunk.product,
        category: chunk.category,
        document_type: chunk.document_type,
        topics: chunk.topics,
        referenced_products: chunk.referenced_products,
        key_features: chunk.key_features,
        use_cases: chunk.use_cases,
        search_keywords: chunk.search_keywords,
        tags: chunk.tags,
        metadata: chunk.metadata
      });
    }
  }

  const dimension = vectors[0].length;
  console.log(`📐 Embedding dimension detected: ${dimension}`);

  let index;
  switch (INDEX_TYPE) {
    case 'flat':
      index = new IndexFlatIP(dimension);
      break;
    default:
      console.warn(`⚠️  Unsupported index type "${INDEX_TYPE}". Falling back to IndexFlatIP.`);
      index = new IndexFlatIP(dimension);
  }

  const flattened = [];
  for (const vector of vectors) {
    for (let i = 0; i < vector.length; i += 1) {
      flattened.push(vector[i]);
    }
  }

  console.log('📥 Adding vectors to FAISS index...');
  index.add(flattened);

  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  const indexPath = path.join(outputDir, 'vectors.index');
  const metadataPath = path.join(outputDir, 'metadata.json');
  const configPath = path.join(outputDir, 'index-config.json');

  index.write(indexPath);
  fs.writeFileSync(metadataPath, JSON.stringify(embeddings, null, 2));
  fs.writeFileSync(configPath, JSON.stringify({
    type: INDEX_TYPE,
    dimension,
    chunkSize: CHUNK_SIZE,
    chunkOverlap: CHUNK_OVERLAP,
    model: MODEL_ID,
    createdAt: new Date().toISOString()
  }, null, 2));

  console.log('✅ FAISS index build complete');
  console.log(`   • Index:    ${indexPath}`);
  console.log(`   • Metadata: ${metadataPath}`);
  console.log(`   • Config:   ${configPath}`);
}

main().catch(error => {
  console.error('❌ Failed to build FAISS index:', error.message);
  process.exitCode = 1;
});
