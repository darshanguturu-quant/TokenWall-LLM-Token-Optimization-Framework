

!pip install groq tiktoken sentence-transformers gradio PyPDF2 numpy -q


import os, re, time, json, hashlib, warnings
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from abc import ABC, abstractmethod

warnings.filterwarnings("ignore")

from groq import Groq
import tiktoken
from sentence_transformers import SentenceTransformer
import PyPDF2
import gradio as gr

print("✅ Imports OK")



GROQ_API_KEY = ""   # ← Paste your key here
GROQ_MODEL   = "llama-3.1-8b-instant"            # or "mixtral-8x7b-32768"

MODE_CONFIG = {
    "HIGH": {
        "top_k_chunks":      10,
        "compress":          False,
        "summarize_history": False,
        "use_cache":         False,
        "pii_enabled":       False,
        "description":       "Max quality — minimal interference",
        "expected_savings":  "0–10%",
        "expected_latency":  "2–3s",
    },
    "MEDIUM": {
        "top_k_chunks":      5,
        "compress":          False,
        "summarize_history": True,
        "use_cache":         True,
        "pii_enabled":       False,
        "description":       "Balanced — dedup + chunk ranking + history summary",
        "expected_savings":  "30–50%",
        "expected_latency":  "5–8s",
    },
    "LOW": {
        "top_k_chunks":      2,
        "compress":          True,
        "summarize_history": True,
        "use_cache":         True,
        "pii_enabled":       True,
        "description":       "Max savings — full pipeline",
        "expected_savings":  "60–90%",
        "expected_latency":  "10–20s",
    },
}

print("✅ Config loaded")
for mode, cfg in MODE_CONFIG.items():
    print(f"  [{mode}] {cfg['description']} | savings: {cfg['expected_savings']}")



@dataclass
class RequestContext:
    """
    Single object passed through every pipeline step.
    Each step reads from it and writes results back.
    """
    prompt:        str
    mode:          str                          # HIGH | MEDIUM | LOW
    chunks:        List[str] = field(default_factory=list)
    history:       List[Dict] = field(default_factory=list)
    pii_action:    str = "MASK"                 # OFF | MASK | REMOVE

    original_tokens:  int  = 0
    optimized_tokens: int  = 0
    selected_chunks:  List[str] = field(default_factory=list)
    cache_hit:        bool = False
    pii_detected:     bool = False
    pii_map:          Dict = field(default_factory=dict)   # placeholder → original value
    steps_applied:    List[str] = field(default_factory=list)
    metadata:         Dict = field(default_factory=dict)

print("✅ RequestContext defined")



class TokenCounter:
    def __init__(self):
        self.enc = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self.enc.encode(str(text)))

    def count_messages(self, messages: List[Dict]) -> int:
        return sum(self.count(m.get("content", "")) for m in messages)

token_counter = TokenCounter()
print(f"✅ TokenCounter ready")



class BaseStep(ABC):
    """Every pipeline step must implement process()."""

    @abstractmethod
    def process(self, ctx: RequestContext) -> RequestContext:
        pass

    def _register(self, ctx: RequestContext, name: str):
        ctx.steps_applied.append(name)

print("✅ BaseStep defined")



class TokenAnalyzerStep(BaseStep):
    """
    Counts tokens across prompt + all chunks + full history.
    Runs at the start to capture the 'before' state.
    """
    def process(self, ctx: RequestContext) -> RequestContext:
        prompt_tokens  = token_counter.count(ctx.prompt)
        chunk_tokens   = sum(token_counter.count(c) for c in ctx.chunks)
        history_tokens = token_counter.count_messages(ctx.history)

        ctx.original_tokens = prompt_tokens + chunk_tokens + history_tokens

        ctx.metadata["token_breakdown_before"] = {
            "prompt":  prompt_tokens,
            "chunks":  chunk_tokens,
            "history": history_tokens,
            "total":   ctx.original_tokens,
        }
        self._register(ctx, "token_analyzer")
        return ctx

print("✅ TokenAnalyzerStep defined")



class PIIGuardStep(BaseStep):
    """
    Detects PII via regex patterns.
    MASK mode: replaces with placeholder, de-anonymizes LLM response later.
    REMOVE mode: strips PII entirely.
    Active only in LOW mode.
    """
    PATTERNS = {
        "EMAIL":   r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
        "PHONE":   r"\b(\+\d{1,3}[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b",
        "SSN":     r"\b\d{3}-\d{2}-\d{4}\b",
        "CC":      r"\b(?:\d{4}[\s\-]?){3}\d{4}\b",
        "AADHAAR": r"\b\d{4}\s\d{4}\s\d{4}\b",
    }

    def process(self, ctx: RequestContext) -> RequestContext:
        cfg = MODE_CONFIG[ctx.mode]
        if not cfg["pii_enabled"] or ctx.pii_action == "OFF":
            return ctx

        text    = ctx.prompt
        pii_map = {}
        counts  = {}

        for pii_type, pattern in self.PATTERNS.items():
            matches = re.findall(pattern, text)
            flat    = [m if isinstance(m, str) else m[0] for m in matches]

            for i, match in enumerate(flat):
                if not match.strip():
                    continue
                placeholder = f"[{pii_type}_{i+1}]"

                if ctx.pii_action == "MASK":
                    pii_map[placeholder] = match
                    text = text.replace(match, placeholder)
                    counts[pii_type] = counts.get(pii_type, 0) + 1
                elif ctx.pii_action == "REMOVE":
                    text = text.replace(match, "")
                    counts[pii_type] = counts.get(pii_type, 0) + 1

        if counts:
            ctx.pii_detected = True
            ctx.pii_map      = pii_map
            ctx.metadata["pii_counts"] = counts

        ctx.prompt = " ".join(text.split())
        self._register(ctx, "pii_guard")
        return ctx

print("✅ PIIGuardStep defined")



class DeduplicationStep(BaseStep):
    """
    Removes exact-duplicate chunks before sending to LLM.
    Normalizes text (lowercase + whitespace) for comparison.
    """
    def process(self, ctx: RequestContext) -> RequestContext:
        seen    = set()
        deduped = []

        for chunk in ctx.chunks:
            key = " ".join(chunk.lower().split())
            if key not in seen:
                seen.add(key)
                deduped.append(chunk)

        removed       = len(ctx.chunks) - len(deduped)
        ctx.chunks    = deduped
        ctx.metadata["dedup_chunks_removed"] = removed
        self._register(ctx, "deduplication")
        return ctx

print("✅ DeduplicationStep defined")



class ChunkRankingStep(BaseStep):
    """
    Uses embedding cosine similarity to rank chunks by relevance.
    Keeps only top-K most relevant chunks.
      HIGH   → top 10
      MEDIUM → top 5
      LOW    → top 2
    """
    def __init__(self, embedder: SentenceTransformer):
        self.embedder = embedder

    def process(self, ctx: RequestContext) -> RequestContext:
        if not ctx.chunks:
            ctx.selected_chunks = []
            self._register(ctx, "chunk_ranking")
            return ctx

        top_k = MODE_CONFIG[ctx.mode]["top_k_chunks"]

        if len(ctx.chunks) <= top_k:
            ctx.selected_chunks = ctx.chunks
            ctx.metadata["chunk_ranking"] = {"before": len(ctx.chunks), "after": len(ctx.chunks), "top_k": top_k}
            self._register(ctx, "chunk_ranking")
            return ctx

        query_emb  = self.embedder.encode([ctx.prompt], normalize_embeddings=True)
        chunk_embs = self.embedder.encode(ctx.chunks,  normalize_embeddings=True)
        scores     = np.dot(chunk_embs, query_emb.T).flatten()
        top_idx    = np.argsort(scores)[::-1][:top_k]

        ctx.selected_chunks = [ctx.chunks[i] for i in sorted(top_idx)]
        ctx.metadata["chunk_ranking"] = {
            "before":     len(ctx.chunks),
            "after":      len(ctx.selected_chunks),
            "top_k":      top_k,
            "top_scores": [round(float(scores[i]), 3) for i in top_idx],
        }
        self._register(ctx, "chunk_ranking")
        return ctx

print("✅ ChunkRankingStep defined")



class HistorySummarizationStep(BaseStep):
    """
    When history grows beyond threshold, summarizes older messages
    into a single compact line. Keeps 2 most recent messages raw.
    """
    THRESHOLD = 6

    def __init__(self, groq_client: Groq):
        self.client = groq_client

    def process(self, ctx: RequestContext) -> RequestContext:
        if not MODE_CONFIG[ctx.mode]["summarize_history"]:
            return ctx
        if len(ctx.history) <= self.THRESHOLD:
            return ctx

        to_summarize = ctx.history[:-2]
        recent       = ctx.history[-2:]

        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in to_summarize
        )
        prompt = (
            "Summarize this conversation in 1-2 sentences. "
            "Focus on key topics and decisions:\n\n" + history_text
        )

        try:
            resp    = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80,
            )
            summary = resp.choices[0].message.content.strip()
            ctx.history = [
                {"role": "system", "content": f"[Prior conversation summary]: {summary}"},
                *recent,
            ]
            ctx.metadata["history_summary"]          = summary
            ctx.metadata["history_messages_before"]  = len(to_summarize) + len(recent)
            ctx.metadata["history_messages_after"]   = len(ctx.history)
        except Exception as e:
            ctx.metadata["history_summary_error"] = str(e)

        self._register(ctx, "history_summarization")
        return ctx

print("✅ HistorySummarizationStep defined")



class CompressionStep(BaseStep):
    """
    Lightweight prompt compression: removes filler/hedge words.
    Phase 2 will use LLM-based compression (LLMLingua).
    """
    FILLERS = [
        r"\bplease\b", r"\bkindly\b", r"\bbasically\b", r"\bactually\b",
        r"\bjust\b", r"\bvery\b", r"\breally\b", r"\bsimply\b",
        r"\bI was wondering if you could\b", r"\bCould you please\b",
        r"\bI would like to know\b", r"\bCan you tell me\b",
        r"\bWould you be able to\b", r"\bI was hoping\b",
    ]

    def process(self, ctx: RequestContext) -> RequestContext:
        if not MODE_CONFIG[ctx.mode]["compress"]:
            return ctx

        text         = ctx.prompt
        original_len = len(text)

        for pattern in self.FILLERS:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        text = " ".join(text.split())
        ctx.metadata["compression"] = {
            "before_chars": original_len,
            "after_chars":  len(text),
            "ratio":        round(len(text) / max(original_len, 1), 3),
        }
        ctx.prompt = text
        self._register(ctx, "compression")
        return ctx

print("✅ CompressionStep defined")



class SemanticCacheStep(BaseStep):
    """
    In-memory semantic cache (Redis in production).
    Embeds incoming query and compares against all cached queries.
    If cosine similarity >= threshold, returns cached response.
    """
    def __init__(self, embedder: SentenceTransformer, threshold: float = 0.92):
        self.embedder  = embedder
        self.threshold = threshold
        self._cache: Dict[str, Dict] = {}

    def process(self, ctx: RequestContext) -> RequestContext:
        if not MODE_CONFIG[ctx.mode]["use_cache"]:
            return ctx

        query_emb  = self.embedder.encode([ctx.prompt], normalize_embeddings=True)[0]
        best_score = 0.0
        best_key   = None

        for key, entry in self._cache.items():
            score = float(np.dot(query_emb, entry["embedding"]))
            if score > best_score:
                best_score = score
                best_key   = key

        if best_score >= self.threshold and best_key:
            ctx.cache_hit = True
            ctx.metadata["cached_response"] = self._cache[best_key]["response"]
            ctx.metadata["cache_score"]     = round(best_score, 4)

        ctx.metadata["_query_embedding"] = query_emb
        self._register(ctx, "semantic_cache")
        return ctx

    def store(self, prompt: str, response: str, embedding: np.ndarray):
        key = hashlib.md5(prompt.encode()).hexdigest()[:10]
        self._cache[key] = {"embedding": embedding, "response": response, "prompt": prompt}

    def size(self) -> int:
        return len(self._cache)

print("✅ SemanticCacheStep defined")



class FinalTokenCountStep(BaseStep):
    """Counts tokens AFTER all optimization steps."""
    def process(self, ctx: RequestContext) -> RequestContext:
        active_chunks  = ctx.selected_chunks if ctx.selected_chunks else ctx.chunks
        prompt_tokens  = token_counter.count(ctx.prompt)
        chunk_tokens   = sum(token_counter.count(c) for c in active_chunks)
        history_tokens = token_counter.count_messages(ctx.history)

        ctx.optimized_tokens = prompt_tokens + chunk_tokens + history_tokens
        ctx.metadata["token_breakdown_after"] = {
            "prompt":  prompt_tokens,
            "chunks":  chunk_tokens,
            "history": history_tokens,
            "total":   ctx.optimized_tokens,
        }
        self._register(ctx, "final_token_count")
        return ctx

print("✅ FinalTokenCountStep defined")



class OptimizationMode(ABC):
    @abstractmethod
    def get_pipeline_steps(self) -> List[str]:
        pass

class HighMode(OptimizationMode):
    def get_pipeline_steps(self):
        return ["token_analyzer", "chunk_ranking", "final_token_count"]

class MediumMode(OptimizationMode):
    def get_pipeline_steps(self):
        return ["token_analyzer", "deduplication", "chunk_ranking",
                "history_summarization", "semantic_cache", "final_token_count"]

class LowMode(OptimizationMode):
    def get_pipeline_steps(self):
        return ["token_analyzer", "pii_guard", "deduplication", "chunk_ranking",
                "history_summarization", "compression", "semantic_cache", "final_token_count"]

print("✅ Mode strategies defined")
print("  HIGH   →", HighMode().get_pipeline_steps())
print("  MEDIUM →", MediumMode().get_pipeline_steps())
print("  LOW    →", LowMode().get_pipeline_steps())



class PipelineRunner:
    """
    Executes steps sequentially. Errors are caught and logged —
    pipeline continues rather than crashing.
    """
    def __init__(self, registry: Dict):
        self.registry = registry

    def run(self, ctx: RequestContext, steps: List[str]) -> RequestContext:
        for step_name in steps:
            handler = self.registry.get(step_name)
            if handler is None:
                ctx.metadata[f"missing_step_{step_name}"] = True
                continue
            try:
                ctx = handler.process(ctx)
            except Exception as e:
                ctx.metadata[f"step_error_{step_name}"] = str(e)
        return ctx

print("✅ PipelineRunner defined")



class TokenWall:
    def __init__(self, groq_api_key: str):
        print("⏳ Loading embedding model (first run ~30s)...")
        self.client    = Groq(api_key=groq_api_key)
        self.embedder  = SentenceTransformer("all-MiniLM-L6-v2")
        self.sem_cache = SemanticCacheStep(self.embedder, threshold=0.92)
        print("✅ Embedding model loaded")

        self.registry = {
            "token_analyzer":        TokenAnalyzerStep(),
            "pii_guard":             PIIGuardStep(),
            "deduplication":         DeduplicationStep(),
            "chunk_ranking":         ChunkRankingStep(self.embedder),
            "history_summarization": HistorySummarizationStep(self.client),
            "compression":           CompressionStep(),
            "semantic_cache":        self.sem_cache,
            "final_token_count":     FinalTokenCountStep(),
        }

        self.mode_map = {
            "HIGH":   HighMode(),
            "MEDIUM": MediumMode(),
            "LOW":    LowMode(),
        }

        self.runner = PipelineRunner(self.registry)
        print("✅ TokenWall engine ready")

    def process(self, prompt: str, mode: str, chunks: List[str],
                history: List[Dict], pii_action: str = "MASK") -> Dict:

        start = time.time()

        ctx = RequestContext(
            prompt=prompt, mode=mode,
            chunks=chunks.copy(), history=history.copy(),
            pii_action=pii_action,
        )

        steps = self.mode_map[mode].get_pipeline_steps()
        ctx   = self.runner.run(ctx, steps)

        if ctx.cache_hit:
            return self._build_result(ctx, ctx.metadata["cached_response"],
                                      int((time.time()-start)*1000), from_cache=True)

        active_chunks = ctx.selected_chunks if ctx.selected_chunks else ctx.chunks
        context_block = "\n\n---\n\n".join(active_chunks)

        system_msg = (
            "You are a helpful assistant. Answer questions ONLY using the context below. "
            "If the answer is not in the context, say 'I don't have enough information.'\n\n"
            f"CONTEXT:\n{context_block}"
        )

        messages = [
            {"role": "system", "content": system_msg},
            *ctx.history,
            {"role": "user", "content": ctx.prompt},
        ]

        try:
            resp     = self.client.chat.completions.create(
                model=GROQ_MODEL, messages=messages,
                max_tokens=512, temperature=0.3,
            )
            response = resp.choices[0].message.content.strip()
        except Exception as e:
            response = f"⚠️ LLM error: {e}"

        if ctx.pii_detected and ctx.pii_map:
            for placeholder, original in ctx.pii_map.items():
                response = response.replace(placeholder, original)

        emb = ctx.metadata.get("_query_embedding")
        if emb is not None:
            self.sem_cache.store(prompt, response, emb)

        return self._build_result(ctx, response, int((time.time()-start)*1000))

    def _build_result(self, ctx, response, latency_ms, from_cache=False):
        saved      = max(0, ctx.original_tokens - ctx.optimized_tokens)
        saving_pct = round((saved / max(ctx.original_tokens, 1)) * 100, 1)
        cost_saved = round((saved / 1000) * 0.005, 5)

        return {
            "response":              response,
            "mode":                  ctx.mode,
            "original_tokens":       ctx.original_tokens,
            "optimized_tokens":      ctx.optimized_tokens,
            "saved_tokens":          saved,
            "saving_percent":        f"{saving_pct}%",
            "estimated_cost_saved":  f"${cost_saved}",
            "latency_ms":            latency_ms,
            "cache_hit":             from_cache or ctx.cache_hit,
            "pii_detected":          ctx.pii_detected,
            "steps_applied":         ctx.steps_applied,
            "cache_size":            self.sem_cache.size(),
            "metadata":              {k: v for k, v in ctx.metadata.items() if not k.startswith("_")},
        }



def load_document(file_path: str) -> str:
    if file_path.lower().endswith(".pdf"):
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def chunk_document(text: str, chunk_size: int = 250, overlap: int = 40) -> List[str]:
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks

print("✅ Document utilities ready")



tokenwall = TokenWall(GROQ_API_KEY)



_doc_chunks: List[str]  = []
_history:    List[Dict] = []

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=JetBrains+Mono:wght@300;400;600&display=swap');

/* ── Reset & Root ── */
* { box-sizing: border-box; }

:root {
    --bg-void:     #060810;
    --bg-deep:     #0c0f1a;
    --bg-card:     #111520;
    --bg-card2:    #161b2e;
    --cyan:        #00e5ff;
    --cyan-dim:    #00a8c0;
    --amber:       #ffab40;
    --green:       #00e676;
    --red:         #ff5252;
    --text-primary:#e8eaf6;
    --text-dim:    #7986a3;
    --border:      #1e2640;
    --border-glow: #00e5ff33;
    --font-display:'Syne', sans-serif;
    --font-mono:   'JetBrains Mono', monospace;
}

/* ── Page Background ── */
body, .gradio-container {
    background: var(--bg-void) !important;
    font-family: var(--font-mono) !important;
    color: var(--text-primary) !important;
}

/* Animated grid background */
.gradio-container::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,229,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,229,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── Header ── */
.tw-header {
    text-align: center;
    padding: 48px 24px 32px;
    position: relative;
}

.tw-logo {
    font-family: var(--font-display);
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -2px;
    background: linear-gradient(135deg, var(--cyan) 0%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin-bottom: 6px;
}

.tw-tagline {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--text-dim);
    letter-spacing: 3px;
    text-transform: uppercase;
}

.tw-badge {
    display: inline-block;
    background: linear-gradient(90deg, #00e5ff22, #7c3aed22);
    border: 1px solid var(--border-glow);
    border-radius: 100px;
    padding: 4px 14px;
    font-size: 0.7rem;
    color: var(--cyan);
    letter-spacing: 2px;
    margin-top: 10px;
}

/* ── Mode Cards ── */
.tw-mode-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin: 16px 0;
}

.tw-mode-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 10px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s ease;
    font-family: var(--font-mono);
}

.tw-mode-card:hover {
    border-color: var(--cyan-dim);
    background: var(--bg-card2);
}

.tw-mode-card .mode-label {
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 4px;
}

.tw-mode-card .mode-saving {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--cyan);
}

/* ── Panels / Cards ── */
.tw-panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
}

.tw-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--cyan), transparent);
    opacity: 0.5;
}

.tw-panel-title {
    font-family: var(--font-display);
    font-size: 0.72rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--cyan);
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.tw-panel-title::before {
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--cyan);
    box-shadow: 0 0 8px var(--cyan);
}

/* ── Stat Cards (inside analytics) ── */
.tw-stats-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 14px;
}

.tw-stat {
    background: var(--bg-deep);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
    position: relative;
    overflow: hidden;
}

.tw-stat.highlight {
    border-color: var(--cyan-dim);
    background: linear-gradient(135deg, #00e5ff0a, var(--bg-deep));
}

.tw-stat .stat-label {
    font-size: 0.62rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 6px;
}

.tw-stat .stat-value {
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--text-primary);
    font-family: var(--font-mono);
}

.tw-stat .stat-value.cyan  { color: var(--cyan); }
.tw-stat .stat-value.amber { color: var(--amber); }
.tw-stat .stat-value.green { color: var(--green); }

/* ── Pipeline Viz ── */
.tw-pipeline {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
    margin-top: 12px;
}

.tw-step {
    background: var(--bg-deep);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.62rem;
    letter-spacing: 1px;
    color: var(--cyan);
    font-family: var(--font-mono);
}

.tw-arrow {
    color: var(--text-dim);
    font-size: 0.75rem;
}

/* ── Gradio component overrides ── */

/* Tabs */
.tab-nav { border-bottom: 1px solid var(--border) !important; }
.tab-nav button {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 1px !important;
    color: var(--text-dim) !important;
    background: transparent !important;
    border: none !important;
    padding: 8px 16px !important;
}
.tab-nav button.selected {
    color: var(--cyan) !important;
    border-bottom: 2px solid var(--cyan) !important;
}

/* Buttons */
button.primary {
    background: linear-gradient(135deg, #00b4cc, #0077aa) !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    letter-spacing: 1px !important;
    font-weight: 600 !important;
    color: #fff !important;
    box-shadow: 0 0 20px #00e5ff30 !important;
    transition: all 0.2s !important;
}
button.primary:hover {
    box-shadow: 0 0 30px #00e5ff50 !important;
    transform: translateY(-1px) !important;
}

button.secondary {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    color: var(--text-dim) !important;
}

/* Textbox / Input */
.block label span {
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: var(--text-dim) !important;
}

textarea, input[type="text"] {
    background: var(--bg-deep) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    transition: border-color 0.2s !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: var(--cyan-dim) !important;
    box-shadow: 0 0 0 2px #00e5ff15 !important;
    outline: none !important;
}

/* Radio buttons */
.wrap.svelte-1p9xokt { gap: 8px !important; }

.wrap input[type=radio] + span {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 1px !important;
    color: var(--text-dim) !important;
    padding: 6px 14px !important;
    transition: all 0.15s !important;
}
.wrap input[type=radio]:checked + span {
    background: linear-gradient(135deg, #00e5ff15, #7c3aed15) !important;
    border-color: var(--cyan) !important;
    color: var(--cyan) !important;
    box-shadow: 0 0 12px #00e5ff20 !important;
}

/* Chatbot */
.chatbot {
    background: var(--bg-deep) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
}

.chatbot .message.user {
    background: linear-gradient(135deg, #00e5ff15, #0077aa20) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: 12px 12px 4px 12px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
}

.chatbot .message.bot {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px 12px 12px 4px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
}

/* Accordion */
.accordion {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
.accordion .label-wrap span {
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    letter-spacing: 1.5px !important;
    color: var(--text-dim) !important;
}

/* File upload */
.file-preview {
    background: var(--bg-deep) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 10px !important;
}

/* Markdown override */
.prose { color: var(--text-primary) !important; font-family: var(--font-mono) !important; }
.prose h1,.prose h2,.prose h3 { font-family: var(--font-display) !important; color: var(--text-primary) !important; }
.prose table { border-collapse: collapse !important; width: 100% !important; font-size: 0.78rem !important; }
.prose th {
    background: var(--bg-card2) !important;
    color: var(--cyan) !important;
    border: 1px solid var(--border) !important;
    padding: 8px 12px !important;
    font-size: 0.68rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}
.prose td {
    border: 1px solid var(--border) !important;
    padding: 7px 12px !important;
    color: var(--text-primary) !important;
}
.prose code {
    background: var(--bg-deep) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    color: var(--amber) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    padding: 1px 6px !important;
}

/* Code block */
.codemirror-wrapper {
    background: var(--bg-deep) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* Hide Gradio footer */
footer { display: none !important; }
.built-with { display: none !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--cyan-dim); }

/* Glow pulse animation for live dot */
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 4px var(--cyan); opacity: 1; }
    50% { box-shadow: 0 0 12px var(--cyan); opacity: 0.6; }
}

.live-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--green);
    animation: pulse-glow 2s infinite;
    margin-right: 6px;
}

/* Savings bar */
.savings-bar-wrap {
    background: var(--bg-deep);
    border: 1px solid var(--border);
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
    margin-top: 6px;
}
.savings-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--cyan), #7c3aed);
    border-radius: 6px;
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
}
"""

HEADER_HTML = """
<div class="tw-header">
  <div class="tw-logo">⬛ TokenWall</div>
  <div class="tw-tagline">Intelligent LLM Token Optimization Middleware</div>
  <div class="tw-badge">FR1 — OPTIMIZATION MODES</div>

  <div class="tw-mode-grid" style="max-width:520px;margin:24px auto 0;">
    <div class="tw-mode-card">
      <div class="mode-label">🟢 HIGH</div>
      <div class="mode-saving" style="color:#00e676;">0–10%</div>
      <div style="font-size:0.6rem;color:#7986a3;margin-top:3px;">top 10 chunks</div>
    </div>
    <div class="tw-mode-card">
      <div class="mode-label">🟡 MEDIUM</div>
      <div class="mode-saving" style="color:#ffab40;">30–50%</div>
      <div style="font-size:0.6rem;color:#7986a3;margin-top:3px;">dedup + cache</div>
    </div>
    <div class="tw-mode-card">
      <div class="mode-label">🔴 LOW</div>
      <div class="mode-saving" style="color:#00e5ff;">60–90%</div>
      <div style="font-size:0.6rem;color:#7986a3;margin-top:3px;">full pipeline</div>
    </div>
  </div>
</div>
"""

def build_analytics_html(result: dict) -> str:
    saved_pct_num = float(result["saving_percent"].replace("%",""))
    bar_width     = min(int(saved_pct_num), 100)

    cache_html = (
        '<span style="color:var(--green)">✅ HIT</span>'
        if result["cache_hit"] else
        '<span style="color:var(--text-dim)">❌ MISS</span>'
    )
    pii_html = (
        '<span style="color:var(--amber)">⚠ DETECTED</span>'
        if result["pii_detected"] else
        '<span style="color:var(--text-dim)">NONE</span>'
    )

    steps_html = ""
    for i, s in enumerate(result["steps_applied"]):
        steps_html += f'<span class="tw-step">{s}</span>'
        if i < len(result["steps_applied"]) - 1:
            steps_html += '<span class="tw-arrow">›</span>'

    mode_colors = {"HIGH": "#00e676", "MEDIUM": "#ffab40", "LOW": "#00e5ff"}
    mc = mode_colors.get(result["mode"], "#00e5ff")

    return f"""
<div class="tw-panel" style="margin-top:0;">
  <div class="tw-panel-title">
    <span class="live-dot"></span>LIVE ANALYTICS
    <span style="margin-left:auto;font-size:0.65rem;color:var(--text-dim);">{result['latency_ms']} ms</span>
  </div>

  <div class="tw-stats-grid">
    <div class="tw-stat">
      <div class="stat-label">Original Tokens</div>
      <div class="stat-value">{result['original_tokens']:,}</div>
    </div>
    <div class="tw-stat highlight">
      <div class="stat-label">Optimized Tokens</div>
      <div class="stat-value cyan">{result['optimized_tokens']:,}</div>
    </div>
    <div class="tw-stat highlight">
      <div class="stat-label">Tokens Saved</div>
      <div class="stat-value green">↓ {result['saved_tokens']:,}</div>
    </div>
    <div class="tw-stat">
      <div class="stat-label">Est. Cost Saved</div>
      <div class="stat-value amber">{result['estimated_cost_saved']}</div>
    </div>
  </div>

  <!-- Savings Bar -->
  <div style="margin-bottom:16px;">
    <div style="display:flex;justify-content:space-between;font-size:0.65rem;color:var(--text-dim);margin-bottom:6px;">
      <span>TOKEN REDUCTION</span>
      <span style="color:var(--cyan);font-weight:600;">{result['saving_percent']}</span>
    </div>
    <div class="savings-bar-wrap">
      <div class="savings-bar-fill" style="width:{bar_width}%;"></div>
    </div>
  </div>

  <!-- Mode + Cache + PII Row -->
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:14px;">
    <div style="background:var(--bg-deep);border:1px solid var(--border);border-radius:8px;padding:10px;text-align:center;">
      <div style="font-size:0.6rem;letter-spacing:2px;color:var(--text-dim);margin-bottom:4px;">MODE</div>
      <div style="font-size:0.85rem;font-weight:700;color:{mc};">{result['mode']}</div>
    </div>
    <div style="background:var(--bg-deep);border:1px solid var(--border);border-radius:8px;padding:10px;text-align:center;">
      <div style="font-size:0.6rem;letter-spacing:2px;color:var(--text-dim);margin-bottom:4px;">CACHE</div>
      <div style="font-size:0.78rem;">{cache_html} <span style="color:var(--text-dim);font-size:0.65rem;">({result['cache_size']})</span></div>
    </div>
    <div style="background:var(--bg-deep);border:1px solid var(--border);border-radius:8px;padding:10px;text-align:center;">
      <div style="font-size:0.6rem;letter-spacing:2px;color:var(--text-dim);margin-bottom:4px;">PII</div>
      <div style="font-size:0.78rem;">{pii_html}</div>
    </div>
  </div>

  <!-- Pipeline -->
  <div style="font-size:0.6rem;letter-spacing:2px;color:var(--text-dim);margin-bottom:8px;text-transform:uppercase;">
    Pipeline Executed
  </div>
  <div class="tw-pipeline">{steps_html}</div>
</div>
"""

def handle_upload(file):
    global _doc_chunks, _history
    if file is None:
        return "⚠ No file selected.", "", "—"
    text        = load_document(file.name)
    _doc_chunks = chunk_document(text)
    _history    = []
    preview     = text[:700] + "…" if len(text) > 700 else text
    total_tok   = token_counter.count(text)
    status      = f"✅  {len(_doc_chunks)} chunks loaded  ·  {total_tok:,} raw tokens"
    return status, preview, f"{len(_doc_chunks)} chunks"


def handle_chat(user_msg, mode, pii_action, chatbot_state):
    global _history
    chatbot_state = chatbot_state or []

    if not _doc_chunks:
        chatbot_state.append({"role": "user",      "content": user_msg})
        chatbot_state.append({"role": "assistant", "content": "⚠ Upload and load a document first."})
        return chatbot_state, "<p style='color:var(--text-dim);font-size:0.8rem;'>Waiting for document…</p>", "", ""

    if not user_msg.strip():
        return chatbot_state, "", "", ""

    result = tokenwall.process(
        prompt=user_msg, mode=mode, chunks=_doc_chunks,
        history=_history, pii_action=pii_action,
    )

    _history.append({"role": "user",      "content": user_msg})
    _history.append({"role": "assistant", "content": result["response"]})

    analytics_html = build_analytics_html(result)
    meta_json      = json.dumps(result["metadata"], indent=2, default=str)

    chatbot_state.append({"role": "user",      "content": user_msg})
    chatbot_state.append({"role": "assistant", "content": result["response"]})

    return chatbot_state, analytics_html, meta_json, ""


def handle_clear():
    global _history
    _history = []
    return [], "<p style='color:var(--text-dim);font-size:0.8rem;'>Cleared.</p>", "", ""


with gr.Blocks(
    title="TokenWall — FR1",
    css=CUSTOM_CSS,
    theme=gr.themes.Base(
        primary_hue="cyan",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("JetBrains Mono"),
    ),
) as demo:

    gr.HTML(HEADER_HTML)

    with gr.Row(equal_height=False):

        with gr.Column(scale=1, min_width=270):

            with gr.Group():
                gr.HTML("""<div class="tw-panel-title" style="margin:16px 16px 4px;">
                    ⚙ SETTINGS</div>""")

                mode_radio = gr.Radio(
                    choices=["HIGH", "MEDIUM", "LOW"],
                    value="MEDIUM",
                    label="OPTIMIZATION MODE",
                    container=True,
                )
                pii_radio = gr.Radio(
                    choices=["OFF", "MASK", "REMOVE"],
                    value="MASK",
                    label="PII ACTION  (LOW mode only)",
                    container=True,
                )

            with gr.Group():
                gr.HTML("""<div class="tw-panel-title" style="margin:16px 16px 4px;">
                    📄 DOCUMENT</div>""")

                file_input = gr.File(
                    label="UPLOAD PDF OR TXT",
                    file_types=[".pdf", ".txt"],
                )
                upload_btn = gr.Button(
                    "⬆  LOAD DOCUMENT",
                    variant="primary",
                    size="lg",
                )
                upload_status = gr.Textbox(
                    label="STATUS",
                    interactive=False,
                    lines=1,
                    placeholder="No document loaded…",
                )
                chunk_info = gr.Textbox(
                    label="CHUNKS",
                    interactive=False,
                    lines=1,
                )

            with gr.Accordion("📖 DOCUMENT PREVIEW", open=False):
                doc_preview = gr.Textbox(interactive=False, lines=10, show_label=False)

        with gr.Column(scale=2):

            chatbot = gr.Chatbot(
                height=400,
                type="messages",
                show_label=False,
                avatar_images=(None, None),
                bubble_full_width=False,
                placeholder=(
                    "<div style='text-align:center;color:#7986a3;padding:60px 20px;"
                    "font-family:JetBrains Mono,monospace;font-size:0.8rem;'>"
                    "⬛ Load a document and ask a question<br>"
                    "<span style='font-size:0.65rem;letter-spacing:2px;'>TOKENWALL READY</span>"
                    "</div>"
                ),
            )

            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="Ask something about your document…",
                    show_label=False,
                    scale=5,
                    lines=1,
                )
                send_btn = gr.Button("SEND ›", variant="primary", scale=1)

            clear_btn = gr.Button("🗑  CLEAR CONVERSATION", size="sm", variant="secondary")

            analytics_out = gr.HTML(
                value="<div style='padding:24px;text-align:center;color:#7986a3;"
                      "font-family:JetBrains Mono,monospace;font-size:0.75rem;"
                      "letter-spacing:2px;'>ANALYTICS APPEAR AFTER FIRST QUERY</div>"
            )

            with gr.Accordion("{ } FULL METADATA — JSON", open=False):
                metadata_out = gr.Code(language="json", lines=14, show_label=False)

    upload_btn.click(
        handle_upload,
        inputs=[file_input],
        outputs=[upload_status, doc_preview, chunk_info],
    )

    send_inputs  = [user_input, mode_radio, pii_radio, chatbot]
    send_outputs = [chatbot, analytics_out, metadata_out, user_input]

    send_btn.click(handle_chat,    inputs=send_inputs, outputs=send_outputs)
    user_input.submit(handle_chat, inputs=send_inputs, outputs=send_outputs)
    clear_btn.click(handle_clear,  outputs=[chatbot, analytics_out, metadata_out, user_input])


import gradio as gr
gr.close_all()
demo.launch(share=True, show_error=True)
