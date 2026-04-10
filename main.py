# =============================================================================
# AutoStream – AI Video Editing SaaS | FastAPI Chatbot v3
# =============================================================================
#
# WHAT'S NEW IN v3:
#   1. LLM INTENT DETECTION  – Claude Haiku classifies intent from the full
#      conversation history, not just keyword matching on the latest message.
#      Keywords are still used as a fast fallback if the LLM call fails.
#
#   2. CONVERSATION HISTORY  – Every turn (user + bot) is appended to
#      state["history"]. History is used in two places:
#        a) Passed to the LLM so it understands context ("what about refunds?"
#           after a pricing question now correctly resolves to "policy").
#        b) Injected into the RAG query so follow-up phrases like "tell me more"
#           or "what about the other one?" retrieve the right chunks.
#
# FILE STRUCTURE:
#   main.py             ← this file
#   knowledge_base.json ← document store (loaded at startup)
#
# HOW TO RUN:
#   pip install fastapi uvicorn httpx
#   uvicorn main:app --reload
#
# INTERACTIVE DOCS:  http://127.0.0.1:8000/docs
# =============================================================================

import json
import math
import os
import random
import re
from collections import defaultdict
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="AutoStream AI Chatbot",
    description="FastAPI chatbot with LLM intent detection, conversation memory, and RAG",
    version="3.0.0",
)


# =============================================================================
# PYDANTIC SCHEMAS
# =============================================================================

class ChatRequest(BaseModel):
    session_id: str
    message: str

    class Config:
        json_schema_extra = {
            "example": {"session_id": "user_001", "message": "What about the cheaper option?"}
        }


class RetrievedChunk(BaseModel):
    id: str
    title: str
    category: str
    score: float
    content: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    intent: Optional[str]
    intent_source: str          # "llm" | "keyword_fallback" — tells you which path fired
    retrieved_chunks: list[RetrievedChunk]
    lead_captured: bool
    session_ended: bool


# =============================================================================
# ─── LLM INTENT DETECTION ────────────────────────────────────────────────────
#
#  Why LLM instead of only keywords?
#
#  Keywords fail on:
#    • Vague follow-ups:  "what about the other one?" → needs prior context
#    • Paraphrasing:      "I'd love to sign up" doesn't contain "buy/subscribe"
#    • Negation:          "I don't want to cancel" looks like "cancel" intent
#    • Multi-intent:      "Is it free and can I upgrade later?" mixes intents
#
#  Claude Haiku is fast (~200 ms) and cheap. We send it:
#    - The last N turns of conversation history (for context)
#    - The current user message
#    - A strict JSON-output prompt (so parsing is deterministic)
#
#  If the Haiku call fails for any reason, we fall back to keyword matching
#  so the bot never breaks.
#
# =============================================================================

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
LLM_MODEL         = "claude-haiku-4-5-20251001"
HISTORY_WINDOW    = 6   # number of recent turns sent to the LLM for context

# System prompt for the intent classifier.
# Strict JSON output keeps parsing simple and reliable.
INTENT_SYSTEM_PROMPT = """You are an intent classifier for AutoStream, an AI video editing SaaS chatbot.

Given the conversation history and the latest user message, classify the user's intent into EXACTLY ONE of these labels:

  greeting      – the user is saying hello or opening the conversation
  pricing       – the user is asking about plans, costs, fees, or comparing tiers
  features      – the user is asking about product capabilities, tools, or how something works
  policy        – the user is asking about refunds, cancellations, support, or guarantees
  high_intent   – the user clearly wants to buy, subscribe, sign up, or get started
  exit          – the user wants to end the conversation
  unknown       – none of the above

IMPORTANT RULES:
- Use the full conversation history to resolve ambiguous follow-ups.
  Example: if the last bot reply was about pricing and the user says "what about the other one?",
  classify as "pricing" not "unknown".
- Short affirmations after a feature explanation ("sounds good", "nice") are "features" not "unknown".
- Expressions of purchase intent ("let's do it", "I'm in", "sign me up") are "high_intent".
- Respond with ONLY a JSON object — no explanation, no markdown, no extra text:
  {"intent": "<label>"}"""


async def llm_detect_intent(user_message: str, history: list[dict]) -> Optional[str]:
    """
    Call Claude Haiku to classify intent using conversation history as context.

    Args:
        user_message : the current user input
        history      : list of {"role": "user"|"assistant", "content": "..."} dicts

    Returns:
        Intent string on success, None on any error (triggers keyword fallback).
    """
    # Build the last HISTORY_WINDOW turns as context for the LLM
    recent = history[-(HISTORY_WINDOW * 2):]   # each turn = 2 items (user + assistant)

    # Compose the messages array: history context + current user message
    messages = recent + [{"role": "user", "content": user_message}]

    payload = {
        "model":      LLM_MODEL,
        "max_tokens": 50,           # intent JSON is tiny — keep cost minimal
        "system":     INTENT_SYSTEM_PROMPT,
        "messages":   messages,
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                ANTHROPIC_API_URL,
                headers={"Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        raw_text = data["content"][0]["text"].strip()

        # Strip accidental markdown fences just in case
        raw_text = re.sub(r"```(?:json)?|```", "", raw_text).strip()

        parsed = json.loads(raw_text)
        intent = parsed.get("intent", "").strip().lower()

        valid_intents = {"greeting", "pricing", "features", "policy",
                         "high_intent", "exit", "unknown"}
        if intent in valid_intents:
            return intent

    except Exception as exc:
        # Log but never crash — keyword fallback will handle it
        print(f"[LLM intent] Call failed ({type(exc).__name__}: {exc}), using keyword fallback.")

    return None   # signals fallback


# ─── KEYWORD FALLBACK ─────────────────────────────────────────────────────────

INTENT_KEYWORDS = {
    "greeting":    ["hi", "hello", "hey", "good morning", "good evening",
                    "good afternoon", "howdy", "what's up", "greetings"],
    "pricing":     ["price", "pricing", "cost", "plan", "plans", "how much",
                    "subscription", "fee", "charge", "basic", "pro", "tier"],
    "features":    ["feature", "features", "what can", "capabilities",
                    "does it", "can it", "caption", "edit", "4k", "720",
                    "storage", "collaboration", "export", "trial"],
    "policy":      ["refund", "money back", "cancel", "cancellation",
                    "return", "support", "help", "contact", "policy"],
    "high_intent": ["buy", "purchase", "subscribe", "sign up", "signup",
                    "start", "get started", "i want", "i'm ready", "let's go",
                    "join", "enroll", "register", "i'd like to", "ready to",
                    "upgrade"],
    "exit":        ["exit", "quit", "bye", "goodbye", "stop", "end", "close"],
}


def keyword_detect_intent(user_input: str) -> str:
    """
    Fast keyword-based fallback intent classifier.
    Priority: exit > high_intent > greeting > pricing > features > policy > unknown
    """
    text = user_input.lower().strip()
    for intent in ["exit", "high_intent", "greeting", "pricing", "features", "policy"]:
        for kw in INTENT_KEYWORDS[intent]:
            if kw in text:
                return intent
    return "unknown"


# ─── END LLM INTENT DETECTION ────────────────────────────────────────────────


# =============================================================================
# ─── RAG ENGINE ──────────────────────────────────────────────────────────────
#
#  The RAG query is now CONTEXT-ENRICHED:
#    enriched_query = current_message + keywords from recent history
#
#  This means:
#    Turn 1 – "Tell me about pricing"   → retrieves pricing docs
#    Turn 2 – "What about refunds?"     → "what about refunds + pricing basic pro plan"
#                                          → retrieves refund doc (correctly)
#    Turn 3 – "Tell me more"            → "tell me more + refund pricing basic"
#                                          → still finds the right context
#
# =============================================================================

KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.json")

_DOCUMENTS: list[dict] = []
_INDEX:     dict[str, set] = defaultdict(set)
_DF:        dict[str, int] = defaultdict(int)


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


def load_knowledge_base() -> None:
    """Load knowledge_base.json and build the inverted index. Called once at startup."""
    global _DOCUMENTS, _INDEX, _DF

    if not os.path.exists(KB_PATH):
        raise FileNotFoundError(f"knowledge_base.json not found at: {KB_PATH}")

    with open(KB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    _DOCUMENTS = data["documents"]

    for doc in _DOCUMENTS:
        searchable = " ".join([
            doc["title"],
            doc["content"],
            " ".join(doc.get("tags", [])),
            doc["category"],
        ])
        tokens = set(_tokenize(searchable))
        for token in tokens:
            _INDEX[token].add(doc["id"])
            _DF[token] += 1

    print(f"[RAG] Loaded {len(_DOCUMENTS)} documents, {len(_INDEX)} index tokens.")


def build_enriched_query(user_message: str, history: list[dict]) -> str:
    """
    Enrich the current user message with topic keywords from recent history.

    How it works:
      - Extract all user turns from the last HISTORY_WINDOW turns.
      - Tokenize them and keep only "content words" (length > 3, not stopwords).
      - Append the top-8 most frequent content words to the current message.

    Result: even vague follow-ups like "tell me more" carry enough signal for
    the TF-IDF retriever to find the right documents.
    """
    STOPWORDS = {
        "what", "about", "that", "this", "the", "and", "for", "you",
        "your", "have", "does", "with", "how", "can", "tell", "more",
        "just", "also", "like", "will", "from", "its", "are", "was",
        "but", "not", "any", "all", "get", "our", "them", "then",
        "said", "know", "want", "okay", "sure", "yes", "yep", "nope",
    }

    # Collect user-side tokens from recent history
    recent_user_turns = [
        turn["content"]
        for turn in history[-(HISTORY_WINDOW * 2):]
        if turn["role"] == "user"
    ]

    freq: dict[str, int] = defaultdict(int)
    for turn_text in recent_user_turns:
        for tok in _tokenize(turn_text):
            if len(tok) > 3 and tok not in STOPWORDS:
                freq[tok] += 1

    # Top-8 context keywords by frequency
    context_keywords = sorted(freq, key=lambda t: freq[t], reverse=True)[:8]

    if context_keywords:
        return f"{user_message} {' '.join(context_keywords)}"
    return user_message


def rag_retrieve(query: str, top_k: int = 3, min_score: float = 0.5) -> list[dict]:
    """
    TF-IDF retrieval over the knowledge base.
    Scores: (1 + log(tf)) * smoothed_idf, sorted descending, filtered by min_score.
    """
    N = len(_DOCUMENTS)
    if N == 0:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scores: dict[str, float] = defaultdict(float)

    for token in query_tokens:
        if token not in _INDEX:
            continue
        df  = _DF[token]
        idf = math.log((N + 1) / (df + 1)) + 1
        for doc_id in _INDEX[token]:
            doc      = next(d for d in _DOCUMENTS if d["id"] == doc_id)
            doc_text = " ".join([
                doc["title"], doc["content"], " ".join(doc.get("tags", []))
            ]).lower()
            tf       = doc_text.count(token)
            tf_score = 1 + math.log(tf) if tf > 0 else 0
            scores[doc_id] += tf_score * idf

    ranked  = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for doc_id, score in ranked[:top_k]:
        if score < min_score:
            break
        doc = next(d for d in _DOCUMENTS if d["id"] == doc_id)
        results.append({**doc, "score": round(score, 4)})

    return results


# ─── END RAG ENGINE ───────────────────────────────────────────────────────────


# =============================================================================
# NATURAL REPLY COMPOSER
# =============================================================================

def compose_natural_reply(chunks: list[dict], intent: str, user_input: str,
                           history: list[dict]) -> str:
    """
    Weave retrieved chunks into fluent prose.

    v3 addition: uses history to detect follow-up turns and adapt the opening.
    If the user is clearly asking a follow-up (short message, references prior topic),
    the opener acknowledges continuity ("Building on that, ..." / "Great follow-up!")
    instead of starting from scratch.
    """
    if not chunks:
        return ""

    # Detect if this looks like a follow-up (short message + prior history)
    is_followup = len(history) >= 2 and len(user_input.strip().split()) <= 6

    # ------------------------------------------------------------------
    # 1. OPENING
    # ------------------------------------------------------------------
    followup_openings = [
        "Great follow-up question! Building on what we just covered —",
        "Good question — to add a bit more detail on that:",
        "Happy to dig deeper! Here's what you should know:",
        "Sure, let me expand on that for you:",
    ]

    fresh_openings = {
        "pricing":  ["Great question on pricing! Here's a quick breakdown:",
                     "Sure thing — let me walk you through your options:",
                     "Happy to help with pricing. Here's what we've got:"],
        "features": ["AutoStream packs in quite a lot — let me highlight what's most relevant:",
                     "Good news — AutoStream's got you covered here. Let me explain:",
                     "Here's what AutoStream can do for you:"],
        "policy":   ["Totally fair to ask — here's the honest answer:",
                     "Here's everything you need to know on that:",
                     "No worries, I'll clear that up for you right now:"],
        "unknown":  ["Let me pull up what I know about that:",
                     "Here's what I found that might help:"],
    }

    if is_followup:
        opening = random.choice(followup_openings)
    else:
        pool    = fresh_openings.get(intent, fresh_openings["unknown"])
        opening = random.choice(pool)

    # ------------------------------------------------------------------
    # 2. BODY — stitch chunks with transitions
    # ------------------------------------------------------------------
    transitions = [
        "On top of that,",
        "Also worth knowing —",
        "And one more thing:",
        "Additionally,",
        "You'll also be glad to hear that",
    ]

    body_parts = []
    for i, chunk in enumerate(chunks):
        text = chunk["content"].strip()
        if i == 0:
            body_parts.append(text)
        else:
            connector = transitions[min(i - 1, len(transitions) - 1)]
            body_parts.append(f"{connector} {text[0].lower()}{text[1:]}")

    body = " ".join(body_parts)

    # ------------------------------------------------------------------
    # 3. CLOSER
    # ------------------------------------------------------------------
    closers = {
        "pricing":  "Want to dive in? Just say 'I want to get started' and I'll set you up. 🚀",
        "features": "Curious to try it? Feel free to ask about pricing or say 'I want to subscribe'. 😊",
        "policy":   "Any other questions? I'm happy to help with pricing, features, or anything else.",
        "unknown":  "Anything else I can help you with?",
    }
    top_category = chunks[0]["category"]
    closer       = closers.get(top_category, closers["unknown"])

    return f"{opening}\n\n{body}\n\n{closer}"


# =============================================================================
# LEAD CAPTURE
# =============================================================================

def mock_lead_capture(name: str, email: str, platform: str) -> None:
    """Simulate saving a lead to CRM. Replace with a real API call in production."""
    print("\n" + "=" * 55)
    print("  ✅ Lead captured successfully!")
    print(f"     Name     : {name}")
    print(f"     Email    : {email}")
    print(f"     Platform : {platform}")
    print("=" * 55 + "\n")


# =============================================================================
# SESSION HELPERS
# =============================================================================

SESSION_STORE: dict = {}


def fresh_state() -> dict:
    return {
        "intent":          None,
        "intent_source":   "keyword_fallback",
        "collecting_lead": False,
        "lead_step":       None,
        "name":            None,
        "email":           None,
        "platform":        None,
        # NEW: full conversation history as list of {"role", "content"} dicts
        "history":         [],
    }


def get_or_create_session(session_id: str) -> dict:
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = fresh_state()
    return SESSION_STORE[session_id]


def append_history(state: dict, role: str, content: str) -> None:
    """
    Add a turn to conversation history.
    Caps history at 40 turns to prevent unbounded memory growth.
    """
    state["history"].append({"role": role, "content": content})
    if len(state["history"]) > 40:
        state["history"] = state["history"][-40:]


def is_valid_email(email: str) -> bool:
    return bool(re.match(r"^[\w\.-]+@[\w\.-]+\.\w{2,}$", email.strip()))


# =============================================================================
# CORE BOT LOGIC
# =============================================================================

async def get_bot_response(user_input: str, state: dict) -> tuple:
    """
    Async because we now await the LLM intent call.

    Returns: (reply, chunks, lead_captured, session_ended)

    Flow:
      1. Record user turn in history.
      2. If collecting lead → step-by-step (no intent needed).
      3. Otherwise → LLM intent detection (keyword fallback on failure)
                   → context-enriched RAG query
                   → compose natural reply.
      4. Record bot reply in history.
    """
    lead_captured = False
    session_ended = False
    chunks: list[dict] = []

    # Always record what the user said first
    append_history(state, "user", user_input)

    # ------------------------------------------------------------------
    # STAGE 1 – Lead capture (no intent detection needed mid-flow)
    # ------------------------------------------------------------------
    if state["collecting_lead"]:
        step = state["lead_step"]

        if step == "name":
            if len(user_input.strip()) < 2:
                reply = "Please enter your full name (at least 2 characters)."
            else:
                state["name"] = user_input.strip().title()
                state["lead_step"] = "email"
                reply = f"Nice to meet you, {state['name']}! 📧 What's your email address?"

        elif step == "email":
            if not is_valid_email(user_input):
                reply = "That doesn't look like a valid email. Please try again (e.g. you@example.com)."
            else:
                state["email"] = user_input.strip().lower()
                state["lead_step"] = "platform"
                reply = (
                    "Great! 📱 Which platform do you primarily create content for?\n"
                    "   (e.g. YouTube, Instagram, TikTok, Facebook, LinkedIn)"
                )

        elif step == "platform":
            if len(user_input.strip()) < 2:
                reply = "Please enter a platform name (e.g. YouTube, Instagram)."
            else:
                state["platform"] = user_input.strip().title()
                mock_lead_capture(state["name"], state["email"], state["platform"])
                lead_captured = True
                state["collecting_lead"] = False
                state["lead_step"] = None
                reply = (
                    f"🎉 You're all set, {state['name']}!\n\n"
                    f"Our team will reach out to you at {state['email']} within 24 hours.\n\n"
                    "Feel free to ask me anything else about AutoStream! 🚀"
                )
        else:
            reply = "Something went wrong. Let's start over — what's your full name?"
            state["collecting_lead"] = False
            state["lead_step"] = None

        append_history(state, "assistant", reply)
        return reply, chunks, lead_captured, session_ended

    # ------------------------------------------------------------------
    # STAGE 2 – Intent detection (LLM first, keyword fallback)
    # ------------------------------------------------------------------

    # Try LLM intent detection with full history context
    # We pass history BEFORE the current user turn (history[-1] is the turn
    # we just appended, so we send history[:-1] as prior context)
    prior_history = state["history"][:-1]
    intent = await llm_detect_intent(user_input, prior_history)

    if intent is not None:
        state["intent_source"] = "llm"
    else:
        # LLM failed or timed out — use keywords
        intent = keyword_detect_intent(user_input)
        state["intent_source"] = "keyword_fallback"

    state["intent"] = intent

    # ------------------------------------------------------------------
    # STAGE 3 – Route by intent
    # ------------------------------------------------------------------

    if intent == "exit":
        session_ended = True
        reply = "👋 Thanks for chatting with AutoStream. See you soon!"
        append_history(state, "assistant", reply)
        return reply, chunks, lead_captured, session_ended

    if intent == "greeting":
        reply = (
            "👋 Hello! Welcome to AutoStream – your AI-powered video editing platform!\n\n"
            "I can help you with:\n"
            "  • 💰 Pricing & plans\n"
            "  • ✨ Features & capabilities\n"
            "  • 📋 Refund & support policies\n"
            "  • 🚀 Getting started\n\n"
            "What would you like to know?"
        )
        append_history(state, "assistant", reply)
        return reply, chunks, lead_captured, session_ended

    if intent == "high_intent":
        state["collecting_lead"] = True
        state["lead_step"] = "name"
        reply = (
            "🌟 Awesome! Let's get you set up with AutoStream!\n\n"
            "I just need a few quick details to create your account.\n\n"
            "👤 First, what's your full name?"
        )
        append_history(state, "assistant", reply)
        return reply, chunks, lead_captured, session_ended

    # ------------------------------------------------------------------
    # STAGE 4 – Context-enriched RAG retrieval + natural reply
    # ------------------------------------------------------------------

    # Build an enriched query that folds in keywords from prior turns
    enriched_query = build_enriched_query(user_input, prior_history)
    print(f"[RAG] Enriched query: {enriched_query!r}")

    chunks = rag_retrieve(query=enriched_query, top_k=3, min_score=0.5)

    if chunks:
        reply = compose_natural_reply(
            chunks,
            intent=intent,
            user_input=user_input,
            history=prior_history,
        )
    else:
        reply = (
            "Hmm, I didn't quite catch that — but I'm here to help! 😊 "
            "I can tell you about our pricing plans, walk you through AutoStream's features, "
            "or answer questions about our refund and support policies. "
            "What would you like to know?"
        )

    append_history(state, "assistant", reply)
    return reply, chunks, lead_captured, session_ended


# =============================================================================
# STARTUP EVENT
# =============================================================================

@app.on_event("startup")
async def startup_event():
    load_knowledge_base()


# =============================================================================
# ROUTES
# =============================================================================

@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a message to the AutoStream chatbot",
)
async def chat(request: ChatRequest):
    """
    **POST /chat** – Main conversational endpoint.

    **v3 improvements:**
    - `intent_source` field in response tells you whether intent came from the
      LLM (`"llm"`) or keyword fallback (`"keyword_fallback"`).
    - Follow-up messages like *"tell me more"*, *"what about refunds?"*, or
      *"and the other plan?"* now resolve correctly using conversation history.
    - RAG queries are enriched with keywords from prior turns.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if not request.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id cannot be empty.")

    state = get_or_create_session(request.session_id)

    reply, chunks, lead_captured, session_ended = await get_bot_response(
        request.message, state
    )

    if session_ended:
        SESSION_STORE.pop(request.session_id, None)

    return ChatResponse(
        session_id=request.session_id,
        reply=reply,
        intent=state.get("intent"),
        intent_source=state.get("intent_source", "keyword_fallback"),
        retrieved_chunks=[
            RetrievedChunk(
                id=c["id"],
                title=c["title"],
                category=c["category"],
                score=c["score"],
                content=c["content"],
            )
            for c in chunks
        ],
        lead_captured=lead_captured,
        session_ended=session_ended,
    )


@app.get("/chat/{session_id}/history", summary="View conversation history for a session")
async def get_history(session_id: str):
    """
    **GET /chat/{session_id}/history** – Return the full turn-by-turn history
    stored for a session. Useful for debugging context-aware responses.
    """
    if session_id not in SESSION_STORE:
        raise HTTPException(status_code=404, detail="Session not found.")
    state = SESSION_STORE[session_id]
    return {
        "session_id": session_id,
        "turn_count": len(state["history"]),
        "history": state["history"],
    }


@app.get("/knowledge", summary="Browse the knowledge base")
async def list_knowledge(category: Optional[str] = None):
    """Return all documents. Optionally filter by category: pricing | features | policy."""
    docs = _DOCUMENTS
    if category:
        docs = [d for d in docs if d["category"] == category]
    return {"total": len(docs), "documents": docs}


@app.get("/knowledge/search", summary="Test RAG retrieval directly")
async def search_knowledge(q: str, top_k: int = 3):
    """Run the RAG retriever directly to test relevance scores."""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query 'q' cannot be empty.")
    results = rag_retrieve(query=q, top_k=top_k)
    return {"query": q, "top_k": top_k, "results": results}


@app.delete("/chat/{session_id}", summary="Clear a chat session")
async def clear_session(session_id: str):
    if session_id not in SESSION_STORE:
        raise HTTPException(status_code=404, detail="Session not found.")
    SESSION_STORE.pop(session_id)
    return {"message": f"Session '{session_id}' cleared."}


@app.get("/", summary="Health check")
async def root():
    return {
        "status": "ok",
        "service": "AutoStream AI Chatbot",
        "version": "3.0.0",
        "kb_documents": len(_DOCUMENTS),
        "kb_index_tokens": len(_INDEX),
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
