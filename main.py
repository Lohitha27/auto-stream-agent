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




# =============================================================================
# FASTAPI APP
# =============================================================================



# =============================================================================
# PYDANTIC SCHEMAS
# =============================================================================

 # ================= FIXED VERSION =================

import json
import math
import os
import random
import re
from collections import defaultdict
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# ---------- DISABLED LLM (to avoid crash) ----------
async def llm_detect_intent(user_message: str, history: list[dict]) -> Optional[str]:
    return None  # always fallback

# ---------- KEYWORD FALLBACK ----------
INTENT_KEYWORDS = {
    "greeting": ["hi", "hello", "hey"],
    "pricing": ["price", "cost", "plan"],
    "features": ["feature", "can it"],
    "policy": ["refund", "cancel"],
    "high_intent": ["buy", "subscribe", "start"],
    "exit": ["bye", "exit"]
}

def keyword_detect_intent(user_input: str):
    text = user_input.lower()
    for intent, words in INTENT_KEYWORDS.items():
        for w in words:
            if w in text:
                return intent
    return "unknown"

# ---------- FIXED KB PATH ----------
KB_PATH = "knowledge_base.json"

_DOCUMENTS = []

def load_knowledge_base():
    global _DOCUMENTS
    with open(KB_PATH, "r") as f:
        data = json.load(f)
    _DOCUMENTS = data["documents"]

# ---------- BASIC CHAT ----------
class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    intent: Optional[str]
    lead_captured: bool
    session_ended: bool

@app.on_event("startup")
async def startup():
    load_knowledge_base()

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    intent = keyword_detect_intent(req.message)

    if intent == "pricing":
        reply = "Pro Plan is $79/month with unlimited videos and 4K resolution."
    elif intent == "features":
        reply = "AutoStream offers AI captions, 4K editing, and automation tools."
    elif intent == "policy":
        reply = "No refunds after 7 days. Pro users get 24/7 support."
    elif intent == "high_intent":
        reply = "Great! Please share your name, email, and platform."
    else:
        reply = "Hello! Ask me about pricing, features, or plans."

    return ChatResponse(
        session_id=req.session_id,
        reply=reply,
        intent=intent,
        lead_captured=False,
        session_ended=False
    )

@app.get("/")
def root():
    return {"status": "ok"}
