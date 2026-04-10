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
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# -------- LOAD KNOWLEDGE BASE --------
with open("knowledge_base.json", "r") as f:
    kb = json.load(f)

documents = kb["documents"]

# -------- SESSION MEMORY --------
sessions = {}

# -------- LEAD CAPTURE FUNCTION --------
def mock_lead_capture(name, email, platform):
    print(f"Lead captured successfully: {name}, {email}, {platform}")

# -------- REQUEST MODEL --------
class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    intent: str
    lead_captured: bool
    session_ended: bool

# -------- SIMPLE RAG --------
def get_rag_response(query):
    query = query.lower()
    for doc in documents:
        if any(word in doc.lower() for word in query.split()):
            return doc
    return "Sorry, I couldn't find relevant info."

# -------- INTENT DETECTION --------
def detect_intent(text):
    text = text.lower()

    if any(x in text for x in ["hi", "hello", "hey"]):
        return "greeting"
    elif any(x in text for x in ["price", "plan", "cost"]):
        return "pricing"
    elif any(x in text for x in ["buy", "subscribe", "start"]):
        return "high_intent"
    else:
        return "general"

# -------- MAIN CHAT API --------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id
    message = req.message

    # Initialize session
    if session_id not in sessions:
        sessions[session_id] = {
            "stage": None,
            "name": None,
            "email": None,
            "platform": None
        }

    session = sessions[session_id]

    # -------- LEAD FLOW --------
    if session["stage"] == "ask_name":
        session["name"] = message
        session["stage"] = "ask_email"
        return ChatResponse(session_id=session_id,
                            reply="Please provide your email.",
                            intent="lead_capture",
                            lead_captured=False,
                            session_ended=False)

    elif session["stage"] == "ask_email":
        session["email"] = message
        session["stage"] = "ask_platform"
        return ChatResponse(session_id=session_id,
                            reply="Which platform do you create content on? (YouTube/Instagram)",
                            intent="lead_capture",
                            lead_captured=False,
                            session_ended=False)

    elif session["stage"] == "ask_platform":
        session["platform"] = message

        # CALL TOOL
        mock_lead_capture(
            session["name"],
            session["email"],
            session["platform"]
        )

        session["stage"] = None

        return ChatResponse(session_id=session_id,
                            reply="🎉 Thank you! You are successfully registered.",
                            intent="lead_capture",
                            lead_captured=True,
                            session_ended=True)

    # -------- NORMAL FLOW --------
    intent = detect_intent(message)

    if intent == "greeting":
        reply = "Hello! How can I help you with AutoStream today?"

    elif intent == "pricing":
        reply = get_rag_response(message)

    elif intent == "high_intent":
        session["stage"] = "ask_name"
        reply = "Great! Let's get you started. What is your name?"

    else:
        reply = get_rag_response(message)

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        intent=intent,
        lead_captured=False,
        session_ended=False
    )

# -------- ROOT --------
@app.get("/")
def root():
    return {"status": "ok"}
