# =============================================================================
# AutoStream Chatbot – Streamlit UI
# =============================================================================
#
# HOW TO RUN:
#   pip install streamlit requests
#   streamlit run streamlit_app.py
#
# Make sure your FastAPI server is running first:
#   uvicorn main:app --reload
#
# The UI will be available at: http://localhost:8501
# =============================================================================

import uuid
import requests
import streamlit as st

# =============================================================================
# CONFIG
# =============================================================================

API_BASE_URL = "http://127.0.0.1:8000"   # change if your FastAPI runs elsewhere
API_TIMEOUT  = 10                          # seconds before giving up on a request

# =============================================================================
# PAGE SETUP  (must be the very first Streamlit call)
# =============================================================================

st.set_page_config(
    page_title="AutoStream AI Assistant",
    page_icon="🎬",
    layout="centered",
)

# =============================================================================
# CUSTOM CSS  – clean, minimal chat aesthetic
# =============================================================================

st.markdown("""
<style>
/* ── Global ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0f0f13;
    color: #e8e8f0;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Chat container ── */
[data-testid="stVerticalBlock"] { gap: 0rem; }

/* ── User bubble ── */
.user-bubble {
    display: flex;
    justify-content: flex-end;
    margin: 6px 0;
}
.user-bubble .bubble-inner {
    background: linear-gradient(135deg, #6c63ff, #4f46e5);
    color: #fff;
    padding: 10px 16px;
    border-radius: 18px 18px 4px 18px;
    max-width: 72%;
    font-size: 0.92rem;
    line-height: 1.5;
    box-shadow: 0 2px 8px rgba(108,99,255,0.35);
    word-wrap: break-word;
}

/* ── Bot bubble ── */
.bot-bubble {
    display: flex;
    justify-content: flex-start;
    margin: 6px 0;
}
.bot-bubble .avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, #f59e0b, #ef4444);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    flex-shrink: 0;
    margin-right: 8px;
    margin-top: 2px;
}
.bot-bubble .bubble-inner {
    background: #1e1e2e;
    color: #e2e2f0;
    padding: 10px 16px;
    border-radius: 18px 18px 18px 4px;
    max-width: 72%;
    font-size: 0.92rem;
    line-height: 1.6;
    border: 1px solid #2a2a3e;
    word-wrap: break-word;
    white-space: pre-wrap;
}

/* ── Intent badge ── */
.intent-badge {
    display: inline-block;
    font-size: 0.68rem;
    padding: 2px 8px;
    border-radius: 20px;
    margin-top: 5px;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}
.badge-llm      { background: #1a3a2a; color: #4ade80; border: 1px solid #166534; }
.badge-keyword  { background: #1a2a3a; color: #60a5fa; border: 1px solid #1e40af; }
.badge-lead     { background: #2a1a3a; color: #c084fc; border: 1px solid #6b21a8; }

/* ── Header card ── */
.header-card {
    background: linear-gradient(135deg, #1e1e2e 0%, #12121c 100%);
    border: 1px solid #2a2a3e;
    border-radius: 16px;
    padding: 20px 24px 16px;
    margin-bottom: 20px;
    text-align: center;
}
.header-card h1 {
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #6c63ff, #f59e0b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 4px 0;
}
.header-card p {
    color: #888;
    font-size: 0.82rem;
    margin: 0;
}

/* ── Status dot ── */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}
.dot-green  { background: #4ade80; box-shadow: 0 0 6px #4ade80; }
.dot-red    { background: #f87171; box-shadow: 0 0 6px #f87171; }

/* ── Input area ── */
[data-testid="stTextInput"] input {
    background: #1e1e2e !important;
    border: 1px solid #3a3a5e !important;
    border-radius: 12px !important;
    color: #e2e2f0 !important;
    font-size: 0.92rem !important;
    padding: 10px 14px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #6c63ff !important;
    box-shadow: 0 0 0 2px rgba(108,99,255,0.2) !important;
}

/* ── Send button ── */
[data-testid="stButton"] button {
    background: linear-gradient(135deg, #6c63ff, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    height: 44px !important;
    transition: opacity 0.2s !important;
}
[data-testid="stButton"] button:hover { opacity: 0.88 !important; }

/* ── Divider ── */
hr { border-color: #2a2a3e !important; margin: 12px 0 !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #12121c !important;
    border-right: 1px solid #2a2a3e !important;
}
[data-testid="stSidebar"] * { color: #c8c8e0 !important; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALISATION
# =============================================================================

if "session_id" not in st.session_state:
    # One stable UUID per browser session — survives page reruns
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    # Each message: {"role": "user"|"bot", "text": str, "meta": dict|None}
    st.session_state.messages = []

if "session_ended" not in st.session_state:
    st.session_state.session_ended = False

if "api_ok" not in st.session_state:
    st.session_state.api_ok = None   # None = unknown, True = up, False = down


# =============================================================================
# API HELPERS
# =============================================================================

def check_api_health() -> bool:
    """Ping the FastAPI health endpoint."""
    try:
        r = requests.get(f"{API_BASE_URL}/", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def call_chat_api(session_id: str, message: str) -> dict | None:
    """
    POST /chat and return the parsed response dict.
    Returns None if the call fails.
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"session_id": session_id, "message": message},
            timeout=API_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to the API. Is the FastAPI server running?"}
    except requests.exceptions.Timeout:
        return {"error": "The API took too long to respond. Please try again."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


def reset_session() -> None:
    """Clear chat history and start a fresh session."""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages   = []
    st.session_state.session_ended = False


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("### 🎬 AutoStream")
    st.markdown("---")

    # API health status
    if st.button("Check API Status", use_container_width=True):
        st.session_state.api_ok = check_api_health()

    if st.session_state.api_ok is True:
        st.markdown('<span class="status-dot dot-green"></span> API is online',
                    unsafe_allow_html=True)
    elif st.session_state.api_ok is False:
        st.markdown('<span class="status-dot dot-red"></span> API is offline',
                    unsafe_allow_html=True)
        st.caption("Run: `uvicorn main:app --reload`")
    else:
        st.caption("Click above to check API status")

    st.markdown("---")

    # Session info
    st.markdown("**Session ID**")
    st.code(st.session_state.session_id[:18] + "...", language=None)
    st.caption(f"{len(st.session_state.messages)} messages in history")

    st.markdown("---")

    # New chat
    if st.button("🔄 New Conversation", use_container_width=True):
        reset_session()
        st.rerun()

    st.markdown("---")

    # Quick-start prompts
    st.markdown("**💡 Try asking:**")
    example_prompts = [
        "What are your pricing plans?",
        "Tell me about AI captions",
        "Do you have a free trial?",
        "What's your refund policy?",
        "I want to subscribe",
    ]
    for prompt in example_prompts:
        if st.button(prompt, use_container_width=True, key=f"ex_{prompt}"):
            st.session_state["prefill"] = prompt
            st.rerun()


# =============================================================================
# HEADER
# =============================================================================

st.markdown("""
<div class="header-card">
    <h1>🎬 AutoStream Assistant</h1>
    <p>AI-powered video editing · Ask me anything about plans, features, or pricing</p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# CHAT HISTORY DISPLAY
# =============================================================================

def render_intent_badge(meta: dict) -> str:
    """Render a small badge showing intent and whether LLM or keyword fired."""
    if not meta:
        return ""
    intent = meta.get("intent") or "unknown"
    source = meta.get("intent_source", "")
    lead   = meta.get("lead_captured", False)

    if lead:
        return f'<div><span class="intent-badge badge-lead">✅ lead captured</span></div>'
    if source == "llm":
        return f'<div><span class="intent-badge badge-llm">🤖 llm · {intent}</span></div>'
    elif source == "keyword_fallback":
        return f'<div><span class="intent-badge badge-keyword">🔑 keyword · {intent}</span></div>'
    return ""


chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        # Empty state hint
        st.markdown("""
        <div style="text-align:center; color:#555; padding: 40px 0 20px;">
            <div style="font-size:2.5rem">💬</div>
            <p style="margin-top:8px; font-size:0.88rem">
                Start by saying <em>hello</em>, or ask about pricing or features.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="user-bubble">
                    <div class="bubble-inner">{msg["text"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                badge = render_intent_badge(msg.get("meta"))
                st.markdown(f"""
                <div class="bot-bubble">
                    <div class="avatar">🎬</div>
                    <div>
                        <div class="bubble-inner">{msg["text"]}</div>
                        {badge}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Small spacer so the last message isn't hidden behind the input
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)


# =============================================================================
# INPUT AREA
# =============================================================================

st.markdown("<hr>", unsafe_allow_html=True)

# Handle sidebar quick-prompt injection
prefill_value = st.session_state.pop("prefill", "")

col_input, col_btn = st.columns([5, 1])

with col_input:
    user_input = st.text_input(
        label="message",
        value=prefill_value,
        placeholder="Type a message…  (Enter to send)",
        label_visibility="collapsed",
        key="chat_input",
        disabled=st.session_state.session_ended,
    )

with col_btn:
    send_clicked = st.button(
        "Send",
        use_container_width=True,
        disabled=st.session_state.session_ended,
    )

# Trigger on Enter key OR button click
should_send = (send_clicked or (user_input and user_input != prefill_value)) and user_input.strip()


# =============================================================================
# SEND LOGIC
# =============================================================================

if should_send and not st.session_state.session_ended:
    message_text = user_input.strip()

    # Add user message immediately (optimistic UI)
    st.session_state.messages.append({"role": "user", "text": message_text, "meta": None})

    # Call the API
    with st.spinner("AutoStream is thinking…"):
        result = call_chat_api(st.session_state.session_id, message_text)

    if result is None or "error" in result:
        # Show error as a bot message so it's visible in-chat
        error_text = result["error"] if result else "Unknown error occurred."
        st.session_state.messages.append({
            "role": "bot",
            "text": f"⚠️ {error_text}",
            "meta": None,
        })
    else:
        # Store bot reply with metadata for badge rendering
        st.session_state.messages.append({
            "role": "bot",
            "text":  result.get("reply", ""),
            "meta": {
                "intent":        result.get("intent"),
                "intent_source": result.get("intent_source"),
                "lead_captured": result.get("lead_captured", False),
            },
        })

        # Lock input if the bot ended the session
        if result.get("session_ended"):
            st.session_state.session_ended = True

    st.rerun()


# =============================================================================
# SESSION ENDED BANNER
# =============================================================================

if st.session_state.session_ended:
    st.markdown("""
    <div style="text-align:center; padding: 16px; background:#1a1a2e;
                border-radius:12px; border:1px solid #2a2a4e; margin-top:12px;">
        <span style="font-size:1.1rem">👋 Conversation ended.</span>
        <p style="color:#888; font-size:0.82rem; margin:6px 0 0">
            Click <strong>New Conversation</strong> in the sidebar to start fresh.
        </p>
    </div>
    """, unsafe_allow_html=True)
