import os
import io
import json
import time
import threading
from datetime import date

import markdown
import streamlit as st
from fpdf import FPDF
from fpdf.fonts import TextStyle
from pypdf import PdfReader
from openai import OpenAI
from tavily import TavilyClient

st.set_page_config(page_title="Insurance Policy Assistant", page_icon="🛡️")

# The application looks for these variables automatically in the cloud environment
def get_setting(name, default=None):
    """Read a setting from environment variables, or from Streamlit secrets."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


TAVILY_API_KEY = get_setting("TAVILY_API_KEY")
GEMINI_API_KEY = get_setting("GEMINI_API_KEY")
# Gemini model to use. Change via the GEMINI_MODEL setting if Google renames it.
MODEL = get_setting("GEMINI_MODEL", "gemini-3.8-flash")

if not TAVILY_API_KEY or not GEMINI_API_KEY:
    st.error("Missing API Keys! Please configure TAVILY_API_KEY and GEMINI_API_KEY in your hosting dashboard.")
    st.stop()

tavily = TavilyClient(api_key=TAVILY_API_KEY)
# Gemini accepts the OpenAI library when pointed at Google's address
llm = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    max_retries=2,
)


# Models to use, in order of preference, with the requests per minute the app will
# send to each (a little under each model's free-tier limit). Each model has its own
# separate free allowance, so when one is busy or used up the app moves to the next.
# If you turn on billing, set GEMINI_RPM in Secrets (e.g. "100") to lift the main model's pace.
MODEL_PLAN = [
    (MODEL, int(get_setting("GEMINI_RPM", 4))),
    ("gemini-2.5-flash-lite", 9),
    ("gemini-2.5-flash", 4),
]


@st.cache_resource
def shared_model_state():
    """Shared by ALL visitors, because they all use the same Gemini quota."""
    return {"lock": threading.Lock(), "times": {}, "cool_until": {}, "reason": {}}


def take_slot(model: str, rpm: int) -> bool:
    """Reserve a request for this model if it is under its per-minute pace and not cooling down."""
    state = shared_model_state()
    with state["lock"]:
        now = time.time()
        if state["cool_until"].get(model, 0) > now:
            return False
        recent = [t for t in state["times"].get(model, []) if now - t < 60]
        if len(recent) >= rpm:
            state["times"][model] = recent
            return False
        recent.append(now)
        state["times"][model] = recent
        return True


def cool_down(model: str, err: str):
    """After a refusal, rest this model: an hour if the daily quota is gone, else a minute."""
    daily = "429" in err and "day" in err.lower()
    if daily:
        reason = "daily free limit used up"
    elif "429" in err:
        reason = "per-minute limit, resting 1 min"
    else:
        reason = "error: " + err[:160]
    state = shared_model_state()
    with state["lock"]:
        minute_limit = "429" in err and not daily
        state["cool_until"][model] = time.time() + (60 if minute_limit else 3600)
        state["reason"][model] = reason


def model_status_text() -> str:
    state = shared_model_state()
    now = time.time()
    parts = []
    for model, _ in MODEL_PLAN:
        if state["cool_until"].get(model, 0) > now:
            parts.append(f"{model}: {state['reason'].get(model, 'resting')}")
    return " | ".join(parts)


def ask_gemini(**kwargs):
    """Send the request to the first model that has room; wait only if none do."""
    kwargs.pop("model", None)
    note = st.empty()
    last_error = "429 all models busy"
    deadline = time.time() + 180  # give up after 3 minutes of waiting
    while time.time() < deadline:
        for model, rpm in MODEL_PLAN:
            if not take_slot(model, rpm):
                continue
            try:
                resp = llm.chat.completions.create(model=model, **kwargs)
                note.empty()
                st.session_state.models_used = st.session_state.get("models_used", set()) | {model}
                return resp
            except Exception as e:
                err = str(e)
                if "429" in err:
                    cool_down(model, err)
                    last_error = err
                    continue
                if model != MODEL_PLAN[0][0] and ("400" in err or "404" in err):
                    # a back-up model couldn't handle this conversation; skip it for now
                    cool_down(model, err)
                    last_error = err
                    continue
                raise
        note.caption("⏳ All free Gemini models are busy, waiting a few seconds... " + model_status_text())
        time.sleep(5)
    note.empty()
    raise Exception(last_error + " || " + model_status_text())

# Sources the bot searches first. Edit this list freely.
TRUSTED_DOMAINS = [
    "irdai.gov.in",
    "policyholder.gov.in",
    "starhealth.in",
    "hdfcergo.com",
    "icicilombard.com",
    "nivabupa.com",
    "careinsurance.com",
    "bajajallianz.com",
    "tataaig.com",
    "adityabirlacapital.com",
    "licindia.in",
    "policybazaar.com",
    "beshak.org",
]

SYSTEM_PROMPT = """You are an insurance research assistant for India.

Rules:
1. Before stating any fact about a specific insurer, policy, premium, or claim ratio,
   use web_search. Never answer such facts from memory.
2. For policy terms, find the official policy wording or brochure (often a PDF on the
   insurer's site) and use read_document on it, with a short "focus" naming what you
   need (e.g. "pre-existing disease waiting period"). Quote the relevant clause briefly
   and name the section it came from.
   Search economy (each search costs money):
   - Write plain search queries. Do NOT use site:, filetype:, OR, or quoted phrases.
   - Use at most 3 searches per question.
   - As soon as you have the policy wording PDF, read it instead of searching again.
     If the focus didn't find the clause, read the same document again with a
     different focus (e.g. "Excl01", "waiting period", "exclusions").
   - Prefer the "Policy Wording"/"Policy Clause" document over the brochure.
3. When comparing policies, use a table with these rows where available:
   sum insured options, initial waiting period, pre-existing disease waiting period,
   specific-disease waiting period, room rent limit, co-payment, sub-limits,
   restoration benefit, no-claim bonus, key exclusions, claim settlement ratio
   (with year and source). Write "Not found" instead of guessing.
4. Explain benefits AND risks/catches (exclusions, sub-limits, co-pay, waiting periods)
   in plain language.
5. Do not tell the user which policy to buy. Explain trade-offs against the needs they
   describe, and suggest they confirm with the insurer or an IRDAI-licensed advisor.
6. For regulatory and market questions (IRDAI circulars, consultation papers, effects on
   insurers or listed companies): give the document's name, date and current status
   (draft / final / in force), summarise what it proposes, then report who analysts
   and news sources say benefits or loses, attributing each view to its source.
   Separate what the regulator said from what commentators expect. Do not give buy or
   sell recommendations on any stock.
7. End every answer with a "Sources" list of the URLs you used, and note that
   terms can change.
"""


def add_credits(n: int):
    st.session_state.credits_used = st.session_state.get("credits_used", 0) + n


def web_search(query: str, trusted_only: bool = True) -> list:
    # "basic" depth costs 1 Tavily credit per search ("advanced" costs 2)
    kwargs = {"query": query, "search_depth": "basic", "max_results": 5}
    if trusted_only:
        kwargs["include_domains"] = TRUSTED_DOMAINS
    res = tavily.search(**kwargs)
    add_credits(1)
    return [
        {
            "title": r.get("title"),
            "url": r.get("url"),
            "snippet": (r.get("content") or "")[:1000],
        }
        for r in res.get("results", [])
    ]


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def fetch_document(url: str) -> str:
    """Download a page/PDF once via Tavily; later reads of the same URL are free for 7 days."""
    res = tavily.extract(urls=[url])
    results = res.get("results", [])
    if not results:
        return ""
    return results[0].get("raw_content") or ""


def pick_relevant_parts(text: str, focus: str, max_chars: int = 12000) -> str:
    """Return the parts of a long document that best match the focus words."""
    chunk_size = 1500
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    words = [w for w in focus.lower().replace("-", " ").split() if len(w) > 2]
    if not words:
        return text[:max_chars]
    scored = []
    for idx, chunk in enumerate(chunks):
        low = chunk.lower().replace("-", " ")
        score = sum(low.count(w) for w in words)
        if score:
            scored.append((score, idx))
    best = sorted(scored, reverse=True)[: max_chars // chunk_size]
    keep = sorted(idx for _, idx in best)  # keep original document order
    if not keep:
        return "(Focus words not found in this document.)\n\n" + text[:3000]
    return "\n\n[...]\n\n".join(chunks[i] for i in keep)


def read_document(url: str, focus: str = "") -> dict:
    cache_key_seen = url in st.session_state.setdefault("fetched_urls", set())
    text = fetch_document(url)
    if not cache_key_seen:
        st.session_state.fetched_urls.add(url)
        add_credits(1)
    if not text:
        return {"url": url, "error": "Could not read this page. Try another source."}
    return {
        "url": url,
        "document_length_chars": len(text),
        "focus": focus,
        "relevant_text": pick_relevant_parts(text, focus),
    }


def read_uploaded_policy(focus: str = "") -> dict:
    text = st.session_state.get("uploaded_text", "")
    if not text:
        return {"error": "No policy document has been uploaded."}
    return {
        "document": st.session_state.get("uploaded_name", "uploaded policy"),
        "focus": focus,
        "relevant_text": pick_relevant_parts(text, focus),
    }


UPLOAD_TOOL = {
    "type": "function",
    "function": {
        "name": "read_uploaded_policy",
        "description": "Read the policy document the user uploaded. Returns the parts most relevant to 'focus'. Free to call; call again with a different focus if needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "focus": {
                    "type": "string",
                    "description": "Key words for what you need, e.g. 'room rent limit'",
                },
            },
            "required": ["focus"],
        },
    },
}


# ---------- PDF download of an answer ----------
PDF_REPLACEMENTS = {
    "₹": "Rs. ", "“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-",
    "•": "-", "…": "...", "≥": ">=", "≤": "<=", "×": "x", " ": " ", "✓": "Yes",
    "✗": "No", "→": "->",
}


def pdf_safe(text: str) -> str:
    """The built-in PDF font only handles basic characters; swap the common others."""
    for old, new in PDF_REPLACEMENTS.items():
        text = text.replace(old, new)
    return text.encode("latin-1", "ignore").decode("latin-1")


def split_tables(text: str) -> list:
    """Split markdown into ('text', ...) and ('table', [rows]) blocks."""
    blocks, buf, rows = [], [], []
    for line in text.splitlines() + [""]:
        if line.strip().startswith("|"):
            if buf:
                blocks.append(("text", "\n".join(buf)))
                buf = []
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not all(set(c) <= set(":- ") for c in cells):  # skip |---|---| line
                rows.append(cells)
        else:
            if rows:
                blocks.append(("table", rows))
                rows = []
            buf.append(line)
    if buf:
        blocks.append(("text", "\n".join(buf)))
    return blocks


def write_markdown_text(pdf: FPDF, text: str):
    if not text.strip():
        return
    heading = dict(color="#1F3A5F", b_margin=2)
    pdf.write_html(
        markdown.markdown(text),
        font_family="helvetica",
        tag_styles={
            "h1": TextStyle(font_size_pt=15, t_margin=6, **heading),
            "h2": TextStyle(font_size_pt=13, t_margin=5, **heading),
            "h3": TextStyle(font_size_pt=12, t_margin=4, **heading),
        },
    )


def draw_table(pdf: FPDF, rows: list):
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 9)
    with pdf.table(
        text_align="LEFT",
        line_height=5,
        first_row_as_headings=True,  # first row is shown in bold
        markdown=True,  # keeps **bold** inside cells
    ) as table:
        for i, r in enumerate(rows):
            row = table.row()
            for cell in r:
                row.cell(cell.replace("**", "") if i == 0 else cell)
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 10)


@st.cache_data(show_spinner=False)
def answer_to_pdf(question: str, answer: str) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Insurance Policy Assistant", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, f"Generated on {date.today():%d %B %Y}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 11)
    pdf.multi_cell(0, 6, pdf_safe("Question: " + question))
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 10)
    try:
        for kind, block in split_tables(pdf_safe(answer)):
            if kind == "table":
                draw_table(pdf, block)
            else:
                write_markdown_text(pdf, block)
    except Exception:
        # Fallback: plain text if the formatted version can't be laid out
        pdf.multi_cell(0, 5, pdf_safe(answer))
    pdf.ln(5)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(
        0, 4,
        "For information only - not financial advice. Policy terms change; "
        "always verify with the insurer or an IRDAI-licensed advisor.",
    )
    return bytes(pdf.output())


def pdf_button(question: str, answer: str, key: str):
    try:
        data = answer_to_pdf(question, answer)
    except Exception:
        return  # never let a PDF problem break the chat
    st.download_button(
        "⬇️ Download as PDF",
        data=data,
        file_name=f"insurance-answer-{date.today():%Y%m%d}.pdf",
        mime="application/pdf",
        key=key,
    )


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for insurers, policies, policy wordings, premiums, claim settlement ratios and regulations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "trusted_only": {
                        "type": "boolean",
                        "description": "True = only trusted insurer/regulator sites. Set False if nothing useful is found.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": "Read a web page or policy document (e.g. a policy wording PDF). Returns the parts most relevant to 'focus'. Re-reading the same URL with a different focus is free.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "focus": {
                        "type": "string",
                        "description": "Key words for what you need, e.g. 'pre-existing disease waiting period'",
                    },
                },
                "required": ["url", "focus"],
            },
        },
    },
]

FUNCTIONS = {
    "web_search": web_search,
    "read_document": read_document,
    "read_uploaded_policy": read_uploaded_policy,
}


def run_agent(history: list) -> str:
    system = SYSTEM_PROMPT
    tools = list(TOOLS)
    if st.session_state.get("uploaded_text"):
        name = st.session_state.get("uploaded_name", "uploaded policy")
        system += (
            f"\nThe user has uploaded a policy document named '{name}'. "
            "For questions about 'my policy', 'this policy' or 'the document', use "
            "read_uploaded_policy first (it is free) and cite it as "
            f"'Uploaded document: {name}' with the section name. Use web_search only "
            "for things the document doesn't cover, such as comparisons with other policies.\n"
        )
        tools.append(UPLOAD_TOOL)
    messages = [{"role": "system", "content": system}] + history[-10:]
    max_steps = 8
    for step in range(max_steps):
        if step == max_steps - 1:
            # Last step: stop researching and answer with what was found
            messages.append({
                "role": "user",
                "content": "Stop researching now. Answer with what you have found so far, "
                           "clearly saying what you could not confirm, and list the sources.",
            })
            resp = ask_gemini(model=MODEL, messages=messages)
            return resp.choices[0].message.content or ""
        resp = ask_gemini(model=MODEL, messages=messages, tools=tools)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content or ""
        messages.append(msg.model_dump(exclude_none=True))
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            label = args.get("query") or args.get("url") or args.get("focus")
            st.caption(f"🔎 {call.function.name}: {label}")
            try:
                result = FUNCTIONS[call.function.name](**args)
            except Exception as e:
                result = {"error": str(e)}
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)}
            )
    return "I couldn't finish researching that. Please try a narrower question."


# ---------------- Chat UI ----------------
st.title("🛡️ Insurance Policy Assistant")
st.caption("For information only — not financial advice. Always verify with the insurer.")

if "history" not in st.session_state:
    st.session_state.history = []

# ---- Sidebar: upload your own policy ----
with st.sidebar:
    st.header("📄 Your policy document")
    st.caption(
        "Upload a policy wording or schedule (PDF) to ask questions about it. "
        "Don't upload documents with personal or medical details: on the free "
        "plan, Google may use what's sent to improve its products."
    )
    uploaded = st.file_uploader("Upload a policy PDF", type="pdf")
    if uploaded is not None and uploaded.name != st.session_state.get("uploaded_name"):
        try:
            reader = PdfReader(io.BytesIO(uploaded.getvalue()))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception:
            text = ""
        st.session_state.uploaded_name = uploaded.name
        st.session_state.uploaded_text = text
    if uploaded is None and st.session_state.get("uploaded_name"):
        st.session_state.uploaded_name = None
        st.session_state.uploaded_text = ""
    if st.session_state.get("uploaded_name"):
        if len(st.session_state.get("uploaded_text", "")) < 200:
            st.warning(
                "No readable text found. This may be a scanned PDF (a picture of pages), "
                "which this version can't read."
            )
        else:
            st.success(f"Loaded: {st.session_state.uploaded_name}")
            st.caption("Try: \"What are the exclusions in my policy?\"")

# ---- If the last question never got an answer, it was stopped ----
stopped_question = None
if st.session_state.history and st.session_state.history[-1]["role"] == "user":
    stopped_question = st.session_state.history.pop()["content"]

# ---- Chat history ----
for i, m in enumerate(st.session_state.history):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and i > 0:
            pdf_button(st.session_state.history[i - 1]["content"], m["content"], key=f"pdf_{i}")

if stopped_question:
    st.info(f"⏹ Stopped. You asked: \"{stopped_question}\". Ask again or try a different question.")

if prompt := st.chat_input("Ask about an insurer, a policy, or compare policies..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    # Clicking any button makes Streamlit restart the page, which ends the current research.
    stop_area = st.empty()
    stop_area.button("⏹ Stop", key="stop_button", help="Stop researching this question")
    with st.chat_message("assistant"):
        st.session_state.credits_used = 0
        st.session_state.models_used = set()
        with st.spinner("Researching..."):
            try:
                answer = run_agent(st.session_state.history)
            except Exception as e:
                err = str(e)
                if "429" in err:
                    st.error(
                        "All free Gemini models are at their limit right now. "
                        "Try again in a few minutes. If it keeps happening, today's free "
                        "quota may be used up; it resets daily."
                    )
                    if " || " in err:
                        st.caption("Details: " + err.split(" || ", 1)[1])
                elif "404" in err or "not found" in err.lower():
                    st.error(f"The model '{MODEL}' wasn't found. Set GEMINI_MODEL to a current Gemini model name.")
                else:
                    st.error(f"Something went wrong: {err}")
                st.session_state.history.pop()  # drop the unanswered question
                st.stop()
        stop_area.empty()  # research finished, hide the Stop button
        st.markdown(answer)
        models = ", ".join(sorted(st.session_state.get("models_used", set()))) or "-"
        st.caption(
            f"Tavily credits used: about {st.session_state.credits_used} · Gemini model(s): {models}"
        )
        pdf_button(prompt, answer, key=f"pdf_{len(st.session_state.history)}")
    st.session_state.history.append({"role": "assistant", "content": answer})
