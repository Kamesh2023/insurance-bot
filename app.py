import os
import json
import streamlit as st
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
)

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
6. End every answer with a "Sources" list of the URLs you used, and note that
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

FUNCTIONS = {"web_search": web_search, "read_document": read_document}


def run_agent(history: list) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history[-10:]
    max_steps = 8
    for step in range(max_steps):
        if step == max_steps - 1:
            # Last step: stop researching and answer with what was found
            messages.append({
                "role": "user",
                "content": "Stop researching now. Answer with what you have found so far, "
                           "clearly saying what you could not confirm, and list the sources.",
            })
            resp = llm.chat.completions.create(model=MODEL, messages=messages)
            return resp.choices[0].message.content or ""
        resp = llm.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content or ""
        messages.append(msg.model_dump(exclude_none=True))
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            label = args.get("query") or args.get("url")
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

for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask about an insurer, a policy, or compare policies..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        st.session_state.credits_used = 0
        with st.spinner("Researching..."):
            try:
                answer = run_agent(st.session_state.history)
            except Exception as e:
                err = str(e)
                if "429" in err:
                    st.error("Free-tier limit reached. Wait a minute and try again.")
                elif "404" in err or "not found" in err.lower():
                    st.error(f"The model '{MODEL}' wasn't found. Set GEMINI_MODEL to a current Gemini model name.")
                else:
                    st.error(f"Something went wrong: {err}")
                st.session_state.history.pop()  # drop the unanswered question
                st.stop()
        st.markdown(answer)
        st.caption(f"Tavily credits used for this answer: about {st.session_state.credits_used}")
    st.session_state.history.append({"role": "assistant", "content": answer})
