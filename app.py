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
   insurer's site) and use read_document on it. Quote the relevant clause briefly and
   name the section it came from.
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


def web_search(query: str, trusted_only: bool = True) -> list:
    kwargs = {"query": query, "search_depth": "advanced", "max_results": 6}
    if trusted_only:
        kwargs["include_domains"] = TRUSTED_DOMAINS
    res = tavily.search(**kwargs)
    return [
        {
            "title": r.get("title"),
            "url": r.get("url"),
            "snippet": (r.get("content") or "")[:1500],
        }
        for r in res.get("results", [])
    ]


def read_document(url: str) -> dict:
    res = tavily.extract(urls=[url])
    results = res.get("results", [])
    if not results:
        return {"url": url, "error": "Could not read this page. Try another source."}
    text = results[0].get("raw_content") or ""
    return {"url": url, "text": text[:15000], "truncated": len(text) > 15000}


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
            "description": "Read the full text of a web page or policy document (e.g. a policy wording PDF) from a URL.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
]

FUNCTIONS = {"web_search": web_search, "read_document": read_document}


def run_agent(history: list) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history[-10:]
    for _ in range(8):  # max research steps per question
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
    st.session_state.history.append({"role": "assistant", "content": answer})
