import streamlit as st
from chat import build_rag_chain, ask

st.set_page_config(
    page_title="Terveyskirjasto AI",
    page_icon="🏥",
    layout="centered",
)

st.markdown(
    """
    <style>
    .tk-header {
        background: #004f9e;
        padding: 18px 32px 14px 32px;
        border-radius: 0 0 8px 8px;
        margin-bottom: 8px;
    }
    .tk-header h1 {
        margin: 0;
        font-size: 1.55rem;
        font-weight: 700;
        color: #ffffff;
    }
    .tk-header p {
        margin: 4px 0 0 0;
        font-size: 0.87rem;
        color: #b8d4ee;
    }
    .welcome-card {
        background: #ffffff;
        border: 1px solid #ccdff0;
        border-left: 4px solid #0072ce;
        border-radius: 8px;
        padding: 24px 28px;
        margin: 20px 0 12px 0;
        color: #1a2a3a;
    }
    .welcome-card h3 {
        margin: 0 0 10px 0;
        color: #004f9e;
        font-size: 1.05rem;
    }
    .welcome-card ul {
        margin: 10px 0 0 0;
        padding-left: 20px;
        color: #4a6070;
        font-size: 0.91rem;
        line-height: 1.8;
    }
    .disclaimer {
        background: #fff8e1;
        border: 1px solid #ffe082;
        border-radius: 6px;
        padding: 10px 16px;
        font-size: 0.82rem;
        color: #5a4000;
        margin-bottom: 20px;
    }
    .source-pill {
        display: inline-block;
        background: #e8f2fb;
        border: 1px solid #ccdff0;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.76rem;
        margin: 3px 3px 0 0;
        color: #004f9e;
        font-family: monospace;
        text-decoration: none;
    }
    .source-pill:hover {
        background: #ccdff0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="tk-header">
        <h1>🏥 Terveyskirjasto AI</h1>
        <p>Tekoälyavusteinen hakupalvelu Duodecimin terveyskirjastoon</p>
    </div>
    """,
    unsafe_allow_html=True,
)

def source_pill_html(src: str) -> str:
    if src.startswith("http"):
        label = src.rstrip("/").split("/")[-1]
        return f'<a href="{src}" target="_blank" class="source-pill">{label}</a>'
    return f'<span class="source-pill">{src}</span>'


# ── Load RAG chain ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Ladataan tietokantaa...")
def load_chain():
    return build_rag_chain()

retriever, chain = load_chain()

# ── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Welcome screen (shown until first message) ───────────────────────────────
if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-card">
            <h3>Tervetuloa – miten voin auttaa?</h3>
            Voit kysyä terveyteen liittyviä kysymyksiä suomeksi tai englanniksi.
            Vastaukset perustuvat Terveyskirjaston lääketieteellisiin artikkeleihin.
            <ul>
                <li>Mitkä ovat diabeteksen oireet?</li>
                <li>Miten verenpainetta voi alentaa?</li>
                <li>Milloin kannattaa hakeutua lääkäriin selkäkivun takia?</li>
            </ul>
        </div>
        <div class="disclaimer">
            ⚠️ <strong>Huomio:</strong> Tämä palvelu tarjoaa yleistä terveysinformaatiota eikä korvaa lääkärin arviota.
            Hakeudu lääkäriin, jos tarvitset henkilökohtaista hoitoa.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Lähteet", expanded=False):
                for src in msg["sources"]:
                    st.markdown(source_pill_html(src), unsafe_allow_html=True)

# ── Chat input ────────────────────────────────────────────────────────────────
if question := st.chat_input("Kirjoita kysymyksesi tähän..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Haetaan tietoa artikkelitietokannasta..."):
            answer, docs = ask(question, retriever, chain)

        st.markdown(answer)

        sources = list({
            d.metadata.get("url") or d.metadata.get("source", "unknown")
            for d in docs
        })
        if sources:
            with st.expander("Lähteet", expanded=False):
                for src in sources:
                    st.markdown(source_pill_html(src), unsafe_allow_html=True)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
