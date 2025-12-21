import os
from typing import List, Dict, Optional

import streamlit as st
from huggingface_hub import InferenceClient
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


st.set_page_config(page_title="Dual Provider Chatbot", page_icon="💬")


def get_secret(name: str) -> Optional[str]:
    """Fetch a secret from Streamlit secrets or environment variables."""
    if name in st.secrets:
        return st.secrets[name]
    return os.getenv(name)


def require_key(value: Optional[str], label: str) -> bool:
    """Guard for required credentials."""
    if not value:
        st.error(f"{label} is missing. Add it to .streamlit/secrets.toml or set it as an environment variable.")
        return False
    return True


def build_prompt(messages: List[Dict[str, str]]) -> str:
    """Flatten chat messages into a simple prompt for text-generation models."""
    lines = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prefix = "Assistant" if role == "assistant" else "User"
        lines.append(f"{prefix}: {content}")
    lines.append("Assistant:")
    return "\n".join(lines)


def call_openai(messages: List[ChatCompletionMessageParam], model: str, api_key: str) -> str:
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(model=model, messages=messages)
    return completion.choices[0].message.content.strip()


def call_hf(messages: List[Dict[str, str]], model: str, token: Optional[str] = None) -> str:
    client = InferenceClient(model=model, token=token)
    prompt = build_prompt(messages)
    result = client.text_generation(prompt, max_new_tokens=256, temperature=0.7)
    return result.strip()


def main() -> None:
    st.title("Dual Provider Chatbot")
    st.caption("Choose between OpenAI Chat Completions or a Hugging Face text-generation model.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("Settings")
        provider = st.selectbox("Provider", ["OpenAI", "Hugging Face"])

        st.markdown(f"**Active provider:** `{provider}`")

        openai_model = st.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
        hf_model = st.selectbox("Hugging Face model", ["gpt2", "tiiuae/falcon-7b-instruct"], index=0)

        if st.button("Clear chat", type="secondary"):
            st.session_state.messages = []
            st.success("Chat cleared.")

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    user_input = st.chat_input("Ask something...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        try:
            if provider == "OpenAI":
                openai_api_key = get_secret("OPENAI_API_KEY")
                if not require_key(openai_api_key, "OpenAI API key"):
                    return
                reply = call_openai(st.session_state.messages, model=openai_model, api_key=openai_api_key)
            else:
                hf_token = get_secret("HF_TOKEN")
                if not hf_token:
                    st.info("Using Hugging Face without a token; add HF_TOKEN for stronger models and higher limits.")
                reply = call_hf(st.session_state.messages, model=hf_model, token=hf_token)

            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.chat_message("assistant").write(reply)
        except Exception as exc:
            st.error(f"{provider} error: {exc}")


if __name__ == "__main__":
    main()
