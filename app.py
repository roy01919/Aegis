import os
from pathlib import Path
from typing import List, Dict, Optional

import streamlit as st
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None  # type: ignore


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


def find_gguf(model_path: Path) -> Optional[Path]:
    """Locate a GGUF file inside a path or use the file directly."""
    if model_path.is_file() and model_path.suffix.lower() == ".gguf":
        return model_path
    if model_path.is_dir():
        ggufs = sorted(model_path.glob("*.gguf"))
        if ggufs:
            return ggufs[0]
    return None


@st.cache_resource(show_spinner="Loading GGUF model...")
def load_gguf_model(gguf_path: str) -> Llama:
    if Llama is None:
        raise ImportError("llama-cpp-python is required for GGUF models.")
    return Llama(model_path=gguf_path, n_ctx=4096)


@st.cache_resource(show_spinner="Loading Transformers model...")
def load_transformer_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        device_map="auto",
    )
    return model, tokenizer


def call_openai(messages: List[ChatCompletionMessageParam], model: str, api_key: str) -> str:
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(model=model, messages=messages)
    return completion.choices[0].message.content.strip()


def call_local_llama(
    messages: List[Dict[str, str]],
    model_path: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    resolved_path = Path(model_path).expanduser()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Model path not found: {resolved_path}")

    gguf_path = find_gguf(resolved_path)
    if gguf_path:
        llama = load_gguf_model(str(gguf_path))
        prompt = build_prompt(messages)
        output = llama(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        text = output["choices"][0]["text"]
        return text.strip()

    model, tokenizer = load_transformer_model(str(resolved_path))
    apply_chat = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat):
        prompt = apply_chat(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = build_prompt(messages)

    inputs = tokenizer(prompt, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
    )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    st.title("Dual Provider Chatbot")
    st.caption("Choose between OpenAI Chat Completions or a Hugging Face text-generation model.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("Settings")
        provider = st.selectbox("Provider", ["OpenAI", "Local Llama 3.1 (8B)"])

        st.markdown(f"**Active provider:** `{provider}`")

        openai_model = st.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
        model_path = st.text_input(
            "Local model path",
            value="/home/roy/.llama/checkpoints/Llama3.1-8B-Instruct",
            help="Path to a GGUF file or a local Transformers model directory.",
        )
        max_new_tokens = st.slider("Max new tokens", min_value=64, max_value=1024, value=256, step=64)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.7, step=0.05)

        if Path(model_path).expanduser().exists():
            gguf_hint = find_gguf(Path(model_path).expanduser())
            ready_text = "GGUF model detected." if gguf_hint else "Transformers model detected."
            st.caption(f"Local model ready: {ready_text}")
        else:
            st.caption("Local model path not found yet.")

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
                reply = call_local_llama(
                    st.session_state.messages,
                    model_path=model_path,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )

            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.chat_message("assistant").write(reply)
        except Exception as exc:
            st.error(f"{provider} error: {exc}")


if __name__ == "__main__":
    main()
