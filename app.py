import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import re

MODEL_DIR = "./distilgpt2_ai_finetuned"

@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model

def clean_answer(text: str) -> str:
    """
    Post-process the raw model output.
    - Keep only the portion after 'Answer:'.
    - Stop at first blank line or new 'Question:' token if it appears.
    - Collapse repeated spaces/newlines.
    """
    if "Answer:" in text:
        text = text.split("Answer:", 1)[-1]

    # Stop if model starts another Question or Answer block
    m = re.split(r"(?:\n\s*\n|Question:|Answer:)", text, maxsplit=1)
    answer = m[0].strip() if m else text.strip()

    # De-duplicate repeated whitespace
    answer = re.sub(r"\s+", " ", answer).strip()
    return answer

st.set_page_config(page_title="AI Q&A Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 AI Q&A Chatbot (Fine-tuned distilgpt2 on 50 Questions)")

tokenizer, model = load_model()

question = st.text_input(
    "Ask one of the 50 AI questions (or something close):",
    "What is machine learning?"
)

with st.sidebar:
    st.header("Generation Controls")
    temperature = st.slider("Temperature", 0.1, 1.2, 0.6, 0.05)
    top_p = st.slider("Top-p (nucleus sampling)", 0.5, 1.0, 0.95, 0.01)
    repetition_penalty = st.slider("Repetition penalty", 1.0, 2.0, 1.2, 0.05)
    no_repeat_ngram_size = st.slider("No repeat n-gram size", 0, 8, 4, 1)
    max_new_tokens = st.slider("Max new tokens", 20, 300, 120, 10)

if st.button("Get Answer"):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = clean_answer(raw)

    st.markdown("**Answer:** " + answer)
