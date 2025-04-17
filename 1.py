import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

# ---------- CONFIG ----------

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(image):
    raw_image = Image.open(image).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# ---------- LANGUAGE MODELS ----------

llm = ChatOpenAI(temperature=0)

# Agent 1: Property Issue Troubleshooter
agent1_prompt = PromptTemplate.from_template(
    "You're a helpful property inspection expert. Based on the image caption and user description, identify the issue and suggest practical next steps.\n\n"
    "🖼️ Image Description: {caption}\n📝 User Description: {text}\n\nWhat could be the issue and how can it be resolved?"
)
agent1_chain = LLMChain(llm=llm, prompt=agent1_prompt)

# Agent 2: Tenancy Law FAQ Assistant
agent2_prompt = PromptTemplate.from_template(
    "You're a friendly tenancy legal assistant. Provide a clear and concise answer to the user's question. If you need more context (like location), ask politely.\n\n"
    "User Question: {input}\n\nAnswer:"
)
agent2_chain = LLMChain(llm=llm, prompt=agent2_prompt)

# ---------- ROUTING ----------

def route_agent(image, text):
    if image is not None:
        return "agent1"
    keywords = ["notice", "evict", "deposit", "landlord", "tenant", "rent", "contract", "agreement", "lease"]
    if any(word in text.lower() for word in keywords):
        return "agent2"
    return "ask"

# ---------- STREAMLIT UI ----------

st.set_page_config(page_title="🏡 Property Assistant Chatbot")
st.title("🏡 Multi-Agent Real Estate Assistant")

st.markdown("""
Welcome! This assistant can help with:
- 🛠️ **Property issues**: Upload a photo to detect problems like cracks or mold.
- 📄 **Tenancy questions**: Ask about rent, evictions, deposits, and more.

Let’s get started!
""")

image = st.file_uploader("📸 Upload an image of the property (optional)", type=["png", "jpg", "jpeg"])
text_input = st.text_area("💬 Describe the issue or ask your question")

if st.button("🧠 Get Answer"):
    if not text_input and not image:
        st.warning("⚠️ Please enter a question or upload an image.")
    else:
        agent = route_agent(image, text_input)

        if agent == "agent1":
            st.info("🛠️ Looks like you're reporting a property issue...")
            with st.spinner("🔍 Analyzing the image and preparing suggestions..."):
                caption = caption_image(image)
                result = agent1_chain.run({"caption": caption, "text": text_input})
            st.success("✅ Here's what I found:")
            st.write(result)

        elif agent == "agent2":
            st.info("📄 Your question seems related to tenancy or legal matters...")
            with st.spinner("🧑‍⚖️ Reviewing legal guidance..."):
                result = agent2_chain.run({"input": text_input})
            st.success("✅ Here's what I found:")
            st.write(result)

        else:
            st.warning("🤖 I'm not sure what you're asking. Are you reporting a property issue or asking a legal question?")
