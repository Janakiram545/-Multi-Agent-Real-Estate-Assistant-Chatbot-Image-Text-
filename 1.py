import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# ---------- CONFIG ----------
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(image):
    raw_image = Image.open(image).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

llm = ChatOpenAI(temperature=0)

agent1_prompt = PromptTemplate.from_template(
    "You are a property inspection assistant. Given an image caption and user query, detect issues and suggest fixes.\n\nImage Caption: {caption}\nUser Text: {text}\nAnswer:"
)
agent1_chain = agent1_prompt | llm | StrOutputParser()

agent2_prompt = PromptTemplate.from_template(
    "You are a legal assistant for tenancy-related questions. Provide jurisdiction-specific, helpful answers.\n\nUser Question: {input}\nAnswer:"
)
agent2_chain = agent2_prompt | llm | StrOutputParser()


def route_agent(image, text):
    if image is not None:
        return "agent1"
    keywords = ["notice", "evict", "deposit", "landlord", "tenant", "rent", "contract"]
    if any(word in text.lower() for word in keywords):
        return "agent2"
    return "ask"


st.set_page_config(page_title="🏠 Real Estate Multi-Agent Bot")
st.title("🏠 Multi-Agent Real Estate Chatbot")

st.write("Ask about property issues or tenancy laws. Upload an image if reporting a physical problem.")

image = st.file_uploader("Upload a property image (optional)", type=["png", "jpg", "jpeg"])
text_input = st.text_area("Enter your question or context")

if st.button("Submit"):
    if not text_input and not image:
        st.warning("Please provide some input.")
    else:
        agent = route_agent(image, text_input)

        if agent == "agent1":
            st.info("Routing to Property Issue Agent 🛠️")
            with st.spinner("Analyzing image..."):
                caption = caption_image(image)
                result = agent1_chain.invoke({"caption": caption, "text": text_input})
            st.success(result)

        elif agent == "agent2":
            st.info("Routing to Tenancy FAQ Agent 📄")
            with st.spinner("Fetching legal advice..."):
                result = agent2_chain.invoke({"input": text_input})
            st.success(result)

            # Optional: Show model usage
            if hasattr(result, "response_metadata"):
                with st.expander("View model debug info"):
                    st.json(result.response_metadata)

        else:
            st.warning("Not sure how to help. Is this about a property issue or a legal question?")
