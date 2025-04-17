# 🏡 Multi-Agent Real Estate Assistant Chatbot (Image + Text)

This project is a **Streamlit-based chatbot** that intelligently routes user queries between two specialized agents:

- **🛠️ Agent 1: Property Issue Troubleshooter**
- **📄 Agent 2: Tenancy FAQ Expert**

The chatbot accepts **text** and optional **images** of properties to provide relevant and helpful responses — from identifying mold or damage in a photo to answering legal tenancy questions like deposits and eviction notices.

---

## ✨ Features

- 🖼️ Image-based issue detection using **BLIP (Salesforce image captioning)**
- 🧠 GPT-powered agents via **LangChain + OpenAI**
- 📄 Tenancy law guidance (location-aware, if given)
- 💬 Smart agent routing based on image and keyword detection
- 🌐 Streamlit UI with interactive response display

---

## 🧰 Tech Stack

| Component        | Tool/Library                             |
|------------------|-------------------------------------------|
| UI               | [Streamlit](https://streamlit.io)        |
| LLM Backend      | [OpenAI API](https://openai.com)         |
| Agent Framework  | [LangChain](https://www.langchain.com)   |
| Image Captioning | [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) |
| Secrets Handling | `.env` file via `python-dotenv`          |


