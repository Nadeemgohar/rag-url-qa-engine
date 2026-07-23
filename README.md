#  🎗️ QA-RAG-GenAi Project

An **AI-powered web application** built with **Streamlit** that fetches content from websites and answers questions based on that content using Google GenAI and ChromaDB.

---

## 🚀 Demo
🔗 [(https://activity-cowljgbrhajxltzpba6qmq.streamlit.app/)](https://activity-cowljgbrhajxltzpba6qmq.streamlit.app/)

---
## screenshots
<img width="1837" height="890" alt="Screenshot from 2026-07-16 16-26-09" src="https://github.com/user-attachments/assets/0853ee79-55db-4c0f-8449-5aee929f885c" />


---

## 📌 Features
- Fetches and parses website content using a URL.
- Converts website text into **vector embeddings** for semantic search.
- Answers user questions based on website content using **Google GenAI**.
- Interactive, user-friendly **Streamlit interface**.
- Displays results in clear **cards** for easy readability.
- Includes an **About Me sidebar** with portfolio links.

---

## 🔍 Usage
1. Open the app in your browser.
2. Enter a website URL in the input field.
3. Click **Fetch** to extract website content.
4. Enter your question in the input box.
5. Click **Ask AI** to get the answer.
6. View the result:
   - ✅ Answer displayed in context cards
   - 📝 Optionally explore the scraped content

---

## 📊 Dataset / Memory
- No explicit dataset required.
- Website content is converted into **vector embeddings** and stored in **ChromaDB** for semantic search.
- Embeddings allow the app to answer questions based on meaning rather than just keywords.

---

## ⚙️ Tech Stack
- **Python 3.9+**
- **Streamlit** – Frontend web app
- **Requests** – Web content fetching
- **BeautifulSoup** – Web content parsing
- **ChromaDB** – Vector database for embeddings
- **Google GenAI SDK** – LLM for text embedding and generation
- **NumPy & Pandas** – Data processing
- **pysqlite3-binary** – Streamlit Cloud SQLite compatibility

## 👨‍💻 Author  
**Nadeem Gohar**  

- 💼 [LinkedIn](https://www.linkedin.com/in/nadeem-gohar-0708382b0/)
- 💻 [GitHub](https://github.com/Nadeemgohar)  


---

## ❤️ Acknowledgements
- [Google GenAI SDK Documentation](https://developers.generativeai.google/)
- [ChromaDB Documentation](https://www.trychroma.com/)
- [BeautifulSoup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- Open-source community for Python ML & AI tools

---

## ⚠️ Disclaimer
This project is for **educational purposes only** and should NOT be used as a substitute for professional advice or factual information verification.
