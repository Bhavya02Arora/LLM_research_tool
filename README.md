# LLM_research_tool

✨ Features

Enter up to 3 news article URLs in the sidebar

Content loaded via SeleniumURLLoader (fallback: UnstructuredURLLoader)

Text automatically split into chunks for efficient embedding

FAISS vector store created and stored locally (faiss_index/)

Ask questions in natural language and get answers with sources

Built with:

Streamlit for UI

LangChain for retrieval & QA

OpenAI API for embeddings & LLM


# Installation
1. Clone the repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

2. Create and activate virtual environment (Windows example)
python -m venv venv
venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Set up environment variables

Create a .env file in the project root:

OPENAI_API_KEY=your_openai_api_key_here


⚠️ Never commit .env to GitHub.

# Usage

Run the Streamlit app:

streamlit run main.py


Enter 1–3 news article URLs in the sidebar

Click Process URLs to fetch and embed content

Ask a question in the input box

See generated answers + sources 🚀

