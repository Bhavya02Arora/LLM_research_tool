import os
import streamlit as st
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_community.document_loaders import SeleniumURLLoader

load_dotenv()
st.title("ReseachTool: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
faiss_path = "faiss_index"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)


if process_url_clicked:
    urls = [u.strip() for u in urls if u.strip()]  # remove empty entries
    if not urls:
        st.error("Please enter at least one valid URL.")
    else:
        try:
            loader = SeleniumURLLoader(urls=urls)
            main_placeholder.text("Data Loading...Started...✅✅✅")
            data = loader.load()
            # loader = UnstructuredURLLoader(urls=urls)

            # data = loader.load()

            st.write(f"Loaded {len(data)} documents")
            for doc in data:
                st.write(doc.metadata.get("source", "No Source"))
                st.write(doc.page_content[:500])  # preview first 500 characters


            if not data:
                st.error("No content could be loaded from the provided URLs.")
            else:
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=['\n\n', '\n', '.', ','],
                    chunk_size=1000
                )
                main_placeholder.text("Text Splitter...Started...✅✅✅")
                docs = text_splitter.split_documents(data)

                if not docs:
                    st.error("No documents to process after splitting.")
                else:
                    embeddings = OpenAIEmbeddings()
                    vectorstore_openai = FAISS.from_documents(docs, embeddings)
                    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
                    time.sleep(2)

                    # Save FAISS index safely (no pickle issue)
                    vectorstore_openai.save_local(faiss_path)
                    st.success("Index saved successfully!")

        except Exception as e:
            st.error(f"An error occurred: {e}")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(faiss_path):
        try:
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)

            st.header("Answer")
            st.write(result["answer"])

            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                for source in sources.split("\n"):
                    st.write(source)

        except Exception as e:
            st.error(f"An error occurred while loading the index: {e}")


# streamlit run main.py
