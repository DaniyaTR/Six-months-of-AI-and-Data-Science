import streamlit as st
from google import genai
from langchain_text_splitters import CharacterTextSplitter
import io
import PyPDF2
import docx


def generate_response(uploaded_file, gemini_api_key, query_text):
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(
                io.BytesIO(uploaded_file.read())
            )
            documents = [
                pdf_reader.pages[i].extract_text() or ""
                for i in range(len(pdf_reader.pages))
            ]

        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc_reader = docx.Document(
                io.BytesIO(uploaded_file.read())
            )
            documents = [
                para.text for para in doc_reader.paragraphs
            ]

        else:
            documents = [
                uploaded_file.read().decode("utf-8")
            ]

        text_splitter = CharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200
        )

        texts = text_splitter.split_text(
            "\n".join(documents)
        )

        context = "\n\n".join(texts[:10])

        client = genai.Client(
            api_key=gemini_api_key
        )

        prompt = f"""
You are a helpful document assistant.

Use the following document to answer the user's question.

DOCUMENT:
{context}

QUESTION:
{query_text}

Give a clear and helpful answer based on the document.
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None


st.set_page_config(
    page_title="🦜🔗 Ask the Doc App"
)

st.title("🦜🔗 Ask the Doc App")

uploaded_file = st.file_uploader(
    "Upload an article",
    type=["txt", "pdf", "docx"]
)

query_text = st.text_input(
    "Enter your question:",
    placeholder="Please provide a short summary.",
    disabled=not uploaded_file
)

result = []

with st.form("myform", clear_on_submit=True):

    gemini_api_key = st.text_input(
        "Gemini API Key",
        type="password",
        disabled=not (uploaded_file and query_text)
    )

    submitted = st.form_submit_button(
        "Submit",
        disabled=not (uploaded_file and query_text)
    )

    if submitted and gemini_api_key:

        with st.spinner("Searching for answers..."):

            response = generate_response(
                uploaded_file,
                gemini_api_key,
                query_text
            )

            result.append(response)


if len(result) and result[0]:
    st.info(result[0])