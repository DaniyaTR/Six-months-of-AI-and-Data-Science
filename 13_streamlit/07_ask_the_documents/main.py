import streamlit as st
import PyPDF2
import google.generativeai as genai
import os


# Gemini API key
gemini_api_key = st.sidebar.text_input(
    "Gemini API Key",
    type="password"
)

if gemini_api_key:
    genai.configure(api_key=gemini_api_key)


# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        return text


# List PDF files
def list_pdf_files(directory):
    pdf_files = []

    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdf"):
            pdf_files.append(
                os.path.join(directory, filename)
            )

    return pdf_files


# Generate question using Gemini
def get_questions_from_gemini(text):
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
Create one useful question from this document:

{text[:12000]}
"""

    response = model.generate_content(prompt)

    return response.text.strip()


# Generate answer using Gemini
def get_answers_from_gemini(text, question):
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
Use the following document to answer the question.

DOCUMENT:
{text[:12000]}

QUESTION:
{question}

Give a clear and helpful answer.
"""

    response = model.generate_content(prompt)

    return response.text.strip()


# Main app
def main():

    st.title("Ask Questions From PDF Documents in Folder")

    if not gemini_api_key:
        st.info("Please enter your Gemini API Key in the sidebar.")
        return

    pdf_folder = st.text_input(
        "Enter the folder path containing PDF files:"
    )

    if pdf_folder and os.path.isdir(pdf_folder):

        pdf_files = list_pdf_files(pdf_folder)

        if not pdf_files:

            st.warning(
                "No PDF files found in the specified folder."
            )

        else:

            st.info(
                f"Number of PDF files found: {len(pdf_files)}"
            )

            selected_pdf = st.selectbox(
                "Select a PDF file",
                pdf_files
            )

            st.info(
                f"Selected PDF: {selected_pdf}"
            )

            text = extract_text_from_pdf(
                selected_pdf
            )

            question = get_questions_from_gemini(
                text
            )

            st.write(
                "Generated Question: " + question
            )

            user_question = st.text_input(
                "Ask a question about the document"
            )

            if user_question:

                answer = get_answers_from_gemini(
                    text,
                    user_question
                )

                st.write(
                    "Answer: " + answer
                )


if __name__ == "__main__":
    main()