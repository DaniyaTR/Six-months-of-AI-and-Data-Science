import streamlit as st
from google import genai

st.title("🦜🔗 Quickstart App")

api_key = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=api_key)

text = st.text_area("Enter text:", "...")

if st.button("Submit"):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=text
    )
    st.info(response.text)