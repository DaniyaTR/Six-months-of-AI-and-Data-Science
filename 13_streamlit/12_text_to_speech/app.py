import streamlit as st
from gtts import gTTS
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="Text to Speech Converter",
    page_icon="🔊",
    layout="centered"
)

# Title
st.title("🔊 Text to Speech Converter 📝")

# Text input
text = st.text_area(
    "Enter text to convert to speech",
    height=180,
    placeholder="Type or paste your text here..."
)

# Language selection
language = st.selectbox(
    "Select Language",
    [
        ("English", "en"),
        ("Urdu", "ur"),
        ("Hindi", "hi"),
        ("Arabic", "ar"),
        ("French", "fr"),
        ("German", "de"),
        ("Spanish", "es")
    ],
    format_func=lambda x: x[0]
)

# Speech speed
speed = st.slider(
    "Speech Speed",
    min_value=50,
    max_value=200,
    value=100,
    step=10
)

# Generate button
if st.button("🔊 Generate Speech", type="primary"):

    if not text.strip():
        st.warning("Please enter some text first.")

    else:
        try:
            with st.spinner("Generating speech..."):

                # Create temporary MP3 file
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp3"
                )
                temp_path = temp_file.name
                temp_file.close()

                # Generate speech
                tts = gTTS(
                    text=text,
                    lang=language[1],
                    slow=False
                )

                tts.save(temp_path)

                st.success("✅ Speech generated successfully!")

                # Play audio
                with open(temp_path, "rb") as audio_file:
                    audio_bytes = audio_file.read()

                st.audio(audio_bytes, format="audio/mp3")

                # Download button
                st.download_button(
                    label="⬇️ Download Speech",
                    data=audio_bytes,
                    file_name="speech.mp3",
                    mime="audio/mp3"
                )

                # Clean up
                try:
                    os.remove(temp_path)
                except:
                    pass

        except Exception as e:
            st.error(f"Unable to generate speech: {e}")