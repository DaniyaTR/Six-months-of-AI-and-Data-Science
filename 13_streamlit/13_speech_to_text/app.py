import streamlit as st
import whisper
import tempfile
import os

st.set_page_config(page_title="Speech to Text", page_icon="🎙️")

st.title("🎙️ Speech to Text")


audio_file = st.file_uploader(
    "Upload Audio File",
    type=["mp3", "wav", "m4a", "mp4", "mpeg", "mpga", "webm"]
)

if audio_file is not None:

    st.audio(audio_file)

    if st.button("🎯 Transcribe Audio"):

        try:

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                f.write(audio_file.getvalue())
                audio_path = f.name

            with st.spinner("🎧 Transcribing..."):

                model = whisper.load_model("base")

                result = model.transcribe(audio_path)

            st.success("✅ Transcription completed!")

            st.subheader("📝 Transcription")

            st.write(result["text"])

            os.remove(audio_path)

        except Exception as e:

            st.error("❌ Error:")
            st.write(str(e))

else:

    st.info("👆 Upload an audio file to begin.")