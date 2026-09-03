import streamlit as st
from huggingface_hub import InferenceClient
from io import BytesIO

st.title("🎨 AI Image Generator")

# Get Hugging Face token from Streamlit secrets
HF_TOKEN = st.secrets["HF_TOKEN"]

prompt = st.text_input(
    "Enter a prompt:",
    "A beautiful sunset over mountains"
)

if st.button("Generate Image"):

    if not prompt:
        st.error("Please enter a prompt.")

    else:
        with st.spinner("Generating image..."):

            try:
                client = InferenceClient(
                    api_key=HF_TOKEN,
                    provider="auto"
                )

                image = client.text_to_image(
                    prompt=prompt,
                    model="black-forest-labs/FLUX.1-schnell"
                )

                st.image(
                    image,
                    caption="Generated Image"
                )

                buffer = BytesIO()
                image.save(buffer, format="PNG")

                st.download_button(
                    label="⬇️ Download Image",
                    data=buffer.getvalue(),
                    file_name="generated_image.png",
                    mime="image/png"
                )

            except Exception as e:
                st.error(f"Image generation failed: {e}")