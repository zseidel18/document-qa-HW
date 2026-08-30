import re
from abc import ABC, abstractmethod

import streamlit as st
from openai import OpenAI, AuthenticationError
from PyPDF2 import PdfReader


class text(ABC):
    """Base interface for text-like document content."""

    def __init__(self, content):
        self.content = content or ""

    @abstractmethod
    def clean(self):
        """Return normalized text suitable for downstream processing."""
        raise NotImplementedError

    @abstractmethod
    def word_count(self):
        """Return the number of words in the text."""
        raise NotImplementedError

    @abstractmethod
    def snippet(self, max_chars=200):
        """Return a compact preview of the material."""
        raise NotImplementedError


class PlainText(text):
    """Concrete text implementation used by the document QA app."""

    def clean(self):
        return re.sub(r"\s+", " ", self.content).strip()

    def word_count(self):
        cleaned = self.clean()
        return len(cleaned.split()) if cleaned else 0

    def snippet(self, max_chars=200):
        cleaned = self.clean()
        if len(cleaned) <= max_chars:
            return cleaned
        if max_chars <= 3:
            return cleaned[:max_chars]
        return cleaned[: max_chars - 3].rsplit(" ", 1)[0] + "..."


def read_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text

# Show title and description.
st.title("MY Document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get "
    "[here](https://platform.openai.com/account/api-keys)."
)

# Ask user for OpenAI API key.
openai_api_key = st.text_input("OpenAI API Key", type="password")

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")

else:
    client = OpenAI(api_key=openai_api_key)

    # Validate API key immediately.
    try:
        client.models.list()
        st.success("API key is valid!", icon="✅")
    except AuthenticationError:
        st.error("Invalid OpenAI API key. Please try again.", icon="❌")
        st.stop()
    except Exception as e:
        st.error(f"Could not validate API key: {e}")
        st.stop()

    # Let the user upload a file.
    uploaded_file = st.file_uploader(
        "Upload a document (.pdf or .txt)", type=("pdf", "txt")
    )

    # Ask the user for a question via `st.text_area`.
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:

        # Process the uploaded file and question.
        file_extension = uploaded_file.name.split('.')[-1]

        if file_extension == 'txt':
            document = uploaded_file.read().decode()

        elif file_extension == 'pdf':
            document = read_pdf(uploaded_file)

        document_text = PlainText(document)
        cleaned_document = document_text.clean()
        snippet = document_text.snippet()

        messages = [
            {
                "role": "user",
                "content": (
                    f"Here's a document: {cleaned_document} \n\n---\n\n"
                    f"Document preview: {snippet} \n\n"
                    f"{question}"
                ),
            }
        ]

        # Generate an answer using the OpenAI API.
        stream = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            stream=True,
        )

        # Stream the response to the app using `st.write_stream`.
        st.write_stream(stream)