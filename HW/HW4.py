import streamlit as st
from openai import OpenAI
import sys
from pathlib import Path
from bs4 import BeautifulSoup

# A fix for working with ChromaDB on Streamlit Community Cloud
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb


#### OPENAI CLIENT ####

if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(
        api_key=st.secrets["OPENAI_API_KEY"]
    )


#### ADD DOCUMENT TO CHROMADB ####

def add_to_collection(collection, text, document_id):

    client = st.session_state.openai_client

    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )

    embedding = response.data[0].embedding

    collection.add(
        documents=[text],
        ids=[document_id],
        embeddings=[embedding]
    )


#### EXTRACT SEMANTIC SECTIONS FROM HTML ####

def extract_sections_from_html(html_path):

    with open(html_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    # Remove code and styling
    for item in soup(["script", "style", "noscript"]):
        item.decompose()

    # Get the main organization information
    main_content = soup.find(
        "div",
        attrs={"role": "main"}
    )

    if main_content is None:
        main_content = soup.body

    if main_content is None:
        main_content = soup

    # Get organization name
    heading = main_content.find("h1")

    if heading:
        organization_name = heading.get_text(
            " ",
            strip=True
        )
    else:
        organization_name = html_path.stem

    # Get all headings
    headings = []

    for heading in main_content.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6"]
    ):
        headings.append(
            heading.get_text(" ", strip=True)
        )

    sections = []
    current_section = []

    # Go through all visible text in page order
    for text in main_content.stripped_strings:

        text = " ".join(text.split())

        # Start a new section when a heading is found
        if text in headings:

            if current_section:
                sections.append(current_section)

            current_section = [text]

        else:
            current_section.append(text)

    # Add final section
    if current_section:
        sections.append(current_section)

    return organization_name, sections


#### SEMANTIC CHUNKING ####

# I am using section-based semantic chunking.
#Each heading is kept together with the text that follows it.
#This allows us to keep the context of each section intact.
#The split headings are then used to create two roughly equal mini-documents for each HTML file.

def chunk_document(organization_name, sections):

    section_texts = [
        "\n".join(section)
        for section in sections
    ]

    if len(section_texts) >= 2:

        total_length = sum(
            len(section)
            for section in section_texts
        )

        halfway = total_length / 2

        chunk1_sections = []
        chunk2_sections = []

        current_length = 0

        for section in section_texts:

            if current_length < halfway:

                chunk1_sections.append(section)

                current_length += len(section)

            else:

                chunk2_sections.append(section)

        # Make sure both chunks contain information
        if len(chunk2_sections) == 0:

            chunk2_sections.append(
                chunk1_sections.pop()
            )

        chunk1 = "\n\n".join(
            chunk1_sections
        )

        chunk2 = "\n\n".join(
            chunk2_sections
        )

    else:

        full_text = "\n".join(
            section_texts
        )

        words = full_text.split()

        midpoint = len(words) // 2

        chunk1 = " ".join(
            words[:midpoint]
        )

        chunk2 = " ".join(
            words[midpoint:]
        )

    # Add organization name to both chunks
    chunk1 = (
        organization_name
        + "\n"
        + chunk1
    )

    chunk2 = (
        organization_name
        + "\n"
        + chunk2
    )

    return chunk1, chunk2


#### LOAD HTML FILES INTO CHROMADB ####

def load_html_to_collection(folder_path, collection):

    html_files = list(
        Path(folder_path).rglob("*.html")
    )

    html_files += list(
        Path(folder_path).rglob("*.htm")
    )

    for html_path in html_files:

        organization_name, sections = (
            extract_sections_from_html(
                html_path
            )
        )

        chunk1, chunk2 = chunk_document(
            organization_name,
            sections
        )

        chunk1_id = (
            html_path.name
            + "_chunk_1"
        )

        chunk2_id = (
            html_path.name
            + "_chunk_2"
        )

        if chunk1.strip():

            add_to_collection(
                collection,
                chunk1,
                chunk1_id
            )

        if chunk2.strip():

            add_to_collection(
                collection,
                chunk2,
                chunk2_id
            )


#### FILE LOCATIONS ####

base_path = Path(__file__).resolve().parent.parent

data_folder = (
    base_path
    / "HW-04-Data"
)

db_path = (
    base_path
    / "ChromaDB_for_HW4"
)


#### LOAD SAVED CHROMADB ####

if "HW4_VectorDB" not in st.session_state:

    chroma_client = chromadb.PersistentClient(
        path=str(db_path)
    )

    # This code was used once to create the collection
    # collection = chroma_client.get_or_create_collection(
    #     "HW4Collection"
    # )
    #
    # load_html_to_collection(
    #     data_folder,
    #     collection
    # )

    # The completed collection is now saved in
    # ChromaDB_for_HW4, so the app only loads it.
    collection = chroma_client.get_collection(
        "HW4Collection"
    )

    st.session_state.HW4_VectorDB = collection

else:

    collection = (
        st.session_state.HW4_VectorDB
    )


#### MAIN APP ####

st.title(
    "HW 4: iSchool Student Organization Chatbot Using RAG"
)


#### INITIALIZE CHAT HISTORY ####

if "messages" not in st.session_state:

    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content":
                "What can I help you with?"
        }
    ]


#### DISPLAY CHAT HISTORY ####

for msg in st.session_state.messages:

    chat_msg = st.chat_message(
        msg["role"]
    )

    chat_msg.write(
        msg["content"]
    )


#### REACT TO USER INPUT ####

if prompt := st.chat_input(
    "Ask a question about student organizations..."
):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):

        st.markdown(prompt)


    #### GET RELEVANT INFORMATION FROM VECTOR DATABASE ####

    client = st.session_state.openai_client

    response = client.embeddings.create(
        input=prompt,
        model="text-embedding-3-small"
    )

    query_embedding = (
        response.data[0].embedding
    )

    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=5
    )


    #### COMBINE RETURNED DOCUMENTS ####

    extra_info = ""

    for doc in results["documents"][0]:

        extra_info += (
            doc
            + "\n\n"
        )


    #### ADD RAG INFORMATION TO THE PROMPT ####

    system_message = {
        "role": "system",
        "content":
            "You are a helpful AI assistant "
            "that answers questions about "
            "Syracuse University student organizations. "
            "Use the following student organization "
            "documents to answer the user's question. "
            "If the answer cannot be found in the "
            "provided documents, say that you could "
            "not find that information in the student "
            "organization documents."
            "\n\n"
            + extra_info
    }


    #### MEMORY CONVERSATION BUFFER ####

    # The initial greeting does not count as an interaction.
    # Keep up to four previous user/assistant interactions
    # plus the current user question.

    conversation_buffer = (
        st.session_state.messages[1:][-9:]
    )


    #### CREATE MESSAGES FOR LLM ####

    messages_for_llm = (
        [system_message]
        + conversation_buffer
    )


    #### CALL THE LLM ####

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages_for_llm,
        stream=True
    )


    #### DISPLAY ASSISTANT RESPONSE ####

    with st.chat_message("assistant"):

        response = st.write_stream(
            stream
        )


    #### STORE ASSISTANT RESPONSE ####

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )


    #### KEEP ONLY LAST 5 INTERACTIONS ####

    st.session_state.messages = (
        [st.session_state.messages[0]]
        + st.session_state.messages[1:][-10:]
    )