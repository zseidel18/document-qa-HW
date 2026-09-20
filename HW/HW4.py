import streamlit as st
from openai import OpenAI
import sys
from pathlib import Path
from bs4 import BeautifulSoup


# A fix for working with ChromaDB on Streamlit Community Cloud
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb


#### USING CHROMA DB WITH OPENAI EMBEDDINGS ####

# Create OpenAI client
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(
        api_key=st.secrets["OPENAI_API_KEY"]
    )


# A function that will add documents to collection
# collection = ChromaDB collection, already established
# text = extracted/chunked text from HTML files
# Embeddings inserted into the collection from OpenAI
def add_to_collection(collection, text, document_id):

    # Create an embedding
    client = st.session_state.openai_client

    response = client.embeddings.create(
        input=text,
        model='text-embedding-3-small'
    )

    # Get the embedding
    embedding = response.data[0].embedding

    # Add embedding and document to ChromaDB
    collection.add(
        documents=[text],
        ids=[document_id],
        embeddings=[embedding]
    )


#### EXTRACT SEMANTIC SECTIONS FROM HTML ####

# This function extracts sections from each HTML file.
# A section begins with a heading and includes the text
# that follows that heading.
def extract_sections_from_html(html_path):

    with open(html_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')

    # Remove script and style information
    for item in soup(['script', 'style']):
        item.decompose()

    sections = []
    current_section = []

    # Look at headings, paragraphs, and list items
    for tag in soup.find_all(
        ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']
    ):

        text = tag.get_text(" ", strip=True)

        if not text:
            continue

        # If the tag is a heading, start a new section
        if tag.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:

            # Save the previous section
            if current_section:
                sections.append(current_section)

            # Start the new section with the heading
            current_section = [text]

        else:

            # Add the text to the current section
            current_section.append(text)

    # Add the final section
    if current_section:
        sections.append(current_section)

    return sections


#### SEMANTIC CHUNKING ####

# I am using section-based semantic chunking.
#Each heading is kept together with the text that follows it.
#This allows us to keep the context of each section intact.
#The split headings are then used to create two roughly equal mini-documents for each HTML file.

def chunk_document(sections):

    # Turn each section into one block of text
    section_texts = [
        "\n".join(section)
        for section in sections
    ]

    # Normal case: there are multiple semantic sections
    if len(section_texts) >= 2:

        # Find the approximate halfway point
        total_length = sum(
            len(section)
            for section in section_texts
        )

        halfway = total_length / 2

        chunk1_sections = []
        chunk2_sections = []

        current_length = 0

        # Keep complete sections together
        for section in section_texts:

            if current_length < halfway:

                chunk1_sections.append(section)
                current_length += len(section)

            else:

                chunk2_sections.append(section)

        # Make sure both chunks have at least one section
        if len(chunk2_sections) == 0:

            chunk2_sections.append(
                chunk1_sections.pop()
            )

        # Create the two mini-documents
        chunk1 = "\n\n".join(chunk1_sections)
        chunk2 = "\n\n".join(chunk2_sections)

    # Fallback if the HTML file only has one section
    else:

        full_text = "\n".join(section_texts)

        words = full_text.split()

        midpoint = len(words) // 2

        chunk1 = " ".join(
            words[:midpoint]
        )

        chunk2 = " ".join(
            words[midpoint:]
        )

    return chunk1, chunk2


#### POPULATE COLLECTION WITH HTML FILES ####

# This function uses extract_sections_from_html,
# chunk_document, and add_to_collection
# to put all HTML files into ChromaDB
def load_html_to_collection(folder_path, collection):

    loaded = 0

    # Get all HTML files
    html_files = list(
        Path(folder_path).glob('*.html')
    )

    # Also include .htm files if there are any
    html_files += list(
        Path(folder_path).glob('*.htm')
    )

    for html_path in html_files:

        # Extract sections from HTML
        sections = extract_sections_from_html(
            html_path
        )

        # Create two mini-documents
        chunk1, chunk2 = chunk_document(
            sections
        )

        # Add first chunk
        if chunk1.strip():

            add_to_collection(
                collection,
                chunk1,
                html_path.name + '_chunk_1'
            )

        # Add second chunk
        if chunk2.strip():

            add_to_collection(
                collection,
                chunk2,
                html_path.name + '_chunk_2'
            )

        loaded += 1

    return loaded


#### FILE LOCATIONS ####

# HW4.py is inside the HW folder.
# parent.parent moves back to the main project folder.
base_path = Path(__file__).resolve().parent.parent

# HTML files are stored in the HW-04-Data folder
data_folder = base_path / 'HW-04-Data'

# Persistent ChromaDB location
db_path = base_path / 'ChromaDB_for_HW4'


#### CREATE CHROMADB ####

# Only create/open ChromaDB once during the Streamlit session
if 'HW4_VectorDB' not in st.session_state:

    # Check whether the database already exists
    db_exists = db_path.exists()

    # Create or open ChromaDB
    chroma_client = chromadb.PersistentClient(
        path=str(db_path)
    )

    # Create or open the collection
    collection = chroma_client.get_or_create_collection(
        'HW4Collection'
    )

    # Only load the HTML files if the database is new
    # or the collection is currently empty.
    # This allows the app to run multiple times without
    # rebuilding the vector database every time.
    if not db_exists or collection.count() == 0:

        loaded = load_html_to_collection(
            data_folder,
            collection
        )

    # Store collection in session state
    st.session_state.HW4_VectorDB = collection

else:

    collection = st.session_state.HW4_VectorDB


#### MAIN APP ####

st.title(
    'HW 4: iSchool Student Organization Chatbot Using RAG'
)


#### INITIALIZE CHAT HISTORY ####

# Initialize chat history
if "messages" not in st.session_state:

    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "What can I help you with?"
        }
    ]


#### DISPLAY CHAT HISTORY ####

# Display chat messages from history on app rerun
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

    # Add user message to chat history
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    # Display user message
    with st.chat_message("user"):

        st.markdown(prompt)


    #### GET RELEVANT INFORMATION FROM VECTOR DATABASE ####

    client = st.session_state.openai_client

    # Create an embedding for the user's question
    response = client.embeddings.create(
        input=prompt,
        model="text-embedding-3-small"
    )

    # Get the embedding
    query_embedding = response.data[0].embedding

    # Get the three closest chunks from the vector database
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )


    #### COMBINE RETURNED DOCUMENTS ####

    extra_info = ""

    for doc in results["documents"][0]:

        extra_info += doc + "\n"


    #### ADD RAG INFORMATION TO THE PROMPT ####

    system_message = {
        "role": "system",
        "content":
            "You are a helpful AI assistant that answers questions "
            "about iSchool student organizations. "
            "Use the following context to answer the question. "
            "If you use information from the context, clearly say that "
            "the information came from the student organization "
            "documents using RAG. "
            "If the answer cannot be found in the provided context, "
            "say that you could not find that information in the "
            "student organization documents.\n\n"
            + extra_info
    }


    #### MEMORY CONVERSATION BUFFER ####

    # The initial assistant greeting does not count as an interaction.
    #
    # Each interaction contains:
    # 1 user message
    # 1 assistant response
    #
    # While answering the current question, keep:
    # 4 previous complete interactions = 8 messages
    # current user question = 1 message
    #
    # This gives the LLM up to the last 5 interactions.

    conversation_buffer = (
        st.session_state.messages[1:][-9:]
    )


    #### CREATE MESSAGES FOR LLM ####

    messages_for_llm = [
        system_message
    ] + conversation_buffer


    #### CALL THE LLM ####

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages_for_llm,
        stream=True
    )


    #### DISPLAY ASSISTANT RESPONSE ####

    with st.chat_message("assistant"):

        response = st.write_stream(stream)


    #### STORE ASSISTANT RESPONSE ####

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )


    #### KEEP ONLY LAST 5 INTERACTIONS ####

    # Do not count the initial greeting.
    #
    # messages[0] = initial greeting
    # messages[1:] = conversation
    # [-10:] = last 5 user/assistant interactions

    st.session_state.messages = (
        [st.session_state.messages[0]]
        + st.session_state.messages[1:][-10:]
    )