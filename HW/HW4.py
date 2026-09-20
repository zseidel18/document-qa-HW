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


#### EXTRACT TEXT FROM HTML ####

# This function extracts text from each HTML file
# to pass to the chunking function
def extract_text_from_html(html_path):

    with open(html_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')

    # Remove script and style information
    for item in soup(['script', 'style']):
        item.decompose()

    # Get text from the HTML page
    text = soup.get_text(separator='\n')

    # Remove blank lines
    paragraphs = [
        line.strip()
        for line in text.split('\n')
        if line.strip()
    ]

    return paragraphs


#### CHUNK HTML DOCUMENT ####

# CHUNKING METHOD:
# I am using paragraph-based semantic chunking.
#
# Each HTML document is split into two mini-documents using paragraph
# boundaries. I chose this method because paragraphs represent natural
# breaks in the information on the HTML pages. This helps keep related
# ideas together instead of cutting sentences in the middle.
#
# If an HTML page does not contain enough separate paragraphs, the
# fallback splits the available text approximately in half so that
# every HTML document still creates the required two mini-documents.
def chunk_document(paragraphs):

    # Normal case: split the document between paragraphs
    if len(paragraphs) >= 2:

        midpoint = len(paragraphs) // 2

        chunk1 = '\n'.join(paragraphs[:midpoint])
        chunk2 = '\n'.join(paragraphs[midpoint:])

    # Fallback if the HTML page only produces one block of text
    else:

        full_text = ' '.join(paragraphs)

        words = full_text.split()

        midpoint = len(words) // 2

        chunk1 = ' '.join(words[:midpoint])
        chunk2 = ' '.join(words[midpoint:])

    return chunk1, chunk2


#### POPULATE COLLECTION WITH HTML FILES ####

# This function uses extract_text_from_html,
# chunk_document, and add_to_collection
# to put all HTML files into the ChromaDB collection
def load_html_to_collection(folder_path, collection):

    loaded = 0

    # Get all .html files
    html_files = list(Path(folder_path).glob('*.html'))

    # Also include .htm files if there are any
    html_files += list(Path(folder_path).glob('*.htm'))

    for html_path in html_files:

        # Extract text from HTML
        paragraphs = extract_text_from_html(html_path)

        # Split each HTML document into two mini-documents
        chunk1, chunk2 = chunk_document(paragraphs)

        # Add chunk 1
        if chunk1.strip():

            add_to_collection(
                collection,
                chunk1,
                html_path.name + '_chunk_1'
            )

        # Add chunk 2
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

# HTML files are in:
# project folder / HW-04-Data
data_folder = base_path / 'HW-04-Data'

# ChromaDB will be stored in:
# project folder / ChromaDB_for_HW4
db_path = base_path / 'ChromaDB_for_HW4'


#### CREATE CHROMADB ####

# Only create/open ChromaDB once during the Streamlit session
if 'HW4_VectorDB' not in st.session_state:

    # Create ChromaDB client
    chroma_client = chromadb.PersistentClient(
        path=str(db_path)
    )

    # Create the collection if it does not already exist
    collection = chroma_client.get_or_create_collection(
        'HW4Collection'
    )

    # Only add the HTML documents if the collection is empty.
    # This prevents the files from being embedded again every
    # time the application runs.
    if collection.count() == 0:

        loaded = load_html_to_collection(
            data_folder,
            collection
        )

    # Store collection in session state
    st.session_state.HW4_VectorDB = collection

else:

    collection = st.session_state.HW4_VectorDB


#### MAIN APP ####

st.title('HW 4: iSchool Student Organization Chatbot Using RAG')


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

    chat_msg = st.chat_message(msg["role"])

    chat_msg.write(msg["content"])


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

    response = client.embeddings.create(
        input=prompt,
        model="text-embedding-3-small"
    )

    # Get the embedding
    query_embedding = response.data[0].embedding

    # Get the text related to this question
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

    # The chatbot stores up to the last 5 interactions.
    #
    # One complete interaction contains:
    # 1 user message
    # 1 assistant response
    #
    # When the current user question is being answered, we keep
    # the previous 4 complete interactions plus the current
    # user question.

    conversation_buffer = st.session_state.messages[1:]

    if len(conversation_buffer) > 9:

        conversation_buffer = conversation_buffer[-9:]


    # Add the RAG system message before the conversation
    messages_for_llm = [
        system_message
    ] + conversation_buffer


    #### CALL THE LLM ####

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages_for_llm,
        stream=True
    )

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

    # Keep the initial greeting plus:
    # 5 user messages
    # 5 assistant responses
    #
    # This gives the chatbot a 5-interaction memory buffer.

    if len(st.session_state.messages) > 11:

        st.session_state.messages = (
            [st.session_state.messages[0]]
            + st.session_state.messages[-10:]
        )