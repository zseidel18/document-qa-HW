import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai


# Re-used from HW2
def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None


# Show title and description.
st.title("HW 3 - URL Question Answering Chatbot")

st.write(
    "Enter up to two URLs in the sidebar and choose an LLM. "
    "The chatbot uses the content from the URLs as context to answer questions. "
    "The chatbot uses a conversation buffer that remembers the last 6 messages "
    "(3 user-assistant exchanges)."
)


# Let the user enter up to two URLs.
url1 = st.sidebar.text_input("Enter URL 1")

url2 = st.sidebar.text_input("Enter URL 2 (optional)")


# Let the user select the LLM.
llm_choice = st.sidebar.selectbox(
    "Choose an LLM",
    [
        "OpenAI - GPT-5.6 Sol",
        "Gemini - Gemini 3.1 Pro"
    ]
)


# Select the specific model.
if llm_choice == "OpenAI - GPT-5.6 Sol":
    model_to_use = "gpt-5.6-sol"

elif llm_choice == "Gemini - Gemini 3.1 Pro":
    model_to_use = "gemini-3.1-pro-preview"


# Read content from the URLs.
url_context = ""

if url1:
    document1 = read_url_content(url1)

    if document1:
        url_context += "URL 1 Content:\n" + document1


if url2:
    document2 = read_url_content(url2)

    if document2:
        url_context += "\n\nURL 2 Content:\n" + document2


# Define the system message.
system_message = {
    "role": "system",
    "content":
        "Answer all questions so that a 10 year old can understand. "
        "After answering a user's question, ask 'Do you want more info?'. "
        "If the user says yes, provide more information about the previous "
        "answer and ask 'Do you want more info?' again. "
        "If the user says no, ask 'What can I help you with?'. "
        "Use the following URL content as context when answering the user's "
        "questions:\n\n" + url_context
}


# Create an OpenAI client.
if 'client' not in st.session_state:
    api_key = st.secrets["OPENAI_API_KEY"]
    st.session_state.client = OpenAI(api_key=api_key)


# Initialize chat history.
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        system_message,
        {
            "role": "assistant",
            "content": "What can I help you with?"
        }
    ]

else:
    # Always keep the current URL content in the system prompt.
    st.session_state.messages[0] = system_message


# Display chat messages from history on app rerun.
for msg in st.session_state.messages:

    if msg["role"] != "system":

        chat_msg = st.chat_message(msg["role"])
        chat_msg.write(msg["content"])


# React to user input.
if prompt := st.chat_input("What is up?"):

    # Add user message to chat history.
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    # Display user message.
    with st.chat_message("user"):
        st.markdown(prompt)


    # Always keep the system prompt.
    system_message = st.session_state.messages[0]

    # Get conversation messages.
    # Start at 2 so the original "What can I help you with?"
    # does not count as part of the 6-message buffer.
    conversation_messages = st.session_state.messages[2:]

    # Keep only the last 6 conversation messages.
    buffer_messages = [
        system_message
    ] + conversation_messages[-6:]


    # OpenAI
    if llm_choice == "OpenAI - GPT-5.6 Sol":

        client = st.session_state.client

        stream = client.chat.completions.create(
            model=model_to_use,
            messages=buffer_messages,
            stream=True
        )

        with st.chat_message("assistant"):
            response = st.write_stream(stream)


    # Gemini
    elif llm_choice == "Gemini - Gemini 3.1 Pro":

        # Get Gemini API key.
        gemini_api_key = st.secrets["GEMINI_API_KEY"]

        # Configure Gemini.
        genai.configure(api_key=gemini_api_key)

        # Create Gemini model using the system prompt.
        model = genai.GenerativeModel(
            model_to_use,
            system_instruction=system_message["content"]
        )

        # Convert the buffered messages to Gemini format.
        gemini_messages = []

        for msg in buffer_messages[1:]:

            if msg["role"] == "user":
                gemini_role = "user"

            else:
                gemini_role = "model"

            gemini_messages.append(
                {
                    "role": gemini_role,
                    "parts": [msg["content"]]
                }
            )


        # Generate a streaming response.
        stream = model.generate_content(
            gemini_messages,
            stream=True
        )


        # Stream Gemini text to Streamlit.
        def gemini_stream():
            for chunk in stream:
                if chunk.text:
                    yield chunk.text


        with st.chat_message("assistant"):
            response = st.write_stream(gemini_stream())


    # Add assistant response to chat history.
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )