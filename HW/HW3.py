import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai


# Re-use read_url_content() from HW2.
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
        "OpenAI",
        "Gemini"
    ]
)


# Let the user select between models.
use_advanced_model = st.sidebar.checkbox("Use advanced model")


# Select the model.
if llm_choice == "OpenAI":

    if use_advanced_model:
        model_to_use = "gpt-5.6-sol"

    else:
        model_to_use = "gpt-5.6-luna"


elif llm_choice == "Gemini":

    if use_advanced_model:
        model_to_use = "gemini-3.1-pro-preview"

    else:
        model_to_use = "gemini-3-flash-preview"


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
        "You are a helpful question answering assistant. "
        "Use the following URL content as context when answering "
        "the user's questions.\n\n"
        + url_context
}


# Create an OpenAI client.
if "client" not in st.session_state:

    openai_api_key = st.secrets["OPENAI_API_KEY"]

    st.session_state.client = OpenAI(
        api_key=openai_api_key
    )


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

    # Always keep the current URL context
    # in the system prompt.
    st.session_state.messages[0] = system_message


# Display chat messages from history on app rerun.
for msg in st.session_state.messages:

    if msg["role"] != "system":

        chat_msg = st.chat_message(msg["role"])

        chat_msg.write(msg["content"])


# Only allow questions after at least one URL
# has been successfully read.
if url_context:

    # React to user input.
    if prompt := st.chat_input("What can I help you with?"):

        # Add user message to chat history.
        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )


        # Display user message.
        with st.chat_message("user"):

            st.markdown(prompt)


        # Always keep the system prompt.
        system_message = st.session_state.messages[0]


        # Get all conversation messages
        # except the system prompt.
        conversation_messages = st.session_state.messages[1:]


        # Keep only the last 6 conversation messages.
        buffer_messages = [
            system_message
        ] + conversation_messages[-6:]


        # OpenAI
        if llm_choice == "OpenAI":

            client = st.session_state.client

            stream = client.chat.completions.create(
                model=model_to_use,
                messages=buffer_messages,
                stream=True
            )


            # Stream the response.
            with st.chat_message("assistant"):

                response = st.write_stream(stream)


        # Gemini
        elif llm_choice == "Gemini":

            # Get Gemini API key.
            gemini_api_key = st.secrets["GEMINI_API_KEY"]


            # Configure Gemini.
            genai.configure(
                api_key=gemini_api_key
            )


            # Select the Gemini model.
            model = genai.GenerativeModel(
                model_to_use,
                system_instruction=system_message["content"]
            )


            # Prepare the conversation history.
            gem_message = ""

            for msg in buffer_messages[1:]:

                if msg["role"] == "user":
                    gem_message += (
                        "User: "
                        + msg["content"]
                        + "\n\n"
                    )

                elif msg["role"] == "assistant":
                    gem_message += (
                        "Assistant: "
                        + msg["content"]
                        + "\n\n"
                    )


            # Generate a streaming response.
            stream = model.generate_content(
                gem_message,
                stream=True
            )


            # Convert Gemini stream into text
            # for st.write_stream().
            def gemini_stream():

                for chunk in stream:

                    if chunk.text:
                        yield chunk.text


            # Stream the response.
            with st.chat_message("assistant"):

                response = st.write_stream(
                    gemini_stream()
                )


        # Add assistant response to chat history.
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )


else:

    st.info(
        "Enter at least one valid URL in the sidebar to begin."
    )