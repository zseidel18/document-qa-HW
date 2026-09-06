import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai


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
st.title("HW 2 - URL Summarizer")
st.write(
    "Provide a URL to summarize using your selected summary type "
    "from the options in the sidebar."
)

# Let the user enter a URL.
url = st.text_input("Enter a URL")

# Get OpenAI API key from Streamlit secrets.
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Initialize the OpenAI client.
client = OpenAI(api_key=openai_api_key)

# Let the user select the type of summary.
summary_type = st.sidebar.selectbox(
    "Choose a summary type",
    [
        "Summarize the document in 100 words",
        "Summarize the document in 2 connecting paragraphs",
        "Summarize the document in 5 bullet points"
    ]
)
# Let the user select the output language.
output_language = st.sidebar.selectbox(
    "Choose an output language",
    [
        "English",
        "French",
        "German",
        "Spanish"
    ]
)

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

if llm_choice == "OpenAI":
    if use_advanced_model:
        model_to_use = "gpt-5-mini"
    else:
        model_to_use = "gpt-5-nano"

elif llm_choice == "Gemini":
    if use_advanced_model:
        model_to_use = "gemini-3.1-pro-preview"
    else:
        model_to_use = "gemini-3-flash-preview"


if url:

    # Process the URL.
    document = read_url_content(url)

    if document:

        # Define a system message.
        system_message = "You are a helpful assistant that summarizes documents."

        # Prepare the messages to the LLM.
        messages = [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {summary_type}. Output the summary in {output_language}."
            }
        ]

        if llm_choice == "OpenAI":

            # Generate an answer using the OpenAI API.
            stream = client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                stream=True,
            )

            # Stream the response to the app using `st.write_stream`.
            st.write_stream(stream)

        elif llm_choice == "Gemini":

            # Get Gemini API key.
            gemini_api_key = st.secrets["GEMINI_API_KEY"]

            # Configure the Google AI library with the API key.
            genai.configure(api_key=gemini_api_key)

            # Select the Gemini model.
            model = genai.GenerativeModel(model_to_use)

            # Prepare the message.
            gem_message = system_message + \
                "\n\nPlease summarize the following document: \n" + document + \
                "\n\n" + summary_type + \
                ". Output the summary in " + output_language + "."

            # Generate the response.
            response = model.generate_content(gem_message)

            # Display the response.
            st.write(response.text)