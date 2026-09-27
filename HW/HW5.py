import streamlit as st
from openai import OpenAI
import sys
import json
from pathlib import Path

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb


if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])


base_path = Path(__file__).resolve().parent.parent
db_path = base_path / 'ChromaDB_for_HW4'

if 'HW5_VectorDB' not in st.session_state:
    chroma_client = chromadb.PersistentClient(path=str(db_path))
    st.session_state.HW5_VectorDB = chroma_client.get_collection('HW4Collection')

collection = st.session_state.HW5_VectorDB


def relevant_club_info(query):
    client = st.session_state.openai_client
    response = client.embeddings.create(
        input=query,
        model='text-embedding-3-small'
    )
    query_embedding = response.data[0].embedding
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )
    documents = results['documents'][0]
    return '\n\n'.join(documents) if documents else 'No relevant student organization documents found.'


tools = [
    {
        'type': 'function',
        'function': {
            'name': 'relevant_club_info',
            'description': 'Search Syracuse University student organization documents for information relevant to a question. Use a query that includes the organization name and the detail needed. Use this when answering questions about clubs, including follow-up questions.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'Search phrase for relevant student organization information.'
                    }
                },
                'required': ['query']
            }
        }
    }
]


st.title('HW 5: iSchool Student Organization Chatbot')

if 'HW5_messages' not in st.session_state:
    st.session_state.HW5_messages = [
        {'role': 'assistant', 'content': 'What can I help you with?'}
    ]

for msg in st.session_state.HW5_messages:
    with st.chat_message(msg['role']):
        st.write(msg['content'])

if prompt := st.chat_input('Ask a question about student organizations...'):
    st.session_state.HW5_messages.append({'role': 'user', 'content': prompt})

    with st.chat_message('user'):
        st.markdown(prompt)

    client = st.session_state.openai_client
    system_message = {
        'role': 'system',
        'content': (
            'You are a helpful AI assistant that answers questions about Syracuse University student organizations. '
            'Use relevant_club_info to look up information needed to answer organization questions. '
            'For a follow-up question, use the conversation to identify which organization the user means and search for the new detail. '
            'Do not invent club facts. If the answer cannot be found in the retrieved documents, say that you could not find that information in the student organization documents.'
        )
    }

    # Initial greeting is excluded. Keep four previous exchanges and this question.
    conversation_buffer = st.session_state.HW5_messages[1:][-9:]
    messages_for_llm = [system_message] + conversation_buffer

    first_response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages_for_llm,
        tools=tools,
        tool_choice='auto'
    )
    assistant_message = first_response.choices[0].message

    if assistant_message.tool_calls:
        messages_for_llm.append(assistant_message)
        for tool_call in assistant_message.tool_calls:
            if tool_call.function.name == 'relevant_club_info':
                try:
                    arguments = json.loads(tool_call.function.arguments)
                    query = arguments['query']
                    extra_info = relevant_club_info(query)
                except (ValueError, KeyError, TypeError) as error:
                    extra_info = f'Could not run the document search: {error}'
            else:
                extra_info = 'Unknown function requested.'

            messages_for_llm.append({
                'role': 'tool',
                'tool_call_id': tool_call.id,
                'content': extra_info
            })

        # Answer with the search results; no tools are provided on this request.
        stream = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=messages_for_llm,
            stream=True
        )
        with st.chat_message('assistant'):
            response = st.write_stream(stream)
    else:
        response = assistant_message.content or 'What can I help you with?'
        with st.chat_message('assistant'):
            st.write(response)

    st.session_state.HW5_messages.append({'role': 'assistant', 'content': response})
    st.session_state.HW5_messages = (
        [st.session_state.HW5_messages[0]]
        + st.session_state.HW5_messages[1:][-10:]
    )