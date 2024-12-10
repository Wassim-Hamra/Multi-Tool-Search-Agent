import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

# Loading environ variables - for local use
# load_dotenv()
# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# for deployment
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]


# Arxiv and Wikipedia Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

duck_wrapper = DuckDuckGoSearchAPIWrapper(max_results=1)
search = DuckDuckGoSearchRun(api_wrapper=duck_wrapper)


# Session State
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {'role':'Searchly', 'content':'Hi! I am Searchly🌐, a chatbot that can search the web to answer all of your questions. How can I help you today ?'}
    ]


# APP
gradient_text_html = """
    <style>
    .gradient-text {
        font-weight: bold;
        background: -webkit-linear-gradient(right, red, orange);
        background: linear-gradient(to left, blue, yellow);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline;
        font-size: 3em;
        text-align: center;
        width: 100%;
        display: block;
    }
    </style>
    <div class="gradient-text">Searchly🌐</div><br>
    <h3 class="gradient-text" style="font-weight: light; font-size: 2em;">Chat with search</h3>
    """

st.markdown(gradient_text_html, unsafe_allow_html=True)


for msg in st.session_state['messages']:
    st.chat_message(msg['role']).write(msg['content'])

if prompt := st.chat_input(placeholder="What is Generative AI ?"):
    st.session_state.messages.append({'role':'user','content':prompt})
    st.chat_message('User').write(prompt)

    # LLM
    llm = ChatGroq(model_name='gemma2-9b-it',streaming=True)
    tools = [arxiv, wiki, search]
    # Agent
    search_agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors = True)
    with st.chat_message('Searchly'):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({'role':'Searchly', 'content':response})
        st.write(response)
