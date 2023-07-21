# Bring in deps
import os 
import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

st.write(
    "Has environment variables been set:",
    os.environ["OPENAI_API_KEY"] == st.secrets["OPENAI_API_KEY"],
)



# App framework
st.title('💪 DOM Gheadle 🤴  Chat Daddy 🔥')
prompt = st.text_input('Plug in your prompt here') 

# img = Image.open("images/dom_geatle.png")
# st.image(img)


# Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='Your name is DOM Gheadle and you are my daddy. Talk to me about {topic}'
)

# Memory 
script_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')


# Llms
llm = OpenAI(temperature=0.9) 
script_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='topic', memory=script_memory)

# wiki = WikipediaAPIWrapper()

# Show stuff to the screen if there's a prompt
if prompt: 
    script = script_chain.run(prompt)

    # st.write(title) 
    # st.write(script) 

    with st.expander('Title History'): 
        st.info(script_memory.buffer)

    # with st.expander('Script History'): 
    #     st.info(script_memory.buffer)

    # with st.expander('Wikipedia Research'): 
    #     st.info(wiki_research)
