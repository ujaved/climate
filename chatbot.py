from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
from langchain.llms import Replicate
from requests.sessions import Session
import time

PROMPT_TEMPLATE = """
You are the system that should help people to evaluate the impact of climate change
on decisions they are taking today (e.g. install wind turbines, solar panels, build a building,
parking lot, open a shop, buy crop land). You are working with data on a local level,
and decisions also should be given for particular locations. You will be given information 
about changes in environmental variables for particular location, and how they will 
change in a changing climate. Your task is to provide assessment of potential risks 
and/or benefits for the planned activity related to change in climate. Use information 
about the country to retrieve information about policies and regulations in the 
area related to climate change, environmental use and activity requested by the user.
You don't have to use all variables provided to you, if the effect is insignificant,
don't use variable in analysis. DON'T just list information about variables, don't 
just repeat what is given to you as input. I don't want to get the code, 
I want to receive a narrative, with your assessments and advice. Format 
your response as MARKDOWN, don't use Heading levels 1 and 2.

Current conversation: {history}
Human: {input}
AI:
"""

REPLICATE_LLAMA_ENDPOINT = "meta/llama-2-70b-chat:2d19859030ff705a87c746f7e96eea03aefb71f166725aee39692f1476566d48"


class StreamlitStreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.text = ""
        self.container = st.empty()

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)

class Chatbot:
    def __init__(self, model_id: str, temperature: float) -> None:
        self.model_id = model_id
        self.temperature = temperature
        self.num_tokens = 0
        self.num_tokens_delta = 0

    def response(self, prompt: str) -> str:
        raise NotImplementedError


class OpenAIChatbot(Chatbot):

    def __init__(self, model_id: str, temperature: float) -> None:
        super().__init__(model_id=model_id, temperature=temperature)
        self.llm = ChatOpenAI(model_name=self.model_id,temperature=self.temperature, streaming=True)
        PROMPT = PromptTemplate(input_variables=["history", "input"], template=PROMPT_TEMPLATE)
        self.chain = ConversationChain(prompt=PROMPT, llm=self.llm, memory=ConversationBufferMemory(), verbose=True)


    def response(self, prompt: str) -> str:
        with get_openai_callback() as cb:
            # for every response, we create a new stream handler; if not, response would use the old container
            self.chain.llm.callbacks = [StreamlitStreamHandler()]
            resp = self.chain.run(prompt)
            self.num_tokens_delta = cb.total_tokens - self.num_tokens
            self.num_tokens = cb.total_tokens
        return resp


class LLamaChatbot(Chatbot):

    def __init__(self, model_id: str, temperature: float) -> None:
        super().__init__(model_id=model_id, temperature=temperature)
        self.llm = Replicate(streaming=True, model=REPLICATE_LLAMA_ENDPOINT, model_kwargs={"temperature": temperature, "max_length": 4000})
        PROMPT = PromptTemplate(input_variables=["history", "input"], template=PROMPT_TEMPLATE)
        self.chain = ConversationChain(prompt=PROMPT, llm=self.llm, memory=ConversationBufferMemory(), verbose=True)

    def response(self, prompt: str) -> str:
        self.chain.llm.callbacks = [StreamlitStreamHandler()]
        resp = self.chain.run(prompt)
        return resp


class ClimateGPTChatbot():
    def __init__(self, endpoint: str, max_new_tokens: int, poll_interval: int):
        self.endpoint = endpoint
        self.max_new_tokens = max_new_tokens
        self.client = Session()
        self.poll_interval = poll_interval
        self.container = st.empty()

    def response(self, prompt: str) -> str:
        self.client.post(
            url=self.endpoint,
            json={"prompt": prompt, "max_new_tokens": self.max_new_tokens, "new_stream": True},
        )
        is_streaming = True
        resp = ""
        while is_streaming:
            time.sleep(self.poll_interval)
            response = self.client.post(
                url=self.endpoint,
                json={"prompt": prompt, "max_new_tokens": self.max_new_tokens, "new_stream": False},
            )
            is_streaming = not response.json()["is_complete"]
            resp += response.json()["generated_text"]
            self.container.markdown(resp)
        return resp
