from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import runnable
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI LLM
openai_llm = OpenAI(
    model_name="gpt-3.5-turbo-instruct",
    # temperature=0.7, 
    openai_api_key = api_key)

# Define the LLM and prompt template
qa_template = PromptTemplate.from_template(
    "Answer the following question concisely:\nQuestion : {question}\nAnswer :"
)

# Initialize the Q&A chain with the LLM and prompt
qa_chain = qa_template | openai_llm

# Run the chain with a sample question
question = "What is the role of artificial intelligence in healthcare?"
answer = qa_chain.invoke({"question":question})
print(answer)

## As gpt-3.5-turbo-instruct is a "/v1/completions" model not a "/v1/chat/completions" model 
## In simpler words gpt-3.5-turbo-instruct is a completion model as gpt-3.5-turbo is a chat model. As the above code is for completion not for chat completion.

