import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
from langserve import add_routes

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(api_key=groq_api_key, model="gemma2-9b-it")

parser = StrOutputParser()

generic_template = "Translate the following into {language}"

prompt = ChatPromptTemplate.from_messages(
    [("system", generic_template), 
     ("user", "{text}")]
)

chain = prompt | model | parser

# App Definition
app = FastAPI(title="LangChain Server", 
              version="1.0", 
              description="A simple API using LangChain and Groq" )

# Route Definition
add_routes(
    app, 
    chain,
    path="/chain"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)