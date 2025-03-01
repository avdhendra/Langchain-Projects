from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

model=ChatOpenAI(model="gpt-40")

prompt_template=ChatPromptTemplate.from_messages(
    [
        ("system","You are a comedian who tells jokes about {topic}."),
        ("human","tell me {joke_count} jokes.")
    ]
)
# create the combined chain using langchain expression language

chain=prompt_template | model | StrOutputParser() #The StrOutputParser() is an output parser in LangChain that takes the raw output from a language model (like ChatOpenAI) and converts it into a simple string format. Specifically:

# Run the chain
result=chain.invoke({"topic":"lawyers","joke_count":3})

print(result)

