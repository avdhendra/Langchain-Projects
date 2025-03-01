from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda,RunnableSequence
from langchain_openai import ChatOpenAI

load_dotenv()

model=ChatOpenAI(model="gpt-4")

prompt_template=ChatPromptTemplate.from_messages(
    [
        ("system","You are a comedian who tells jokes about {topic}."),
        ("human","Tell me {joke_count} jokes.")
    ]
)

#Create individual runnable (steps in the chain)
# The **x is a Python syntax feature called keyword argument unpacking. It’s used here to unpack a 
# dictionary (x) into keyword arguments that are passed to the format_prompt method of the prompt_template object.

fromat_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model= RunnableLambda(lambda x: model.invoke(x.to_message()))
parse_output= RunnableLambda(lambda x:x.content)

chain= RunnableSequence(first=fromat_prompt,middle=[invoke_model],last=parse_output)

response = chain.invoke({"topic":"lawyer","joke_count":3})

print(response)