#Sources
#PandasAI, OpenAI and Streamlit - Analyzing File Uploads with User Prompts:
#https://cobusgreyling.medium.com/creating-a-basic-openai-assistant-notebook-546dc7cf8fb6

#Downloading packages (run the below command in terminal)
#pip install -r requirements.txt

#Import required libraries
import os 
import openai
import requests
import json
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv, find_dotenv

#Load OpenAIKey
load_dotenv(find_dotenv())
API_KEY = os.environ['OPENAI_API_KEY']

#Initalise Large Language Model (LLM)

llm = OpenAI(api_token = API_KEY)

#Create assistant description (prompt / instructions)

prmpt = '''
You are a data specialist who will answer general knowledge questions regarding artificial intelligence. 
'''

#Create Assistant
assistant = llm.beta.assistants.create(
    name="Data Specialist Assistant",
    instructions=prmpt,
    model="gpt-4-1106-preview"
)

#Create Thread
thread = llm.beta.threads.create()


#Create Message
message = llm.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="How will artificial intelligence change the way employees work?"
)

#Run
run = llm.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id
)


#Check Run Status
# run = llm.beta.threads.runs.retrieve(
#   thread_id=thread.id,
#   run_id=run.id
# )

#Retrieve Messages
messages = llm.beta.threads.messages.list(
  thread_id=thread.id
)

print (messages)