#————————————————————

# Name: OpenAI Assistant Basic (V1)
# Purpose: Create an openai assistant using the GPT4 LLM that is tuned to act as a data specialist and answers general questions related to AI.
# Company: Allgeier Schweiz AG
# Author: Nicolas Rehder (nrehder@allgeier.ch)
# Create for: SDSC 2024
# Date Created: 08.01.2024
# Last Updated: 08.01.2024
# Python Version: 3.10.4

#General Sources:
#https://platform.openai.com/docs/api-reference?lang=python
#https://platform.openai.com/docs/assistants/how-it-works
#https://github.com/openai/openai-python/blob/main/examples/assistant.py

#Openai Uasage:
#https://platform.openai.com/usage

#Additionals:
#Alphavantage
#Open Interpreter

#————————————————————

# Download Python packages (run the below command in terminal if packages have not yet been installed)
#pip install -r C:\Python\openai-lab\support\requirements\requirements.txt

# Import required libraries
import os
import time
import requests
import json

import openai
from openai import OpenAI

import pandas as pd

from dotenv import load_dotenv, find_dotenv

# Load OpenAIKey
load_dotenv(find_dotenv())
API_KEY = os.environ['OPENAI_API_KEY']

# Initalise Large Language Model (LLM)

llm = OpenAI(api_key = API_KEY)

# Create assistant description (prompt / instructions)

prmpt = '''
You are a data specialist who will answer general knowledge questions regarding artificial intelligence and its applications. 
'''

# Create Assistant
assistant = llm.beta.assistants.create(
    name="Data Specialist Assistant",
    instructions=prmpt,
    model="gpt-3.5-turbo-1106" #"gpt-4-1106-preview"
)

# Create Thread
thread = llm.beta.threads.create()


# Create Message
message = llm.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="How will artificial intelligence change the way employees work?"
)

# Run
run = llm.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id
)



#Retrieve Messages & Check Run Status
print("checking assistant status. ")
while True:
    run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

    if run.status == "completed":
        print("done!")
        messages = client.beta.threads.messages.list(thread_id=thread.id)

        print("messages: ")
        for message in messages:
            assert message.content[0].type == "text"
            print({"role": message.role, "message": message.content[0].text.value})

        client.beta.assistants.delete(assistant.id)

        break
    else:
        print("in progress...")
        time.sleep(5)

print (messages)