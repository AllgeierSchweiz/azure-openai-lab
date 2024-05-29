----------

# A Hands-on Exploration of the Azure OpenAI API (Part 4 of 5)

![](https://cdn-images-1.medium.com/max/800/1*PV9Eh3WpAhnW9NZCijZPzw.png)

## 1. Retrieval-Augmented Generation

Let’s continue our journey and learn more about enhancing our model output using Retrieval-Augmented Generation (RAG).

As mentioned in Part 1, Chapter 4.3 of this workshop: **Retrieval Augmented Generation (RAG)**, rather than solely depending on the default model’s existing knowledge, we can supplement the model with our data and boost the overall comprehension without re-training.

The two Jupyter Notebooks we are going to work with are:

1.  _P4-azure-openai-assistant-rag-data-preprocessing.ipynb_
2.  _P4-azure-openai-rag.ipynb_

In Notebook **P4-azure-openai-assistant-rag-data-preprocessing.ipynb,** we will conduct data pre-processing steps using the Azure OpenAI Assistant API using the GPT-4 model version **1106 Preview**. We will input data-wrangling instructions in text format and have the code interpreter of the Azure OpenAI Assistant execute our requirements.

In Notebook **P4-azure-openai-rag.ipynb,** we will implement RAG using **ChromaDB** as our Vector Database together with **LangChain** and the Azure OpenAI embedding model **Text-Embedding-3-Large**. Subsequently, we will use the **GPT-3.5-Turbo** model with the aid of RAG to create well-thought-out and flavourful vegan recipes for special occasions.

<br/><br/>

### 1.2 What we are doing in a nutshell

![](https://cdn-images-1.medium.com/max/800/1*42TBGtmSVNIjjuVK-i6VqQ.png)

1.  Import pre-processed data in text format and split the text into chunks.
2.  Convert each block of text into a vector using the embedding model `text-embedding-3-large` .
3.  Store the vectors in a vector database.
4.  Create a user input i.e. list of ingredients.
5.  Convert the user input into a vector using the embedding model `text-embedding-3-large` .
6.  Index the user input vector in the vector database.
7.  Use a vector search function to find the most semantically similar information to the user input vector from the vector database.
8.  Inject the information of the most similar vector from the embedding space into the original prompt, augmenting the prompt input, using the conversational model `gpt-35-turbo` .
9.  The model creates the desired answer i.e. vegan recipe.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

----------

## 2. Vector Database

Vector databases operate in a similar manner to traditional databases, but they have been specifically designed to store, index, and search for vectors. These vectors represent information such as text or images in numerical format from proprietary data. This numerical representation is created by using an embedding model such as the Azure OpenAI `text-embedding-3-large` model.

It is important to understand the rationale behind RAG. The objective is to enhance the prompt with relevant information to support the underlying model. This necessitates a fit between the user input and the supplemented data. We can verify this using similarity search methods. The database calculates the distances between the user input embedding and the embeddings of the underlying data. The closer these vectors are, the more similar and relevant they are to one another. The output with the highest degree of similarity is selected for use, thereby enhancing the value of the prompt sent to the model.

There are different types of vector databases. For our implementation, we’ll be using the open-source, locally instantiated [ChromaDB](https://www.trychroma.com/) vector database.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

----------

## 3. LangChain

LangChain is a Python library and an open-source framework for developing applications using large language models, such as those available on Azure OpenAI Services. As the name suggests, the capabilities of LangChain enable users to combine multiple key components to streamline the process of working with models.

In our case, we will use LangChain to create our text chunks, generate the embeddings of our data, persist the vector database, create prompt templates, and call the Azure OpenAI model to generate a vegan recipe. All these components are combined to form a logical sequence creating a coherent workflow.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

----------

## 4. Data Pre-Processing with Azure OpenAI Assistants

Before starting our RAG implementation, we need to certify the data used to augment our prompt is correctly structured and contains the correct information.

To ensure our model assists in creating vegan recipes, we need to use data containing only vegan recipe examples. We must also adjust the data structure by removing unnecessary columns and trimming values. For RAG, we don’t need 200'000 examples — 50 will suffice. Therefore, we will create a subset of the data.

All of this will be accomplished using the Azure OpenAI Assistant. We will provide a set of instructions with the outlined requirements, and the Assistant’s Code Interpreter will use Python to transform the data accordingly.

The Jupyter Notebook we are going to work with is called **P4-azure-openai-assistant-rag-data-preprocessing.ipynb.**

<br/><br/>

### 4.1 Open the Notebook

-   On the left of the codespace environment, select the **Explorer** icon.
-   Open the Notebook **P4-azure-openai-assistant-rag-data-preprocessing.ipynb.**

![](https://cdn-images-1.medium.com/max/800/1*xNmcuV7vtNkHxQMetZmB3A.png)

<br/><br/>

### 4.2 Initializing the Azure OpenAI Client

We will start by importing the necessary Python packages to run our code.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Import Python packages  
import os  
import io  
import time  
from io import StringIO  
import json  
from dotenv import load_dotenv  
from pathlib import Path  
import pandas as pd  
from openai import AzureOpenAI
```

-   If necessary, select a kernel to work with.

![](https://cdn-images-1.medium.com/max/800/1*-Vz32KNx4eQ1b5PS5cM_2g.png)

Next, we will load the Azure OpenAI Key and Endpoint as variables.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Load required variables from .env file.  
load_dotenv(dotenv_path=Path("/workspaces/azure-openai-lab/.venv/.env")) #Error sometimes due to \ or \\. Try one or the other.  
  
# Load Azure OpenAI Key and Endpoint. These values can be found within the Azure OpenAI Service resource in portal.azure.com under Keys and Endpoint  
azure_oai_key = os.environ['AZURE_OPENAI_KEY']  
azure_oai_endpoint = os.environ['AZURE_OPENAI_ENDPOINT']
```

Once the credentials are available as variables, we can initialize the Azure OpenAI client.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Initialize the Azure OpenAI client  
client = AzureOpenAI(  
    api_key = azure_oai_key,    
    api_version = "2024-02-15-preview",  
    azure_endpoint = azure_oai_endpoint  
    )
```

**_NOTE: The latest API version for the AzureOpenAI client when using the Azure OpenAI Assistant API can be found_** [**_here_**](https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation#latest-ga-api-release)**_._**

<br/><br/>

### 4.3 Import File from Azure OpenAI Service

The data used for our RAG implementation in Chapter 5 is based on a CSV file containing 200'000 example recipes. We first need to do some data wrangling. Let’s import that file into our Notebook.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Import existing uploaded file on Azure OpenAI Service
for i in client.files.list():
    if i.filename == "recipes.csv":
        file__id = i.id
        print(i.id)
```

**_NOTE: For convenience, we have pre-uploaded this file into Azure OpenAI Service for you to import into your Codespace environment._**

<br/><br/>

### 4.4 Data Wrangling Instructions

To perform our data pre-processing steps using the Azure OpenAI Assistant, we need to provide the model with clear instructions outlining our requirements.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Create data transformation instructions
instructions = '''
### INSTRUCTIONS
You are a senior data analyst who will work with data in a csv file in your files. 
You have access to a sandbox environment for writing Python code.
When the user asks you to perform your actions, use the csv file to read the data into a pandas dataframe.
Execute each of the steps listed below in your ACTIONS section.

---

### ACTIONS:
1. Read the tab separated comma file data into a pandas DataFrame. 
2. Drop columns "id", "contributor_id" and "submitted".
3. Trim column "name" by removing irregular text spacing at the front or back of each value while keeping single spaces between words.
4. Filter the data where column "tags" contains the word vegan.
5. Create a new column named "dense_feature" combining the values of the columns "name", "tags", "nutrition", "ingredients" and "steps" separated by a semicolon.
6. Prepare a final data set named "recipes-preprocessed" that only has 50 rows randomly sampled from the dataframe from step 5.
7. Prepare recipes-preprocessed as csv files for download by the user. 

---

### DO NOT:
1. Do not return any images. 
2. Do not return any other file types.
'''
```

<br/><br/>

### 4.4 Create an Azure OpenAI Assistant

The Azure OpenAI Assistant requires C**ode Interpreter** enabled to perform our data transformations. The Code Interpreter enables Assistants to write and run Python code in a sandbox environment. It can process files with various data and formats and iteratively solve complex coding problems. In our case, we want to perform data transformation using `pandas` .

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Create an Azure OpenAI Assistant  
assistant = client.beta.assistants.create(  
    name = "data analyst assistant",  
    instructions = instructions,  
    tools = [{"type": "code_interpreter"}],  
    model = "gpt-4-1106-preview", #You must replace this value with the deployment name for your model.  
    file_ids=[file__id]  
)  
  
# Get the file id  
fileId = assistant.file_ids[0]  
  
# Create a thread  
thread = client.beta.threads.create()
```
**_NOTE: A thread is a conversation session between an assistant and a user. Threads simplify application development by storing message history and truncating it when the conversation gets too long for the model’s context length._**

We can now initialize the thread and send our instructions to the Azure OpenAI Assistant for evaluation.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Initalize thread and start data transformation using the Azure OpenAI Assistant Code Interpreter
prompt = "Please execute the INSTRUCTIONS and ACTIONS on the data stored in the csv file " + fileId

message = client.beta.threads.messages.create(
    thread_id = thread.id,
    role = "user",
    content = prompt
)
```

<br/><br/>

### 4.5 Run the Azure OpenAI Assistant

All variables have been setup, we can now run the Azure OpenAI Assistant

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Run the Azure OpenAI Assistant  
run = client.beta.threads.runs.create(  
  thread_id=thread.id,  
  assistant_id=assistant.id,  
  #instructions="New instructions" #You can optionally provide new instructions but these will override the default instructions  
)
```

Let’s check the status of the run while we wait.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Check status of Azure OpenAI Assistant run
while True:
    sec = 30
    # Wait for 30 seconds
    time.sleep(sec)  
    # Retrieve the run status
    run_status = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
    )
    # If run is completed, get messages
    if run_status.status == 'completed':
        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )
        # Loop through messages and print content based on role
        for msg in messages.data:
            role = msg.role
            try:
                content = msg.content[0].text.value
                print(f"{role.capitalize()}: {content}")
            except AttributeError:
                # This will execute if .text does not exist
                print(f"{role.capitalize()}: [Non-text content, possibly an image or other file type]")
        break
    elif run.status == "requires_action":
        # handle function calling and continue with the execution
        pass
    elif run.status == "expired" or run.status=="failed" or run.status=="cancelled":
        # run failed, expired, or was cancelled
        break    
    # elif run.last_error != "None":
    #     # run failed, expired, or was cancelled
    #     break     
    else:
        print("In progress...")
```

<br/><br/>

### 4.6 Import Generated CSV

Let’s view the newly created file and check the transformations.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Functions to read csv files from Azure OpenAI Service
output_path = r"/workspaces/azure-openai-lab/data/generated_output/"

def read_and_save_file(first_file_id, file_name):    
    # its binary, so read it and then make it a file like object
    file_data = client.files.content(first_file_id)
    file_data_bytes = file_data.read()
    file_like_object = io.BytesIO(file_data_bytes)
    #now read as csv to create df
    returned_data = pd.read_csv(file_like_object)
    returned_data.to_csv(output_path + file_name, index=False)
    return returned_data
    # file = read_and_save_file(first_file_id, "analyst_output.csv")
    
def files_from_messages():
    messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )
    first_thread_message = messages.data[0]  # Accessing the first ThreadMessage
    message_ids = first_thread_message.file_ids
    # Loop through each file ID and save the file with a sequential name
    for i, file_id in enumerate(message_ids):
        file_name = f"recipes-preprocessed.csv"  # Generate a sequential file name
        read_and_save_file(file_id, file_name)
        print(f'saved {file_name}')  

# Extract the file names from the response, retrieve the content and save the data as a csv file 
files_from_messages()
```

-   To open the file, navigate the **Explorer** icon and select the file named **recipes-preprocessed.csv** in the **generated_output** tab located under the **data** tab.

![](https://cdn-images-1.medium.com/max/800/1*FkaP2MMYmLvODihN52K9Hg.png)

<br/><br/>

### 4.7 Clean up Azure OpenAI Assistant Environment

It’s important to make sure there aren’t any unnecessary artifacts lying around in Azure OpenAI Service. Let’s delete the assistant, thread, and generated CSV file from the Azure OpenAI environment.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
#Clean up Azure OpenAI environment  
client.beta.assistants.delete(assistant.id)  
client.beta.threads.delete(thread.id)  
client.files.delete(messages.data[0].file_ids[0])
```

**_NOTE: Since the data transformation steps use a random sample on the original data, we will use a standardized version of the generated file named recipes-preprocessed.csv to ascertain everyone has the same copy of the data._**

<p align="right">(<a href="#readme-top">back to top</a>)</p>

----------

## 5. Getting started with RAG

The Jupyter Notebook we are going to work with is called **P4-azure-openai-rag.ipynb.**

<br/><br/>

### 5.1 Open the Notebook

-   On the left of the codespace environment, select the **Explorer** icon.
-   Open the Notebook **P4-azure-openai-rag.ipynb.**

![](https://cdn-images-1.medium.com/max/800/1*gWw4ZL9mFw5ImbTuhLiwoA.png)

<br/><br/>

### 5.2 Initializing the Azure OpenAI Client

We will start by importing the necessary Python packages to run our code.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Import Python packages  
import os  
import io  
import time  
from io import StringIO  
import json  
from dotenv import load_dotenv  
from pathlib import Path  
import pandas as pd  
from openai import AzureOpenAI  
import chromadb  
import chromadb.utils.embedding_functions as embedding_functions  
from langchain.vectorstores import Chroma  
from langchain_openai import AzureOpenAIEmbeddings  
from langchain.document_loaders import DataFrameLoader  
from langchain.text_splitter import CharacterTextSplitter  
from langchain_openai import AzureChatOpenAI  
from langchain.chains import RetrievalQA  
from langchain import PromptTemplate  
from langchain_core.prompts import (  
    ChatPromptTemplate,  
    FewShotChatMessagePromptTemplate,  
)
```

Next, we will load the Azure OpenAI Key and Endpoint as variables.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Load required variables from .env file.  
load_dotenv(dotenv_path=Path("/workspaces/azure-openai-lab/.venv/.env")) #Error sometimes due to \ or \\. Try one or the other.  
  
# Load Azure OpenAI Key and Endpoint. These values can be found within the Azure OpenAI Service resource in portal.azure.com under Keys and Endpoint  
azure_oai_key = os.environ['AZURE_OPENAI_KEY']  
azure_oai_endpoint = os.environ['AZURE_OPENAI_ENDPOINT']
```

Once the credentials are available as variables, we can initialize the Azure OpenAI client.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Initialize the Azure OpenAI client  
client = AzureOpenAI(  
        azure_endpoint = azure_oai_endpoint,   
        api_key=azure_oai_key,    
        api_version="2024-02-01"  
        )
```

**_NOTE: The latest API version for the AzureOpenAI client can be found_** [**_here_**](https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation#latest-ga-api-release)**_._**

<br/><br/>

### 5.3 Import CSV

RAG relies on underlying data to supplement the model. In our case, we will use a CSV file containing vegan recipes. Let’s import that file into our Notebook.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Import recipes from CSV file  
path_input = r"/workspaces/azure-openai-lab/data/recipes-preprocessed.csv"  
df = pd.read_csv(path_input , sep=',', on_bad_lines='skip', low_memory=False)
```

<br/><br/>

### 5.4 Vector Database Preparations

Before embedding and saving our data into a vector database such as chromadb, we first conduct some pre-processing steps. We will use the column **dense_feature** as our vector database input. This column combines the values of the columns: **name, tags, nutrition, ingredients,** and **steps** separated by a semicolon.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Create dataframe input compatible with langchain chroma  
df_text_input = pd.DataFrame(df["dense_feature"])  
df_loader = DataFrameLoader(df_text_input, page_content_column="dense_feature")  
df_document = df_loader.load()
```

Next, we will chunk our text. Chunking allows us to break down large pieces of text into smaller segments. This technique helps optimize the relevance of the content retrieved from the vector database ensuring our search results accurately capture the essence of the user’s query. Therefore, finding the optimal chunk size is crucial.

The most common and straightforward approach to chunking is Fixed-sized chunking. With this strategy we decide the number of tokens in our chunk and, optionally, whether there should be any overlap between them. In general, we will want to keep some overlap between chunks to ensure the semantic context doesn’t get lost between chunks.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Chunk input  
text_splitter = CharacterTextSplitter(  
    separator = "\n\n",  
    chunk_size = 256,  
    chunk_overlap  = 20  
)  
df_document_split = text_splitter.split_documents(df_document)
```

**_NOTE: We opted for a smaller chunk size of 256 to capture more granular semantic information._**

To generate the word embeddings of the chunked data , we use the Azure OpenAI embedding model `text-embedding-3-large`. To use this model, we use LangChains `AzureOpenAIEmbeddings` constructor.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Generate the Word Embeddings for the Dataset using Azure OpenAI with model text-embedding-ada-002
openai_ef = AzureOpenAIEmbeddings(
                deployment = "text-embedding-3-large",
                openai_api_key = azure_oai_key,
                azure_endpoint = azure_oai_endpoint,
                openai_api_version = "2024-02-01",
            )
```

We are now ready to create our vector database using `chromadb` . The database runs locally on our codespace environment.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Create the ChromaDB Vector Database collection based on the Azure OpenAI embeddings model. Vector Database is created locally.  
# Cant run code using Proxy API.  
  
vectordb = Chroma.from_documents(  
                documents = df_document_split,  
                embedding = openai_ef,  
                collection_name = "recipes",  
                persist_directory = r"/workspaces/azure-openai-lab/data/chromadb",
                collection_metadata={"hnsw:space": "cosine"}  
            )
```

<br/><br/>

### 5.5 Zero-Shot learning

Let’s run a zero-shot learning prompt and see if our RAG implementation assists us in creating vegan recipes.

First, we need to initialize the Azure OpenAI client. We will use the LangChain `AzureChatOpenAI` constructor and call the GPT-3.5-Turbo model.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Initalize Azure Openai through Langchain  
client = AzureChatOpenAI(  
                deployment_name = "gpt-35-turbo",  
                openai_api_key = azure_oai_key,  
                azure_endpoint = azure_oai_endpoint,  
                openai_api_version = "2024-02-01"  
        ) 
```

We will create a prompt template for our LangChain function using the same advanced prompt input as Part 3.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Zero-shot learning Prompt
prompt_template = \
"""
### INSTRUCTIONS
Persona: Act as a head chef such as Joël Robuchon who specializes in simple contemporary cuisine.
Action: Create well-thought-out and flavourful vegan recipes from a list of ingredients {question}, implementing classic culinary techniques.
Target Audience: The recipients of these vegan recipes are couples who want to cook a special meal at least once a week.

### EXAMPLE
{context}

### OUTPUT FORMAT
Output only one vegan recipe and return it as a JSON object with the following format:
{{"name":"","minutes":,"tags":"[]","nutrition":"[]","n_steps":"","steps":"[]","description":"","ingredients":"[]", "n_ingredients":}}

The variables should contain the following information:
- name: the name of the recipe.
- minutes: the time in minutes to prepare the recipe.
- tags: a list of words that characterize the recipe.
- nutrition: a list of numeric values representing calories, total fat, sugar, sodium, protein, saturated fat, and carbohydrates.
- n_steps: the number of steps to prepare the recipe.
- steps: a list of steps to prepare the recipe.
- description: a summary of the recipe.
- ingredients: a list of the ingredient names in the recipe.
- n_ingredients: the total number of ingredients used in the recipe.
"""


simple_prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
```

We can now generate our recipes.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Run chain to call Azure OpenAI using ChromaDB vector database data to enrich the prompt (RAG).  
ingredients = """'Capsicum', 'flour', 'Soy Sauce', 'Chili', 'Coconut', 'Broccoli'"""  
  
chain = RetrievalQA.from_chain_type(  
       llm=client,  
       retriever = vectordb.as_retriever(),  
       chain_type="stuff",  
       chain_type_kwargs={"prompt": simple_prompt}  
)  
result = chain.invoke({"query": ingredients})  
  
# View Azure OpenAI output  
display(result)
```

The generated vegan recipe looks great! The recipe variables look plausible with vegan as one of the recipe tag and the format is JSON as specified.

<br/><br/>

### 5.6 Create Dataframe from generated Recipes

We can now transform our generated output into a data frame to check the overall structure. Alternatively, we could save the generated output directly to a CSV file.

**_NOTE: uncomment the last two lines of the Notebook code to save as CSV file._**

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
#Transform output to pandas dataframe and save as CSV file

# Clean up Azure OpenAI Output
json_data = result['result'].strip('` \n')

if json_data.startswith('json'):
    json_data = json_data[4:]  # Remove the first 4 characters 'json'

recipes_from_rag_json = json.loads(json_data)
recipes_from_rag = pd.json_normalize(recipes_from_rag_json)
# path_output = r"/workspaces/azure-openai-lab/data/generated_output/recipes-from-rag.csv" #r"C:\Python\azure-openai-lab\data\generated_output\recipes-from-rag.csv"
# recipes_from_rag.to_csv(path_output, sep='\t', encoding='utf-8', index=False)
```

<br/><br/>

Congratulations! You’ve made it through Part 4 of this workshop. We’ve learned how to use RAG to enhance our model using ChromaDB and LangChain. In [Part 5][A-Hands-on-Exploration-of-the-Azure-OpenAI-APIs-part-5], we’ll learn more about fine-tuning and even try pairing RAG with a fine-tuned model.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

----------

## 6. Questions, Feedback, Support?

Reach out to us! We are happy to answer any questions you might have or use your feedback to optimize this series!

<p align="right">(<a href="#readme-top">back to top</a>)</p>

----------

## 7. References

[1] [Embeddings and RAG with Azure OpenAI API — CSC Blog (ethz.ch)](https://cscblog.ethz.ch/index.php/2024/02/06/az-open-ai-rag-chromadb-langchain/)

[2] [Tutorial: Use Chroma and OpenAI to Build a Custom Q&A Bot — The New Stack](https://thenewstack.io/tutorial-use-chroma-and-openai-to-build-a-custom-qa-bot/)

[3] [Ask your documents with LangChain, VectorDB & HF (kaggle.com)](https://www.kaggle.com/code/peremartramanonellas/ask-your-documents-with-langchain-vectordb-hf)

[4] [langchain — Azure OpenAI api — returning additional information than the asked question — Stack Overflow](https://stackoverflow.com/questions/77087460/langchain-azure-openai-api-returning-additional-information-than-the-asked-q)

[5] [Chunking Strategies for LLM Applications | Pinecone](https://www.pinecone.io/learn/chunking-strategies/)

[6] [Evaluating the Ideal Chunk Size for a RAG System using LlamaIndex — LlamaIndex, Data Framework for LLM Applications](https://www.llamaindex.ai/blog/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5)

[7] [AzureChatOpenAI | 🦜️🔗 LangChain](https://python.langchain.com/v0.1/docs/integrations/chat/azure_chat_openai/)

[8] [Approaches to AI: When to Use Prompt Engineering, Embeddings, or Fine-tuning | Entry Point AI](https://www.entrypointai.com/blog/approaches-to-ai-prompt-engineering-embeddings-or-fine-tuning/)

[9] [What is a Vector Database & How Does it Work? Use Cases + Examples | Pinecone](https://www.pinecone.io/learn/vector-database/)

[10] [What is LangChain? Getting Started with LangChain | DataStax](https://www.datastax.com/guides/what-is-langchain)

<!-- MARKDOWN LINKS & IMAGES -->
[A-Hands-on-Exploration-of-the-Azure-OpenAI-APIs-part-5]: https://github.com/AllgeierSchweiz/azure-openai-lab/blob/main/series/A%20Hands-on%20Exploration%20of%20the%20Azure%20OpenAI%20API%20(Part%205%20of%C2%A06).md
