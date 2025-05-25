----------

# A Hands-on Exploration of the Azure OpenAI API (Part 5 of 6)

![](https://cdn-images-1.medium.com/max/800/1*PV9Eh3WpAhnW9NZCijZPzw.png)

## 1. Fine-Tuning

Let’s continue our journey and learn more about enhancing our model output using an alternative technique called Fine-Tuning.

As mentioned in Part 1, Chapter 4.2 of this workshop, **Fine-Tuning** is the process of taking a pre-trained model and further training it on proprietary data to suit a specific task.

The four Jupyter Notebooks we are going to work with are:

1.  _P5-azure-openai-assistant-fine-tuning-data-preprocessing.ipynb_
2.  _P5-azure-openai-fine-tuning-training-deployment.ipynb_
3.  _P5-azure-openai-fine-tuning.ipynb_
4.  _P5-azure-openai-rag-with-fine-tuning.ipynb_

In this Notebook, we will implement Fine-Tuning using the GPT-4o-mini model version **2024–07–18** to create  well-thought-out and flavourful vegan recipes for special occasions. This model is optimized for conversational interfaces and expects an input formatted in a [specific chat-like transcript](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/chatgpt) stored inside an array of dictionaries, just like in Part 3. Additionally, we need training and a validation dataset to generate a fine-tuned model. We create these datasets from our vegan recipes CSV file you pre-processed in Part 4.

Unfortunately, creating a fine-tuned model is very resource-intensive and costly. It may take several minutes, if not hours, to finalize. For this reason, you should only create one if you need it. In this workshop, we want to showcase all possibilities and will make one for educational purposes.

<br/><br/>

### 1.2 What we are doing in a nutshell

![Fine-Tuned][Fine-Tuned]

1.  Format the data for Fine-Tuning.
2.  Split the data into a training and validation set.
3.  Import the datasets.
4.  Re-train the base model using the training and validation set.
5.  Create a user input, i.e.,prompt (list of ingredients).
6.  The fine-tuned model creates an output.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

----------

## 2. Data Pre-Processing with Azure OpenAI Assistant

Before creating a fine-tuned model, we must certify the data is correctly formatted.

Since we will be using the `gpt-4o-mini` model, the training and validation datasets must be structured in the conversational format used by the [Chat completions](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/chatgpt) API. Once formatted, both datasets must be saved as a JSON Lines (JSONL) file.

```sql
{"messages": [{"role": "system", "content": "This is a vegan recipe generator. The vegan recipe generated should be output as a JSON object in the format:{\"name\":\"\",\"minutes\":,\"tags\":\"[]\",\"nutrition\":\"[]\",\"n_steps\":\"\",\"steps\":\"[]\",\"description\":\"\",\"ingredients\":\"[]\", \"n_ingredients\":}"}, {"role": "user", "content": "Create well-thought-out and flavourful recipes from a list of ingredients implementing classic culinary techniques."}, {"role": "user", "content": "fresh snow peas, sesame oil, minced garlic cloves, salt & pepper"}, {"role": "assistant", "content": "{\"name\":\"garlic snow peas\",\"minutes\":5,\"tags\":\"['15-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'cuisine', 'preparation', 'occasion', 'low-protein', 'healthy', '5-ingredients-or-less', 'appetizers', 'side-dishes', 'vegetables', 'asian', 'chinese', 'easy', 'dinner-party', 'romantic', 'vegan', 'vegetarian', 'dietary', 'low-sodium', 'low-cholesterol', 'stir-fry', 'comfort-food', 'low-carb', 'inexpensive', 'healthy-2', 'low-in-something', 'taste-mood', 'presentation', 'served-hot', 'technique']\",\"nutrition\":\"[46.6, 5.0, 5.0, 0.0, 2.0, 2.0, 1.0]\",\"n_steps\":6,\"steps\":\"['heat wok on medium-hi heat', 'add oil get it hot , dump in peas', 'about 1 minute after stir frying add garlic and seasoning', 'quick fry just till bright green and still crisp', 'cooking time is an estimate -- do not over cook or they will turn a dull green and go limp', 'remove and serve while still hot']\",\"description\":\"i love the ones at p f changs..so this is my version\",\"ingredients\":\"['fresh snow peas', 'sesame oil', 'minced garlic cloves', 'salt & pepper']\",\"n_ingredients\":4}\n"}]}  
{"messages": [{"role": "system", "content": "This is a vegan recipe generator. The vegan recipe generated should be output as a JSON object in the format:{\"name\":\"\",\"minutes\":,\"tags\":\"[]\",\"nutrition\":\"[]\",\"n_steps\":\"\",\"steps\":\"[]\",\"description\":\"\",\"ingredients\":\"[]\", \"n_ingredients\":}"}, {"role": "user", "content": "Create well-thought-out and flavourful recipes from a list of ingredients implementing classic culinary techniques."}, {"role": "user", "content": "peaches, dates, ground cinnamon, water, almonds, coconut oil"}, {"role": "assistant", "content": "{\"name\":\"summertime peach crumble raw food\",\"minutes\":5,\"tags\":\"['15-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'cobblers-and-crisps', 'desserts', 'fruit', 'easy', 'vegan', 'vegetarian', 'food-processor-blender', 'dietary', 'low-sodium', 'low-in-something', 'pitted-fruit', 'peaches', 'equipment', 'small-appliance']\",\"nutrition\":\"[330.8, 32.0, 108.0, 3.0, 15.0, 34.0, 11.0]\",\"n_steps\":5,\"steps\":\"['for the base ~ place chopped peaches in large bowl & set aside', 'in blender or food processor , blend 5 dates , cinnamon & enough water to get a syrupy consistency', 'pour this date syrup over peaches & refrigerate at least 1 hour', 'for the topping ~ when peach \\/ date mixture has chilled , in a food processor , blend almonds , 4 dates & coconut oil to desired consistency', 'top peach \\/ date mixture with this topping & enjoy']\",\"description\":\"this recipe comes the goneraw internet site.\",\"ingredients\":\"['peaches', 'dates', 'ground cinnamon', 'water', 'almonds', 'coconut oil']\",\"n_ingredients\":6}\n"}]}  
{"messages": [{"role": "system", "content": "This is a vegan recipe generator. The vegan recipe generated should be output as a JSON object in the format:{\"name\":\"\",\"minutes\":,\"tags\":\"[]\",\"nutrition\":\"[]\",\"n_steps\":\"\",\"steps\":\"[]\",\"description\":\"\",\"ingredients\":\"[]\", \"n_ingredients\":}"}, {"role": "user", "content": "Create well-thought-out and flavourful recipes from a list of ingredients implementing classic culinary techniques."}, {"role": "user", "content": "grape-nuts cereal, apple juice concentrate"}, {"role": "assistant", "content": "{\"name\":\"fat free pie crust\",\"minutes\":13,\"tags\":\"['15-minutes-or-less', 'time-to-make', 'course', 'cuisine', 'preparation', 'north-american', 'healthy', '5-ingredients-or-less', 'pies-and-tarts', 'desserts', 'american', 'easy', 'low-fat', 'vegan', 'vegetarian', 'pies', 'dietary', 'low-cholesterol', 'low-saturated-fat', 'healthy-2', 'low-in-something']\",\"nutrition\":\"[88.9, 0.0, 27.0, 5.0, 4.0, 0.0, 6.0]\",\"n_steps\":6,\"steps\":\"['preheat the oven to 350 f \\/ 180 deg c', 'mix together the grape nuts and apple juice concentrate', 'pat into a thin layer on the bottom and sides of a 9 pie pan', \\\"don't worry if there are some gaps\\\", 'bake for 8 minutes', 'cool before filling']\",\"description\":\"from the table of the physicians committee for responsible medicine\",\"ingredients\":\"['grape-nuts cereal', 'apple juice concentrate']\",\"n_ingredients\":2}\n"}]}  
{"messages": [{"role": "system", "content": "This is a vegan recipe generator. The vegan recipe generated should be output as a JSON object in the format:{\"name\":\"\",\"minutes\":,\"tags\":\"[]\",\"nutrition\":\"[]\",\"n_steps\":\"\",\"steps\":\"[]\",\"description\":\"\",\"ingredients\":\"[]\", \"n_ingredients\":}"}, {"role": "user", "content": "Create well-thought-out and flavourful recipes from a list of ingredients implementing classic culinary techniques."}, {"role": "user", "content": "lemon juice, granulated sugar, water, lemon slice"}, {"role": "assistant", "content": "{\"name\":\"refreshing lemonade\",\"minutes\":5,\"tags\":\"['15-minutes-or-less', 'time-to-make', 'course', 'preparation', 'occasion', 'low-protein', 'healthy', '5-ingredients-or-less', 'beverages', 'easy', 'beginner-cook', 'low-fat', 'summer', 'vegan', 'vegetarian', 'dietary', 'low-sodium', 'low-cholesterol', 'seasonal', 'low-saturated-fat', 'low-calorie', 'inexpensive', 'low-in-something', 'presentation', 'served-cold']\",\"nutrition\":\"[116.5, 0.0, 115.0, 0.0, 0.0, 0.0, 10.0]\",\"n_steps\":4,\"steps\":\"['mix the lemon juice and sugar in a 2 quart pitcher', 'fill the container up with water until it reaches 2 quarts', 'float the lemon slices on top', 'chill , then enjoy !']\",\"description\":\"this is a really easy lemonade recipe.  i use bottled lemon juice because it's cheaper, but you could definately use fresh.\",\"ingredients\":\"['lemon juice', 'granulated sugar', 'water', 'lemon slice']\",\"n_ingredients\":4}\n"}]}
```

The data in the CSV will be formatted according to the specified structure using Azure OpenAI Assistant. We will provide a set of instructions outlining the requirements. The Assistant’s Code Interpreter will use Python to transform the data accordingly.

The Jupyter Notebook we are going to work with is called **P5-azure-openai-assistant-fine-tuning-data-preprocessing.ipynb**

<br/><br/>

### 2.1 Open the Notebook

-   On the left of the codespace environment, select the **Explorer** icon.
-   Open the Notebook **P5-azure-openai-assistant-fine-tuning-data-preprocessing.ipynb**

![](https://cdn-images-1.medium.com/max/800/1*lfiQhZ_KNid07f9lHvFxKA.png)

<br/><br/>

### 2.2 Initializing the Azure OpenAI Client

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
# Load required variables from env file.  
load_dotenv(dotenv_path=Path("/workspaces/azure-openai-lab/.venv/.env")) #Error sometimes due to \ or \\. Try one or the other. "C:\\Python\\azure-openai-lab\\.venv\\.env"  
  
# Load Azure OpenAI Key and Endpoint. These values can be found within the Azure OpenAI Service resource in portal.azure.com under Keys and Endpoint  
azure_oai_key = os.environ['AZURE_OPENAI_KEY_P34']  
azure_oai_endpoint = os.environ['AZURE_OPENAI_ENDPOINT_P34']
```

Once the credentials are available as variables, we can initialize the Azure OpenAI client.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Initialize the Azure OpenAI client
client = AzureOpenAI(
    api_key = azure_oai_key,  
    api_version = "2024-05-01-preview",
    azure_endpoint = azure_oai_endpoint
    )
```

**_NOTE: The latest API version for the AzureOpenAI client when using the Azure OpenAI Assistant API can be found_** [**_here_**](https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation#latest-ga-api-release)**_._**

<br/><br/>

### 2.3 Upload CSV File to Azure OpenAI Service

We will be using the pre-processed data generated by the Azure OpenAI Assistant in Part 4 Chapter 4. Let’s upload the file to Azure OpenAI Service.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Upload file into Azure OpenAI Service [NOT USED IN WORKSHOP]  
path_input = r"/workspaces/azure-openai-lab/data/recipes-preprocessed.csv" #Change path if required  
  
# send the csv file to the assistant purpose files  
response = client.files.create(  
  file=open(path_input, "rb"),  
  purpose="assistants"  
)  
print(response)  
file__id = response.id
```

<br/><br/>

### 2.4 Data Wrangling Instructions

To perform our data pre-processing steps using the Azure OpenAI Assistant, we need to provide the model with clear instructions outlining our requirements.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Create data transformation instructions
instructions = '''
### INSTRUCTIONS
You are a senior data analyst who will work with data in an csv file.
You have access to a sandboxed environment for writing python code.
The objective is to create a datset for fine-tuning. The dataset must be formatted in the conversational format that is used by the Chat completions API.
An example of the conversational format is available in the EXAMPLES section.
When the user asks you to perform your actions, you will use the provided csv file and examples in the EXAMPLE section.
Execute each of the steps listed below in your ACTIONS section.

---

### EXAMPLES:

{"messages": [{"role": "system", "content": "This is a vegan recipe generator. The vegan recipe generated should be output as a JSON object in the format:{\"name\":\"\",\"minutes\":,\"tags\":\"[]\",\"nutrition\":\"[]\",\"n_steps\":\"\",\"steps\":\"[]\",\"description\":\"\",\"ingredients\":\"[]\", \"n_ingredients\":}"}, {"role": "user", "content": "Create well-thought-out and flavourful recipes from a list of ingredients implementing classic culinary techniques."}, {"role": "user", "content": "fresh snow peas, sesame oil, minced garlic cloves, salt & pepper"}, {"role": "assistant", "content": "{\"name\":\"garlic snow peas\",\"minutes\":5,\"tags\":\"['15-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'cuisine', 'preparation', 'occasion', 'low-protein', 'healthy', '5-ingredients-or-less', 'appetizers', 'side-dishes', 'vegetables', 'asian', 'chinese', 'easy', 'dinner-party', 'romantic', 'vegan', 'vegetarian', 'dietary', 'low-sodium', 'low-cholesterol', 'stir-fry', 'comfort-food', 'low-carb', 'inexpensive', 'healthy-2', 'low-in-something', 'taste-mood', 'presentation', 'served-hot', 'technique']\",\"nutrition\":\"[46.6, 5.0, 5.0, 0.0, 2.0, 2.0, 1.0]\",\"n_steps\":6,\"steps\":\"['heat wok on medium-hi heat', 'add oil get it hot , dump in peas', 'about 1 minute after stir frying add garlic and seasoning', 'quick fry just till bright green and still crisp', 'cooking time is an estimate -- do not over cook or they will turn a dull green and go limp', 'remove and serve while still hot']\",\"description\":\"i love the ones at p f changs..so this is my version\",\"ingredients\":\"['fresh snow peas', 'sesame oil', 'minced garlic cloves', 'salt & pepper']\",\"n_ingredients\":4}\n"}]}
{"messages": [{"role": "system", "content": "This is a vegan recipe generator. The vegan recipe generated should be output as a JSON object in the format:{\"name\":\"\",\"minutes\":,\"tags\":\"[]\",\"nutrition\":\"[]\",\"n_steps\":\"\",\"steps\":\"[]\",\"description\":\"\",\"ingredients\":\"[]\", \"n_ingredients\":}"}, {"role": "user", "content": "Create well-thought-out and flavourful recipes from a list of ingredients implementing classic culinary techniques."}, {"role": "user", "content": "peaches, dates, ground cinnamon, water, almonds, coconut oil"}, {"role": "assistant", "content": "{\"name\":\"summertime peach crumble raw food\",\"minutes\":5,\"tags\":\"['15-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'cobblers-and-crisps', 'desserts', 'fruit', 'easy', 'vegan', 'vegetarian', 'food-processor-blender', 'dietary', 'low-sodium', 'low-in-something', 'pitted-fruit', 'peaches', 'equipment', 'small-appliance']\",\"nutrition\":\"[330.8, 32.0, 108.0, 3.0, 15.0, 34.0, 11.0]\",\"n_steps\":5,\"steps\":\"['for the base ~ place chopped peaches in large bowl & set aside', 'in blender or food processor , blend 5 dates , cinnamon & enough water to get a syrupy consistency', 'pour this date syrup over peaches & refrigerate at least 1 hour', 'for the topping ~ when peach \\/ date mixture has chilled , in a food processor , blend almonds , 4 dates & coconut oil to desired consistency', 'top peach \\/ date mixture with this topping & enjoy']\",\"description\":\"this recipe comes the goneraw internet site.\",\"ingredients\":\"['peaches', 'dates', 'ground cinnamon', 'water', 'almonds', 'coconut oil']\",\"n_ingredients\":6}\n"}]}
{"messages": [{"role": "system", "content": "This is a vegan recipe generator. The vegan recipe generated should be output as a JSON object in the format:{\"name\":\"\",\"minutes\":,\"tags\":\"[]\",\"nutrition\":\"[]\",\"n_steps\":\"\",\"steps\":\"[]\",\"description\":\"\",\"ingredients\":\"[]\", \"n_ingredients\":}"}, {"role": "user", "content": "Create well-thought-out and flavourful recipes from a list of ingredients implementing classic culinary techniques."}, {"role": "user", "content": "grape-nuts cereal, apple juice concentrate"}, {"role": "assistant", "content": "{\"name\":\"fat free pie crust\",\"minutes\":13,\"tags\":\"['15-minutes-or-less', 'time-to-make', 'course', 'cuisine', 'preparation', 'north-american', 'healthy', '5-ingredients-or-less', 'pies-and-tarts', 'desserts', 'american', 'easy', 'low-fat', 'vegan', 'vegetarian', 'pies', 'dietary', 'low-cholesterol', 'low-saturated-fat', 'healthy-2', 'low-in-something']\",\"nutrition\":\"[88.9, 0.0, 27.0, 5.0, 4.0, 0.0, 6.0]\",\"n_steps\":6,\"steps\":\"['preheat the oven to 350 f \\/ 180 deg c', 'mix together the grape nuts and apple juice concentrate', 'pat into a thin layer on the bottom and sides of a 9 pie pan', \\\"don't worry if there are some gaps\\\", 'bake for 8 minutes', 'cool before filling']\",\"description\":\"from the table of the physicians committee for responsible medicine\",\"ingredients\":\"['grape-nuts cereal', 'apple juice concentrate']\",\"n_ingredients\":2}\n"}]}
{"messages": [{"role": "system", "content": "This is a vegan recipe generator. The vegan recipe generated should be output as a JSON object in the format:{\"name\":\"\",\"minutes\":,\"tags\":\"[]\",\"nutrition\":\"[]\",\"n_steps\":\"\",\"steps\":\"[]\",\"description\":\"\",\"ingredients\":\"[]\", \"n_ingredients\":}"}, {"role": "user", "content": "Create well-thought-out and flavourful recipes from a list of ingredients implementing classic culinary techniques."}, {"role": "user", "content": "lemon juice, granulated sugar, water, lemon slice"}, {"role": "assistant", "content": "{\"name\":\"refreshing lemonade\",\"minutes\":5,\"tags\":\"['15-minutes-or-less', 'time-to-make', 'course', 'preparation', 'occasion', 'low-protein', 'healthy', '5-ingredients-or-less', 'beverages', 'easy', 'beginner-cook', 'low-fat', 'summer', 'vegan', 'vegetarian', 'dietary', 'low-sodium', 'low-cholesterol', 'seasonal', 'low-saturated-fat', 'low-calorie', 'inexpensive', 'low-in-something', 'presentation', 'served-cold']\",\"nutrition\":\"[116.5, 0.0, 115.0, 0.0, 0.0, 0.0, 10.0]\",\"n_steps\":4,\"steps\":\"['mix the lemon juice and sugar in a 2 quart pitcher', 'fill the container up with water until it reaches 2 quarts', 'float the lemon slices on top', 'chill , then enjoy !']\",\"description\":\"this is a really easy lemonade recipe.  i use bottled lemon juice because it's cheaper, but you could definately use fresh.\",\"ingredients\":\"['lemon juice', 'granulated sugar', 'water', 'lemon slice']\",\"n_ingredients\":4}\n"}]}

---

### ACTIONS:

1. Read the tab separated comma file data
2. Transform the data and create a jsonl file formatted in the conversational format as shown in the EXAMPLES section
3. The conversational format has a system role, user role and assistant role, each with text content stored inside an array of dictionaries
4. The first system role content is always: "This is a vegan recipe generator. The vegan recipe generated should be output as a JSON object in the format: {
  "name": "",
  "minutes": 0,
  "tags": [],
  "nutrition": [],
  "n_steps": 0,
  "steps": [],
  "description": "",
  "ingredients": [],
  "n_ingredients": 0
}"
5. The first user role content is always: "Create well-thought-out and flavourful vegan recipes from a list of ingredients implementing classic culinary techniques"
6. The subsequent user role content takes the list of ingredients in the column "ingredients" of the CSV file
7. The assistant role content uses all the column values of the CSV file as a JSON object with the format: {
  "name": "",
  "minutes": 0,
  "tags": [],
  "nutrition": [],
  "n_steps": 0,
  "steps": [],
  "description": "",
  "ingredients": [],
  "n_ingredients": 0
}
8. Split the data set into training and testing data sets with a "75%" and "25%" split respectively
9. Make sure both data sets have the same format provided by the EXAMPLES section
10. Name the data set with "75%" of the data "recipes-training-set"
11. Name the data set with "25%" of the data "recipes-validation-set"
12. Prepare both data sets as a jsonl file for download by the user

---

### DO NOT:
1. Do not return any images. 
2. Do not return any other file types.
'''
```

<br/><br/>

### 2.5 Create an Azure OpenAI Assistant

Just like in Part 4, let’s start the Azure OpenAI Assistant using C**ode Interpreter** enabled to perform our data transformations.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Create an Azure OpenAI Assistant
assistant = client.beta.assistants.create(
    name="data analyst assistant",
    instructions=instructions,
    tools=[{"type": "code_interpreter"}],
    model="gpt-4o-mini",  # Replace this value with the deployment name for your model.
    tool_resources={
        "code_interpreter": {"file_ids": [file__id]}
    }
)

# Create a thread
thread = client.beta.threads.create()
```

We can now initialize the thread and send our instructions to the Azure OpenAI Assistant for evaluation.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Initalize thread and start data transformation using the Azure OpenAI Assistant Code Interpreter
prompt = "Please execute the INSTRUCTIONS and ACTIONS on the data stored in the CSV file using the EXAMPLES as reference for the output format"

message = client.beta.threads.messages.create(
    thread_id = thread.id,
    role = "user",
    content = prompt
)
```

<br/><br/>

### 2.6 Run the Azure OpenAI Assistant

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
        print("in progress...")
```

<br/><br/>

### 2.7 Import Generated CSV

Let’s view the newly created file and check the transformations.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Functions to read xlsx files from Azure Openai

output_path = r"/workspaces/azure-openai-lab/data/generated_output/" #r"C:\\Python\\azure-openai-lab\\data\\generated_output\\"

# Write to jsonl
def write_jsonl(data_list: list, filename: str) -> None:
    with open(filename, "w") as out:
        for ddict in data_list:
            jout = json.dumps(ddict) + "\n"
            out.write(jout)


def read_and_save_file(first_file_id, file_name, output_path):   
    # its binary, so read it and then make it a file like object
    file_data = client.files.content(first_file_id)
    file_data_bytes = file_data.read()
    file_data_decoded = file_data_bytes.decode('utf8').replace("'", '"')
    file_data_list = file_data_decoded.splitlines()
    write_jsonl(file_data_list, output_path + file_name)

    
def files_from_messages():
    messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )
    first_thread_message = messages.data[0]  # Accessing the first ThreadMessage
    message_ids = first_thread_message.attachments
    print(message_ids)
    # Loop through each file ID and save the file with a sequential name
    for i, file_id in enumerate(message_ids):
        if i == 1:
            file_name = f"recipes-training-set.jsonl"  # Generate a sequential file name
            read_and_save_file(file_id.file_id, file_name, output_path)
        else:
            file_name = f"recipes-validation-set.jsonl"  # Generate a sequential file name
            read_and_save_file(file_id.file_id, file_name, output_path)

# Extract the file names from the response, retrieve the content and save the data as a jsonl file
files_from_messages()
```

<br/><br/>

### 2.8 Clean up Azure OpenAI Assistant Environment

You want to make sure there aren’t any unnecessary artifacts laying around in Azure OpenAI Service. Let’s delete the assistant, thread, and generated CSV file from the Azure OpenAI environment.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
#Clean up Azure OpenAI environment
client.beta.assistants.delete(assistant.id)
client.beta.threads.delete(thread.id)
for i in range(0, 2):
    client.files.delete(messages.data[0].attachments[i].file_id)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

----------

## 3. Getting started with Fine-Tuning [NOT USED IN THE WORKSHOP]

Now that the training and validation datasets are ready, we can start fine-tuning a base model.

The Jupyter Notebook we are going to work with to achieve this is called **P5-azure-openai-fine-tuning-training-deployment.ipynb**

<br/><br/>

### 3.1 Open the Notebook

-   On the left of the codespace environment, select the **Explorer** icon.
-   Open the Notebook **P5-azure-openai-fine-tuning-training-deployment.ipynb**

![](https://cdn-images-1.medium.com/max/800/1*pkSd77WrUp23siYY7d44ZA.png)

<br/><br/>

### 3.2 Initializing the Azure OpenAI Client

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
import json  
from IPython.display import clear_output  
from IPython.core.display import HTML  
import requests  
import random
```

-   If necessary, select a kernel to work with.

![](https://cdn-images-1.medium.com/max/800/1*-Vz32KNx4eQ1b5PS5cM_2g.png)

Next, we will load the Azure OpenAI Key and Endpoint as variables.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Load required variables from .env file.
load_dotenv(dotenv_path=Path("/workspaces/azure-openai-lab/.venv/.env")) #Error sometimes due to \ or \\. Try one or the other. "C:\\Python\\azure-openai-lab\\.venv\\.env"

# Load Azure OpenAI Key and Endpoint. These values can be found within the Azure OpenAI Service resource in portal.azure.com under Keys and Endpoint
azure_oai_key = os.environ['AZURE_OPENAI_KEY_P34']
azure_oai_endpoint = os.environ['AZURE_OPENAI_ENDPOINT_P34']
```

Once the credentials are available as variables, we can initialize the Azure OpenAI client.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Initialize the Azure OpenAI client
client = AzureOpenAI(
    api_key = azure_oai_key,  
    azure_endpoint = azure_oai_endpoint,
    api_version = "2024-05-01-preview"
    )
```

**_NOTE: The latest API version for the AzureOpenAI client when using the Azure OpenAI Assistant API can be found_** [**_here_**](https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation#latest-ga-api-release)**_._**

<br/><br/>

### 3.3 Upload JSONL Training and Validation Files

To train our base model, we must upload the training and validation JSONL formatted datasets into Azure OpenAI Service.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
#Load JSONL to Azure OpenAI Service  
  
training_file_path = r"/workspaces/azure-openai-lab/data/recipes-training-set.jsonl"  
validation_file_path = r"/workspaces/azure-openai-lab/data/recipes-validation-set.jsonl"  
  
# Upload the training and validation dataset files to Azure OpenAI with the SDK.  
  
training_response = client.files.create(  
    file=open(training_file_path, "rb"), purpose="fine-tune"  
)  
training_file_id = training_response.id  
  
validation_response = client.files.create(  
    file=open(validation_file_path, "rb"), purpose="fine-tune"  
)  
validation_file_id = validation_response.id  
  
print("Training file ID:", training_file_id)  
print("Validation file ID:", validation_file_id)
```

<br/><br/>

### 3.4 Run Fine-Tuning

We are now ready to run the fine-tuning job.

**_IMPORTANT: Running a fine-tuning job will incur training costs. Depending on the size of your training and validation datasets, these costs may be higher than expected. Only run a fine-tuning job if you plan on using a fine-tuned model and always check the current_** [**_pricing_**](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) **_to ensure you understand the charges._**

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Initalize Fine-Tuning Job
finetune = client.fine_tuning.jobs.create(
    training_file=training_file_id,
    validation_file=validation_file_id,
    model="gpt-4o-mini",
)

job_id = finetune.id
```

To continuously check the status of the job, we can run the code below.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql  
# Model deployment current takes 45 - 60 minutes  
  
# Track training status  
start_time = time.time()  
  
# Get the status of our fine-tuning job.  
finetune = client.fine_tuning.jobs.retrieve(job_id)  
  
status = finetune.status  
  
# If the job isn't done yet, poll it every 10 seconds.  
while status not in ["succeeded", "failed"]:  
    time.sleep(10)  
      
    finetune = client.fine_tuning.jobs.retrieve(job_id)  
    print(finetune.model_dump_json(indent=2))  
    print("Elapsed time: {} minutes {} seconds".format(int((time.time() - start_time) // 60), int((time.time() - start_time) % 60)))  
    status = finetune.status  
    print(f'Status: {status}')  
    clear_output(wait=True)  
  
print(f'Fine-tuning job {job_id} finished with status: {status}')  
  
# List all fine-tuning jobs for this resource.  
print('Checking other fine-tune jobs for this resource.')  
finetune = client.fine_tuning.jobs.list()  
print(f'Found {len(finetune.data)} fine-tune jobs.')
```

**_NOTE: It may take 45–60 minutes for the job to finalize. If your job takes longer than 60 minutes, check to see if you’re not running into any tokens per minute rate limit issues._**

Once the job has been finalized, you can deploy the fine-tuned model and make it available for usage.

You can programmatically deploy the model in Python using an authentication key. To make things easier, we will showcase how to do this directly in Azure OpenAI Studio.

**_IMPORTANT: Deploying a model can incur high costs. Only deploy a fine-tuned model if you plan on using it and always check the current_** [**_pricing_**](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) **_to ensure you understand the charges._**

-   In the Azure Portal, open the Azure OpenAI Service resource you created located in **Sweden Central** and select the **Explore Azure AI Foundry Portal** button

![](https://cdn-images-1.medium.com/max/800/1*PLejFC9jozUzVhWlzLYjUA.png)

-   In Azure AI Foundry, select the **Fine-Tuning** tab on the left panel, click the fine-tuned model you created and select **Deploy**.

![](https://cdn-images-1.medium.com/max/800/1*4jCTsPTzWTURBUZGTjF2Ww.png)

-   Once you selected the Deploy button, a new window will pop up. Adjusted the name of the model to `gpt-4o-mini-ft`and select **Deploy**.

![](https://cdn-images-1.medium.com/max/800/1*aEaXBfEL3pKX1imIgNAZ4Q.png)

-   Head over to the **Deployments** tab on the left panel. You should see the fine-tuned model you created. The initial status will be **Creating** and should move to status **Succeeded** after a few minutes.

![](https://cdn-images-1.medium.com/max/800/1*2xOPRKmZUHcyZgYu_BSSSQ.png)

You are now ready to start creating recipes using your fine-tuned model!

<p align="right">(<a href="#readme-top">back to top</a>)</p>

----------

## 4. Using the Fine-Tuned Model

### 4.1 Open the Notebook

-   On the left of the codespace environment, select the **Explorer** icon.
-   Open the Notebook **P5-azure-openai-fine-tuning.ipynb.**

![](https://cdn-images-1.medium.com/max/800/1*xabLofXM2SoB0AxOjCGxkg.png)

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
import json  
from IPython.display import clear_output  
from IPython.core.display import HTML  
import requests  
import random
```

Next, we will load the Azure OpenAI Key and Endpoint as variables.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Load required variables from .env file.
load_dotenv(dotenv_path=Path("/workspaces/azure-openai-lab/.venv/.env")) #Error sometimes due to \ or \\. Try one or the other. "C:\\Python\\azure-openai-lab\\.venv\\.env"

# Load Azure OpenAI Key and Endpoint. These values can be found within the Azure OpenAI Service resource in portal.azure.com under Keys and Endpoint
azure_oai_key = os.environ['AZURE_OPENAI_KEY_P34']
azure_oai_endpoint = os.environ['AZURE_OPENAI_ENDPOINT_P34']
```

Once the credentials are available as variables, we can initialize the Azure OpenAI client.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```
# Initialize the Azure OpenAI client
client = AzureOpenAI(
    api_key = azure_oai_key,  
    azure_endpoint = azure_oai_endpoint,
    api_version = "2024-12-01-preview"
    )
```

**_NOTE: The latest API version for the AzureOpenAI client when fine-tuning can be found_** [**_here_**](https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation#latest-ga-api-release)**_._**

<br/><br/>

### 4.3 Zero-Shot learning

The client has been initialized so that we can create our recipe using our fine-tuned model.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Zero-Shot learning. Model has a token limit of 4096.

# Create advanced System prompt
systemcontent = \
"""
### INSTRUCTIONS
Persona: Act as a head chef such as Joël Robuchon who specializes in simple contemporary cuisine.
Action: Create well-thought-out and flavourful vegan recipes from a list of ingredients implementing classic culinary techniques.
Target Audience: The recipients of these vegan recipes are couples who want to cook a special meal at least once a week.

---

### OUTPUT FORMAT
Output only one vegan recipe and return it as a JSON object with the following format:
{
  "name": "",
  "minutes": 0,
  "tags": [],
  "nutrition": [],
  "n_steps": 0,
  "steps": [],
  "description": "",
  "ingredients": [],
  "n_ingredients": 0
}

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

# Create a prompt of ingredients the model should create a recipe from
ingredients = """'Tofu', 'Avocado', 'Soy Sauce', 'Chili', 'Coconut Milk', 'Broccoli'"""

# Send request to Azure OpenAI model
completion = client.chat.completions.create(
  model = "gpt-4o-mini-ft",
  #response_format={ "type": "json_object" }, # Not support for fine tuned models
  messages = [    
    {"role": "system", "content": systemcontent},
    {"role": "user", "content": ingredients}
  ]
)

# View generated recipe
result = completion.choices[0].message.content
print(result)

```

The generated recipe is indeed vegan and it looks plausible! The format fits our specified JSON object too! We are all set at this point, but out of curiosity, let’s add RAG to the mix.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

----------

## 5. Fine-Tuning and RAG

Large Language models are generally trained to be helpful generalists by default. We learned that fine-tuning can narrow their focus and instill specific behavior patterns. With RAG we can enhance the model by providing relevant context with new information before generating a response. Therefore, retrieval strategies such as RAG complement fine-tuning rather than serving as an alternative.

Let’s use our fine-tuned model together with RAG. We will re-use the embeddings saved in the vector database we created in Part 4.

<br/><br/>

### 5.1 Open the Notebook

-   On the left of the codespace environment, select the **Explorer** icon.
-   Open the Notebook **P5-azure-openai-rag-with-fine-tuning.ipynb.**

![](https://cdn-images-1.medium.com/max/800/1*7ryYdSENmcz-IqwmB6AhtQ.png)

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
#from langchain.vectorstores import Chroma
from langchain_chroma import Chroma
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
load_dotenv(dotenv_path=Path("/workspaces/azure-openai-lab/.venv/.env")) #Error sometimes due to \ or \\. Try one or the other. "C:\\Python\\azure-openai-lab\\.venv\\.env"

# Load Azure OpenAI Key and Endpoint. These values can be found within the Azure OpenAI Service resource in portal.azure.com under Keys and Endpoint
azure_oai_key = os.environ['AZURE_OPENAI_KEY_P34']
azure_oai_endpoint = os.environ['AZURE_OPENAI_ENDPOINT_P34']
```

Once the credentials are available as variables, we can initialize the Azure OpenAI client using our new fine-tuned model `gpt-35-turbo-0613-ft`.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Initalize Azure Openai using LangChain
client = AzureChatOpenAI(
                deployment_name = "gpt-4o-mini-ft",
                openai_api_key = azure_oai_key,
                azure_endpoint = azure_oai_endpoint,
                openai_api_version = "2024-12-01-preview"
        )
```

**_NOTE: The latest API version for the AzureOpenAI client can be found_** [**_here_**](https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation#latest-ga-api-release)**_._**

<br/><br/>

### 5.3 Reload Vector Database

Since we have already created a vector database with the embeddings of our data in Part 4, we can reload and re-use this database for our RAG implementation.

To do this, we need to re-initialize the `AzureOpenAIEmbeddings` constructor to define the model retriever and pass that to the chain. This will allow us to use our previously persisted Database with new prompts.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Generate the Word Embeddings for the Dataset using Azure OpenAI with model text-embedding-ada-002
openai_ef = AzureOpenAIEmbeddings(
                deployment = "text-embedding-3-large",
                openai_api_key = azure_oai_key,
                azure_endpoint = azure_oai_endpoint,
                openai_api_version = "2024-10-21",
            )
```

We are now ready to reload the vector database.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Re-load an existing vector database from a local path  
vectordb = Chroma(persist_directory=r"/workspaces/azure-openai-lab/data/chromadb", embedding_function=openai_ef)
```

<br/><br/>

### 5.4 Zero-Shot learning

Let’s run a zero-shot learning prompt using our fine-tuned model together with our RAG implementation.

We will use the same prompt template as in Part 4.

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
{{
  "name": "",
  "minutes": 0,
  "tags": [],
  "nutrition": [],
  "n_steps": 0,
  "steps": [],
  "description": "",
  "ingredients": [],
  "n_ingredients": 0
}}

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
and generate a vegan recipe.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```
# Run chain to call Azure OpenAI using ChromaDB vector database data to enrich the prompt (RAG).
ingredients = """'Tofu', 'Avocado', 'Soy Sauce', 'Chili', 'Coconut Milk', 'Broccoli'"""

chain = RetrievalQA.from_chain_type(
       llm=client,
       retriever = vectordb.as_retriever(),
       chain_type="stuff",
       chain_type_kwargs={"prompt": simple_prompt}
)

# View generated recipe
result = chain.invoke({"query": ingredients})
print(result)
```
The generated vegan recipe looks great! The recipe variables look plausible with vegan as one of the recipe tag and the format is JSON as specified.

Congratulations! You have completed Part 5 of this workshop. We have learned how to use the Azure OpenAI Assistant API for data pre-processing, and have used fine-tuned models to generate better recipe outputs. We will continue with [Part 6][A-Hands-on-Exploration-of-the-Azure-OpenAI-APIs-part-6] of the workshop and learn how we can use the new GPT-4o model to generate a list of ingredients from images.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

----------

## 6. Questions, Feedback, Support?

Reach out to us! We are happy to answer any questions you might have or use your feedback to optimize this series!

<p align="right">(<a href="#readme-top">back to top</a>)</p>

----------

## 7. References

[1] [Full Fine-Tuning, PEFT, Prompt Engineering, or RAG? (deci.ai)](https://deci.ai/blog/fine-tuning-peft-prompt-engineering-and-rag-which-one-is-right-for-you/)

<!-- MARKDOWN LINKS & IMAGES -->
[A-Hands-on-Exploration-of-the-Azure-OpenAI-APIs-part-6]: https://github.com/AllgeierSchweiz/azure-openai-lab/blob/main/series/A%20Hands-on%20Exploration%20of%20the%20Azure%20OpenAI%20API%20(Part%206%20of%C2%A06).md
[Fine-Tuned]: https://github.com/AllgeierSchweiz/azure-openai-lab/blob/main/images/Fine_Tuning.jpg

[A-Hands-on-Exploration-of-the-Azure-OpenAI-APIs-part-6]: https://github.com/AllgeierSchweiz/azure-openai-lab/blob/main/series/A%20Hands-on%20Exploration%20of%20the%20Azure%20OpenAI%20API%20(Part%206%20of%C2%A06).md
