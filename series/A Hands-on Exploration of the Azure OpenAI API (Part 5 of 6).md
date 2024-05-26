----------

### A Hands-on Exploration of the Azure OpenAI API (Part 5 of 5)

![](https://cdn-images-1.medium.com/max/800/1*PV9Eh3WpAhnW9NZCijZPzw.png)

### 1. Fine-Tuning

Let’s continue our journey and learn more about enhancing our model output using an alternative technique called Fine-Tuning.

As mentioned in Part 1, Chapter 4.2 of this workshop: **Fine-Tuning** is the process of taking a pre-trained model and further training it on proprietary data to suit a specific task.

The three Jupyter Notebooks we are going to work with are:

1.  _P5-azure-openai-assistant-fine-tuning-data-preprocessing.ipynb_
2.  _P5-azure-openai-fine-tuning.ipynb_
3.  _P5-azure-openai-rag-with-fine-tuning.ipynb_

In this Notebook, we will implement Fine-Tuning using the GPT-3.5-Turbo model version **0613** to create  well-thought-out and flavourful vegan recipes for special occasions. This model is optimized for conversational interfaces and expects an input formatted in a [specific chat-like transcript](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/chatgpt) stored inside an array of dictionaries just like in Part 3. Additionally, to generate a fine-tuned model, we need a training and a validation dataset. We create these datasets from our vegan recipes CSV file.

Unfortunately, creating a fine-tuned model is very resource-intensive and may take several minutes, if not hours, to finalize. For this reason, we have already created a model for you to play with. The model is named `gpt-35-turbo-0613-ft`.

Lets take a look at the vegan recipes it generates!

<br/><br/>

#### 1.2 What we are doing in a nutshell

![Fine-Tuned][Fine-Tuned]

1.  Format the data for Fine-Tuning
2.  Split the data into a training and validation set.
3.  Import the datasets.
4.  Re-train the base model using the training and validation set.
5.  Create a user input i.e. prompt (list of ingredients).
6.  The fine-tuned model creates an output.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

----------

### 2. Data Pre-Processing with Azure OpenAI Assistant

Before creating a fine-tuned model, we need to certify the data is correctly formatted. Since we will be using the `gpt-35-turbo-0613`model, the training and validation datasets must use the conversational format that is used by the [Chat completions](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/chatgpt) API. Once formatted, both files must be saved as a JSON Lines (JSONL) document.

```sql
{"messages": [{"role": "system", "content": "Act as a head chef and create a flavourful recipe from a list of ingredients"}, {"role": "user", "content": "noodles, celery, bamboo shoots, peanut oil, garlic cloves, onion, light soy sauce, dark soy sauce, fresh ginger, vegetable stock, sugar, bean sprouts"}, {"role": "assistant", "content": "{\"name\":\"vegetarian chow mein\",\"minutes\":32,\"tags\":\"['60-minutes-or-less', 'time-to-make', 'cuisine', 'preparation', 'asian', 'chinese', 'vegan', 'vegetarian', 'dietary']\",\"nutrition\":\"[316.6, 14.0, 20.0, 32.0, 22.0, 9.0, 15.0]\",\"n_steps\":12,\"steps\":\"['if you are using fresh noodles , blanch them first in a large pot of boiling water for 3 to 5 minutes', 'if you are using the dried noodles , cook in boiling water for 4 to 5 minutes', 'drain the noodles , then put them into cold water until required', 'string the celery and slice diagonally', 'shred the bamboo shoots', 'heat a wok or large frying-pan and add the oil', 'when moderately hot , add the garlic and stir-fry for 10 seconds', 'add the onion , celery , and bamboo shoots and stir-fry for about 5 minutes', 'drain the noodles thoroughly and put into the wok', 'continue to stir-fry for 1 minute then add the rest of the ingredients except the bean sprouts', 'continue to stir-fry for another 2 minutes and then add the bean sprouts', 'give the mixture a good stir and turn it onto a serving platter']\",\"description\":\"335 calories (kcal); 10 g total fat (15 percent calories from fat); 12 g protein; 51 g carbohydrate; 54 mg cholesterol; 901 mg sodium; 4g fiber\\r\\n(based on a medium sized onion; also, the database does not have a category for dark soy sauce. just to give an idea of how much it helps to cut back on oil in cooking, using 3 tablespoons instead of 2 increases the calorie count to 364 calories per serving).\",\"ingredients\":\"['noodles', 'celery', 'bamboo shoots', 'peanut oil', 'garlic cloves', 'onion', 'light soy sauce', 'dark soy sauce', 'fresh ginger', 'vegetable stock', 'sugar', 'bean sprouts']\",\"n_ingredients\":12,\"dense_feature\":\"vegetarian chow mein;['60-minutes-or-less', 'time-to-make', 'cuisine', 'preparation', 'asian', 'chinese', 'vegan', 'vegetarian', 'dietary'];[316.6, 14.0, 20.0, 32.0, 22.0, 9.0, 15.0];['noodles', 'celery', 'bamboo shoots', 'peanut oil', 'garlic cloves', 'onion', 'light soy sauce', 'dark soy sauce', 'fresh ginger', 'vegetable stock', 'sugar', 'bean sprouts'];['if you are using fresh noodles , blanch them first in a large pot of boiling water for 3 to 5 minutes', 'if you are using the dried noodles , cook in boiling water for 4 to 5 minutes', 'drain the noodles , then put them into cold water until required', 'string the celery and slice diagonally', 'shred the bamboo shoots', 'heat a wok or large frying-pan and add the oil', 'when moderately hot , add the garlic and stir-fry for 10 seconds', 'add the onion , celery , and bamboo shoots and stir-fry for about 5 minutes', 'drain the noodles thoroughly and put into the wok', 'continue to stir-fry for 1 minute then add the rest of the ingredients except the bean sprouts', 'continue to stir-fry for another 2 minutes and then add the bean sprouts', 'give the mixture a good stir and turn it onto a serving platter']\"}\n"}]}  
{"messages": [{"role": "system", "content": "Act as a head chef and create a flavourful recipe from a list of ingredients"}, {"role": "user", "content": "whole wheat flour, rye flour, rye meal, compressed yeast, salt, molasses, oil, warm water, caraway seeds"}, {"role": "assistant", "content": "{\"name\":\"delicious german rye bread\",\"minutes\":230,\"tags\":\"['time-to-make', 'course', 'main-ingredient', 'cuisine', 'preparation', 'occasion', 'breads', 'german', 'scandinavian', 'easy', 'european', 'picnic', 'vegan', 'vegetarian', 'finnish', 'dietary', 'inexpensive', 'egg-free', 'yeast', 'free-of-something', 'pasta-rice-and-grains', 'taste-mood', 'savory', 'to-go', '4-hours-or-less']\",\"nutrition\":\"[1937.9, 55.0, 98.0, 195.0, 117.0, 27.0, 123.0]\",\"n_steps\":18,\"steps\":\"['place wholewheat flour , rye flour , rolled rye , salt and caraway seeds in a bowl', 'make a well in center', 'add the remaining ingredients to the well', 'mix all ingredients initially with a wooden spoon , until roughly mixed', 'if mixture seems too dry , add a little more water \\/ beer', 'kneed the mixture for 10 minutes , until smooth and elastic , adding a little more flour from time to time if the mixture is too sticky', 'form dough into a ball , and cover the mixing bowl with plastic wrap or a damp teatowel', 'leave to prove for 2 hours', 'after 2 hours , preheat oven to 200c', 'take the cover off the mixing bowl', 'punch the dough down in the center', 'lightly oil a baking tray or loaf tin and dust with flour', 'shape dough into desired form', 'leave alone until nearly doubled in size', 'place in hot oven for 45 minutes to 1 hour , depending on how crusty you like it', 'the loaf should be cooked when it gives a hollow sound when tapped', 'allow to cool before serving - it is usually better the day after baking', 'enjoy !']\",\"description\":\"this recipe for rye bread makes a medium-sized, heavy loaf. it is great served with butter, mustard, and cold meats. it's also delicious when toasted in the morning, and spread with lashings of butter and honey. mmmm!\",\"ingredients\":\"['whole wheat flour', 'rye flour', 'rye meal', 'compressed yeast', 'salt', 'molasses', 'oil', 'warm water', 'caraway seeds']\",\"n_ingredients\":9,\"dense_feature\":\"delicious german rye bread;['time-to-make', 'course', 'main-ingredient', 'cuisine', 'preparation', 'occasion', 'breads', 'german', 'scandinavian', 'easy', 'european', 'picnic', 'vegan', 'vegetarian', 'finnish', 'dietary', 'inexpensive', 'egg-free', 'yeast', 'free-of-something', 'pasta-rice-and-grains', 'taste-mood', 'savory', 'to-go', '4-hours-or-less'];[1937.9, 55.0, 98.0, 195.0, 117.0, 27.0, 123.0];['whole wheat flour', 'rye flour', 'rye meal', 'compressed yeast', 'salt', 'molasses', 'oil', 'warm water', 'caraway seeds'];['place wholewheat flour , rye flour , rolled rye , salt and caraway seeds in a bowl', 'make a well in center', 'add the remaining ingredients to the well', 'mix all ingredients initially with a wooden spoon , until roughly mixed', 'if mixture seems too dry , add a little more water \\/ beer', 'kneed the mixture for 10 minutes , until smooth and elastic , adding a little more flour from time to time if the mixture is too sticky', 'form dough into a ball , and cover the mixing bowl with plastic wrap or a damp teatowel', 'leave to prove for 2 hours', 'after 2 hours , preheat oven to 200c', 'take the cover off the mixing bowl', 'punch the dough down in the center', 'lightly oil a baking tray or loaf tin and dust with flour', 'shape dough into desired form', 'leave alone until nearly doubled in size', 'place in hot oven for 45 minutes to 1 hour , depending on how crusty you like it', 'the loaf should be cooked when it gives a hollow sound when tapped', 'allow to cool before serving - it is usually better the day after baking', 'enjoy !']\"}\n"}]}  
{"messages": [{"role": "system", "content": "Act as a head chef and create a flavourful recipe from a list of ingredients"}, {"role": "user", "content": "semi-sweet chocolate chips, blueberry preserves, soymilk, sugar, canola oil, vanilla extract, almond extract, all-purpose flour, unsweetened cocoa powder, baking powder, baking soda, salt, fresh blueberries"}, {"role": "assistant", "content": "{\"name\":\"fudgy wudgy blueberry brownies\",\"minutes\":55,\"tags\":\"['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'cuisine', 'preparation', 'occasion', 'north-american', 'desserts', 'fruit', 'vegan', 'vegetarian', 'cookies-and-brownies', 'chocolate', 'bar-cookies', 'brownies', 'dietary', 'low-sodium', 'comfort-food', 'low-in-something', 'berries', 'blueberries', 'taste-mood', 'sweet']\",\"nutrition\":\"[259.7, 16.0, 103.0, 3.0, 4.0, 13.0, 13.0]\",\"n_steps\":13,\"steps\":\"['preheat oven to 325', 'grease a 9x13 inch baking pan', 'melt 2 \\/ 3 cup chocolate chips over a double boiler or in microwave , your choice', 'in a large mixing bowl combine bluberry spreadable fruit , soy milk , sugar , canola oil and extracts', 'mix on high until no large clumps are visible , about 2-3 minutes', 'sift in flour , cocoa powder , baking powder , baking soda , and salt', 'stir until well mixed , batter will be thick', 'mix in melted chocolate', 'fold in remaining 1 \\/ 2 cup of whole chocolate chips and fresh blueberries', 'spread batter in baking pan and bake for 45 minutes', 'toothpick test is null and void here because the whole chocolate chips will stick anyway !', 'remove from oven and let cool', 'enjoy !']\",\"description\":\"from veganomicon!  these are intensely fudgey and amazing!  this was one of my first attempts at vegan baking, and i am quite proud of how these turned out.\",\"ingredients\":\"['semi-sweet chocolate chips', 'blueberry preserves', 'soymilk', 'sugar', 'canola oil', 'vanilla extract', 'almond extract', 'all-purpose flour', 'unsweetened cocoa powder', 'baking powder', 'baking soda', 'salt', 'fresh blueberries']\",\"n_ingredients\":13,\"dense_feature\":\"fudgy wudgy blueberry brownies;['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'cuisine', 'preparation', 'occasion', 'north-american', 'desserts', 'fruit', 'vegan', 'vegetarian', 'cookies-and-brownies', 'chocolate', 'bar-cookies', 'brownies', 'dietary', 'low-sodium', 'comfort-food', 'low-in-something', 'berries', 'blueberries', 'taste-mood', 'sweet'];[259.7, 16.0, 103.0, 3.0, 4.0, 13.0, 13.0];['semi-sweet chocolate chips', 'blueberry preserves', 'soymilk', 'sugar', 'canola oil', 'vanilla extract', 'almond extract', 'all-purpose flour', 'unsweetened cocoa powder', 'baking powder', 'baking soda', 'salt', 'fresh blueberries'];['preheat oven to 325', 'grease a 9x13 inch baking pan', 'melt 2 \\/ 3 cup chocolate chips over a double boiler or in microwave , your choice', 'in a large mixing bowl combine bluberry spreadable fruit , soy milk , sugar , canola oil and extracts', 'mix on high until no large clumps are visible , about 2-3 minutes', 'sift in flour , cocoa powder , baking powder , baking soda , and salt', 'stir until well mixed , batter will be thick', 'mix in melted chocolate', 'fold in remaining 1 \\/ 2 cup of whole chocolate chips and fresh blueberries', 'spread batter in baking pan and bake for 45 minutes', 'toothpick test is null and void here because the whole chocolate chips will stick anyway !', 'remove from oven and let cool', 'enjoy !']\"}\n"}]}
```

The data in the CSV will be formatted to the specified structure using the Azure OpenAI Assistant. We will provide a set of instructions outlining the requirements. The Assistant’s Code Interpreter will use Python to transform the data accordingly.

The Jupyter Notebook we are going to work with is called **P5-azure-openai-assistant-fine-tuning-data-preprocessing.ipynb**

<br/><br/>

#### 2.1 Open the Notebook

-   On the left of the codespace environment, select the **Explorer** icon.
-   Open the Notebook **P5-azure-openai-assistant-fine-tuning-data-preprocessing.ipynb**

![](https://cdn-images-1.medium.com/max/800/1*iWZz2zDi4DOxlhPNdh2kAg.png)

<br/><br/>

#### 2.2 Initializing the Azure OpenAI Client

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

#### 2.3 Import File from Azure OpenAI Service

We will be using the transformed data generated by Azure OpenAI in Part 4 Chapter 4. Let’s import that file into our Notebook.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Import existing uploaded file on Azure OpenAI Service  
for i in client.files.list():  
    if "recipes_preprocessed" in i.filename:  
        file__id = i.id  
        print(i.id)
```

**_NOTE: For convenience, we have pre-uploaded this file into Azure OpenAI Service for you to import into your Codespace environment._**

<br/><br/>

#### 2.4 Data Wrangling Instructions

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

{"messages": [{"role": "system", "content": "Act as a head chef and create a flavourful recipe from a list of ingredients"}, {"role": "user", "content": "'pork spareribs', 'soy sauce', 'fresh garlic', 'fresh ginger', 'chili powder', 'fresh coarse ground black pepper', 'salt', 'fresh cilantro leaves', 'tomato sauce', 'brown sugar', 'yellow onion', 'white vinegar', 'honey', 'a.1. original sauce', 'liquid smoke', 'cracked black pepper', 'cumin', 'dry mustard', 'cinnamon sticks', 'orange, juice of', 'mirin', 'water'"}, {"role": "assistant", "content": "{"name":"backyard style  barbecued ribs","minutes":120,"tags":"['weeknight', 'time-to-make', 'course', 'main-ingredient', 'cuisine', 'preparation', 'occasion', 'north-american', 'south-west-pacific', 'main-dish', 'pork', 'oven', 'holiday-event', 'stove-top', 'hawaiian', 'spicy', 'copycat', 'independence-day', 'meat', 'pork-ribs', 'super-bowl', 'novelty', 'taste-mood', 'savory', 'sweet', 'equipment', '4-hours-or-less']","nutrition":"[1109.5, 83.0, 378.0, 275.0, 96.0, 86.0, 36.0]","n_steps":10,"steps":"['in a medium saucepan combine all the ingredients for sauce#1 , bring to a full rolling boil , reduce heat to medium low and simmer for 1 hour , stirring often', 'rub the ribs with soy sauce , garlic , ginger , chili powder , pepper , salt and chopped cilantro , both sides !', 'wrap ribs in heavy duty foil', 'let stand 1 hour', 'preheat oven to 350 degrees', 'place ribs in oven for 1 hour , turning once after 30 minutes', '3 times during cooking the ribs open foil wrap and drizzle ribs with sauce#1', 'place all the ingredients for sauce#2 in a glass or plastic bowl , whisk well and set aside', 'remove ribs from oven and place on serving platter', 'offer both sauces at table to drizzle over ribs']","description":"this recipe is posted by request and was originaly from chef sam choy's cookbook ","ingredients":"['pork spareribs', 'soy sauce', 'fresh garlic', 'fresh ginger', 'chili powder', 'fresh coarse ground black pepper', 'salt', 'fresh cilantro leaves', 'tomato sauce', 'brown sugar', 'yellow onion', 'white vinegar', 'honey', 'a.1. original sauce', 'liquid smoke', 'cracked black pepper', 'cumin', 'dry mustard', 'cinnamon sticks', 'orange, juice of', 'mirin', 'water']","n_ingredients":22}"}]}
{"messages": [{"role": "system", "content": "Act as a head chef and create a flavourful recipe from a list of ingredients"}, {"role": "user", "content": "'lean pork chops', 'flour', 'salt', 'dry mustard', 'garlic powder', 'oil', 'chicken rice soup'"}, {"role": "assistant", "content": "{"name":"chicken lickin  good  pork chops","minutes":500,"tags":"['weeknight', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'main-dish', 'pork', 'crock-pot-slow-cooker', 'dietary', 'meat', 'pork-chops', 'equipment']","nutrition":"[105.7, 8.0, 0.0, 26.0, 5.0, 4.0, 3.0]","n_steps":5,"steps":"['dredge pork chops in mixture of flour , salt , dry mustard and garlic powder', 'brown in oil in a large skillet', 'place browned pork chops in a crock pot', 'add the can of soup , undiluted', 'cover and cook on low for 6-8 hours']","description":"here's and old standby i enjoy from time to time. it's from an old newspaper clipping i cut out years ago. very tasty.","ingredients":"['lean pork chops', 'flour', 'salt', 'dry mustard', 'garlic powder', 'oil', 'chicken rice soup']","n_ingredients":7}"}]}

---

### ACTIONS:

1. Read the tab separated comma file data.
2. Transform the data and create a jsonl file formatted in the conversational format as shown in the EXAMPLES section.
3. The conversational format has a system, user and assistant text input stored inside an array of dictionaries.
4. The system text input is always "Act as a head chef and create a flavourful recipe from a list of ingredients".
5. The user text input takes the list of ingredients in the column "ingredients" of the CSV file.
6. The assistant input takes all the columns of the CSV file.
7. Split the data set into training and testing data sets with a 25% split.
8. Make sure both data sets have the same format provided by the EXAMPLES section.
9. Name the data set with 75% of the data "recipes-training-set".
10. Name the data set with 25% of the data "recipes-validation-set".
11. Prepare both data sets as a jsonl file for download by the user.
12. Provide a summary paragraph explaining the preparation of the dataset.

---

### DO NOT:
1. Do not return any images. 
2. Do not return any other file types.
'''
```

<br/><br/>

#### 2.5 Create an Azure OpenAI Assistant

Just like in Part 4, let’s start the Azure OpenAI Assistant using **Code Interpreter** enabled to perform our data transformations.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Create an Azure OpenAI Assistant  
assistant = client.beta.assistants.create(  
    name = "data analyst assistant",  
    instructions = instructions,  
    tools = [{"type": "code_interpreter"}],  
    model = "gpt-4-1106-preview", 
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
prompt = "Please execute the INSTRUCTIONS and ACTIONS on the data stored in the CSV file using the EXAMPLES as reference for the output format " + fileId

message = client.beta.threads.messages.create(
    thread_id = thread.id,
    role = "user",
    content = prompt
)
```

<br/><br/>

#### 2.6 Run the Azure OpenAI Assistant

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

#### 2.7 Import Generated CSV

Let’s view the newly created file and check the transformations.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Functions to read xlsx files from Azure Openai

output_path = r"/workspaces/azure-openai-lab/data/generated_output/"

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
    message_ids = first_thread_message.file_ids
    print(message_ids)
    # Loop through each file ID and save the file with a sequential name
    for i, file_id in enumerate(message_ids):
        if i == 1:
            file_name = f"recipes-training-set.jsonl"  # Generate a sequential file name
            read_and_save_file(file_id, file_name, output_path)
        else:
            file_name = f"recipes-validation-set.jsonl"  # Generate a sequential file name
            read_and_save_file(file_id, file_name, output_path)

# Extract the file names from the response, retrieve the content and save the data as a jsonl file
files_from_messages()
```

<br/><br/>

#### 2.8 Clean up Azure OpenAI Assistant Environment

You want to make sure there aren’t any unnecessary artifacts laying around in Azure OpenAI Service. Let’s delete the assistant, thread, and generated CSV file from the Azure OpenAI environment.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
#Clean up Azure OpenAI environment
client.beta.assistants.delete(assistant.id)
client.beta.threads.delete(thread.id)
for i in range(0, 2):
    client.files.delete(messages.data[0].file_ids[i])
```

**_NOTE: In Chapter 4 we will use a standardized version of the generated training and validation files named recipes-training-set.jsonl and recipes-validation-set.jsonl_** **_to ascertain everyone has the same copy of the data._**

<p align="right">(<a href="#readme-top">back to top</a>)</p>

----------

### 3. Getting started with Fine-Tuning

#### 3.1 Open the Notebook

-   On the left of the codespace environment, select the **Explorer** icon.
-   Open the Notebook **P5-azure-openai-fine-tuning.ipynb.**

![](https://cdn-images-1.medium.com/max/800/1*NPEPXixYCKwHwwpwTtGoEQ.png)

<br/><br/>

#### 3.2 Initializing the Azure OpenAI Client

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
    azure_endpoint = azure_oai_endpoint,  
    api_version = "2024-02-15-preview"  
    )
```

**_NOTE: The latest API version for the AzureOpenAI client when fine-tuning can be found_** [**_here_**](https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation#latest-ga-api-release)**_._**

<br/><br/>

#### 3.3 Zero-Shot learning

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
{"name":"","minutes":,"tags":"[]","nutrition":"[]","n_steps":"","steps":"[]","description":"","ingredients":"[]", "n_ingredients":}

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
ingredients = """'veal roast', 'butter', 'oil', 'carrots', 'onions', 'parsley sprigs', 'bay leaf', 'thyme', 'salt', 'pepper', 'bacon'"""

# Send request to Azure OpenAI model
completion = client.chat.completions.create(
  model = "gpt-35-turbo-0613-ft",
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
The generated recipe is indeed vegan and it looks plausible! The format fits our specified JSON object too! We are all set at this point, but out of curiosity, let's add RAG to the mix.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

----------

### 4. Fine-Tuning and RAG

Large Language models are generally trained to be helpful generalists by default. We learned that fine-tuning can narrow their focus and instill specific behavior patterns. With RAG we can enhance the model by providing relevant context with new information before generating a response. Therefore, retrieval strategies such as RAG complement fine-tuning rather than serving as an alternative.

Let’s use our fine-tuned model together with RAG. We will re-use the embeddings saved in the vector database we created in Part 4.

<br/><br/>

#### 4.1 Open the Notebook

-   On the left of the codespace environment, select the **Explorer** icon.
-   Open the Notebook **P5-azure-openai-rag-with-fine-tuning.ipynb.**

![](https://cdn-images-1.medium.com/max/800/1*-hULLRqxuzGJsG6AyBTHkA.png)

<br/><br/>

#### 4.2 Initializing the Azure OpenAI Client

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

#### 4.3 Reload Vector Database

Since we have already created a vector database with the embeddings of our data in Part 4, we can reload and re-use this database for our RAG implementation.

To do this, we need to re-initialize the `AzureOpenAIEmbeddings` constructor to define the model retriever and pass that to the chain. This will allow us to use our previously persisted Database with new prompts.

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

We are now ready to reload the vector database.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Re-load an existing vector database from a local path  
vectordb = Chroma(persist_directory=r"/workspaces/azure-openai-lab/data/chromadb", embedding_function=openai_ef)
```

<br/><br/>

#### 4.4 Zero-Shot learning

Let’s run a zero-shot learning prompt using our fine-tuned model together with our RAG implementation.

First, we need to initialize the Azure OpenAI client. We will use the LangChain `AzureChatOpenAI` constructor and call the Fine-Tuned GPT-3.5-Turbo model called `gpt-35-turbo-0613-ft`.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Initalize Azure Openai using LangChain (Default gpt-35-turbo and fine-tuned gpt-35-turbo-0613-ft)  
client = AzureChatOpenAI(  
                deployment_name = "gpt-35-turbo-0613-ft",  
                openai_api_key = azure_oai_key,  
                azure_endpoint = azure_oai_endpoint,  
                openai_api_version = "2024-02-01"  
        ) 
```

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

and generate a vegan recipe.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Run chain to call Azure OpenAI using ChromaDB vector database data to enrich the prompt (RAG).  
ingredients = """'Tofu', 'Avocado', 'Soy Sauce', 'Chili', 'Coconut Milk', 'Broccoli'"""  
  
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

Congratulations! You have completed Part 5 of this workshop. We have learned how to use the Azure OpenAI Assistant API for data pre-processing, and have used fine-tuned models to generate better recipe outputs. We will continue with [Part 6][A-Hands-on-Exploration-of-the-Azure-OpenAI-APIs-part-6] of the workshop and learn how we can use the new GPT-4o model to generate a list of ingredients from images.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

----------

### 5. Questions, Feedback, Support?

Reach out to us! We are happy to answer any questions you might have or use your feedback to optimize this series!

<p align="right">(<a href="#readme-top">back to top</a>)</p>

----------

### 6. References

[1] [Full Fine-Tuning, PEFT, Prompt Engineering, or RAG? (deci.ai)](https://deci.ai/blog/fine-tuning-peft-prompt-engineering-and-rag-which-one-is-right-for-you/)

<!-- MARKDOWN LINKS & IMAGES -->
[A-Hands-on-Exploration-of-the-Azure-OpenAI-APIs-part-6]: https://github.com/AllgeierSchweiz/azure-openai-lab/blob/main/series/A%20Hands-on%20Exploration%20of%20the%20Azure%20OpenAI%20API%20(Part%206%20of%C2%A06).md
[Fine-Tuned]: https://github.com/AllgeierSchweiz/azure-openai-lab/blob/main/images/Fine_Tuning.jpg
