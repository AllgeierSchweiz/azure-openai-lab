
----------

# A Hands-on Exploration of the Azure OpenAI API (Part 3 of 5)

![](https://cdn-images-1.medium.com/max/800/1*PV9Eh3WpAhnW9NZCijZPzw.png)

## 1. Getting started with the Azure OpenAI API

Lets explore how to create complex text inputs to enhance the model’s performance, also known as prompt engineering. This can be reinforced with techniques like **one-shot** and **few-shot learning**, where we provide relevant data or information to give the model additional context for the conversation. Alternatively, we create prompts without examples called **zero-shot learning.** In all cases, we can enhance the prompt input by using prompt engineering frameworks to further guide the models output.

The Jupyter Notebook we are going to work with is:

1.  _P3-azure-openai-prompt-engineering.ipynb_

In this Notebook, we will call the Azure OpenAI API using the Chat Completion API using the GPT-4o-mini model. This model is optimized for conversational interfaces and expects an input formatted in a [specific chat-like transcript](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/chatgpt) stored inside an array of dictionaries. The structure looks as follows:

```sql
{"role": "system", "content": "Provide context and / or instructions to the model."},  
{"role": "user", "content": "Example question goes here."},  
{"role": "assistant", "content": "Example answer goes here."},  
{"role": "user", "content": "First question/message for the model to respond to."}
```
What exactly do these roles mean?

-   **System**: Allows you to specify the way the model answers questions and / or should behave. Prompt Engineering is done on this level.
-   **User**: Equivalent to the queries made by the user.
-   **Assistant**: The model’s responses, based on the user input. Examples for one-shot and few-shot learning are added here.

**_NOTE: You may have more than one set of user/assistant roles in the prompt. This is the case when implementing few-shot learning._**

<br/><br/>

### 1.1 Open the Notebook

-   On the left of the codespace environment, select the **Explorer** icon.
-   Open the Notebook **P3-azure-openai-prompt-engineering.ipynb**.

![](https://cdn-images-1.medium.com/max/800/1*xz0by11chN-zudVuEtj_BA.png)

<br/><br/>

### 1.2 Running the Code

There are multiple options to run the code in a Notebook. You can:

1.  Run the entire code block by selecting the **Execute Cell** button or **CTRL+ALT+ENTER**.
2.  You can also run the code within a code block line-by-line by selecting the **Run by Line** button or **F10.**
3.  Or  you can run the entire Notebook code all at once starting from the top by selecting the **Execute Cell and Below** button.

![](https://cdn-images-1.medium.com/max/800/1*KCPLLwydrB1sJJR9dXMlHQ.png)

**_NOTE: We will run the code block-by-block using the Execute Cell button for the entire workshop series._**

<br/><br/>

### 1.3 Initializing the Azure OpenAI Client

We will start by importing the necessary Python packages to run our code.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Import Python packages  
import os  
import io  
import time  
from io import StringIO  
import json  
from dotenv import load_dotenv, find_dotenv  
from pathlib import Path  
import pandas as pd  
from openai import AzureOpenAI  
import sys
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
        azure_endpoint = azure_oai_endpoint,   
        api_key=azure_oai_key,    
        api_version="2024-02-01"  
        )
```

**_NOTE: The latest API version for the AzureOpenAI client can be found_** [**_here_**](https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation#latest-ga-api-release)**_._**

<br/><br/>

### 1.4 Zero-Shot learning

The client has been initialized so that we can create our first prompt following the Chat Completion transcript format mentioned in Chapter 1.

We will not include any examples to aid the model, we are therefore following a zero-shot learning approach. This means our prompt only requires a **system** and a **user** role. In our case, the prompt looks like this:

```sql
{"role": "system", "content": "You are a helpful cook"},  
{"role": "user", "content": "Create a flavourful recipe using the following ingredients:" + "\n" + "---" + "\n" + ingredients + "\n" + "---"}
```
**_NOTE: Using clear syntax for your prompts, such as headings, section markers and separators, helps communicate intent. We use separators, in this case_** `---` **_._**

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Create a prompt of ingredients the model should create a recipe from  
ingredients = """'veal roast', 'butter', 'oil', 'carrots', 'onions', 'parsley sprigs', 'bay leaf', 'thyme', 'salt', 'pepper', 'bacon'"""  
  
# Send request to Azure OpenAI model  
response = client.chat.completions.create(  
   model="gpt-35-turbo",  
   temperature=0.7,  
   #max_tokens=120,  
   messages=[  
       {"role": "system", "content": "You are a helpful cook"},  
       {"role": "user", "content": "Create a flavourful recipe using the following ingredients:" + "\n" + "---" + "\n" + ingredients + "\n" + "---"}  
   ]  
)  
  
print("Summary: " + response.choices[0].message.content + "\n")
```

**_NOTE: The model name being called in the chat completion client is the name of the model you deployed in your Azure OpenAI Service cockpit. For the SDSC workshop, all model deployments have already been provisioned and nothing needs to be changed._**

The generated recipe seems plausible, however, we want to guide the model into creating more creative recipes for a specific use case. Additionally, we would prefer the output in a semi-structured format like JSON. Let’s adjust the system prompt to take our wishes into account.

<br/><br/>

### 1.5 Prompt Engineering

The fundamental concept underlying prompt engineering is that generic inputs will inevitably result in the generation of generic outputs. In order to generate outputs that are of any use, it is necessary to provide the model with direction and guidance.

To achieve this, we use prompt engineering frameworks. These frameworks can be considered as user input blueprints. They provide users with a systematic approach, comprising a number of actions, ideas, instructions and examples, which can be passed on to the model.

There are a number of frameworks that can assist users in creating the optimal input prompt. In this instance, our focus will be on the following key elements:

-   The **instructions** for the **persona** we want the model to emulate.
-   The **instructions** for the **actions** we want the model to execute.
-   The **instructions** for the **target audience** we want the output to be geared towards.
-   The output format we want our model to generate for us.

With these 4 elements in mind, here is what our prompt might look like:

```sql
"""
### INSTRUCTIONS
Persona: Act as a head chef such as Joël Robuchon who specializes in simple contemporary cuisine.
Action: Create well-thought-out and flavourful recipes from a list of ingredients implementing classic culinary techniques.
Target Audience: The recipients of these recipes are couples who want to cook a special meal at least once a week.

---

### OUTPUT FORMAT
Output only one recipe and return it as a JSON object with the following format:
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
```

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Zero-Shot learning. Model has a token limit of 4096.

# Create advanced System prompt
systemcontent = \
"""
### INSTRUCTIONS
Persona: Act as a head chef such as Joël Robuchon who specializes in simple contemporary cuisine.
Action: Create well-thought-out and flavourful recipes from a list of ingredients implementing classic culinary techniques.
Target Audience: The recipients of these recipes are couples who want to cook a special meal at least once a week.

---

### OUTPUT FORMAT
Output only one recipe and return it as a JSON object with the following format:
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
response = client.chat.completions.create(
   model="gpt-35-turbo",
   temperature=0.7,
   #max_tokens=120,
   messages=[
       {"role": "system", "content": systemcontent},
       {"role": "user", "content": "Create a flavourful recipe using the following ingredients:" + "\n" + "---" + "\n" + ingredients + "\n" + "---"}
   ]
)

result = response.choices[0].message.content
print(result + "\n")
```

The generated output looks much better, but we can see that certain output attributes are sometimes left empty. Let’s provide the model with an example to guide it towards our desired output.

<br/><br/>

### 1.6 One-Shot learning

To give the Chat Completion Client an example, we need to supply an assistant role to the transcript format. This means our prompt has a **system, assistant,** and **user** role.

In our case, the prompt looks like this:

```sql
{"role": "system", "content": systemcontent},  
{"role": "user", "content": "Create a flavourful recipe using the following ingredients:" + "\n" + "---" + "\n" + ingredients + "\n" + "---"},  
{"role": "assistant", "content": assistant_content_1},  
{"role": "user", "content": ingredients}
```

**_NOTE: The variable assistant_content_1 has the example we want to feed the model._**

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# One-Shot learning. Model has a token limit of 4096.

# Create a prompt of ingredients the model should create a recipe from
ingredients = """'veal roast', 'butter', 'oil', 'carrots', 'onions', 'parsley sprigs', 'bay leaf', 'thyme', 'salt', 'pepper', 'bacon'"""

# Create advanced System prompt
systemcontent = \
"""
### INSTRUCTIONS
Persona: Act as a head chef such as Joël Robuchon who specializes in simple contemporary cuisine.
Action: Create well-thought-out and flavourful recipes from a list of ingredients implementing classic culinary techniques.
Target Audience: The recipients of these recipes are couples who want to cook a special meal at least once a week.

---

### OUTPUT FORMAT
Output only one recipe and return it as a JSON object with the following format:
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

# One-Shot learning example
ingredients_1 = """'pork spareribs', 'soy sauce', 'fresh garlic', 'fresh ginger', 'chili powder', 'fresh coarse ground black pepper', 'salt', 'fresh cilantro leaves', 'tomato sauce', 'brown sugar', 'yellow onion', 'white vinegar', 'honey', 'a.1. original sauce', 'liquid smoke', 'cracked black pepper', 'cumin', 'dry mustard', 'cinnamon sticks', 'orange, juice of', 'mirin', 'water'"""
assistant_content_1 = """{"name":"backyard style  barbecued ribs","minutes":120,"tags":"['weeknight', 'time-to-make', 'course', 'main-ingredient', 'cuisine', 'preparation', 'occasion', 'north-american', 'south-west-pacific', 'main-dish', 'pork', 'oven', 'holiday-event', 'stove-top', 'hawaiian', 'spicy', 'copycat', 'independence-day', 'meat', 'pork-ribs', 'super-bowl', 'novelty', 'taste-mood', 'savory', 'sweet', 'equipment', '4-hours-or-less']","nutrition":"[1109.5, 83.0, 378.0, 275.0, 96.0, 86.0, 36.0]","n_steps":10,"steps":"['in a medium saucepan combine all the ingredients for sauce#1 , bring to a full rolling boil , reduce heat to medium low and simmer for 1 hour , stirring often', 'rub the ribs with soy sauce , garlic , ginger , chili powder , pepper , salt and chopped cilantro , both sides !', 'wrap ribs in heavy duty foil', 'let stand 1 hour', 'preheat oven to 350 degrees', 'place ribs in oven for 1 hour , turning once after 30 minutes', '3 times during cooking the ribs open foil wrap and drizzle ribs with sauce#1', 'place all the ingredients for sauce#2 in a glass or plastic bowl , whisk well and set aside', 'remove ribs from oven and place on serving platter', 'offer both sauces at table to drizzle over ribs']","description":"this recipe is posted by request and was originaly from chef sam choy's cookbook ","ingredients":"['pork spareribs', 'soy sauce', 'fresh garlic', 'fresh ginger', 'chili powder', 'fresh coarse ground black pepper', 'salt', 'fresh cilantro leaves', 'tomato sauce', 'brown sugar', 'yellow onion', 'white vinegar', 'honey', 'a.1. original sauce', 'liquid smoke', 'cracked black pepper', 'cumin', 'dry mustard', 'cinnamon sticks', 'orange, juice of', 'mirin', 'water']","n_ingredients":22}"""

# Send request to Azure OpenAI model
response = client.chat.completions.create(
   model="gpt-35-turbo",
   temperature=0.7,
   #max_tokens=120,
   messages=[
       {"role": "system", "content": systemcontent},
       {"role": "user", "content": "Create a flavourful recipe using the following ingredients:" + "\n" + "---" + "\n" + ingredients + "\n" + "---"},
       {"role": "assistant", "content": assistant_content_1},
       {"role": "user", "content": ingredients}
   ]
)

result = response.choices[0].message.content
print(result + "\n")
```

The generated output looks great! The recipe is plausible, the output is in JSON format and none of the output attributes are empty. However, to ensure the model fully understands the direction we want the output to take, we can provide additional examples. Let’s do that!

<br/><br/>

### 1.7 Few-Shot learning

The prompt structure remains the same, but we can now add two or more examples using the **user** and **assistant** roles.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Few-Shot learning. Model has a token limit of 4096.

# Create a prompt of ingredients the model should create a recipe from
ingredients = """'veal roast', 'butter', 'oil', 'carrots', 'onions', 'parsley sprigs', 'bay leaf', 'thyme', 'salt', 'pepper', 'bacon'"""

# Create advanced System prompt
systemcontent = \
"""
### INSTRUCTIONS
Persona: Act as a head chef such as Joël Robuchon who specializes in simple contemporary cuisine.
Action: Create well-thought-out and flavourful recipes from a list of ingredients implementing classic culinary techniques.
Target Audience: The recipients of these recipes are couples who want to cook a special meal at least once a week.

---

### OUTPUT FORMAT
Output only one recipe and return it as a JSON object with the following format:
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

# Few-Shot learning examples

ingredients_1 = """'pork spareribs', 'soy sauce', 'fresh garlic', 'fresh ginger', 'chili powder', 'fresh coarse ground black pepper', 'salt', 'fresh cilantro leaves', 'tomato sauce', 'brown sugar', 'yellow onion', 'white vinegar', 'honey', 'a.1. original sauce', 'liquid smoke', 'cracked black pepper', 'cumin', 'dry mustard', 'cinnamon sticks', 'orange, juice of', 'mirin', 'water'"""
assistant_content_1 = """{"name":"backyard style  barbecued ribs","minutes":120,"tags":"['weeknight', 'time-to-make', 'course', 'main-ingredient', 'cuisine', 'preparation', 'occasion', 'north-american', 'south-west-pacific', 'main-dish', 'pork', 'oven', 'holiday-event', 'stove-top', 'hawaiian', 'spicy', 'copycat', 'independence-day', 'meat', 'pork-ribs', 'super-bowl', 'novelty', 'taste-mood', 'savory', 'sweet', 'equipment', '4-hours-or-less']","nutrition":"[1109.5, 83.0, 378.0, 275.0, 96.0, 86.0, 36.0]","n_steps":10,"steps":"['in a medium saucepan combine all the ingredients for sauce#1 , bring to a full rolling boil , reduce heat to medium low and simmer for 1 hour , stirring often', 'rub the ribs with soy sauce , garlic , ginger , chili powder , pepper , salt and chopped cilantro , both sides !', 'wrap ribs in heavy duty foil', 'let stand 1 hour', 'preheat oven to 350 degrees', 'place ribs in oven for 1 hour , turning once after 30 minutes', '3 times during cooking the ribs open foil wrap and drizzle ribs with sauce#1', 'place all the ingredients for sauce#2 in a glass or plastic bowl , whisk well and set aside', 'remove ribs from oven and place on serving platter', 'offer both sauces at table to drizzle over ribs']","description":"this recipe is posted by request and was originaly from chef sam choy's cookbook ","ingredients":"['pork spareribs', 'soy sauce', 'fresh garlic', 'fresh ginger', 'chili powder', 'fresh coarse ground black pepper', 'salt', 'fresh cilantro leaves', 'tomato sauce', 'brown sugar', 'yellow onion', 'white vinegar', 'honey', 'a.1. original sauce', 'liquid smoke', 'cracked black pepper', 'cumin', 'dry mustard', 'cinnamon sticks', 'orange, juice of', 'mirin', 'water']","n_ingredients":22}"""

ingredients_2 = """'lean pork chops', 'flour', 'salt', 'dry mustard', 'garlic powder', 'oil', 'chicken rice soup'"""
assistant_content_2 = """{"name":"chicken lickin  good  pork chops","minutes":500,"tags":"['weeknight', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'main-dish', 'pork', 'crock-pot-slow-cooker', 'dietary', 'meat', 'pork-chops', 'equipment']","nutrition":"[105.7, 8.0, 0.0, 26.0, 5.0, 4.0, 3.0]","n_steps":5,"steps":"['dredge pork chops in mixture of flour , salt , dry mustard and garlic powder', 'brown in oil in a large skillet', 'place browned pork chops in a crock pot', 'add the can of soup , undiluted', 'cover and cook on low for 6-8 hours']","description":"here's and old standby i enjoy from time to time. it's from an old newspaper clipping i cut out years ago. very tasty.","ingredients":"['lean pork chops', 'flour', 'salt', 'dry mustard', 'garlic powder', 'oil', 'chicken rice soup']","n_ingredients":7}"""

ingredients_3 = """'boneless skinless chicken breast halves', 'condensed cream of chicken soup', 'egg', 'seasoning salt', 'all-purpose flour', 'cornstarch', 'garlic powder', 'paprika', 'salt and pepper', 'oil'"""
assistant_content_3 = """{"name":"crispy crunchy  chicken","minutes":35,"tags":"['60-minutes-or-less', 'time-to-make', 'course', 'preparation', 'healthy', 'main-dish', 'dietary', 'low-saturated-fat', 'low-in-something']","nutrition":"[335.8, 11.0, 2.0, 24.0, 64.0, 10.0, 10.0]","n_steps":8,"steps":"['combine soup , egg and seasoned salt in a bowl and set aside', 'mix together flour , cornstarch , garlic powder , paprika , salt and pepper in a resealable plastic bag', 'dip chicken pieces into soup mixture and turn so as to coat all over', 'place chicken pieces in bag with flour mixture , seal bag and shake to coat chicken', 'place coated pieces of chicken on a platter and allow to set until the coating becomes doughy', 'heat oil in a deep fryer or in a skillet over medium heat , using enough oil to cover chicken pieces when fried', 'once chicken is doughy , fry pieces in oil for approx 5-8 minutes or until cooked through and juices run clear', 'drain pieces on paper towel and serve']","description":"delicious, crunchy fried chicken. this recipe came from the ","ingredients":"['boneless skinless chicken breast halves', 'condensed cream of chicken soup', 'egg', 'seasoning salt', 'all-purpose flour', 'cornstarch', 'garlic powder', 'paprika', 'salt and pepper', 'oil']","n_ingredients":10}"""

# Send request to Azure OpenAI model
response = client.chat.completions.create(
   model="gpt-35-turbo",
   temperature=0.7,
   #max_tokens=120,
   messages=[
       {"role": "system", "content": systemcontent},
       {"role": "user", "content": "Create a flavourful recipe using the following ingredients:" + "\n" + "---" + "\n" + ingredients_1 + "\n" + "---"},
       {"role": "assistant", "content": assistant_content_1},
       {"role": "user", "content": "Create a flavourful recipe using the following ingredients:" + "\n" + "---" + "\n" + ingredients_2 + "\n" + "---"},
       {"role": "assistant", "content": assistant_content_2},
       {"role": "user", "content": "Create a flavourful recipe using the following ingredients:" + "\n" + "---" + "\n" + ingredients_3 + "\n" + "---"},
       {"role": "assistant", "content": assistant_content_3},
       {"role": "user", "content": ingredients}
   ]
)

result = response.choices[0].message.content
print(result + "\n")
```
<br/><br/>

Congratulations! You have completed Part 3 of this workshop. We have learned how to initialize the Azure OpenAI client to use the Azure OpenAI API with Python, and have implemented zero-shot, one-shot, and few-shot learning techniques, including advanced prompting, to optimize our model output. We will continue with [Part 4][A-Hands-on-Exploration-of-the-Azure-OpenAI-APIs-part-4] of the workshop.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

----------

## 2. Questions, Feedback, Support?

Reach out to us! We are happy to answer any questions you might have or use your feedback to optimize this series!

<p align="right">(<a href="#readme-top">back to top</a>)</p>

----------

## 3. References

[1] [Work with the GPT-35-Turbo and GPT-4 models — Azure OpenAI Service | Microsoft Learn](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/chatgpt?tabs=python-new)

[2] [The Perfect Prompt: A Prompt Engineering Cheat Sheet | by Maximilian Vogel | The Generator | Apr, 2024 | Medium](https://medium.com/the-generator/the-perfect-prompt-prompt-engineering-cheat-sheet-d0b9c62a2bba)

[3] [Ultimate guide to 44 different AI prompt engineering strategies for best output! (beeazt.com)](https://beeazt.com/knowledge-base/prompt-frameworks/)

<!-- MARKDOWN LINKS & IMAGES -->
[A-Hands-on-Exploration-of-the-Azure-OpenAI-APIs-part-4]: https://github.com/AllgeierSchweiz/azure-openai-lab/blob/main/series/A%20Hands-on%20Exploration%20of%20the%20Azure%20OpenAI%20API%20(Part%204%20of%C2%A06).md
