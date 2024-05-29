----------

# A Hands-on Exploration of the Azure OpenAI API (Part 6 of 6)

![](https://cdn-images-1.medium.com/max/800/1*PV9Eh3WpAhnW9NZCijZPzw.png)

## 1. Omni Image-to-Text

Let’s continue our journey and learn how we can use the new GPT-4o model to translate images of food into text, i.e. a list of ingredients, which we can use to create new recipes.

1.  _P6-azure-openai-omni-image-to-text.ipynb_

In this Notebook, we will call the Azure OpenAI API with the Chat Completion API using the GPT-4o model. This is a multimodal model which can use both text and images to generate natural language outputs.

<br/><br/>

### 1.1 Open the Notebook

-   On the left of the codespace environment, select the **Explorer** icon.
-   Open the Notebook **P6-azure-openai-omni-image-to-text.ipynb**

![](https://cdn-images-1.medium.com/max/800/1*fTpm5q4wh08nfl6r9KzLzA.png)

<br/><br/>

### 1.2 Initializing the Azure OpenAI Client

We will start by importing the necessary Python packages to run our code.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Import Python packages  
import os  
import io  
import time  
import base64  
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
        api_version= "2024-04-01-preview" #"2024-05-13"  
        )
```

**_NOTE: The latest API version for the AzureOpenAI client can be found_** [**_here_**](https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation#latest-ga-api-release)**_. Currently the designated GPT-4o API version 2024–05–13 is not working. We will fall back to version 2024–04–01-preview, which is used in GPT-4 vision models._**

<br/><br/>

### 1.3 Image Processing

We have the images of six different food products, one image of the contents of a refrigerator, and one image of a recipe book page. We will be sending these images to the Azure OpenAI API as a base64 format. A bit of encoding will do the trick.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Prepare image for Azure OpenAI model
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Image of individual food products  
avocado = encode_image(f"/workspaces/azure-openai-lab/images/products/avocado.jpg") #f"C:\\Python\\azure-openai-lab\\images\\products\\avocado.jpg"
tofu = encode_image(f"/workspaces/azure-openai-lab/images/products/tofu.jpg") #f"C:\\Python\\azure-openai-lab\\images\\products\\tofu.jpg"
broccoli = encode_image(f"/workspaces/azure-openai-lab/images/products/broccoli.jpg") #f"C:\\Python\\azure-openai-lab\\images\\products\\broccoli.jpg"
chili = encode_image(f"/workspaces/azure-openai-lab/images/products/chili.jpg") #f"C:\\Python\\azure-openai-lab\\images\\products\\chili.jpg"
coconut_milk = encode_image(f"/workspaces/azure-openai-lab/images/products/coconut_milk.jpg") #f"C:\\Python\\azure-openai-lab\\images\\products\\coconut_milk.jpg"
soy_sauce = encode_image(f"/workspaces/azure-openai-lab/images/products/soy_sauce.jpg") #f"C:\\Python\\azure-openai-lab\\images\\products\\soy_sauce.jpg"

# Image of a refrigerator with foods
refrigerator = encode_image(f"/workspaces/azure-openai-lab/images/products/refrigerator.jpg") #f"C:\\Python\\azure-openai-lab\\images\\products\\refrigerator.jpg"

# Image of a recipe book page
recipe = encode_image(f"/workspaces/azure-openai-lab/images/recipes/biscotti.jpg") #f"C:\\Python\\azure-openai-lab\\images\\products\\refrigerator.jpg"
```

<br/><br/>

### 1.4 Individual Food Product Images

The client has been initialized and the images have imported and reformatted. We will not include any examples to aid the model, we are therefore following a zero-shot learning approach.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Generate a list of ingredients from individual food product images.  
  
# Send request to Azure OpenAI model  
response = client.chat.completions.create(  
    model="gpt-4o",  
    temperature=0.7,  
    #max_tokens=120,  
    messages=[  
            {  
            "role": "user",  
            "content": [  
                {"type": "text", "text": "You are a helpful cook. Analyze the provided images, determine the food product being depicted and create a list of all products"},  
                {  
                "type": "image_url",  
                "image_url": {  
                    "url": f"data:image/jpeg;base64,{avocado}"  
                    },  
                },  
                {  
                "type": "image_url",  
                "image_url": {  
                    "url": f"data:image/jpeg;base64,{tofu}"  
                    },  
                },  
                {  
                "type": "image_url",  
                "image_url": {  
                    "url": f"data:image/jpeg;base64,{broccoli}"  
                    },  
                },  
                {  
                "type": "image_url",  
                "image_url": {  
                    "url": f"data:image/jpeg;base64,{chili}"  
                    },  
                },  
                {  
                "type": "image_url",  
                "image_url": {  
                    "url": f"data:image/jpeg;base64,{coconut_milk}"  
                    },  
                },  
                {  
                "type": "image_url",  
                "image_url": {  
                    "url": f"data:image/jpeg;base64,{soy_sauce}"  
                    },  
                },                                                                  
            ],  
        }  
    ]  
)  
  
result = response.choices[0].message.content  
print(result + "\n")
```

**_NOTE: The model name being called in the chat completion client is the name of the model you deployed in your Azure OpenAI Service cockpit. For the SDSC workshop, all model deployments have already been provisioned and nothing needs to be changed._**

Awesome! The model managed to decipher the images and generated a correct list of food products based on our images. There is still room for improvement. We would prefer the output in a semi-structured format like JSON. We also want the model to add some additional attributes to our output such as: **amount**, **units** and **expiration** of the food products. Let’s do some prompt engineering to tweak the generated output.

<br/><br/>

### 1.5 Prompt Engineering

We are going to adjust our system prompt with additional instructions and output format details, to guide the model.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Generate a list of ingredients and additional attributes from individual food product images.

# Create advanced System prompt
systemcontent = \
"""
### INSTRUCTIONS
1. Analyze the provided images.
2. Determine the food product being depicted.
3. Count the numbers of invidual food product items in bowls or vessels.

---

### OUTPUT FORMAT
Return a JSON array with the following format:
{"name":"",amount:"", units:"", expiration_days:}

The variables should contain the following information:
- name: the name of the product in each image.
- amount: the number of products in each image.
- units: the unit of the product in each image using the metric system.
- expiration_days: the expiration date of the product in each image in average number of days.
"""

# Send request to Azure OpenAI model
response = client.chat.completions.create(
    model="gpt-4o",
    temperature=0.7,
    #max_tokens=120,
    messages=[
            {
            "role": "user",
            "content": [
                {"type": "text", "text": systemcontent},
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{avocado}"
                    },
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{tofu}"
                    },
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{broccoli}"
                    },
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{chili}"
                    },
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{coconut_milk}"
                    },
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{soy_sauce}"
                    },
                },                                                                
            ],
        }
    ]
)

result = response.choices[0].message.content
print(result + "\n")
```

The model did a great job! We got precise food product names and additional attributes exactly as instructed.

### 1.6 Single Image with Multiple Food Products

What if we have one image containing multiple food products in different amounts? For example, the contents of a refrigerator. Could the model still produce an accurate output? Lets find out!

We are going to change the image input and re-run the code using the same system prompt.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Generate a list of ingredients and additional attributes from a single image containing multiple food product.

# Create advanced System prompt
systemcontent = \
"""
### INSTRUCTIONS
1. Analyze the provided images.
2. Determine the food product being depicted.
3. Count the numbers of invidual food product items in bowls or vessels.

---

### OUTPUT FORMAT
Return a JSON array with the following format:
{"name":"",amount:"", units:"", expiration_days:}

The variables should contain the following information:
- name: the name of the product in each image.
- amount: the number of products in each image.
- units: the unit of the product in each image using the metric system.
- expiration_days: the expiration date of the product in each image in average number of days.
"""

# Send request to Azure OpenAI model
response = client.chat.completions.create(
    model="gpt-4o",
    temperature=0.7,
    #max_tokens=120,
    messages=[
            {
            "role": "user",
            "content": [
                {"type": "text", "text": systemcontent},
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{refrigerator}"
                    },
                },                                                               
            ],
        }
    ]
)

result = response.choices[0].message.content
print(result + "\n")
```

Brilliant! The model did a great job even with images containing multiple food products scattered around. There is still room for improvement, but impressive nonetheless.

<br/><br/>

### 1.7 Single Image of Recipe Book Page

Often we have an image of a scanned document that we'd like to convert to a structured format. What if we have a page from a recipe book. Can we convert the information on that page into a structured table?

We are going to adjust the system prompt, change the image input, and re-run the code.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
# Generate a list of ingredients and additional attributes from a recipe book page containing text.

# Create advanced System prompt
systemcontent = \
"""
### INSTRUCTIONS
1. Analyze the provided recipe image.
2. Extract the recipe name located at the top of the image with the largest orange font.
3. Extract the recipe description located at the top of the image in quotes.
4. Extract the recipe ingredients located under the "INGREDIENTS" header.
5. Extract the recipe steps located under the "INSTRUCTIONS" header.
6. Extract the recipe nutritions at the bottom of the image starting after "Nutrition Tip" in the grey font.

---

### OUTPUT FORMAT
Return a JSON array with the following format:
{"name":"","description":"", "ingredients":"[]", amount:"[]", units:"[]", "steps":"[]", "nutritions":"", expiration_days:}

The variables should contain the following information:
- name: the name of the recipe.
- description: the description of the recipe.
- ingredients: a list of the ingredients of the recipe.
- amount: a list of the number of each ingredient used in the recipe in numeric form. If no number is visible, define the value as "variable".
- units: a list of the units of each ingredient using the metric system. If no unit is visible, define the value as "null".
- steps: a list of the preparation instructions of the recipe.
- nutritions: a summary of the nutritional information of the recipe. If no nutritional information is found, define the value as "null".
- expiration_days: the expiration date of each ingredient in average number of days. If no expiration date is known, define the value as "null".

"""

# Send request to Azure OpenAI model
response = client.chat.completions.create(
    model="gpt-4o",
    temperature=0.7,
    #max_tokens=120,
    messages=[
            {
            "role": "user",
            "content": [
                {"type": "text", "text": systemcontent},
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{recipe}"
                    },
                },                                                               
            ],
        }
    ]
)

result = response.choices[0].message.content
print(result + "\n")
```
This is remarkable! The output looks fantastic! Optical Character Recognition (OCR) is an important implementation in any business scenario. Let's have a look at the generated output as a dataframe. We'll also save it as a CSV for later use.

### 1.8 Create CSV file from generated Ingredients

We can now transform the JSON into a pandas data frame and create a CSV file to save our newly generated ingredients.

-   In your codespace environment, click on the code block and select the **Execute Cell** button to run the code.

```sql
#Transform output to pandas dataframe and save as CSV file  
  
# Clean up Azure OpenAI Output  
json_data = result.strip('` \n')  
  
if json_data.startswith('json'):  
    json_data = json_data[4:]  # Remove the first 4 characters 'json'  
  
omni_ingredients_from_json = json.loads(json_data)  
omni_ingredients = pd.json_normalize(omni_ingredients_from_json)  
path_output = r"/workspaces/azure-openai-lab/data/omni-ingredients.csv"  
omni_ingredients.to_csv(path_output, sep='\t', encoding='utf-8', index=False)
```
We can now use this as an input for our implementations in [Part 3][A-Hands-on-Exploration-of-the-Azure-OpenAI-APIs-part-3], [Part 4][A-Hands-on-Exploration-of-the-Azure-OpenAI-APIs-part-4], and [Part 5][A-Hands-on-Exploration-of-the-Azure-OpenAI-APIs-part-5].

<br/><br/>

Congratulations! You have finished the workshop!

<p align="right">(<a href="#readme-top">back to top</a>)</p>

----------

## 2. Questions, Feedback, Support?

Reach out to us! We are happy to answer any questions you might have or use your feedback to optimize this series!

<p align="right">(<a href="#readme-top">back to top</a>)</p>

----------

## 3. References

[1] [https://community.openai.com/t/issue-with-useage-of-json-output-an-citation/584189](https://community.openai.com/t/issue-with-useage-of-json-output-an-citation/584189)

[2] [https://alexholmeset.blog/2024/05/22/use-the-azure-openai-gpt-4o-all-in-one-model-with-powershell/](https://alexholmeset.blog/2024/05/22/use-the-azure-openai-gpt-4o-all-in-one-model-with-powershell/)

<!-- MARKDOWN LINKS & IMAGES -->
[A-Hands-on-Exploration-of-the-Azure-OpenAI-APIs-part-3]: https://github.com/AllgeierSchweiz/azure-openai-lab/blob/main/series/A%20Hands-on%20Exploration%20of%20the%20Azure%20OpenAI%20API%20(Part%203%20of%C2%A06).md
[A-Hands-on-Exploration-of-the-Azure-OpenAI-APIs-part-4]: https://github.com/AllgeierSchweiz/azure-openai-lab/blob/main/series/A%20Hands-on%20Exploration%20of%20the%20Azure%20OpenAI%20API%20(Part%204%20of%C2%A06).md
[A-Hands-on-Exploration-of-the-Azure-OpenAI-APIs-part-5]: https://github.com/AllgeierSchweiz/azure-openai-lab/blob/main/series/A%20Hands-on%20Exploration%20of%20the%20Azure%20OpenAI%20API%20(Part%205%20of%C2%A06).md
