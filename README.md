<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- ABOUT THE PROJECT -->
# A Hands-on Exploration of Azure OpenAI API

![Azure OpenAI API](images/SDSC2024.png)

## Introduction

* Overview of the OpenAI API capabilities and features

This code snippet configures the OpenAI API key and endpoint for the Azure platform. It depends on the `os` module to entry the values of three surroundings variables: `AZURE_OPEN_KEY`, `AZURE_END_POINT`, and `DEPLOYMENT_NAME`. These variables are essential in authenticating and establishing a reference to the OpenAI API.

The `openai.api_key` variable is assigned the worth of the `AZURE_OPEN_KEY` surroundings variable, serving as the key key for authenticating API requests.
The `openai.api_base` variable takes its worth from the `AZURE_END_POINT` surroundings variable, which designates the endpoint URL for the OpenAI API.
The `openai.api_type` variable is explicitly set to “azure”, signaling the utilization of the OpenAI API throughout the Azure platform.
The `openai.api_version` variable is configured with the worth “2023–07–01-preview”, indicating the particular model of the OpenAI API in use.
Moreover, the `deployment_name` variable obtains its worth from the `DEPLOYMENT_NAME` surroundings variable. This variable assumes significance because it specifies the identify of the deployment utilized for the OpenAI API. This identify performs a task in connecting to the exact deployment occasion of the API that’s operational.


* Setting up API credentials

0. Data pre-processing with Assistant - create Fine-tune Dataset for step 3 (Code interpreter) [Currently problematic, openai=1.12 seems to be stable. Assistant not creating dataset - unclear why)
1. Give me a recipe based on a list of ingredients (Zero-Shot Learning & Few-Shot Learning)
2. Give me a specific [vegan] recipe based on a list of ingredients and preferences (RAG & Few-Shot Learning)
3. Give me a specific [vegan] recipe based on a list of ingredients and preferences (Fine-Tuning).
4. Give me a specific [vegan] recipe based on a list of ingredients and preferences (RAG & Few-Shot Learning & Fine-Tuning)
5. Given recipe, list ingredients and look up nutritional values and create caloric summary (RAG & Code interpreter)

* Data Preparation
   
- **0-data-cleansing-csv-jsonl-v1.ipynb**


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Azure Openai Assistant

- **0-azure-openai-assistant-csv-jsonl-v1.ipynb**
- **4-azure-openai-assistant-ingredients-recipes-v1.ipynb**

* Use Code Interpreter to manipulate structured data (Data wrangling steps)


<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Prompt Engineering 

- **1-azure-openai-simple-ingredients-recipes-v1.ipynb**

#### Format
 A specific technique for formatting instructions is to split the instructions at the beginning or end of the prompt, and have the user content contained within --- or ### blocks. These tags allow the model to more clearly differentiate between instructions and content. For example:

 ```sql
Create a flavourful recipe using the following ingredients:
---
Beef, Butter, Mushrooms, Onions, Cream
---
```
#### Grounding
Grounding content allows the model to provide reliable answers by providing content for the model to draw answer from. Grounding content could be an essay or article that you then ask questions about, a company FAQ document, or information that is more recent than the data the model was trained on. If you need more reliable and current responses, or you need to reference unpublished or specific information, grounding content is highly recommended.

#### Zero Shot

#### One Shot

#### Few Shot Learning
Using a user defined example conversation is what is called few shot learning, which provides the model examples of how it should respond to a given query. These examples serve to train the model how to respond.

* Prompt Engineering for output optimization (Simple / Complex)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## RAG (Embeddings/Vector Database)

- **2-azure-openai-rag-ingredients-recipes-v2.ipynb**

* Use RAG to augment LLM queries with additional information contained in Recipes csv using LangChain.
* Based off of recipes table, create embeddings and feed them into a Vector Database (ChromaDB).
* Send query using traditional gpt3 model and later on to fine-tuned model.


<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Fine Tuning

- **3-azure-openai-fine-tuning-ingredients-recipes-v1.ipynb**

* Adjusting an LLM for use with proprietary data.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- FILES -->
## Documentation, Data & Support Files

Download and unzip the file on your local computer.

Have fun!

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

If there are any questions, feel free to reach out!

Nicolas Rehder - nrehder@allgeier.ch

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- REFERENCES -->
## References

The following documentation was used to source the information contained in this workshop.

* [How to Fine-Tune in Azure](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/fine-tuning?tabs=turbo%2Cpython&pivots=programming-language-python)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/AllgeierSchweiz/openai-lab.svg?style=for-the-badge
[contributors-url]: https://github.com/AllgeierSchweiz/openai-lab/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/AllgeierSchweiz/openai-lab.svg?style=for-the-badge
[forks-url]: https://github.com/AllgeierSchweiz/openai-lab/network/members
[stars-shield]: https://img.shields.io/github/stars/AllgeierSchweiz/openai-lab.svg?style=for-the-badge
[stars-url]: https://github.com/AllgeierSchweiz/openai-lab/stargazers
[issues-shield]: https://img.shields.io/github/issues/AllgeierSchweiz/openai-lab.svg?style=for-the-badge
[issues-url]: https://github.com/AllgeierSchweiz/openai-lab/issues
[license-shield]: https://img.shields.io/github/license/AllgeierSchweiz/openai-lab.svg?style=for-the-badge
[license-url]: https://github.com/AllgeierSchweiz/openai-lab/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/nicolas-a-rehder/
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
[Creating-a-modern-data-lakehouse-zip]: https://github.com/AllgeierSchweiz/azure-data-lakehouse-lab/raw/main/Creating-a-Modern-Data-Lakehouse-with-Azure-Synapse.zip
[Creating-a-modern-data-lakehouse-pdf]: https://downgit.github.io/#/home?url=https://github.com/AllgeierSchweiz/azure-data-lakehouse-lab/blob/main/documentation/Creating-a-Modern-Data-Lakehouse-with-Azure-Synapse.pdf
[Creating-a-modern-data-lakehouse-series-part-1]: https://github.com/AllgeierSchweiz/azure-data-lakehouse-lab/blob/main/series/Creating%20a%20Data%20Lakehouse%20with%20Azure%20Synapse%20Analytics%20(Part%201%20of%205).md
[Creating-a-modern-data-lakehouse-series-part-2]: https://github.com/AllgeierSchweiz/azure-data-lakehouse-lab/blob/main/series/Creating%20a%20Data%20Lakehouse%20with%20Azure%20Synapse%20Analytics%20(Part%202%20of%205).md
[Creating-a-modern-data-lakehouse-series-part-3]: https://github.com/AllgeierSchweiz/azure-data-lakehouse-lab/blob/main/series/Creating%20a%20Data%20Lakehouse%20with%20Azure%20Synapse%20Analytics%20(Part%203%20of%205).md
[Creating-a-modern-data-lakehouse-series-part-4]: https://github.com/AllgeierSchweiz/azure-data-lakehouse-lab/blob/main/series/Creating%20a%20Data%20Lakehouse%20with%20Azure%20Synapse%20Analytics%20(Part%204%20of%205).md
[Creating-a-modern-data-lakehouse-series-part-5]: https://github.com/AllgeierSchweiz/azure-data-lakehouse-lab/blob/main/series/Creating%20a%20Data%20Lakehouse%20with%20Azure%20Synapse%20Analytics%20(Part%205%20of%205).md
[Creating-a-free-azure-account-part-1]: https://github.com/AllgeierSchweiz/azure-data-lakehouse-lab/blob/main/series/Creating%20a%20Free%20Azure%20Account%20(Part%201%20of%201).md
[FactProductCategoryPredictions-csv]: https://downgit.github.io/#/home?url=https://github.com/AllgeierSchweiz/azure-data-lakehouse-lab/blob/main/data/FactProductCategoryPredictions.csv
[FactProductSales-csv]: https://downgit.github.io/#/home?url=https://github.com/AllgeierSchweiz/azure-data-lakehouse-lab/blob/main/data/FactProductSales.csv
[FactProductSales-changes-csv]: https://downgit.github.io/#/home?url=https://github.com/AllgeierSchweiz/azure-data-lakehouse-lab/blob/main/data/changes/FactProductSales.csv
[Dataflow-zip]: https://downgit.github.io/#/home?url=https://github.com/AllgeierSchweiz/azure-data-lakehouse-lab/blob/main/support/pipeline/TransformDeltaFormat.zip
[Setup Bronze Database-sql]: https://downgit.github.io/#/home?url=https://github.com/AllgeierSchweiz/azure-data-lakehouse-lab/blob/main/support/notebooks/Setup-Bronze-Database.sql
[Setup Silver Database-ipynb]: https://downgit.github.io/#/home?url=https://github.com/AllgeierSchweiz/azure-data-lakehouse-lab/blob/main/support/notebooks/Setup-Silver-Database.ipynb
[Setup Gold Database-ipynb]: https://downgit.github.io/#/home?url=https://github.com/AllgeierSchweiz/azure-data-lakehouse-lab/blob/main/support/notebooks/Setup-Gold-Database.ipynb
