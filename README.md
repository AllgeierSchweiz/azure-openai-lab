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
## Introduction

* Overview of the OpenAI API capabilities and features

This code snippet configures the OpenAI API key and endpoint for the Azure platform. It depends on the `os` module to entry the values of three surroundings variables: `AZURE_OPEN_KEY`, `AZURE_END_POINT`, and `DEPLOYMENT_NAME`. These variables are essential in authenticating and establishing a reference to the OpenAI API.

The `openai.api_key` variable is assigned the worth of the `AZURE_OPEN_KEY` surroundings variable, serving as the key key for authenticating API requests.
The `openai.api_base` variable takes its worth from the `AZURE_END_POINT` surroundings variable, which designates the endpoint URL for the OpenAI API.
The `openai.api_type` variable is explicitly set to “azure”, signaling the utilization of the OpenAI API throughout the Azure platform.
The `openai.api_version` variable is configured with the worth “2023–07–01-preview”, indicating the particular model of the OpenAI API in use.
Moreover, the `deployment_name` variable obtains its worth from the `DEPLOYMENT_NAME` surroundings variable. This variable assumes significance because it specifies the identify of the deployment utilized for the OpenAI API. This identify performs a task in connecting to the exact deployment occasion of the API that’s operational.


* Setting up API credentials

1. Data Science pre-processing with Assistant - Clean OpenFoodFacts (Code interpreter)
2. GPT4 give me a recipe based on a list of ingredients (Prompt Engineering)
3. Given recipe, list ingredients and look up via RAG nutritional values (Data Enrichment)
4. Based on nutritional values, sum up and create caloric summary [carbs, fats, protein, etc] (Code interpreter)
5. Create Fine-tuned model to output specifically crafted recipates (Master Chef).

https://portal.azure.com/#@algdev.ch/resource/subscriptions/fade7a40-9037-4aeb-82c9-e70f8b49217a/resourceGroups/rgopenaisweden/providers/Microsoft.CognitiveServices/accounts/mssp-openai-sweden/overview

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Azure Openai Assistant

* Use Code Interpreter to manipulate structured data (Data wrangling steps)

**azure-openai-assistant-transformations-v1.ipynb**

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Prompt Engineering 

#### Format of instructions
 A specific technique for formatting instructions is to split the instructions at the beginning or end of the prompt, and have the user content contained within --- or ### blocks. These tags allow the model to more clearly differentiate between instructions and content. For example:

 ```sql
Create a flavourful recipe using the following ingredients:
---
Beef, Butter, Mushrooms, Onions, Cream
---
```

#### Zero Shot
#### One Shot
#### Few Shot Learning

* Prompt Engineering for output optimization (Simple / Complex)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## RAG (Embeddings/Vector Database)

* Use RAG to augment LLM queries with additional information contained in OpenFoodFacts.
* Based off of OpenFoodFacts table, create embeddings and feed them into a Vector Database (ChromaDB).

**azure-openai-rags-v1.ipynb**

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Fine Tuning

* Adjusting an LLM for use with proprietary data.

**azure-openai-fine-tuning-v1.ipynb**

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

* [Azure Architectures](https://learn.microsoft.com/en-us/azure/architecture/browse/)
* [Medallion Structure](https://learn.microsoft.com/en-us/azure/databricks/lakehouse/medallion)
* [Medallion Structure Best Practices](https://piethein.medium.com/medallion-architecture-best-practices-for-managing-bronze-silver-and-gold-486de7c90055)
* [Azure Pipelines](https://aarfahrayees.medium.com/delta-lake-26e76469322c)
* [Data Lakehouse Strategy](https://techcommunity.microsoft.com/t5/azure-synapse-analytics-blog/building-the-lakehouse-implementing-a-data-lake-strategy-with/ba-p/3612291)
* [SQL Database & Lake Database](https://learn.microsoft.com/en-us/answers/questions/784144/what-is-the-difference-between-sql-database-and-la)

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
