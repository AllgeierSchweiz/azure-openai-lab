----------

### A Hands-on Exploration of the Azure OpenAI API (Part 2 of 5)

![](https://cdn-images-1.medium.com/max/800/1*PV9Eh3WpAhnW9NZCijZPzw.png)

### 1. Preparations

The following prerequisites must be met to successfully start this series:

-   You must be connected to the internet.
-   You should ideally use two monitors (**optional**)
-   You have Microsoft Edge or any other reliable web browser installed.
-   You must have a GitHub account (we will use Codespaces to run our Notebooks). If you do not have one, register yourself on [GitHub](https://github.com/).

----------

### 2. Files & Data

All relevant files are in the Allgeier Schweiz [GitHub repository](https://github.com/AllgeierSchweiz/azure-openai-lab).

#### 2.1 Notebooks

The workshop has 6 Jupyter Notebooks we will work through:

1.  _P3-azure-openai-prompt-engineering.ipynb_
2.  _P4-azure-openai-assistant-rag-data-preprocessing.ipynb_
3.  _P4-azure-openai-rag.ipynb_
4.  _P5-azure-openai-assistant-fine-tuning-data-preprocessing.ipynb_
5.  _P5-azure-openai-fine-tuning.ipynb_
6.  _P5-azure-openai-rag-with-fine-tuning.ipynb_

#### 2.2 Files

The workshop has 4 prepared files. Only the **recipes.csv** _w_ill be actively used throughout this series. The other 3 files are backups.

The CSV file **recipes.csv**  consists of 200K recipes covering 18 years of user interactions and uploads on Food.com. The original dataset comes from the paper: [Generating Personalized Recipes from Historical User Preferences](https://aclanthology.org/D19-1613.pdf).

1.  _recipes.csv_
2.  _recipes-preprocessed.csv (Backup)_
3.  _recipes-training-set.jsonl (Backup)_
4.  _recipes-validation-set.jsonl (Backup)_

----------

### 3. Azure Resources

To use Azure OpenAI models you must first request access to the Azure OpenAI Service using the [Microsoft registration form](https://customervoice.microsoft.com/Pages/ResponsePage.aspx?id=v4j5cvGGr0GRqy180BHbR7en2Ais5pxKtso_Pz4b1_xUNTZBNzRKNlVQSFhZMU9aV09EVzYxWFdORCQlQCN0PWcu).

The Azure OpenAI Service sits in an Azure Resource Group within an Azure Subscription. The [region](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models#standard-deployment-model-availability) of the Azure OpenAI Service must be selected carefully since only a handful of regions grant access to specific language model versions and functionalities.

**_NOTE: For the SDSC workshop, all Azure resources, requests, and regions have been pre-selected and provisioned. No additional steps are required._**

To access the Azure OpenAI REST API in Python an API Key and an API Endpoint URL are required. This information will be used in the GitHub Codespaces environment.

**_NOTE: For the SDSC workshop, all Azure resources, requests, and regions have been pre-selected and provisioned. No additional steps are required._**

----------

### 4. GitHub Codespaces

To run the Python Notebooks we will use GitHub Codespaces. This convenient web-based environment mimics Visual Studio Code directly in your web browser. The environment offers users 60 hours a month of free computing and 15GB of free storage. This is more than enough to get us through the workshop!

**_NOTE: Make sure you have logged into GitHub using your GitHub account. Should you not have one, please_** [**_register_**](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiznL-AgpeGAxX8gf0HHf-mA7UQFnoECAYQAQ&url=https%3A%2F%2Fgithub.com%2Fjoin&usg=AOvVaw0H9TK-nu7JfXaoNeNMgJEk&opi=89978449) **_yourself before proceeding._**

#### 4.1 Starting GitHub Codespaces

-   Open the [GitHub Repository](https://github.com/AllgeierSchweiz/azure-openai-lab) and select the green button named **Code**  on the top right corner of the page.
-   In the newly opened tab, select the button **Create codespace on main**.

![](https://cdn-images-1.medium.com/max/800/1*XZjdWF14xVu6mUrGLo_YLw.png)

  

-   Once selected, a new browser tab will appear, and the codespace environment with all our required files will be prepared.

![](https://cdn-images-1.medium.com/max/800/1*zDHra7oxKefa0pFc0wQe3Q.png)

You should see the following codespace environment once the setup has been finalized:

![](https://cdn-images-1.medium.com/max/800/1*G6GtqmB-Wi-1S6Joa0NDuQ.png)

Explain GUI

#### 4.2 Installing Visual Studio Code Extensions

To work with Python and Jupyter Notebooks, we need to install their respective extensions in codespace.

-   On the left of the codespace environment, select the **Extensions** icon.
-   In the search bar, input **Python** and select the **install** button of the first entry.

![](https://cdn-images-1.medium.com/max/800/1*kCtzqjBrC1x2lubMP5Sl8A.png)

**_NOTE: The installation may take 1–2 minutes. Please be patient._**

-   Once the Python extension installation is complete, go back to the search bar and input **Jupyter. S**elect the **install** button of the first entry.

![](https://cdn-images-1.medium.com/max/800/1*Uxs172Dq0D2a7SlNX_rhzA.png)

-     
    

#### 4.3 Creating a Python Kernel Source

To work with Python, we need to select a Python Kernel. We will create a new source using a virtual environment.

-   On the left of the codespace environment, select the **Explorer** icon.
-   Within the explorer, open the tabs to the folders **support** and **notebooks**.
-   Select the Jupyter Notebook **P3-azure-openai-prompt-engineering.ipynb.**

![](https://cdn-images-1.medium.com/max/800/1*2i3DaW-4blqaxexerVxFlQ.png)

  

-   Within the newly opened Jupyter Notebook, select the button **Select Kernel** on the top right corner.

![](https://cdn-images-1.medium.com/max/800/1*wJ-9hVxiTlTUmDGdrZn66Q.png)

-   A new tab will open on the top of the environment. Select **Python Environments**.

![](https://cdn-images-1.medium.com/max/800/1*4P3VpIkmzWSVYw12sHquBw.png)

-   An additional option will appear in the tab. Select **+ Create Python Environment**.

![](https://cdn-images-1.medium.com/max/800/1*3WldmZZYswXXTKnjaSgplg.png)

-   An additional option will appear in the tab. Select `Venv`.

![](https://cdn-images-1.medium.com/max/800/1*Bm_1VA_dCBUMWGgY-6yDiw.png)

-   An additional option will appear in the tab. Select **Use Python from python.defaultinterpreterPath setting.**

![](https://cdn-images-1.medium.com/max/800/1*aJSL8FnSs8vsUGHONIvz2w.png)

-   An additional option will appear in the tab. Tick the first option named **support/requirements/requirements.txt** and select the button **OK.**

![](https://cdn-images-1.medium.com/max/800/1*ZGDK77ul1QzYeeWrR2_agw.png)

Once the environment creation starts, you will receive a message on the bottom right of the screen.

![](https://cdn-images-1.medium.com/max/800/1*rNeNSG1x6DsJD1seNINwIg.png)

**_NOTE:_** `Venv`**_creates a lightweight virtual environment on top of an existing Python installation. This allows us to install Python packages that are isolated from the packages in the base environment. In this manner, we can work with different environments, each with its own independent Python version and set of packages._**

#### 4.4 Setting up Azure OpenAI API credentials

To work with the Azure OpenAI API we need to configure the codespace environment to use the **Azure OpenAI API Key** and **Azure OpenAI Endpoint URL**. These credentials can be found in the Azure OpenAI Service overview within the Azure Portal. We have provided these credentials on the paper in front of you.

We will save these credentials in a `.env` file within our Python environment.

**_NOTE: A_** `.env` **_file is a text file containing values of all the environment variables required for our notebooks. This file is included with your project locally but not saved to source control so that you aren't putting potentially sensitive information at risk._**

-   On the left of the codespace environment, select the **Explorer** icon.
-   Right-click on the **.venv** tab and select **New File**.

![](https://cdn-images-1.medium.com/max/800/1*easwgOsy_ntOd8aCbL_uuw.png)

-   Name this file `.env`  and hit enter.

![](https://cdn-images-1.medium.com/max/800/1*PRYnCNnYHoRn4Ys-_1RYuA.png)

-   Copy the code below and paste it into your newly created `.env` file.

```sql
# Azure Tenant: Azure Pass
# Resource Group name: rg-sdsc2024-x
# Azure OpenAI Resource name: oai-sdsc2024-x

AZURE_OPENAI_KEY = ""
AZURE_OPENAI_ENDPOINT = ""
```

-   Change the entries of **AZURE_OPENAI_KEY** and **AZURE_OPENAI_ENDPOINT** to match the credentials supplied to you.

![](https://cdn-images-1.medium.com/max/800/1*P9hRF15NkZOhPu0v-HHnqg.png)

#### 4.5 ChromaDB troubleshooting

GitHub Codespace currently throws an error when importing the `chromadb` package. The system uses an unsupported version of `sqlite3`, which we need to create our local database.

```sql
>>> import chromadb  
Traceback (most recent call last):  
  File "<stdin>", line 1, in <module>  
  File "/home/vscode/.local/lib/python3.10/site-packages/chromadb/__init__.py", line 69, in <module>  
    raise RuntimeError(  
RuntimeError: Your system has an unsupported version of sqlite3. Chroma requires sqlite3 >= 3.35.0.  
Please visit https://docs.trychroma.com/troubleshooting#sqlite to learn how to upgrade.
```

We need to implement a workaround to fix this issue.

-   On the left of the codespace environment, select the **Explorer** icon.
-   Within the explorer, open the tabs to the folders **.venv**, **lib,** and **chromadb**.
-   Select the Python script named **\_init\_.py** within the **chromadb** tab.

![](https://cdn-images-1.medium.com/max/800/1*VH5gkNvFyc79jx3F1wK33Q.png)

-   Within the **\_init\_.py** file go to line 67 and comment out lines **68 to 85** by highlighting the rows and selecting the keyboard combination **CTRL+K+C**. The area should become green with “#” symbols in front of each code line.

![](https://cdn-images-1.medium.com/max/800/1*J3KmFvgYOSM4pNxxLDVdBw.png)

-   Copy and paste the code below into the **\_init\_.py** file on line 68. Make sure to indent the code by highlighting it and clicking the tab button on your keyboard.

```sql
__import__('pysqlite3')  
import sys  
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
```

![](https://cdn-images-1.medium.com/max/800/1*W6HkQTxIVM0QFvxiwmyuLA.png)

-   Once you have made the changes close the *\_init\_.py** file. All changes are automatically saved. Not to worry!

![](https://cdn-images-1.medium.com/max/800/1*QsMvkmRyEEvlmajyOGcUnA.png)

Perfect! You have now finalized the required preparations to start running the Jupyter notebooks in Codespace. Lets get our hands dirty! We will continue with Part 3 of the workshop.

----------

### 5. Questions, Feedback, Support?

Reach out to us! We are happy to answer any questions you might have or use your feedback to optimize this series!

----------

### 6. References

[1] [venv — Creation of virtual environments — Python 3.12.3 documentation](https://docs.python.org/3/library/venv.html)

[2] [Using .env Files for Environment Variables in Python Applications — DEV Community](https://dev.to/jakewitcher/using-env-files-for-environment-variables-in-python-applications-55a1)

[3] [How to override an old sqlite3 module with pysqlite3 in django settings.py (github.com)](https://gist.github.com/defulmere/8b9695e415a44271061cc8e272f3c300?permalink_comment_id=4711478)

[4] [python — AttributeError: module ‘chromadb’ has no attribute ‘config’ — Stack Overflow](https://stackoverflow.com/questions/76921252/attributeerror-module-chromadb-has-no-attribute-config)

###
