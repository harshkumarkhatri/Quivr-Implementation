# Quivr Implementation

Implementing Quivr to parse documents and use it as a chat bot to answer questions. Using SQLite for persistent Embedding storage

### Setup
* Install the below Python and PIP version
    * Python 3.11.10
    * pip 24.3.1
* Create a venv and activate it
    * python -m venv venv
    * source venv/bin/activate
* Install Quivr and other required dependencies. You might get errors regarding the missing dependencies. Install them and you will be good to go.
    * pip install quivr-core
* Run the below command with openAPI key in the terminal
    * export OPENAI_API_KEY=''
* Run the below command to make the script work.
    * python test_quivr.py
If you get any errors related to dependencies missing, go ahead and install them.

Once the chat bot is active you can ask questions from the document and it will give you satisfactory answers for the same. 

For the first time, It will give you `Storage Miss` as the db embedding will be created. From next time it will give you `Storage Hit`. This means that it is loading the embedding from the pre stored data.