# The Vault App
## Demo for [Build with AI - A Hands-On Introduction to Retrieval Augmented Generation (RAG)](https://gdg.community.dev/events/details/google-gdg-on-campus-heriot-watt-university-dubai-dubai-united-arab-emirates-presents-build-with-ai-a-hands-on-introduction-to-retrieval-augmented-generation-rag/) @ Google Developer Group - Heriot-Watt University, Dubai

View the UI of the live app at [VaultApp](https://vaultapp.streamlit.app) <br>
View the Google Colab version at [GoogleColab](https://colab.research.google.com/drive/1Hqru_V6wlqE686eDngfkK_9dAZHkMrIt?usp=sharing)

## Instructions

### Supported Python versions

![Python 3.12](https://github.com/jonathanjthomas/GDG-RAG-Demo/actions/workflows/python-3.12.yml/badge.svg)

### Recommended system specifications

- **GPU:** 8 GB VRAM
- **RAM:** 16 GB RAM

**Don't worry if you have a GPU with a lesser amount of VRAM, 8 GB is recommended for use with the Gemma:9b model, however, this application works perfectly fine with the Gemma:2b as well, so a minimum of 4 GB VRAM should suffice.**

If the application is not fast enough on your device, **try the Google Colab version** [here](https://colab.research.google.com/drive/1Hqru_V6wlqE686eDngfkK_9dAZHkMrIt?usp=sharing).

### Demo Set-Up (For Build with AI Attendees)
- Download and install [Ollama](https://ollama.com/download)
- Pull the required Ollama models (gemma2, gemma2:2b and nomic-embed-text)
  
 ```shell
  ollama pull gemma2:2b
  ollama pull gemma2
  ollama pull nomic-embed-text
  ```

- Create a folder for the project, and make a Python script inside it named "app.py"
- Set up a virtual environment using the below command (Recommended)

  ```shell
  python -m venv venv
  ```

- Activate your virtual environment using the following command:

  - On Windows:
    ```shell
    venv\Scripts\Activate
    ```
  - On Linux/MacOS:
    ```shell
    source venv/bin/activate
    ```

- Install all the required libraries and dependencies

  ```shell
  pip install langchain-ollama langchain-chroma>=0.1.2 langchain-community pypdf jq streamlit
  ```
  
- Run app.py with streamlit
  
  `streamlit run code\app.py` or `python -m streamlit run code\app.py`

## Debugging

- If you face any conflicts with existing dependencies, make sure you have activated your virtual environment

- If you run into the following error:
  ```shell
  httpx.ConnectError: [WinError 10061] No connection could be made because the target machine actively refused it
  ```
  Then try running the following to resolve the issue
  ```shell
  ollama serve
  ```
- **If you run into any other issues which have not been listed above, please feel free to reach out to us.**

## Reach Out

Have any doubts? Feel free to reach out to us at:

- Aditya S (as2397@hw.ac.uk)
- Jonathan John Thomas (jjt2002@hw.ac.uk)

## Additional Resources

1.	Using Chroma vector store with LangChain: https://python.langchain.com/docs/integrations/vectorstores/chroma/
2.	Ollama Chat models with LangChain: https://python.langchain.com/docs/integrations/chat/ollama/
3.	Ollama Embeddings with LangChain: https://python.langchain.com/docs/integrations/text_embedding/ollama/#indexing-and-retrieval
4.	LangChain text splitters: https://python.langchain.com/docs/how_to/recursive_text_splitter/
5.	Streamlit UI for LLM Chat Apps: https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps
