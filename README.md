# GDG-RAG-Demo

View the UI of the live app at [VaultApp](https://vaultapp.streamlit.app) <br>
View the code in Google Colab at [GoogleColab](https://colab.research.google.com/drive/1Hqru_V6wlqE686eDngfkK_9dAZHkMrIt?usp=sharing)

## Supported Python versions

![Python 3.12](https://github.com/jonathanjthomas/GDG-RAG-Demo/actions/workflows/python-3.12.yml/badge.svg)

# Instructions

## Recommended system specifications

- **GPU:** 8 GB VRAM
- **RAM:** 16 GB RAM

## How to run the Streamlit application

- Make sure you have Python and Git installed on your system
- Clone the repository on to your local machine using Git

  ```shell
  git clone https://github.com/jonathanjthomas/GDG-RAG-Demo.git
  ```

- Navigate to the repository directory
- Set up a virtual environment using the below command

  ```shell
  python -m venv venv
  ```

- Activate your virtual environment using

  - Windows
    ```shell
    venv\Scripts\Activate
    ```
  - Linux and MacOS
    ```shell
    source venv/bin/activate
    ```

- Install all the required libraries and dependencies

  ```shell
  pip install -r requirements.txt
  ```

- Download and install [Ollama](https://ollama.com/download)
- Pull the required Ollama models (gemma2:2b and nomic-embed-text)

  ```shell
  ollama pull gemma2:2b
  ollama pull nomic-embed-text
  ```

- Run vault.py with streamlit

  `streamlit run code\vault.py` or `python -m streamlit run code\vault.py`

## Debugging

- If you face any conflicts with existing dependencies, make sure you have activated your virtual environment

- If you run into an error showing:
  ```shell
  httpx.ConnectError: [WinError 10061] No connection could be made because the target machine actively refused it
  ```
  Then try running the following to resolve the issue
  ```shell
  ollama serve
  ```
- **If you run into any other issues which have not been listed above, please feel free to reach out to us**

## Reach Out

Have any doubts? Feel free to reach out to us at:

- Aditya S (as2397@hw.ac.uk)
- Jonathan John Thomas (jjt2002@hw.ac.uk)

## Additional Resources

[Source code on Google Colab](https://colab.research.google.com/drive/1Hqru_V6wlqE686eDngfkK_9dAZHkMrIt?usp=sharing) <br>
[Gemma 2 - Local RAG with Ollama and LangChain](https://www.youtube.com/watch?v=daZOrbMs61I&t=8s) <br>
[How to Build a Local RAG Knowledge Base with Google Gemma 2 2B](https://sebastian-petrus.medium.com/building-a-local-rag-knowledge-base-with-google-gemma-2-2b-5823cdfe884d) <br>
