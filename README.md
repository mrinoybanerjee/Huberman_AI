# Huberman_AI 🔬

## Project Overview

"Huberman_AI: A Conversational Bot Channeling Dr. Andrew Huberman's Expertise"

Huberman_AI is a RAG-based chatbot that distills Dr. Andrew Huberman's podcast, Huberman Lab into actionable advice. Designed for those looking to use science-based tools and protocols to enhance longevity, mental health, focus, and well-being.

## Installation

To get Huberman_AI up and running on your local machine, follow these steps:

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/Huberman_AI.git
cd Huberman_AI
```

2. **Set up a virtual environment (optional but recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. **Install required dependencies:**

```bash
pip install -r requirements.txt
```

## Configuration

Before launching Huberman_AI, configure the environment variables:

1. Rename the `.env.example` file to `.env`.
2. Fill in the necessary values in `.env`, such as API keys, database connection strings, or other configuration details specific to your environment.

## Usage

To start Huberman_AI, navigate to the `src` directory and run the main script:

```bash
cd src
python main.py
```

Follow the on-screen instructions to interact with the bot.

## Data Management

The RAG is using Huberman Lab newsletters as well as transcripts directly from the podcast to use as the RAG corpus.

## Contact

For support, further information, or to contribute to the project, please reach out to me at mrinoybanerjee@gmail.com

## Project Walkthrough

### Huberman_AI: Using RAG Based LLM

This section provides a step-by-step guide to integrating a Retrieval-Augmented Generation (RAG) based Large Language Model (LLM) into Huberman_AI, enhancing its ability to provide accurate and relevant science-based tools to boost longevity.


#### Text Extraction from Book

1. **Extract Data**: Functionality is provided to extract text from PDF files, which involves:
   - Opening the PDF file.
   - Removing headers/footers to clean the text.
   - Saving the cleaned text for further processing.

#### Store Chunks in MongoDB Database

1. **Database Connection**: Connect to MongoDB database. Update the connection string as per your setup.
2. **Text Chunking**: Using nltk package to implement chunking.
3. **Data Storage**: Stored these chunks in MongoDB, creating a document for each chunk.


#### Create Word Embeddings

1. **Embeddings Generation**: Using a sentence transformer model, generate embeddings for each text chunk stored in MongoDB.
2. **Database Update**: Update each document in MongoDB with its corresponding embedding for later retrieval.

#### RAG: Semantic Search Retrieval

1. **Semantic Search Functionality**: Implemented a function to perform semantic search. This involves generating a query embedding, retrieving and comparing all stored embeddings from MongoDB, and returning the most relevant documents based on similarity.

#### LLM Model

1. **LLM Integration**: Integrated the LLama2-7B model from Replicate to generate answers based on the context provided by the semantic search.
   - This includes handling context truncation and ensuring the generation of non-empty answers.

## Deployment

The Huberman_AI application is deployed on Streamlit and is available for public use. This deployment allows users to interact with Huberman_AI through a user-friendly web interface.

## Repository Structure

Below is the structure of the Huberman_AI repository, providing a clear overview of its organization and contents:

```
.
├── README.md
├── notebooks
│   └── rag_model.ipynb
├── requirements.txt
└── src
    ├── app.py
    ├── evaluate.py
    ├── main.py
    ├── model.py
    └── preprocess.py
```

This structure is designed to facilitate easy navigation and understanding of the project for developers and users alike.

