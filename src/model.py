import pymongo
import numpy as np
from scipy.spatial.distance import cosine
from replicate.client import Client
import google.generativeai as genai  # Import Google GenAI
from sentence_transformers import SentenceTransformer

class RAGModel:
    """
    A class to perform Retrieval-Augmented Generation (RAG) for generating answers
    based on semantic search within a MongoDB database and using the Gemini Pro model
    for text generation.
    """
    def __init__(self, mongo_connection_string, mongo_database, mongo_collection, api_token):
        """
        Initializes the RAGModel with MongoDB connection settings and the API token for Gemini Pro.
        
        :param mongo_connection_string: MongoDB connection string.
        :param mongo_database: MongoDB database name.
        :param mongo_collection: MongoDB collection name.
        :param api_token: Gemini API token.
        """
        self.mongo_connection_string = mongo_connection_string
        self.mongo_database = mongo_database
        self.mongo_collection = mongo_collection
        self.api_token = api_token
        self.client = pymongo.MongoClient(self.mongo_connection_string)
        self.db = self.client[self.mongo_database]
        self.chunks_collection = self.db[self.mongo_collection]
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Load sentence transformer model for embeddings
        genai.configure(api_key=self.api_token)  # Configure GenAI with the provided API key
        self.engineered_context = "[INST]\nYou weild the knowledge of Dr. Andrew Huberman."

    def semantic_search(self, query, top_k=5):
        """
        Performs semantic search to find the most relevant text chunks based on the query.
        
        :param query: The query string for which to find relevant documents.
        :param top_k: The number of top results to return.
        :return: A list of tuples containing the document ID, similarity score, and text for each result.
        """
        query_embedding = self.model.encode(query, convert_to_tensor=False)
        similarities = []
        for document in self.chunks_collection.find():
            doc_embedding = np.array(document['embedding'])
            similarity = 1 - cosine(query_embedding, doc_embedding)
            similarities.append((document['_id'], similarity, document['text']))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    

    def generate_answer(self, question, max_context_length=2000):
        """
        Generates an answer to a given question using the best context found by semantic search
        and the Gemini Pro model.
        
        :param question: The question to generate an answer for.
        :param max_context_length: Maximum length of the context to be considered.
        :return: Generated answer as a string.
        """
        context_results = self.semantic_search(question, top_k=5)
        if context_results:
            # Concatenate the top-k context results into a single string
            context = " ".join([result[2] for result in context_results])
            if len(context) > max_context_length:
                context = context[:max_context_length]
            # Use engineered context with the best context found by semantic search
            prompt = f"[INST]\nQuestion: {question}\nContext: {context}\n{self.engineered_context}\n[/INST]"
        else:
            # Use only the engineered context
            prompt = f"[INST]\nQuestion: {question}\n{self.engineered_context}\n[/INST]"

        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)

        # Handle the case where the response is empty
        if not response.text:
            return "Sorry, I don't have an answer for that."
        
        return response.text
    

    def generate_RAG_answer(self, question, max_context_length=500):
        """
        Generates an answer to a given question using the best context found by RAG based semantic search
        and the Gemini Pro model.
        
        :param question: The question to generate an answer for.
        :param max_context_length: Maximum length of the context to be considered.
        :return: Generated answer as a string.
        """
        context_results = self.semantic_search(question, top_k=1)
        context = context_results[0][2]
        if len(context) > max_context_length:
            context = context[:max_context_length]
        prompt = f"[INST]\nQuestion: {question}\nContext: {context}\n[/INST]"

        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)

        # Handle the case where the response is empty
        if not response.text:
            return "Sorry, I don't have an answer for that."
        
        return response.text
    

    def generate_non_RAG_answer(self, question):
        """
        Generates an answer to a given question using only the Gemini Pro model.
        
        :param question: The question to generate an answer for.
        :return: Generated answer as a string.
        """
        prompt = f"[INST]\nQuestion: {question}\n[/INST]"

        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)

        # Handle the case where the response is empty
        if not response.text:
            return "Sorry, I don't have an answer for that."
        
        return response.text
