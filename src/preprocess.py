import re
import fitz  # PyMuPDF, used for reading PDF files
import pymongo
from sentence_transformers import SentenceTransformer  # For generating text embeddings
import nltk
from nltk.tokenize import sent_tokenize

# Ensure the necessary NLTK model is downloaded
nltk.download('punkt', quiet=True)

class PreprocessPDF:
    """
    A class for preprocessing PDF documents, storing the processed text in MongoDB,
    and updating the MongoDB documents with sentence embeddings.
    """
    def __init__(self, pdf_path, mongo_connection_string, mongo_database, mongo_collection):
        """
        Initializes the PreprocessPDF object with paths, database settings, and loads the sentence transformer model.
        
        :param pdf_path: Path to the PDF document to be processed.
        :param mongo_connection_string: MongoDB connection string.
        :param mongo_database: Name of the MongoDB database.
        :param mongo_collection: Name of the MongoDB collection.
        """
        self.pdf_path = pdf_path
        self.mongo_connection_string = mongo_connection_string
        self.mongo_database = mongo_database
        self.mongo_collection = mongo_collection
        self.client = pymongo.MongoClient(self.mongo_connection_string)
        self.db = self.client[self.mongo_database]
        self.collection = self.db[self.mongo_collection]
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Load the model for embeddings
    
    @staticmethod
    def preprocess_text_mupdf(text):
        """
        Preprocesses and cleans the text extracted from a PDF.
        
        :param text: Raw text extracted from the PDF document.
        :return: Cleaned and preprocessed text.
        """
        text = re.sub(r'\n\s*\n', '\n', text)  # Remove empty lines
        text = re.sub(r'[^A-Za-z0-9.,;:!?()\'\"\n]+', ' ', text)  # Keep only certain punctuation and alphanumeric characters
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        return text.strip()
    
    def extract_and_clean_text(self):
        """
        Extracts text from the PDF document, cleans it, and returns the cleaned text.
        
        :return: Cleaned text from the entire PDF document.
        """
        pdf_document = fitz.open(self.pdf_path)
        cleaned_text_mupdf = ""
        for page_number in range(pdf_document.page_count):
            page = pdf_document.load_page(page_number)
            text = page.get_text()
            cleaned_text_mupdf += self.preprocess_text_mupdf(text)
        pdf_document.close()
        return cleaned_text_mupdf
    
    def chunk_by_sentence(self, text):
        """
        Splits the text into chunks based on sentence boundaries using NLTK's sent_tokenize.
        
        :param text: Cleaned text to be split into sentences.
        :return: A list of sentences extracted from the text.
        """
        return [sentence.strip() for sentence in sent_tokenize(text)]
        
    def store_chunks_in_mongodb(self, chunks):
        """
        Stores each chunk of text as a separate document in a MongoDB collection.
        
        :param chunks: List of text chunks (sentences) to be stored.
        """
        for chunk in chunks:
            document = {"text": chunk}
            self.collection.insert_one(document)
        print(f"Total chunks stored in MongoDB: {len(chunks)}")
    
    def update_documents_with_embeddings(self):
        """
        Updates each document in the MongoDB collection with an embedding generated from its text.
        """
        for document in self.collection.find():
            # Generate embedding for the document's text
            embedding = self.model.encode(document['text'], convert_to_tensor=False)
            # Update the document with the generated embedding
            self.collection.update_one({'_id': document['_id']}, {'$set': {'embedding': embedding.tolist()}})
        print("All documents updated with sentence embeddings.")