import streamlit as st
import os
from dotenv import load_dotenv
from model import RAGModel

# Load the environment variables
load_dotenv()

MONGODB_CONNECTION_STRING = str(os.getenv("MONGODB_URI"))
MONGODB_DATABASE = str(os.getenv("MONGODB_DATABASE"))
MONGODB_COLLECTION = str(os.getenv("MONGODB_COLLECTION"))
API_TOKEN = str(os.getenv("GEMINI_API_KEY"))

# Initialize the RAGModel with your MongoDB and API settings
@st.cache(allow_output_mutation=True)
def load_model():
    return RAGModel(
        mongo_connection_string=MONGODB_CONNECTION_STRING,
        mongo_database=MONGODB_DATABASE,
        mongo_collection=MONGODB_COLLECTION,
        api_token=API_TOKEN
    )

# Streamlit app customization
st.set_page_config(page_title="HubermanAI", page_icon="🔬", layout="centered", initial_sidebar_state="auto")

# Custom CSS
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       .reportview-container {
            background: #000000
       }
       .logo-img {
           display: flex;
           justify-content: center;
       }
       div.stButton > button:first-child {
           background-color: #00b8a9;
           color: white;
           font-size: 18px;
           font-weight: bold;
       }
       .reportview-container .main footer {visibility: hidden;}  
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

# Display the logo and title
st.markdown("<div class='logo-img'>", unsafe_allow_html=True)
st.image("src/.streamlit/HubermanAI.jpeg", width=200)  # Adjust the path and width as needed
st.markdown("</div>", unsafe_allow_html=True)

st.title("HubermanAI")
st.write("HubermanAI is a conversational chat bot channeling Dr. Andrew Huberman's scientific expertise 🧠 🚀")

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Define your query
query = st.text_input("Message HubermanAI: ", key="query_input")

rag_model = load_model()

# Submit query and update chat history
def handle_query():
    if query:  # Ensure the query is not empty
        st.session_state.chat_history.append(f"You: {query}")
        answer = rag_model.generate_answer(query)
        st.session_state.chat_history.append(f"HubermanAI: {answer}")
        st.session_state.query_input = ""  # Clear the input box after submission

# Button to submit the query
if st.button("Ask HubermanAI"):
    handle_query()

# Display chat history
for message in st.session_state.chat_history:
    st.text_area("", value=message, height=75, disabled=True)

