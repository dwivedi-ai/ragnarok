import os 
import re
import logging
import atexit
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
import speech_recognition as sr
from gtts import gTTS
import tempfile
import grpc
from google.cloud import vision_v1

# Configure logging to suppress absl warnings
logging.basicConfig(level=logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

app = Flask(__name__)

# Initialize global variables
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def init_vision_client():
    """Initialize Google Cloud Vision client using credentials from environment variables"""
    try:
        credentials_path = os.getenv('GOOGLE_VISION_CREDENTIALS_PATH')
        if not credentials_path:
            raise ValueError("GOOGLE_VISION_CREDENTIALS_PATH not set in environment variables")
        
        if not os.path.exists(credentials_path):
            raise FileNotFoundError(f"Credentials file not found at: {credentials_path}")
        
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        client_options = {'api_endpoint': os.getenv('GOOGLE_VISION_API_ENDPOINT', 'vision.googleapis.com')}
        return vision_v1.ImageAnnotatorClient(client_options=client_options)
    except Exception as e:
        print(f"Error initializing Vision client: {e}")
        return None

# Initialize vision client
vision_client = init_vision_client()

# Initialize speech recognizer
recognizer = sr.Recognizer()

def cleanup_grpc():
    """Cleanup gRPC channels on shutdown"""
    try:
        for channel in grpc._channel._channel_pool:
            try:
                channel.close()
            except:
                continue
    except:
        pass

# Register cleanup function
atexit.register(cleanup_grpc)

def detect_text_in_image(image_path):
    """Extract text from image using Google Cloud Vision API"""
    if not vision_client:
        return "Vision client not initialized"
    
    try:
        with open(image_path, 'rb') as image_file:
            content = image_file.read()
        
        image = vision_v1.Image(content=content)
        response = vision_client.text_detection(
            image=image,
            timeout=30  # Set explicit timeout
        )
        
        if not response.text_annotations:
            return "No text detected"
        
        return response.text_annotations[0].description
    except Exception as e:
        print(f"Error in text detection: {e}")
        return f"Error detecting text: {str(e)}"

def initialize_gemini_chat():
    """Initialize Gemini chat model"""
    try:
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        model = genai.GenerativeModel("gemini-1.5-flash")
        return model.start_chat(history=[])
    except Exception as e:
        print(f"Error initializing Gemini chat: {e}")
        return None

def get_text_chunks(text):
    """Split text into chunks"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def get_vector_store(text_chunks):
    """Create vector store from text chunks"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        print(f"Error creating vector store: {e}")

def get_vector_store_image(text_chunks):
    """Create vector store for image text"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index_image")
    except Exception as e:
        print(f"Error creating image vector store: {e}")

def get_conversational_chain():
    """Create conversation chain"""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the context, just say, "I don't have enough information to answer that question."

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def process_document_query(user_question):
    """Process document queries"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        return response["output_text"]
    except Exception as e:
        return f"Error processing document query: {str(e)}"

def process_image_query(user_question):
    """Process image queries"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("faiss_index_image", embeddings, allow_dangerous_deserialization=True)
        images = vector_store.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": images, "question": user_question},
            return_only_outputs=True
        )
        return response["output_text"]
    except Exception as e:
        return f"Error processing image query: {str(e)}"

def classify_intent(user_input):
    """Classify user intent"""
    doc_keywords = r"explain|describe|document|pdf|file|text|read|extract|analyze"
    image_keywords = r"image|picture|photo|explain|describe|text|read|extract|analyze"
    if re.search(doc_keywords, user_input, re.IGNORECASE):
        return "document"
    elif re.search(image_keywords, user_input, re.IGNORECASE):
        return "image"
    return "chat"

# Routes
@app.route('/')
def home():
    """Home page route"""
    return render_template('chat.html')  # Changed to directly serve chat.html

@app.route('/chat', methods=['POST'])  # Changed route to match frontend
def chat_message():
    """Handle chat messages"""
    try:
        user_input = request.form.get('user_input')
        if not user_input:
            return jsonify({"error": "No input provided"}), 400
            
        intent = classify_intent(user_input)
        chat_instance = initialize_gemini_chat()
        
        if not chat_instance:
            return jsonify({"error": "Failed to initialize chat"}), 500
        
        if intent == "document":
            response = process_document_query(user_input)
        elif intent == "image":
            response = process_image_query(user_input)
        else:
            response = chat_instance.send_message(user_input).text
            
        return jsonify({"response": response})
    except Exception as e:
        print(f"Chat error: {str(e)}")  # Add logging
        return jsonify({"error": str(e)}), 500

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Handle image uploads"""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    image = request.files['image']
    if not image.filename:
        return jsonify({"error": "No selected file"}), 400
        
    filename = secure_filename(image.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        image.save(file_path)
        extracted_text = detect_text_in_image(file_path)
        
        if extracted_text and extracted_text != "No text detected":
            text_chunks = get_text_chunks(extracted_text)
            get_vector_store_image(text_chunks)
            return jsonify({
                "message": "Image processed successfully",
                "extracted_text": extracted_text
            })
        else:
            return jsonify({
                "message": "No text detected in image",
                "extracted_text": ""
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload_document', methods=['POST'])
def upload_document():
    """Handle document uploads"""
    if 'document' not in request.files:
        return jsonify({"error": "No document provided"}), 400
    
    document = request.files['document']
    if not document.filename:
        return jsonify({"error": "No selected file"}), 400
        
    filename = secure_filename(document.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        document.save(file_path)
        pdf_reader = PdfReader(file_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
            
        if not text.strip():
            return jsonify({"error": "No text found in PDF"}), 400
            
        text_chunks = get_text_chunks(text)
        get_vector_store(text_chunks)
        return jsonify({"message": "Document processed successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/voice_input', methods=['POST'])
def voice_input():
    """Handle voice input"""
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            text = recognizer.recognize_google(audio)
            return jsonify({"text": text})
    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand audio"}), 400
    except sr.RequestError as e:
        return jsonify({"error": f"Speech recognition error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize services before running the app
    if not vision_client:
        print("Warning: Vision client failed to initialize")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)