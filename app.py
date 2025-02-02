import os
import logging
import certifi
from flask import Flask, request, render_template, redirect, jsonify
from pymongo import MongoClient
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings


from bson.objectid import ObjectId
from datetime import datetime

# Configure logging to output to both console and file
logging.basicConfig(
    level=logging.INFO, 
    format='[%(asctime)s] %(levelname)s-%(lineno)d-%(message)s',
    handlers=[
        logging.FileHandler('app.log'), 
        logging.StreamHandler() 
        ]
    )





app = Flask(__name__)

load_dotenv()

huggingface_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")


uri = os.getenv("MONGODB_URI")

client = MongoClient(uri, tlsCAFile=certifi.where())


# Create database and collection
db = client['demo']
collection = db['tasks']  

from langchain_huggingface import HuggingFaceEndpoint

# Define the LLM
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-xxl",
    task="text-generation",  # define the task
    temperature=0.5,
    model_kwargs={"max_length": 512},
    huggingfacehub_api_token=huggingface_api_key  
)





# Folder to store uploaded PDFs
UPLOAD_FOLDER = 'knowledge_base'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Function to check if the file is a PDF
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

#load pdf text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

#split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks





#Define the embeddings for storing the text chunks
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2")


    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Define the conversation chain
def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

pdf_docs = ["knowledge_base/MembersFAQ.pdf"]
#get pdf files
raw_text = get_pdf_text(pdf_docs)

# get the text chunks
text_chunks = get_text_chunks(raw_text)

# create vector store
vectorstore = get_vectorstore(text_chunks)

conversation_chain = get_conversation_chain(vectorstore)


# Route to display tasks and add a new task
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        task_content = request.form.get('content')
        if not task_content:
            logging.warning("Content is required")
            return jsonify({"status": 0, "msg": "Content is required"}), 400
        

        response = conversation_chain.run(task_content)
        if not response:
            response='Sorry, I could not find a response to your query. Please try again.'

        new_task = {
            "content": task_content,
            "response": response,
            "date_created": datetime.now()
        }
        logging.info(f"New task added: {new_task} and response: {response}")

        try:
            collection.insert_one(new_task)
            return redirect('/')
        except Exception as e:
            logging.error(f"There was an issue adding your task: {e}")  
            return f"There was an issue adding your task: {e}"

    else:
        tasks = list(collection.find().sort("date_created", -1).limit(5))  # Fetch latest 5 responses
        for task in tasks:
            task['_id'] = str(task['_id'])  
        return render_template('index.html', tasks=tasks)

@app.route('/admin/update_kb', methods=['GET', 'POST'])
def update_kb():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'pdf_file' not in request.files:
            logging.warning("No file part")
            return jsonify({"status": 0, "msg": "No file part"}), 400
        
        file = request.files['pdf_file']

        # If user does not select a file
        if file.filename == '':
            return jsonify({"status": 0, "msg": "No selected file"}), 400

        # If the file is allowed
        if file and allowed_file(file.filename):
            # Generate a safe filename and save the file
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Ensure the knowledge_base folder exists
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            
            file.save(file_path)
            logging.info(f"File {filename} successfully uploaded!")
            return jsonify({"status": 1, "msg": f"File {filename} successfully uploaded!"}), 200
        else:
            return jsonify({"status": 0, "msg": "Invalid file type. Only PDF allowed."}), 400

    
    return render_template('upload_kb.html')





# Route to delete a task
@app.route('/delete/<task_id>', methods=['GET'])
def delete(task_id):
    try:
        result = collection.delete_one({"_id": ObjectId(task_id)})
        if result.deleted_count == 0:
            logging.warning("Task not found")
            return jsonify({"status": 0, "msg": "Task not found"}), 404
        return redirect('/')
    except Exception as e:
        logging.error(f"There was a problem deleting the task: {e}")
        return f"There was a problem deleting the task: {e}"


# Route to update a task
@app.route('/update/<task_id>', methods=['GET', 'POST'])
def update(task_id):
    if request.method == 'POST':
        new_content = request.form.get('content')
        if not new_content:
            return jsonify({"status": 0, "msg": "Content is required"}), 400

        try:
            collection.update_one({"_id": ObjectId(task_id)}, {"$set": {"content": new_content}})
            return redirect('/')
        except Exception as e:
            logging.error(f"There was an issue updating your task: {e}")
            return f"There was an issue updating your task: {e}"

    else:
        task = collection.find_one({"_id": ObjectId(task_id)})
        if not task:
            return "Task not found", 404
        task['_id'] = str(task['_id'])  
        return render_template('update.html', task=task)


if __name__ == '__main__':
    app.run(debug=True, port=5001)  
