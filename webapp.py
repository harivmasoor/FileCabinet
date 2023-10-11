from flask import Flask, request, render_template, redirect, url_for, flash 
from uuid import uuid4
import openai
import pinecone
import PyPDF2
from io import BytesIO
from dotenv import load_dotenv
import os

load_dotenv()



app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')


# OpenAI API Key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Pinecone Setup
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))
pinecone_index = pinecone.Index("pdf-embeddings")
pdf_data_store = {}
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        pdfs = request.files.getlist('pdf')
        # Generate unique IDs
        doc_ids = [str(uuid4()) for _ in range(len(pdfs))]

        # Extract text and get embeddings
        vectors_to_upsert = []
        for doc_id, pdf in zip(doc_ids, pdfs):
            reader = PyPDF2.PdfReader(BytesIO(pdf.read()))
            text = " ".join([page.extract_text() for page in reader.pages])
            pdf_data_store[doc_id] = text 
            embedding = get_openai_embedding(text)

            # Ensure embedding is a list of floats
            if isinstance(embedding, str):
                embedding = [float(val) for val in embedding.split(',')]  # Adjust as per the actual format

            vectors_to_upsert.append({
                'id': doc_id,
                'values': embedding,
                'metadata': {
                    'text_excerpt': text[:1000],  # Store the first 1000 characters as an example
                    # Or you could store the entire text if desired:
                    # 'full_text': text
                }
            })

        # Upsert to Pinecone
        pinecone_index.upsert(vectors=vectors_to_upsert)

        flash('PDFs uploaded and indexed!', 'success')
        return redirect(url_for('index'))

    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():

  query = request.form['query']
  embedding = get_openai_embedding(query)  

  results = pinecone_index.query(queries=[embedding], top_k=50)
  return render_template('results.html', results=results)

def get_openai_embedding(text):
  response = openai.Embedding.create(
    input=[text],
    model="text-embedding-ada-002", 
    return_embeddings=True
  )
  return response['data'][0]['embedding'] 

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['message']
    embedding = get_openai_embedding(user_message)
    
    results = pinecone_index.query(queries=[embedding], top_k=5, include_metadata=True)
    print("Pinecone Response:", results)

    if results['results'] and results['results'][0]['matches']:
        matched_metadata = results['results'][0]['matches'][0].get('metadata', {})
        matched_id = results['results'][0]['matches'][0]['id']
        matched_text = matched_metadata.get('text_excerpt', 'No matched text found')
    else:
        matched_id = None
        matched_text = "No matches found"

    response_data = {
        "response": f"Top matched documents: {matched_id}",
        "matched_text": matched_text
    }
    
    return render_template('results.html', results=response_data)


if __name__ == '__main__':
  app.run(debug=True)