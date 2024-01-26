from flask import Flask, render_template, request
import pandas as pd
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Load data and model
Data_Frame = pd.read_csv("DataNeuron_Text_Similarity.csv")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def calculate_semantic_similarity(sentence1, sentence2):
    """
    Calculates the semantic similarity between two input sentences.
    """
    embeddings1 = model.encode(sentence1, convert_to_tensor=True)
    embeddings2 = model.encode(sentence2, convert_to_tensor=True)
    cosine_similarity = util.pytorch_cos_sim(embeddings1, embeddings2)[0][0].item()
    return cosine_similarity

def interpret_score(score):
    """
    This function takes in a score and returns a description based on its value.
    """
    if score >= 1:
        return "Highly Similar!"
    elif score >= 0:
        return "Highly Dissimilar"
    else:
        return "Invalid Score"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        sentence1 = request.form['sentence1']
        sentence2 = request.form['sentence2']
        similarity_score = calculate_semantic_similarity(sentence1, sentence2)
        interpretation = interpret_score(similarity_score)
        return render_template('index.html', similarity_score=similarity_score, interpretation=interpretation)

    return render_template('index.html')

#if __name__ == "__main__":
    #app.run(debug=True)

