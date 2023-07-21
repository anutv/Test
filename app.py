import os
import nltk
from flask import Flask, render_template, request
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Download the NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the text file and parse it into questions and answers
def read_qna_file(file_path):
    questions = []
    answers = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

    current_question = None
    for line in lines:
        line = line.strip()
        if line.startswith("- "):
            current_question = line[2:]
        elif line.startswith("-- "):
            if current_question:
                questions.append(current_question)
                answers.append(line[3:])

    return questions, answers

# Tokenize text into sentences and words
def tokenize_text(text):
    sentences = sent_tokenize(text)
    words = [word.lower() for sentence in sentences for word in word_tokenize(sentence)]
    return words

# Remove stopwords and perform stemming
def preprocess_text(words):
    stop_words = set(stopwords.words('english'))
    porter_stemmer = PorterStemmer()

    return [porter_stemmer.stem(word) for word in words if word not in stop_words]

# Calculate the similarity between two strings
def calculate_similarity(text1, text2):
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])

# Main function to analyze questions and answers
def analyze_qna(file_path, question):
    questions, answers = read_qna_file(file_path)

    question_words = tokenize_text(question)
    processed_question = ' '.join(preprocess_text(question_words))

    similarities = []
    for answer in answers:
        answer_words = tokenize_text(answer)
        processed_answer = ' '.join(preprocess_text(answer_words))
        similarities.append((answer, calculate_similarity(processed_question, processed_answer)))

    # Sort answers by similarity
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

    # Return the most relevant answer
    return sorted_similarities[0][0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    user_question = request.form['question']
    file_path = "data.txt"
    answer = analyze_qna(file_path, user_question)
    return render_template('index.html', question=user_question, answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
