import os
import nltk
import openai
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

# Get the most relevant answer using OpenAI API
def get_openai_answer(user_question):
    openai.api_key = "YOUR_OPENAI_API_KEY"  # Replace with your OpenAI API key
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=user_question,
        max_tokens=150,
        stop=["\n"],
        temperature=0.7
    )
    return response.choices[0].text.strip()

# Main function to analyze questions and answers
def analyze_qna(file_path, user_question, memory):
    questions, answers = read_qna_file(file_path)
    question_similarities = []

    user_question_words = tokenize_text(user_question)
    processed_user_question = ' '.join(preprocess_text(user_question_words))

    # Check if the user's question is similar to any previous question in memory
    for i, (mem_question, mem_answer) in enumerate(memory):
        similarity = calculate_similarity(processed_user_question, mem_question)
        question_similarities.append((i, similarity[0][0]))

    max_similarity = max(question_similarities, key=lambda x: x[1])

    if max_similarity[1] >= 0.75:
        # If the highest similarity is above 0.75, retrieve the previous response from memory
        prev_response = memory[max_similarity[0]][1]
        return prev_response
    else:
        # Otherwise, use OpenAI API to get the response
        new_response = get_openai_answer(user_question)
        # Store the user's question and the new response in memory
        memory.append((processed_user_question, new_response))
        return new_response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    user_question = request.form['question']
    file_path = "data.txt"  # Update with the path to your Q&A text file

    # Initialize or retrieve the memory list from session
    if 'memory' not in session:
        session['memory'] = []
    memory = session['memory']

    answer = analyze_qna(file_path, user_question, memory)

    # Store the updated memory back to session
    session['memory'] = memory

    return render_template('index.html', question=user_question, answer=answer)

if __name__ == "__main__":
    app.secret_key = 'your_secret_key'  # Replace with your own secret key
    app.run(debug=True)