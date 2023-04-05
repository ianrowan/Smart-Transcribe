from flask import Flask, jsonify, request
import spacy
import json
import scipy.sparse as sp
from nltk import WordNetLemmatizer
from smart_transcribe import gpt_keyword_query, nlp
import ast

app = Flask(__name__)

def parse_request(request):
    request_json = request.form.to_dict()
    return ast.literal_eval(list(request_json.keys())[0])

@app.route('/prompt', methods=['POST'])
def api():
    request_data = parse_request(request)
    # Parse the JSON data from the request body
    doc = nlp(request_data["text"])
    gpt_keyword_query(doc.ents, False)
    with open("definitions.json", "r") as f:
        response = json.load(f)
        return jsonify(response)

if __name__ == '__main__':
    app.run(debug=False)