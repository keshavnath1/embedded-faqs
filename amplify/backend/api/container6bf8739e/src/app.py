import json
import openai
import tiktoken
import pandas as pd
import numpy as np
import os

from flask import Flask, jsonify, request
from transformers import GPT2TokenizerFast
app = Flask(__name__)

openai.api_key = "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"

COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"
embeddings_path = "./data/embeddings.json"
fname_path = "./data/selling_qa.csv"

COMPLETIONS_API_PARAMS = {
    "temperature": 0.0,
    "max_tokens": 300,
    "model": COMPLETIONS_MODEL,
}

MAX_SECTION_LEN = 500
SEPARATOR = "\n* "

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
separator_len = len(tokenizer.tokenize(SEPARATOR))

f"Context separator contains {separator_len} tokens"

@app.route("/")
def hello_world():
    return "<h1>Starter Flask App</h1>"


@app.route('/', methods=['POST'])
def get_answer():
    try:
        query = request.get_json(force=True)['query']
        document_embeddings, df = load_embeddings(embeddings_path, fname_path)
        df['tokens'] = df.apply(lambda row : num_tokens(row['context']), axis = 1)
        answer = answer_query_with_context(query, df, document_embeddings)
        return jsonify(Contents=answer)

    except Exception as error:
        raise error

def jsonKeys2int(x):
    if isinstance(x, dict):
        return {int(k):v for k,v in x.items()}
    return x

def get_embedding(text: str, model: str=EMBEDDING_MODEL):
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def load_embeddings(embeddings_path, fname_path):
    """
    Read the document embeddings and their keys from a CSV.
    
    """
    df = pd.read_csv(fname_path, header=0)
    with open(embeddings_path, 'r') as fp:
        embeddings = json.load(fp, object_hook=jsonKeys2int)
    return embeddings, df

def num_tokens(text: str, model: str = "gpt-4") -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def vector_similarity(x, y):
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query, contexts):
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        
        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))
    
    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    
    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"

def answer_query_with_context(
        query: str,
        df: pd.DataFrame,
        document_embeddings,
        show_prompt: bool = False
    ) -> str:

    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )
    
    if show_prompt:
        print(prompt)

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )

    return response["choices"][0]["text"].strip(" \n")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
