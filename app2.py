from flask import Flask,request,render_template,jsonify
app = Flask(__name__)
## app.run(debug=True)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/process',methods= ['POST'])

def process():

    q  = request.form['asked_question']

    #_______0_imports________#

    import pandas as pd
    import openai
    import numpy as np
    import pickle
    from transformers import GPT2TokenizerFast
    COMPLETIONS_MODEL = "text-davinci-002"
    openai.api_key = "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"

    #_______1_create df________#

    df = pd.read_csv('/home/ubuntu/workspace/openai_qa/openai-cookbook/saba/dmv_booklet - Sheet1.csv')
    df = df.set_index(["title", "heading"])

    #_______2_create context_embeddings________#

    MODEL_NAME = "davinci"

    DOC_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-doc-001"
    QUERY_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-query-001"

    def get_embedding(text: str, model: str) -> list[float]:
        result = openai.Embedding.create(
        model=model,
        input=text
        )
        return result["data"][0]["embedding"]

    def get_query_embedding(text: str) -> list[float]:
        return get_embedding(text, QUERY_EMBEDDINGS_MODEL)

    def load_embeddings(fname: str) -> dict[tuple[str, str], list[float]]:
        """
        Read the document embeddings and their keys from a CSV.
        
        fname is the path to a CSV with exactly these named columns: 
            "title", "heading", "0", "1", ... up to the length of the embedding vectors.
        """
        
        df = pd.read_csv(fname, header=0)
        max_dim = max([int(c) for c in df.columns if c != "title" and c != "heading"])
        return {
            (r.title, r.heading): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
        }

    context_embeddings = load_embeddings("/home/ubuntu/workspace/openai_qa/openai-cookbook/saba/context_embeddings_davinci.csv")

    #_______3_find the most similar document embeddings to the question embedding________#

    def vector_similarity(x: list[float], y: list[float]) -> float:
        """
        We could use cosine similarity or dot product to calculate the similarity between vectors.
        In practice, we have found it makes little difference. 
        """
        return np.dot(np.array(x), np.array(y))

    def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
        """
        Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
        to find the most relevant sections. 
        
        Return the list of document sections, sorted by relevance in descending order.
        """
        query_embedding = get_query_embedding(query)

        document_similarities = sorted([
            (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
        ], reverse=True)
        
        return document_similarities    

    MAX_SECTION_LEN = 1200
    SEPARATOR = "\n* "

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    separator_len = len(tokenizer.tokenize(SEPARATOR))

    def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
        """
        Fetch relevant          
        """
        most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
        print(len(most_relevant_document_sections))
        
        chosen_sections = []
        chosen_sections_len = 0
        chosen_sections_indexes = []
   
        for _, section_index in most_relevant_document_sections:

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
    
    COMPLETIONS_API_PARAMS = {
        # We use temperature of 0.0 because it gives the most predictable, factual answer.
        "temperature": 0.0,
        "max_tokens": 1200,
        "model": COMPLETIONS_MODEL,
    }

    def answer_query_with_context(
        query: str,
        df: pd.DataFrame,
        document_embeddings: dict[(str, str), np.array],
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

    output = answer_query_with_context(q, df, context_embeddings)
    print('+')
    # return jsonify({'error' : 'Missing data!'})
    return jsonify({'output':'Answer: ' + output})
    # return jsonify({'A:': result})
    
if __name__ =='__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)