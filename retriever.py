'''UPSC Mains Retriever'''

# imports
import os
from google import genai
from google.genai import types
import numpy as np
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
import psycopg2
import psycopg2.extras
import pickle

# config
DB_CONFIG = {
    "host": os.getenv('host'),
    "database": os.getenv('db'),
    "user": os.getenv('user'),
    "password": os.getenv('pwd'),
    "port": os.getenv('port')
}


# clients
os.environ['GEMINI_API_KEY'] = os.getenv('api_key')
try:
    gemini_client = genai.Client()
except Exception as e:
    print("Error: Could not initialize Gemini Client.")
    print("Please ensure the GEMINI_API_KEY environment variable is set.")
    exit()

# postgress db
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
cur.execute("SELECT question_id, question, embedding FROM pyqs WHERE embedding IS NOT NULL")
pyq_rows = cur.fetchall()


# read pickle tmp solution
with open('pyq_rows_new.pkl','rb') as f:
    pyq_rows_new = pickle.load(f)

def retrieve(query, k, threshold):
    '''--retriever--'''
    # # main all data list
    # cur.execute("SELECT question_id, question, embedding FROM pyqs WHERE embedding IS NOT NULL")
    # pyq_rows = cur.fetchall()

    # convert query to embeddings
    result = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY",
                                        output_dimensionality=768 ))
    query_embedding = result.embeddings[0].values
    embedding_values_np = np.array(query_embedding)
    input_norm = np.linalg.norm(embedding_values_np)
    input_norm = input_norm if input_norm != 0 else 1
    normed_embedding = embedding_values_np / input_norm


    ## get pyq embeddings
    # all_embeddings = [d[2] for d in pyq_rows]
    all_embeddings = pyq_rows_new
    embeddings_matrix = np.array(all_embeddings)
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1 
    normalized_matrix = embeddings_matrix / norms

    ## check similarity
    similarity_scores = cosine_similarity(normalized_matrix, normed_embedding.reshape(1, -1)).flatten()
    # sorted_indices = np.argsort(similarity_scores)[::-1]
    
    # # Return the top k indices
    # top_k_indices = sorted_indices[:k].tolist()

    ##################
    threshold_indices = np.where(similarity_scores >= threshold)[0]
    filtered_scores = similarity_scores[threshold_indices]
    sorted_filtered_indices = np.argsort(filtered_scores)[::-1]
    final_sorted_indices = threshold_indices[sorted_filtered_indices]
    top_k_indices = final_sorted_indices[:k].tolist()
    ######################

    return [pyq_rows[i][1]['english'] for i in top_k_indices]


def generate(context):

    system_instruction = '''You are an expert **UPSC Syllabus Analyst** and **Subject Matter Expert**. You are highly skilled at extracting key conceptual phrases from text.

Your task is to analyze the input context and extract **key query phrases**. These phrases must represent topics that are **either mentioned explicitly or referred to conceptually** within the text AND are **directly relevant to the official UPSC syllabus/PYQs**.

The goal is to generate query terms that can be used to find relevant Previous Year Questions (PYQs).

## Output Format
You MUST output a single string. All extracted phrases must be concatenated and separated by a `|` character.
'''

    user_prompt = f'''## Input Context
    {context}

Note: No need to try to parse/generate phrase exactly from mentioned UPSC syllabus terms.
'''

    generation_config = types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=65000,
            system_instruction=system_instruction
        )

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[user_prompt],
        config=generation_config,
    )


    return response.text


from pydantic import BaseModel, Field
from typing import List
class Output(BaseModel):
    out: List[int] = Field(
        description="A list of question indices (integers) ranked by relevance to the context, in descending order (most relevant first)."
    )

def rank_items(questions, context):

    system_instruction = '''# You are an expert **UPSC PYQ Relevance Engine** and **Contextual Ranking Specialist**. Your primary function is to determine the precise relevance of a set of Previous Year Questions (PYQs) to a given input context.

Your task is to:
1.  Analyze the conceptual and factual overlap between the provided input context and each question in the question list.
2.  **Rank the questions** strictly based on their relevance to the context, where **the most relevant question receives the highest rank (first position)**.
3.  The final output must be **only** the ordered indices of the questions.
4.  You are more than welcomed to miss the questions which seems not relevant to the context as per your expertise.

## Output Format
You MUST output **only** a single Python-style list of integers. The integers must correspond to the original 1-based index (or 0-based, choose one and be consistent—**let's use the provided index from the user prompt for consistency**). Do not include any text, explanation, or preamble.

**Example Output:**
If Question at original index 4 is the most relevant, followed by 1, 3, and then 2, the output is:
`[4, 1, 3, 2]`
'''

    user_prompt = f'''# Please analyze the following context and the list of UPSC Previous Year Questions (PYQs).

## Input Context
{context}

---

## Question List (PYQs)
{questions}

---

**Task:** Rank the PYQs based on their conceptual relevance to the **Input Context**. Provide the indices of the questions in descending order of relevance.
'''

    generation_config = types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=65000,
            system_instruction=system_instruction,
            response_schema=Output,
            response_mime_type="application/json",
        )

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[user_prompt],
        config=generation_config,
    )

    x: Output = response.parsed

    print(x)

    filtered_questions = [questions[i-1] for i in x.model_dump()['out']]
    return filtered_questions


def main_2(context, k, threshold):

    phrases = generate(context)

    print(phrases)
    print('----------')

    all_questions = []
    if phrases:
        for i in phrases.split('|'):
            questions = retrieve(i, 3, threshold)
            all_questions.extend(questions)
    else:
        all_questions.extend(retrieve(context, k))

    all_questions = all_questions[:k]

    print(all_questions)
    print('----------')

    ordered_questions = rank_items(all_questions, context)

    return ordered_questions


def main_1(context, k, threshold):

    all_questions = retrieve(context, k, threshold)

    return all_questions

#################### STREAMLIT ############################

import streamlit as st

# --- Configuration and Setup ---
st.set_page_config(layout="wide", page_title="UPSC PYQ Finder")

# --- Streamlit App Layout ---

st.title("UPSC PYQ Finder 💡")
st.markdown("Use the boxes below to test and compare **Approach 1** and **Approach 2** for question generation.")

# Use columns to place the two approaches side-by-side
col1, col2 = st.columns(2)

# ====================================================================
# Approach 1 Section (Left Column)
# ====================================================================
with col1:
    st.header("### ⚙️ Approach 1")
    # Use st.form to group the inputs and button
    with st.form(key='approach_1_form'):
        
        # Input 1: Context (Text Area for larger input)
        context_1 = st.text_area(
            "**Context/Source Text**", 
            placeholder="Paste the document or text for which you want to find PYQs.", 
            height=150
        )
        
        # Input 2: Number of Questions (Number Input)
        num_questions_1 = st.number_input(
            "**Number of PYQ to Find.**", 
            min_value=1, 
            max_value=20, 
            value=3, 
            step=1
        )
        
        # Input 3: Relevancy Checker Threshold (Slider)
        threshold_1 = st.slider(
            "**Relevancy Checker Threshold**", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.7, 
            step=0.01,
            help="1.0 is an exact match (highly relevant), 0.0 is random."
        )
        
        # Submit Button
        submit_button_1 = st.form_submit_button(label='Run Approach 1')
        
        # Logic to run the function when the button is clicked
        if submit_button_1:
            if context_1:
                output = main_1(context_1, num_questions_1, threshold_1)
                if output:
                    numbered_questions = [f"{i}. {question}" for i, question in enumerate(output, 1)]
                    list_markdown = "\n".join(numbered_questions)
                    st.write("### Questions :")
                    st.markdown(list_markdown)
                else:
                    st.write("### Questions:")
                    st.write("No questions were found with provided settings or context.")
            else:
                st.error("Please provide **Context** for Approach 1.")

# Horizontal line to visually separate the input columns from potential output
st.markdown("---") 

# ====================================================================
# Approach 2 Section (Right Column)
# ====================================================================
with col2:
    st.header("### 🛠️ Approach 2: Advanced")
    # Use st.form to group the inputs and button
    with st.form(key='approach_2_form'):
        
        # Input 1: Context (Text Area)
        context_2 = st.text_area(
            "**Context/Source Text**", 
            placeholder="Paste the document or text for which you want to find PYQs.", 
            height=150
        )
        
        # Input 2: Number of Questions (Number Input)
        num_questions_2 = st.number_input(
            "**Number of PYQ to Find**", 
            min_value=1, 
            max_value=20, 
            value=3, 
            step=1
        )
        
        # Input 3: Relevancy Checker Threshold (Slider)
        threshold_2 = st.slider(
            "**Relevancy Checker Threshold**", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.8, 
            step=0.01,
            help="1.0 is an exact match (highly relevant), 0.0 is random."
        )
        
        # Submit Button
        submit_button_2 = st.form_submit_button(label='Run Approach 2')
        
        # Logic to run the function when the button is clicked
        if submit_button_2:
            if context_2:
                output = main_1(context_1, num_questions_1, threshold_1)
                if output:
                    numbered_questions = [f"{i}. {question}" for i, question in enumerate(output, 1)]
                    list_markdown = "\n".join(numbered_questions)
                    st.write("### Questions :")
                    st.markdown(list_markdown)
                else:
                    st.write("### Questions:")
                    st.write("No questions were found with provided settings or context.")
            else:
                st.error("Please provide **Context** for Approach 2.")