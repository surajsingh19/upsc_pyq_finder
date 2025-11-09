'''UPSC Mains Retriever'''

# imports
import os
from google import genai
from google.genai import types
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pydantic import BaseModel, Field
from typing import List


class Output(BaseModel):
    out: List[int] = Field(
        description="A list of question indices (integers) ranked by relevance to the context, in descending order (most relevant first)."
    )


# clients
os.environ['GEMINI_API_KEY'] = os.getenv('api_key')
try:
    gemini_client = genai.Client()
except Exception as e:
    print("Error: Could not initialize Gemini Client.")
    print("Please ensure the GEMINI_API_KEY environment variable is set.")
    exit()


# read pickle tmp solution
with open('p_embedd.pkl','rb') as f:
    prelims_pyqs = pickle.load(f)

with open('m_embedd.pkl','rb') as f:
    mains_pyqs = pickle.load(f)


def retrieve(query, k, threshold, embeddings_db):
    '''--retriever--'''

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
    all_embeddings = [i['embedding'] for i in embeddings_db]
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

    return [embeddings_db[i] for i in top_k_indices]


def generate(context):

    system_instruction = '''### Role
You are an expert **UPSC Syllabus Analyst** and **Subject Matter Expert**. You possess deep knowledge of the UPSC CSE General Studies syllabus (Prelims and Mains) and are skilled in identifying core academic concepts within diverse texts.

### Objective
Your primary task is to analyze the input context and extract **key conceptual phrases** that accurately represent the **main topics or central concepts** relevant to the UPSC syllabus. These phrases will be used to generate highly effective search queries for Previous Year Questions (PYQs).

### Extraction Guidelines
1.  **Syllabus Relevance:** Phrases MUST be recognizable, core topics directly aligned with the official UPSC syllabus (e.g., "Fundamental Rights," "Fiscal Federalism," "Global Warming," "Montagu-Chelmsford Reforms").
2.  **Focus on Crux:** Extract topics that are **explicitly mentioned** OR are the **main conceptual crux** being referred to.
3.  **Avoid Peripheral Terms:** **Strictly avoid** extracting terms that are merely specific examples, temporary facts, incidental details, or illustrative words, rather than the core syllabus topic itself.
    * **Example Context:** "Implementation Gaps In Mid-Day Meal Scheme Exposed As Children Served Food On Scrap Paper."
    * **GOOD Phrase:** `Mid-Day Meal Scheme|Children Nutritional Security|Social Sector Schemes Implementations`
    * **BAD Phrases (Must Avoid):** `scrap paper tech|scrap paper industry|children served food` (These are non-conceptual details, not crux of the context and also UPSC topics.).
    * **Avoid too broad Terms(Must Avoid):** `Education System` is too broad to consider for above context. Be strict on this.
4.  **Query Quality:** Phrases must be concise, specific, and optimized for searching (i.e., "query-able").

## Output Format
You MUST output **only** a single string. All extracted phrases must be concatenated and separated by a single pipe `|` character. Do not include any preamble, explanation, or concluding remarks.

**Example Output:**
`Cooperative Federalism|Judicial Review|Basic Structure Doctrine|Monetary Policy Committee`
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


def rank_items(questions, context, phrases):

    only_questions = [i['que']['english'] for i in questions]

    system_instruction = '''### Role
You are an expert **UPSC PYQ Relevance Engine** and **Context-Based Ranking Specialist**. You are responsible for precisely matching Previous Year Questions (PYQs) to the core conceptual and factual content of a given input text.

---

### Objective
Your task is to analyze the provided **Input Context** and the list of **PYQs**. Your goal is to generate a final ranked list of PYQ indices based on a strict, two-tiered relevance analysis of the context.

---

### Ranking Guidelines
1.  **Primary Relevance (Highest Priority):** Rank questions primarily based on their direct conceptual match to the **crux or main UPSC-relevant theme** of the Input Context. A strong match to the central idea receives the highest priority.
2.  **Secondary Relevance (Tie-breaker):** If two questions have a similar match to the main theme, use the **specific factual data, granular details, or sub-topics** mentioned within the Input Context as a tie-breaker. Questions that align with these finer points of the text should be ranked higher.
3.  **Strict Ordering:** Rank all selected questions in **descending order of relevance** (Most Relevant = First Index).
4.  **Exclusion Criteria (Mandatory Filter):** You are required to perform a **stringent filter** and **omit (not include)** any question index from the final output list if, based on your UPSC subject matter expertise, the question is deemed irrelevant or too peripheral to the core concepts and themes represented in the **Input Context**. Only truly relevant PYQs should pass this filter.

---

## Output Format
You MUST output **only** a single string representing a Python-style list of integers.
* The integers MUST correspond to the original **1-based index** of the questions as provided in the user prompt.
* The list must be strictly ordered by descending relevance.
* Do not include any text, explanation, or preamble.

**Example Output:**
If Question at index 4 is the most relevant, followed by 1, and questions 3 and 2 were excluded for being irrelevant, the output is:
`[4, 1]`
'''

    user_prompt = f'''# Below are the inputs to process

## Input Context
{context}


---

## Question List (PYQs)
{only_questions}

---
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

    filtered_questions = [questions[i-1] for i in x.model_dump()['out']]
    return filtered_questions


############# Mains #################

def mains_run(context, k, threshold):

    phrases = generate(context)

    mains_pyqs_filtered = [i for i in mains_pyqs if i['subject']!='Essay']

    all_questions = []
    if phrases:
        for i in phrases.split('|'):
            questions = retrieve(i, k, threshold, mains_pyqs_filtered)
            all_questions.extend(questions)
    else:
        phrases = ''
        all_questions.extend(retrieve(context, k, threshold, mains_pyqs_filtered))

    ordered_questions = rank_items(all_questions, context, phrases)

    ordered_questions = ordered_questions[:k]

    # creating unique
    seen = set()
    unique_ordered_questions=[]
    for i in ordered_questions:
        if i['id'] not in seen:
            unique_ordered_questions.append(i)
            seen.add(i['id'])

    return unique_ordered_questions


############# Prelims #################

def prelims_run(context, k, threshold):

    phrases = generate(context)

    all_questions = []
    if phrases:
        for i in phrases.split('|'):
            questions = retrieve(i, k, threshold, prelims_pyqs)
            all_questions.extend(questions)
    else:
        phrases = ''
        all_questions.extend(retrieve(context, k, threshold, prelims_pyqs))

    ordered_questions = rank_items(all_questions, context, phrases)

    ordered_questions = ordered_questions[:k]

    # creating unique
    seen = set()
    unique_ordered_questions=[]
    for i in ordered_questions:
        if i['id'] not in seen:
            unique_ordered_questions.append(i)
            seen.add(i['id'])

    return unique_ordered_questions



#################### STREAMLIT ############################

import streamlit as st
st.set_page_config(layout="wide", page_title="UPSC PYQ Finder")
st.title("UPSC PYQ Finder")
st.markdown("**Usage Guide: **")
st.markdown("Context: The content around which you need UPSC PYQs.")
st.markdown("Que Count: Number of relevant PYQs to Fetch.")
st.markdown("Strictness: If irrelevant PYQs are coming, try increasing this parameter slightly.")

# Use columns to place the two approaches side-by-side
col1, col2 = st.columns(2)

# ====================================================================
# Approach 1 Section (Left Column)
# ====================================================================
with col1:
    st.header("Mains PYQ Finder")
    with st.form(key='mains'):
        
        # Input 1: Context (Text Area for larger input)
        context_1 = st.text_area(
            "**Context/Source Text**", 
            placeholder="Paste the document or text for which you want to find PYQs.", 
            height=150
        )
        
        # Input 2: Number of Questions (Number Input)
        num_questions_1 = st.number_input(
            "**Que Count**", 
            min_value=1, 
            max_value=20, 
            value=3, 
            step=1
        )
        
        # Input 3: Relevancy Checker Threshold (Slider)
        threshold_1 = st.slider(
            "**Strictness**", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.7, 
            step=0.01,
            help="1.0 is an exact match (highly relevant), 0.0 is random."
        )
        
        # Submit Button
        submit_button_1 = st.form_submit_button(label='Submit')
        
        # Logic to run the function when the button is clicked
        if submit_button_1:
            if context_1:
                output = mains_run(context_1, num_questions_1, threshold_1)
                st.write("### Output:")
                st.markdown("---")
                if output:
                    for i, question in enumerate(output, 1):
                        xx = f"Que {i}.  {question['que']['english']}"
                        st.markdown(xx)  
                        st.markdown("\n")
                        st.markdown("---")
                else:
                    st.write("No questions were found with provided settings or context.")
            else:
                st.error("Please provide **Context** for Mains PYQs.")

# Horizontal line to visually separate the input columns from potential output
st.markdown("---") 

# ====================================================================
# Approach 2 Section (Right Column)
# ====================================================================
with col2:
    st.header("Prelims PYQ Finder")
    # Use st.form to group the inputs and button
    with st.form(key='prelims'):
        
        # Input 1: Context (Text Area)
        context_2 = st.text_area(
            "**Context/Source Text**", 
            placeholder="Paste the document or text for which you want to find PYQs.", 
            height=150
        )
        
        # Input 2: Number of Questions (Number Input)
        num_questions_2 = st.number_input(
            "**Que Count**", 
            min_value=1, 
            max_value=20, 
            value=3, 
            step=1
        )
        
        # Input 3: Relevancy Checker Threshold (Slider)
        threshold_2 = st.slider(
            "**Strictness**", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.8, 
            step=0.01,
            help="1.0 is an exact match (highly relevant), 0.0 is random."
        )
        
        # Submit Button
        submit_button_2 = st.form_submit_button(label='Submit')
        
        # Logic to run the function when the button is clicked
        if submit_button_2:
            if context_2:
                output = prelims_run(context_2, num_questions_2, threshold_2)
                st.write("### Output:")
                st.markdown("---")
                if output:
                    for i, item in enumerate(output, 1):
                        question_data = item.get('que', {})
                        question_text = question_data.get('english', 'No question text found.')
                        st.markdown(f"**Que {i}. {question_text}**")
                        options = item.get('options', {})
                        if isinstance(options, dict) and options:
                            option_list = [f"- **{k}**: {v}" for k,v in options['english'].items()]
                            options_markdown = "\n".join(option_list)
                            st.markdown(options_markdown)  
                        st.markdown("\n\n")
                        st.markdown("---") 
                else:
                    st.write("No questions were found with provided settings or context.")
            else:
                st.error("Please provide **Context** for Prelims PYQs.")
