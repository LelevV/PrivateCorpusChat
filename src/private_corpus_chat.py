import openai
import numpy as np
import pandas as pd
from numpy.linalg import norm
import json
import os
import datetime


def get_txt_file_as_str(file):
    with open(file, 'r', encoding='utf-8') as f:
        string = f.read()
    return string 


def write_str_to_txt_file(file_string, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(file_string)


def get_json_file_as_dict(file):
    with open(file, 'r', encoding='utf-8') as f:
        d = json.load(f)
        return d


def write_dict_to_json_file(file, data):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, sort_keys=True, indent=2)


def chunk_text(text, chunk_word_size=500):
    # to avoid splitting within a word, create chunks on words list
    words = text.split(' ')
    chunks_words = [words[i:i+chunk_word_size] for i in range(0, len(words), chunk_word_size)]
    # transform back to strings 
    chuncks_text = [' '.join(words) for words in chunks_words]
    return chuncks_text 


def process_raw_files():
    raw_data_dir = '..//corpus//raw_files//'
    files = os.listdir(raw_data_dir)
    processed_dir = '..//corpus//processed//'
    for file in files:
        if file[-4:] == '.txt':
            # read file as str
            file_str = get_txt_file_as_str(raw_data_dir+file)
            # chunk str into substrings of 500 tokens
            chunks = chunk_text(file_str)
            # write chunks to files in processed folder
            n_chunks = len(chunks)
            for i, chunk in enumerate(chunks, start=1):
                chunk_name = f'{file[:-4]}__chunk_{i}_{n_chunks}.txt'
                write_str_to_txt_file(chunk, processed_dir+chunk_name) 


def get_embedding(text):
     # to avoid using chars that GPT doesnt like
    text = text.encode(encoding='ASCII', errors='ignore').decode()
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    embeddings = response['data'][0]['embedding']
    return embeddings


def get_cosine_sim(x, y):
    cosine_sim = np.dot(x,y)/(norm(x)*norm(y))
    return cosine_sim


def create_embedding_index():
    # write embeddings to json for all processed files 
    processed_dir = '..//corpus//processed//'
    embedding_index_dir = '..//corpus//embedding_index//'
    text_files = os.listdir(processed_dir)
    for file in text_files:
        text = get_txt_file_as_str(processed_dir+file)
        embedding = get_embedding(text)
        embedding_dict = {
            'creation_dt':str(datetime.datetime.now()),
            'source_file':file,
            'source_text':text,
            'embedding':embedding
        }
        json_name = f'{file[:-4]}_embedding.json'
        write_dict_to_json_file(embedding_index_dir+json_name, embedding_dict)


def gpt3_text_completion(prompt, model, max_tokens=60):
    # check if valid model
    gpt3_models = ['text-davinci-003', 'text-curie-001', 'text-babbage-001', 'text-ada-001']
    assert model in gpt3_models, 'Not a valid model!'
    # to avoid using chars that GPT doesnt like
    prompt = prompt.encode(encoding='ASCII', errors='ignore').decode()
    # prompt model via api call
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=0.7,
        max_tokens=max_tokens,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=1
    )
    text = response['choices'][0]['text']
    return text, response 



def log_prompt(prompt, prompt_type, add_embedding=False):

    log_dir = '..//logs//'
    dt_str = str(datetime.datetime.now())
    log_i = 0
    log_i = len(os.listdir(log_dir))
    json_name = f'log_entry__{prompt_type}__{log_i}.json'
    d = {
        'creation_dt':dt_str,
        'prompt_type':prompt_type, 
        'prompt':prompt,
        'file_name':json_name,
        }
    if add_embedding: # only add embedding if specified 
        d['embedding'] = get_embedding(prompt)
    log_file_name = log_dir+json_name
    write_dict_to_json_file(log_file_name, d)
    return log_file_name


def retrieve_top_n_simular_docs(prompt_log_file, corpus_embedding_dir, top_n):
    # load prompt data 
    prompt_dict = get_json_file_as_dict(prompt_log_file)
    prompt_embedding_vector = prompt_dict['embedding']

    # get query simularity for all files in corpus embedding dir
    simularities = [] # [doc_embedding_source_file, simularity]
    docs_embedding_files = os.listdir(corpus_embedding_dir)
    for doc_embedding_file in docs_embedding_files:
        # load doc data
        doc_embedding_dict = get_json_file_as_dict(corpus_embedding_dir+doc_embedding_file)
        doc_embedding_source_file = doc_embedding_dict['source_file']
        doc_embedding_vector = doc_embedding_dict['embedding']
        doc_embedding_text = doc_embedding_dict['source_text']
        # get simularity
        simularity = get_cosine_sim(prompt_embedding_vector, doc_embedding_vector)
        simularities.append([doc_embedding_source_file, simularity])
    
    # only retrieve top n simular docs 
    sim_df = pd.DataFrame(simularities, columns=['source_file', 'simularity'])
    sim_df = sim_df.sort_values('simularity', ascending=False)
    top_n_docs = list(sim_df['source_file'][:top_n])
    return top_n_docs


if __name__ == '__main__':

    TOP_N_RETRIEVAL = 3
    GPT3_MODEL = 'text-davinci-003'
    MAX_TOKENS = 300
    CORPUS_EMBEDDING_DIR = '..//corpus//embedding_index//'
    CORPUS_PROCESSED_DIR = '..//corpus//processed//'
    CONTEXT_PROMPT_FILE = '..//base_prompts//base_context_prompt_dutch.txt'

    # the main chat loop 
    while True:
        # ask for initial user prompt
        user_prompt = input('\n\nUSER: ')
        # log the prompt
        prompt_log_file = log_prompt(user_prompt, 'initial_prompt', add_embedding=True)
        # retrieve top N docs 
        relevant_docs = retrieve_top_n_simular_docs(prompt_log_file, CORPUS_EMBEDDING_DIR, TOP_N_RETRIEVAL)
        # read the docs as str
        relevant_docs_str = [get_txt_file_as_str(CORPUS_PROCESSED_DIR+file) for file in relevant_docs]
        relevant_docs_str = '\n'.join(relevant_docs_str) 
        # generate prompt with context
        context_prompt = get_txt_file_as_str(CONTEXT_PROMPT_FILE)
        context_prompt = context_prompt.replace('###USER_PROMPT###', user_prompt)
        context_prompt = context_prompt.replace('###CONTEXT###', relevant_docs_str)
        context_prompt_log_file = log_prompt(context_prompt, 'context_prompt')
        # gpt completion 
        text, response = gpt3_text_completion(context_prompt, GPT3_MODEL, max_tokens=MAX_TOKENS)
        response_prompt_log_file = log_prompt(text, 'response')
        print('\n\n\nBOT:\n', text)
       
