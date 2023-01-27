import openai
import numpy as np
import pandas as pd
from numpy.linalg import norm
import json
import pyfiglet
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
    """Process the .txt files in the raw_files folder to workable chunks."""
    raw_data_dir = '..//corpus//raw_files//'
    raw_files = os.listdir(raw_data_dir)
    processed_dir = '..//corpus//processed//'
    for raw_file in raw_files:
        if raw_file[-4:] == '.txt':
            # read file as str
            file_str = get_txt_file_as_str(raw_data_dir+raw_file)
            # chunk str into substrings of 500 tokens
            chunks = chunk_text(file_str)
            # write chunks to files in processed folder
            n_chunks = len(chunks)
            for i, chunk in enumerate(chunks, start=1):
                chunk_name = f'{raw_file[:-4]}__chunk_{i}_{n_chunks}.txt'
                write_str_to_txt_file(chunk, processed_dir+chunk_name) 


def get_embedding(text):
    """Generate embedding for text using OpenAI ada api"""
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
    """write embeddings to json for all processed files"""
    # TODO: implement rate limit (default OpenAI is 60/min)
    processed_dir = '..//corpus//processed//'
    embedding_index_dir = '..//corpus//embedding_index//'
    text_files = os.listdir(processed_dir)
    embedding_files = os.listdir(embedding_index_dir)
    for file in text_files:
        # check if files were already embedded
        json_name = f'{file[:-4]}_embedding.json'
        if json_name in embedding_files:
            # embedding file already exists
            continue 
        text = get_txt_file_as_str(processed_dir+file)
        embedding = get_embedding(text)
        embedding_dict = {
            'creation_dt':str(datetime.datetime.now()),
            'source_file':file,
            'source_text':text,
            'embedding':embedding
        }
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


def log_prompt(prompt, prompt_type, add_embedding=False, extra_info_dict=None):
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
    
    # add extra info to d, if given 
    if extra_info_dict:
        for k in extra_info_dict:
            d[k] = extra_info_dict[k]

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
    top_n_docs = list(sim_df['source_file'].iloc[:top_n])
    return top_n_docs


def get_query_summary(content, user_prompt, model, summary_prompt):
        """Summarize and retrieve txt based on given user_prompt and summarization prompt 
        and write result to summary dir
        """
        summary_prompt = get_txt_file_as_str(summary_prompt)
        summary_prompt = summary_prompt.replace('###USER_PROMPT###', user_prompt)
        summary_prompt = summary_prompt.replace('###CONTENT###', content)
        # log prompt 
        summary_prompt_log_file = log_prompt(summary_prompt, 'summary_prompt')
        # completion
        summary, response = gpt3_text_completion(summary_prompt, model, max_tokens=300)
        return summary


def list_files(startpath):
    """To print all files in a directory"""
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))


def ask_question():
    ### For now only one question:
    
    logo_banner = pyfiglet.figlet_format('Private Corpus Search')
    print(logo_banner)
    
    print(''' 
    --------------------------------------------------------------
    ------ Ask a question about your private text corpus!!! ------
    --------------------------------------------------------------     

    '''
    )

    print('Your private corpus contains the following files:')
    list_files(CORPUS_RAW_DIR)

    # ask for initial user prompt
    user_prompt = input('\n\n\nQuestion: ')
    print('\n\n')
    # log the prompt
    prompt_log_file = log_prompt(user_prompt, 'initial_prompt', add_embedding=True)
    # retrieve top N docs 
    relevant_docs = retrieve_top_n_simular_docs(prompt_log_file, CORPUS_EMBEDDING_DIR, TOP_N_RETRIEVAL)
    # read the docs as str
    relevant_docs_str = [get_txt_file_as_str(CORPUS_PROCESSED_DIR+file) for file in relevant_docs]
    relevant_docs_str = '\n'.join(relevant_docs_str) 

    ### Summarize if context is too long
    if len(relevant_docs_str) > MAX_CONTEXT_LENGTH:
        print('Context is too long; summarize...\n')
        # summarize relevant docs 
        print(f'Context summary:')
        relevant_docs_str = []
        for i, file in enumerate(relevant_docs, start=1):
            file_str = get_txt_file_as_str(CORPUS_PROCESSED_DIR+file)
            file_summary = get_query_summary(file_str, user_prompt, GPT3_MODEL, SUMMARY_PROMPT_FILE)
            context_prompt_log_file = log_prompt(file_summary, "summary_response", extra_info_dict={'user_prompt':user_prompt})
            relevant_docs_str.append(file_summary)
            print(f"\n({i}/{len(relevant_docs)}) [Notes of relevant (chunk) file] {file}:")
            print(file_summary.strip(), '/n')
        relevant_docs_str = '\n'.join(relevant_docs_str)
        
    else:
        print(f'Context: {relevant_docs_str} \n\n')

    ### Generate prompt with context
    context_prompt = get_txt_file_as_str(CONTEXT_PROMPT_FILE)
    context_prompt = context_prompt.replace('###USER_PROMPT###', user_prompt)
    context_prompt = context_prompt.replace('###CONTEXT###', relevant_docs_str)
    context_prompt_log_file = log_prompt(context_prompt, 'context_prompt')
    # gpt completion 
    text, response = gpt3_text_completion(context_prompt, GPT3_MODEL, max_tokens=MAX_TOKENS)
    response_prompt_log_file = log_prompt(text, 'final_response')

    print()
    response_banner = pyfiglet.figlet_format('Response')
    print(response_banner)
    print('Question:', user_prompt)
    print('\nAnswer:\n\n', text.strip())



if __name__ == '__main__':
    # global vars
    TOP_N_RETRIEVAL = 3
    GPT3_MODEL = 'text-davinci-003'
    MAX_TOKENS = 500
    MAX_CONTEXT_LENGTH = 3000
    
    # corpus files 
    CORPUS_EMBEDDING_DIR = '..//corpus//embedding_index//'
    CORPUS_PROCESSED_DIR = '..//corpus//processed//'
    CORPUS_RAW_DIR = '..//corpus//raw_files//'

    # prompt files 
    CONTEXT_PROMPT_FILE = '..//base_prompts//base_context_prompt_dutch.txt'
    SUMMARY_PROMPT_FILE = '..//base_prompts//base_summarize_prompt_dutch.txt'

    # check if raw_files is empty
    assert len(os.listdir(CORPUS_RAW_DIR)) > 0, f'No corpus present in the {CORPUS_RAW_DIR} folder!'

    # check if processed files are present, otherwise start processing 
    if len(os.listdir(CORPUS_PROCESSED_DIR)) == 0:
        print('\nNo processed files. Start processing...')
        process_raw_files()
        print('Done processing!\n')

    # check embedding_index is present, otherwise; generate 
    if len(os.listdir(CORPUS_EMBEDDING_DIR)) == 0:
        print('\nNo embedding index. Start creating index...')
        create_embedding_index()
        print('Done creating embedding index!\n')
    
    # check if embedding_index is complete, otherwise; generate remainging
    if len(os.listdir(CORPUS_EMBEDDING_DIR)) < len(os.listdir(CORPUS_PROCESSED_DIR)):
        print('\nEmbedding index not complete. Finish index...')
        create_embedding_index()
        print('Done creating embedding index!\n')

    # ask question
    ask_question()


    
       
