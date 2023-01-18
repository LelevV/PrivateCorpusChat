import openai
import numpy as np
from numpy.linalg import norm
import json
import os


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
        json_name = f'{file}_embedding'
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


