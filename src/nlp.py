"""All NLP releted funcs."""

import datetime
import numpy as np
from numpy.linalg import norm
import openai
from src.utils import get_txt_file_as_str, os_list_dir_filetype, write_dict_to_json_file


def chunk_text(text: str, chunk_word_size=500) -> list:
    """Chunk text into chunks of size chunk_word_size."""
    # to avoid splitting within a word, create chunks on words list
    words = text.split(' ')
    chunks_words = [words[i:i+chunk_word_size] for i in range(0, len(words), chunk_word_size)]
    # transform back to strings
    chuncks_text = [' '.join(words) for words in chunks_words]
    return chuncks_text


def get_embedding(text: str) -> list:
    """Generate embedding for text using OpenAI ada api."""
     # to avoid using chars that GPT doesnt like
    text = text.encode(encoding='ASCII', errors='ignore').decode()
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    embeddings = response['data'][0]['embedding']
    return embeddings


def get_cosine_sim(x_vector, y_vector) -> float:
    """Get the cosine simularity of two vectors."""
    cosine_sim = np.dot(x_vector, y_vector)/(norm(x_vector)*norm(y_vector))
    return cosine_sim


def gpt3_text_completion(prompt: str, model: str, max_tokens=60) -> tuple:
    """complete the prompt using the OpenAI API using a GPT-3 model."""
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


def get_query_summary(content: str, user_prompt: str, model: str, summary_prompt: str) -> str:
    """
    Summarize and retrieve txt based on given user_prompt and summarization prompt 
    and write result to summary dir.
    """
    summary_prompt = get_txt_file_as_str(summary_prompt)
    summary_prompt = summary_prompt.replace('###USER_PROMPT###', user_prompt)
    summary_prompt = summary_prompt.replace('###CONTENT###', content)
    # completion
    summary, _ = gpt3_text_completion(summary_prompt, model, max_tokens=300)
    return summary