# PrivateCorpusChat (Dutch)
A GPT based command-line tool using the OpenAI API to ask a question about your (Dutch) private corpus.

## How it works 
### Once 
1. Files from your private corpus are split up into chunks that are small enough to generate embeddings
2. A embedding index is created from the chunked files 
### Per question
1. A prompt/question is given by the user 
2. The user prompt is transformed to a embedding 
3. The user prompt embedding is used to retrieve the top N relevant documents from the embedding index 
4. The top N documents are summarized and added together to create a 'context' for the final prompt
5. The context and user prompt are combined into the final prompt to generate a answer to the question

## Usage 
1. Create conda env: conda env create -f environment.yml
2. Fill the corpus//raw_files folder with your private corpus (for now only .txt files work)
3. Run private_corpus_chat.py 

