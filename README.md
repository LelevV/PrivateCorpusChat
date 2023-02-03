# PrivateCorpusChat (Dutch)
A simple GPT-3 based command-line tool using the OpenAI API to ask a question about your (Dutch) private corpus. The base-prompts are for now only formulated in Dutch.

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
2. Make sure that you set your OpenAI API secret key to the 'OPENAI_API_KEY' environment variable ([link](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety)).
3. Fill the corpus//raw_files folder with your private corpus (for now only .txt files work)
4. Run private_corpus_chat.py 


## Example
### Ask a question about your private corpus (in this case the 'algemene deel van de WFT')
![Question](/images/intro.png "Question")

### The program retrieves top N chunks of text for the corpus (3 docs is the default)
![Retrieve](/images/retrieve.png "Retrieve")

### The program summarizes the chunks retrieved in the previous step
![Summarize](/images/summarize.png "Summarize")

### The program generates an answer by incorporating the summaries from the previous step into the final user prompt.
![Response](/images/summarize.png "Response")