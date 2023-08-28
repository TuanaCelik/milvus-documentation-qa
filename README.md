# Milvus Documentation Search

This repo includes en example application that makes use of a Retrieval Augmented Generative architecture built with [Haystack](https://haystack.deepset.ai) to do search on the Milvus documentation.

## Install dependencies

```bash
pip install -r requirements.txt
```

## The Indexing Pipeline

An indexing pipeline is used to write documents to a database. In this example, we use the `MilvusDocumentStore` as our database for the RAG pipeline. So, we need to write the Milvus documentation into our Milvus database. For demonstration purposes, we use the `Crawler` component to crawl everything under https://milvus.io/docs/

Once you have Milvus running locally on localhost:19530, you can use the indexing pipeline as follows:

```bash
python scripts/index_files.py
```

## The RAG Pipeline

The RAG pipeline that we use is the following:

```python
from haystack import Pipeline
from haystack.nodes import EmbeddingRetriever, PromptNode, PromptTemplate, AnswerParser
from milvus_haystack import MilvusDocumentStore

document_store = MilvusDocumentStore()

retriever = EmbeddingRetriever(document_store=document_store, embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")
template = PromptTemplate(prompt="deepset/question-answering", output_parser=AnswerParser())
prompt_node = PromptNode(model_name_or_path="gpt-4", default_prompt_template=template, api_key=OPENAI_API_KEY, max_length=500)

query_pipeline = Pipeline()
query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
query_pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["Retriever"])
```

### To run it as a Streamlit App
```bash
streamlit run app.py
```

### To run it as a standalone script
```bash
python scripts/documentation_qa.py
```