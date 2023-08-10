import os
from dotenv import load_dotenv
from haystack import Pipeline
from haystack.nodes import EmbeddingRetriever, PromptNode
from milvus_haystack import MilvusDocumentStore

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

document_store = MilvusDocumentStore()

retriever = EmbeddingRetriever(document_store=document_store, embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")
prompt_node = PromptNode(model_name_or_path="gpt-4", api_key=OPENAI_API_KEY, default_prompt_template="deepset/question-answering")

query_pipeline = Pipeline()
query_pipeline.add_node(component=retriever, name="retriever", inputs=["Query"])
query_pipeline.add_node(component=prompt_node, name="prompt_node", inputs=["retriever"])

result = query_pipeline.run(query="How do I install milvus?")
