import os
from dotenv import load_dotenv
from index_files import index_to_document_store
from haystack import Pipeline
from haystack.nodes import PromptNode

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

document_store, retriever = index_to_document_store()

prompt_node = PromptNode(model_name_or_path="gpt-4", api_key=OPENAI_API_KEY, default_prompt_template="deepset/question-answering")

query_pipeline = Pipeline()
query_pipeline.add_node(component=retriever, name="retriever", inputs=["Query"])
query_pipeline.add_node(component=prompt_node, name="prompt_node", inputs=["retriever"])

result = query_pipeline.run(query="How do I install milvus?")
# def query(query):
#     query_pipeline.run(query=query)
