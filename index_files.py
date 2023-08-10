from haystack import Pipeline
from haystack.nodes import Crawler, PreProcessor, BM25Retriever
from milvus_haystack import MilvusDocumentStore

def index_to_document_store():
    document_store = MilvusDocumentStore(recreate_index=True, return_embedding=True, similarity="cosine")
    crawler = Crawler(urls=["https://milvus.io/docs/"], crawler_depth=1, overwrite_existing_files=True, output_dir="crawled_files")
    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=False,
        clean_header_footer=True,
        split_by="word",
        split_length=500,
        split_respect_sentence_boundary=True,
    )
    
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_node(component=crawler, name="crawler", inputs=['File'])
    indexing_pipeline.add_node(component=preprocessor, name="preprocessor", inputs=['crawler'])
    indexing_pipeline.add_node(component=document_store, name="document_store", inputs=['retriever'])

    indexing_pipeline.run()

    return document_store