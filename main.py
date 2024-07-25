from pymilvus import AnnSearchRequest, connections, Collection, CollectionSchema, DataType, FieldSchema, model, RRFRanker, utility, WeightedRanker
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from fastapi import FastAPI
import uvicorn, json

# Connect to server
connections.connect(
    port="19530"
)

file_path = "documents.json"
app = FastAPI()

# Initialize collection with schemas
collection_name = "main_collection"
fields = [
    FieldSchema(name="primary", dtype=DataType.INT64, is_primary=True, auto_id = True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="dense", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR)
]
schema = CollectionSchema(fields=fields)

# Check if collection exists 
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

# Create collection
collection = Collection(collection_name, schema)

# Dictionary of index parameters for dense vectors
dense_index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128},
}

# Dictionary of index parameters for sparse vectors
sparse_index_params = {
    "metric_type": "IP",
    "index_type": "SPARSE_INVERTED_INDEX",
    "params": {"drop_ratio_build": 0.2},
}

# Create two indexes 
collection.create_index("dense", dense_index_params)
collection.create_index("sparse", sparse_index_params)

collection.load()

bge_m3_ef = BGEM3EmbeddingFunction(
    model_name='BAAI/bge-m3',
    device='cpu', 
    use_fp16=False 
)

@app.get("/insert")
async def create_item(item: dict):
    json_object = json.dumps(item["landmarks"], indent=4)
    with open("documents.json", "w") as outfile:
        outfile.write(json_object)

    with open('documents.json', 'r') as f:
        data = json.load(f)

    docs = [doc['face'] for doc in data]
    docs_embeddings = bge_m3_ef.encode_documents(docs)

    entities = [docs, docs_embeddings["dense"], docs_embeddings["sparse"]]
    collection.insert(entities)
    collection.flush()

    # Temporary vectors
    queries = ["1234"]

    query_embeddings = bge_m3_ef.encode_queries(queries)

    search_param_dense = {
        "data": query_embeddings["dense"],
        "anns_field": "dense",
        "param": {
            "metric_type": "L2"
        },
        "limit": 2
    }
    request_1 = AnnSearchRequest(**search_param_dense)

    search_param_sparse = {
        "data": query_embeddings["sparse"],
        "anns_field": "sparse",
        "param": {
            "metric_type": "IP"
        },
        "limit": 2
    }
    request_2 = AnnSearchRequest(**search_param_sparse)

    # Store requests in a list
    reqs = [request_1, request_2]

    # Weight the sparse and dense vectors evenly

    collection.load()
    result = collection.hybrid_search(
        reqs, 
        rerank = RRFRanker(),
        limit = 1, 
        output_fields=["text"]
    )

    print(result)

    return {"result":item["landmarks"],"message": "Text processed successfully"}

if __name__=="__main__":
    uvicorn.run(app)