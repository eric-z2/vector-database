from pymilvus import AnnSearchRequest, connections, Collection, CollectionSchema, DataType, FieldSchema, model, RRFRanker, utility, WeightedRanker
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from fastapi import FastAPI
import json, uvicorn

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

    # Load skin tone colours
    with open('skin_tone_reference.json', 'r') as f:
        data = json.load(f)

    # Embedding values into database
    docs = [doc['rgb'] for doc in data]
    docs_embeddings = bge_m3_ef.encode_documents(docs)

    entities = [docs, docs_embeddings["dense"], docs_embeddings["sparse"]]
    collection.insert(entities)
    collection.flush()

    # Writes and reads skin tones of parts of the face
    json_object = json.dumps(item["landmarks"], indent=4)
    with open('skin_tone_colour.json', 'w') as outfile:
        outfile.write(json_object)

    with open('skin_tone_colour.json', 'r') as f:
        query_data = json.load(f)

    q = query_data[0]
    first_rgb_value = (q["face"][0] + q["forehead"][0] + q["left_cheek"][0] + q["right_cheek"][0] + q["nose"][0] + q["jaw"][0])/6
    second_rgb_value = (q["face"][1] + q["forehead"][1] + q["left_cheek"][1] + q["right_cheek"][1] + q["nose"][1] + q["jaw"][1])/6
    third_rgb_value = (q["face"][2] + q["forehead"][2] + q["left_cheek"][2] + q["right_cheek"][2] + q["nose"][2] + q["jaw"][2])/6
    query_value = str(first_rgb_value) + ", " + str(second_rgb_value) + ", " + str(third_rgb_value)

    print(query_value)

    # Skin tone rgb values inserted into query
    queries = [query_value]

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

    collection.load()
    result = collection.hybrid_search(
        reqs, 
        rerank = RRFRanker(),
        limit = 1, 
        output_fields=["text"]
    )

    # Isolates for rgb value of the identified skin tone
    data_string = result[0][0]
    temp_str = str(data_string)
    split = temp_str.split(", ")
    connect = split[2] + ", " + split[3] + ", " + split[4]
    connect_split = connect.split(": ")
    temp2 = connect_split[2]
    first_strip = temp2.rstrip('}')
    second_strip = first_strip.strip("'")

    skin_tone = next(item["name"] for item in data if item["rgb"] == second_strip)
    print("Skin tone: " + skin_tone)

    dictionary_dump = {
        "Skin Tone" : skin_tone,
        "RGB value" : query_value
    }
    
    with open('output.json', 'w') as outfile:
        json.dump(dictionary_dump, outfile, indent=0)

    return skin_tone

if __name__=="__main__":
    uvicorn.run(app)