import os
import pinecone
from dotenv import load_dotenv

# dotenv to load environment variables
load_dotenv()

def initialize_pinecone():
    """
    Initializes pinecone with API key and environment.
    """
    
    # Initialize pinecone with API key and environment
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
        environment=os.getenv("PINECONE_ENV")  # next to api key in console
    )

def create_index(index_name, dimension=1536, metric="cosine"):
    """
    Create an index with the given name, dimension, and metric.

    Parameters:
        index_name (str): The name of the index to be created.
        dimension (int, optional): The dimension of the index. Defaults to 384.
        metric (str, optional): The metric used for similarity search. Defaults to "cosine".

    Returns:
        None
    """
    # First, check if our index already exists. If it doesn't, we create it
    if index_name not in pinecone.list_indexes():
        # Create a new index
        pinecone.create_index(name=index_name, metric=metric, dimension=dimension)

if __name__ == "__main__":
    # Change this index name to the name you want to use
    index_name_to_create = os.getenv("PINECONE_INDEX_NAME") # Index name must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character
    dimension = 1536 # 1536 for OpenAI embedding, 768 for Vertex AI embedding
    metric = "cosine"
    
    # Initialize pinecone
    initialize_pinecone()

    # Create an index with the specified name
    create_index(index_name = index_name_to_create,
                 dimension = dimension,
                 metric = metric
    )
