import sys
import os
from transformers import AutoModel, AutoTokenizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

print("Loading model manually...")
try:
    model_name = "/home/aelhidal/protos/rag_get_started/simple_rag/all-MiniLM-L6-v2"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    embedding = HuggingFaceEmbedding(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
    )
    print("Success!")
except Exception as e:
    import traceback
    traceback.print_exc()
