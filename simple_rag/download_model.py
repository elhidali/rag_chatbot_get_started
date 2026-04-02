import os
import urllib.request
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

files = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "modules.json",
    "1_Pooling/config.json",
    "model.safetensors"
]
base_url = "https://yellow-shadow-89d2.abdallah-elhidali.workers.dev/sentence-transformers/all-MiniLM-L6-v2/resolve/main/"

os.makedirs("all-MiniLM-L6-v2/1_Pooling", exist_ok=True)
for f in files:
    url = base_url + f
    dest = os.path.join("all-MiniLM-L6-v2", f)
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    print(f"Downloading {f}...")
    with urllib.request.urlopen(req) as response, open(dest, 'wb') as out_file:
        out_file.write(response.read())
print("Download complete.")
