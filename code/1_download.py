import requests
import os
import gzip

output_dir = "data/raw"
os.makedirs(output_dir, exist_ok=True)

URL = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_32/chembl_32_chemreps.txt.gz"
response = requests.get(URL)
open(output_dir + "/chembl_32_chemreps.txt.gz", "wb").write(response.content)

with gzip.open(output_dir + "/chembl_32_chemreps.txt.gz", 'rb') as f:
    file_content = f.read()

with open(output_dir + "/chembl_32_chemreps.txt", 'wb') as f:
    f.write(file_content)

os.remove(output_dir + "/chembl_32_chemreps.txt.gz")
