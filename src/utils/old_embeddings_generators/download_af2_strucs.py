import requests
import os
import argparse
import pathlib
from tqdm import tqdm

# Ваш API-ключ AlphaFold (если требуется)
API_KEY = "AIzaSyCeurAJz7ZGjPQUtEaerUkBZ3TaBkXrY94"

def extract_ids_from_fasta(fasta_path):
    ids = []
    with open(fasta_path, 'r') as file:
        for line in file:
            if line.startswith('>'):
                # Удаляем символ '>' и обрезаем пробелы
                # Обычно ID — первое слово в заголовке
                id = line[1:].strip().split()[0]
                ids.append(id)
    return ids

def download_pdb(uniprot_id, output_dir):
    url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}?key={API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Ошибка запроса для {uniprot_id}: HTTP {response.status_code}")
        return
    data = response.json()
    if not data:
        print(f"Пустой ответ для {uniprot_id}")
        return
    # Берём первый объект (обычно один)
    prediction = data[0]
    pdb_url = prediction.get("pdbUrl")
    if not pdb_url:
        print(f"PDB URL не найден для {uniprot_id}")
        return
    pdb_response = requests.get(pdb_url)
    if pdb_response.status_code != 200:
        print(f"Ошибка загрузки PDB для {uniprot_id}: HTTP {pdb_response.status_code}")
        return
    pdb_path = os.path.join(output_dir, f"{uniprot_id}.pdb")
    with open(pdb_path, "wb") as f:
        f.write(pdb_response.content)
    # print(f"PDB структура для {uniprot_id} сохранена в {pdb_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('sequences_file',
                        type=pathlib.Path,
                        help='Fasta file with input sequences with uniprot IDs')
    parser.add_argument('out_dir',
                        type=pathlib.Path)
    
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ids = extract_ids_from_fasta(args.sequences_file)

    for id_ in tqdm(ids):
        if os.path.isfile(f'{args.out_dir}/{id_}.pdb'):
            # print("Already processed sequence")
            continue
        download_pdb(id_, args.out_dir)
