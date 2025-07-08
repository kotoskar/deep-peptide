# ВКЛЮЧИ ОКРУЖЕНИЕ через командную строку !!!
# conda activate clustering_env
from Bio import SeqIO
import subprocess

# Пути к файлам
input_ids_path = 'protein_id_homo.txt'         # Файл со списком UniProt ID
fasta_input_path = '/home/user14/DeepPeptide/data/protein_sequences.fasta'   # Исходный FASTA-файл с последовательностями
fasta_output_path = 'sequences_homo.fasta'     # Куда сохранить найденные последовательности

# 1. Считываем UniProt ID из файла
with open(input_ids_path, 'r') as f:
    uniprot_ids = set(line.strip() for line in f if line.strip())

# 2. Загружаем последовательности из fasta и фильтруем по нужным ID
filtered_seqs = (
    rec for rec in SeqIO.parse(fasta_input_path, 'fasta')
    if rec.id in uniprot_ids
)

# 3. Сохраняем найденные последовательности в новый fasta-файл
count = SeqIO.write(filtered_seqs, fasta_output_path, 'fasta')
print(f'Сохранено последовательностей: {count}')

# 4. Запуск команды graphpart needle через subprocess
command = [
    'graphpart', 'needle',
    '--fasta-file', fasta_output_path,
    '--threshold', '0.3',
    '--out-file', 'graphpart_assignments_homo.csv',
    '--labels-name', 'label',
    '--partitions', '5',
    '--threads', '4'
]

print("Запуск команды GraphPart:")
try:
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    print("Вывод команды:")
    print(result.stdout)
    if result.stderr:
        print("Ошибки команды:", result.stderr)
except subprocess.CalledProcessError as e:
    print(f"Ошибка при выполнении команды: {e}")
    print(f"Вывод ошибки: {e.stderr}")
