import csv
from collections import defaultdict
from Bio import SeqIO

# Пути к файлам
csv_path = 'graphpart_assignments_homo.csv'
fasta_path = 'sequences_homo.fasta'

# 1. Считываем из csv соответствие AC → cluster
ac_to_cluster = {}
with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        ac = row['AC']
        cluster = row['cluster']
        # Приводим cluster к int (если нужно)
        cluster_id = int(float(cluster))
        ac_to_cluster[ac] = cluster_id

# 2. Загружаем все последовательности из fasta в словарь по ID
seqs = SeqIO.to_dict(SeqIO.parse(fasta_path, 'fasta'))

# 3. Группируем последовательности по кластерам
clusters = defaultdict(list)
for ac, cluster_id in ac_to_cluster.items():
    if ac in seqs:
        clusters[cluster_id].append(seqs[ac])
    else:
        print(f'Внимание: последовательность с ID {ac} не найдена в {fasta_path}')

# 4. Сохраняем последовательности каждого кластера в отдельный файл
for cluster_id, records in clusters.items():
    out_filename = f'cluster_{cluster_id}.txt'
    count = SeqIO.write(records, out_filename, 'fasta')
    print(f'Кластер {cluster_id}: сохранено {count} последовательностей в файл {out_filename}')
