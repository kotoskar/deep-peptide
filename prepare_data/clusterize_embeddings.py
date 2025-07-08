import torch
import numpy as np
from sklearn.cluster import KMeans

# СНАЧАЛА АКТИВИРУЙ ОКРУЖЕНИЕ!!!!!
# conda activate clustering_env

# Параметры
file_list_path = 'emb_list.txt'  # НА ВХОД ТОЛЬКО файл со списком имён файлов эмбеддингов
num_clusters = 52  # число кластеров k

# Читаем список файлов
with open(file_list_path, 'r') as f:
    file_names = [line.strip() for line in f if line.strip()]

# Загружаем эмбеддинги из файлов
print("Загрузка файлов...")
embeddings = []
for fn in file_names:
    emb = torch.load(fn)  # загружаем тензор из .pt файла
    #emb_np = emb.numpy() if isinstance(emb, torch.Tensor) else np.array(emb)
    emb_np = emb.detach().cpu().numpy() if isinstance(emb, torch.Tensor) else np.array(emb)
    embeddings.append(emb_np)

embeddings = np.vstack(embeddings)  # формируем матрицу (число файлов x размер эмбеддинга)

print(f"Количество файлов: {len(file_names)}")
print(f"Количество эмбеддингов: {embeddings.shape[0]}")

# Кластеризация k-means
print("Кластеризация...")
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(embeddings)

# Сохраняем имена файлов по кластерам
for cluster_idx in range(num_clusters):
    cluster_files = [file_names[i] for i, label in enumerate(labels) if label == cluster_idx]
    with open(f'cluster_{cluster_idx}.txt', 'w') as f:
        for fname in cluster_files:
            f.write(fname + '\n')

print("Кластеризация завершена. Файлы с именами кластеров эмбеддингов сохранены в cluster_*.txt")
