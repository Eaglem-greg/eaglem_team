import sys
import os

# Получаем путь к текущей директории, где лежит ml_server.py
current_dir = os.path.dirname(os.path.abspath(__file__))

# Добавляем путь к папке iTransformer в список путей поиска модулей
sys.path.append(os.path.join(current_dir, 'iTransformer'))

# Теперь импорт заработает
from model.iTransformer import Model as iTransformerModel
import torch
import joblib
import numpy as np
from flask import Flask, request, jsonify
import types
from model.iTransformer import Model as iTransformerModel
from collections import deque

app = Flask(__name__)

# --- Загрузка модели и метаданных ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
metadata = joblib.load("model_metadata.joblib")
configs = types.SimpleNamespace(**metadata['config'])

# Инициализируем архитектуру и загружаем веса
model = iTransformerModel(configs).to(device)
model.load_state_dict(torch.load("itransformer_weights.pth", map_location=device))
model.eval()

# Хранилище истории: {route_id: deque([status_vector1, status_vector2, ...], maxlen=96)}
route_history = {}
TARGET_MEAN = 68.42  # ПРИМЕР! Замени на свое
TARGET_STD = 67.15   # ПРИМЕР! Замени на свое

@app.route('/same_as_yours', methods=['POST'])
def predict():

    data = request.get_json()
    # route_id берем из корня
    route_id = data.get('route_id')
    
    # СТАТУСЫ берем из вложенного словаря 'statuses'
    statuses_dict = data.get('statuses', {})
    current_vector = [statuses_dict.get(f'status_{i}', 0)/200 for i in range(1, 9)]
    
    print(f"DEBUG: Processed Vector: {current_vector}") # Проверим, что тут не нули
    
    # Логика истории
    if route_id not in route_history:
        # Если маршрута нет в истории, заполняем текущим вектором
        route_history[route_id] = deque([current_vector] * configs.seq_len, maxlen=configs.seq_len)
    else:
        route_history[route_id].append(current_vector)
    
    # Превращаем в тензор
    input_array = np.array(route_history[route_id])
    input_tensor = torch.FloatTensor(input_array).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor, None, None, None)
        # iTransformer может возвращать [1, 10, 8], берем прогноз для первого канала
        prediction_scaled = output[0, 0, 0].item() 
        
    prediction_final = (prediction_scaled * TARGET_STD) + TARGET_MEAN
    
    print(f"DEBUG: Scaled output: {prediction_scaled}")
    print(f"DEBUG: Real volume prediction: {prediction_final}")
    
    # Возвращаем результат (ограничиваем снизу нулем, если всё равно минус)
    return jsonify({"predicted_volume": float(max(0, prediction_final))})

def preload_history(file_path):
    import pandas as pd
    global route_history
    
    if not os.path.exists(file_path):
        print(f"⚠️ Файл {file_path} не найден. История будет пустой.")
        return

    print(f"⌛ Загрузка истории из {file_path}...")
    # Читаем только нужные колонки для экономии памяти
    cols = ['route_id', 'timestamp'] + [f'status_{i}' for i in range(1, 9)]
    df = pd.read_parquet(file_path, columns=cols)
    print(cols)
    print(df.info)
    # Сортируем по времени, чтобы история была в правильном порядке
    df = df.sort_values(['route_id', 'timestamp'])
    
    status_cols = [f'status_{i}' for i in range(1, 9)]
    
    # Группируем по маршрутам и берем последние 96 строк для каждого
    for rid, group in df.groupby('route_id'):
        last_values = group[status_cols].tail(configs.seq_len).values.tolist()
        
        # Если данных в истории меньше 96, дополняем начало первым доступным значением
        if len(last_values) < configs.seq_len:
            padding = [last_values[0]] * (configs.seq_len - len(last_values))
            last_values = padding + last_values
            
        route_history[rid] = deque(last_values, maxlen=configs.seq_len)
    
    print(f"✅ История загружена для {len(route_history)} маршрутов.")

# Вставь это перед if __name__ == '__main__':
# Укажи путь к своему тренировочному файлу
TRAIN_DATA_PATH = "train_team_track.parquet" 
preload_history(TRAIN_DATA_PATH)

if __name__ == '__main__':
    print("🚀 iTransformer Inference Server running on port 5000...")
    app.run(host='0.0.0.0', port=5000)