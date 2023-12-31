import os
import librosa
import numpy as np
from keras import layers
from keras import models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Получаем путь к рабочей директории проекта
project_dir = os.path.dirname(os.path.abspath(__file__))

# Создаем временную директорию внутри проекта
temp_dir = os.path.join(project_dir, "temp_audio")
os.makedirs(temp_dir, exist_ok=True)

def extract_features(file_path):
    # Загружаем данные из аудиофайла
    audio, _ = librosa.load(file_path, sr=None)

    # Остальной код остается без изменений
    sample_rate = librosa.get_samplerate(file_path)
    y = np.array(audio)

    # Остальной код остается без изменений
    mfccs = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Загрузка данных и извлечение признаков
data = []
labels = []

# Новые эмоции и пути к файлам записей
new_emotions = ["anger", "disgust", "fear", "enthusiasm", "happiness", "neutral", "sadness"]
new_data_path = "C:/Users/Михаил/Desktop/Все/Учеба/train/"

# Преобразование меток в числовой формат
label_encoder = LabelEncoder()

# Извлечение признаков для новых эмоций
for emotion in new_emotions:
    emotion_path = os.path.join(new_data_path, emotion)

    # Добавьте эту проверку
    if not os.path.exists(emotion_path):
        print(f"Папка {emotion_path} не существует.")
        continue

    for filename in os.listdir(emotion_path):
        filepath = os.path.join(emotion_path, filename)
        features = extract_features(filepath)
        data.append(features)
        labels.append(emotion.split("_")[0])  # Используем первое слово в эмоции

# Преобразование меток в числовой формат
encoded_labels = label_encoder.fit_transform(labels)

# Разделение данных на тренировочные и тестовые наборы
X_train, X_test, y_train, y_test = train_test_split(np.array(data), encoded_labels, test_size=0.2, random_state=42)

# Построение модели нейронной сети
model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(len(new_emotions), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Сохранение модели (опционально)
model.save("voice_emotion_model.h5")

# Замените путь на ваш голосовой файл
test_file_path = "C:/Users/Михаил/Desktop/Все/Учеба/Запись1.wav"

# Извлечение признаков из тестового файла
test_features = extract_features(test_file_path)

# Прогнозирование эмоции с использованием обученной модели
predicted_probs = model.predict(np.array([test_features]))

# Получение предсказанного класса
predicted_label = np.argmax(predicted_probs)

# Обратное преобразование числовой метки в текстовую эмоцию
predicted_emotion = label_encoder.inverse_transform([predicted_label])[0]

print(f"Predicted Emotion: {predicted_emotion}")
