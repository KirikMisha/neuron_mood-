import os
import librosa
import numpy as np
from keras import layers
from keras import models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras import regularizers
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

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

# Замените путь на папку, где у вас лежат голосовые файлы
data_path = "C:/Users/Михаил/Desktop/Все/Учеба/mood/"

# Предполагается, что у вас есть подпапки для разных эмоций (например, "joy", "sadness", "anger")
emotions = ["joy", "sadness", "anger", "neutral"]
for emotion in emotions:
    emotion_path = os.path.join(data_path, emotion)

    # Добавьте эту проверку
    if not os.path.exists(emotion_path):
        print(f"Папка {emotion_path} не существует.")
        continue

    for filename in os.listdir(emotion_path):
        filepath = os.path.join(emotion_path, filename)
        features = extract_features(filepath)
        data.append(features)
        labels.append(emotion)

# Преобразование меток в числовой формат
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Разделение данных на тренировочные и тестовые наборы
X_train, X_test, y_train, y_test = train_test_split(np.array(data), encoded_labels, test_size=0.2, random_state=42)

# Построение модели нейронной сети
model = models.Sequential()
model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(X_train.shape[1],)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(len(emotions), activation='softmax'))

# Уменьшение learning rate
model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Добавление ранней остановки
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Добавление сохранения лучшей модели
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# Обучение модели
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), callbacks=[early_stop, model_checkpoint])

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