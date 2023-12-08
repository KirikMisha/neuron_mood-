import librosa
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Путь к ранее сохраненной модели
model_path = "voice_emotion_model.h5"

# Загрузка модели
loaded_model = load_model(model_path)

def extract_features(file_path):
    # Загрузка данных из аудиофайла
    audio, _ = librosa.load(file_path, sr=None)

    # Остальной код остается без изменений
    sample_rate = librosa.get_samplerate(file_path)
    y = np.array(audio)

    # Остальной код остается без изменений
    mfccs = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

def predict_emotion(file_path):
    # Извлечение признаков из тестового файла
    test_features = extract_features(file_path)

    # Прогнозирование эмоции с использованием обученной модели
    predicted_probs = loaded_model.predict(np.array([test_features]))

    # Получение предсказанного класса
    predicted_label = np.argmax(predicted_probs)

    # Создание и обучение нового LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(["joy", "sadness", "anger", "neutral"])

    # Обратное преобразование числовой метки в текстовую эмоцию
    predicted_emotion = label_encoder.inverse_transform([predicted_label])[0]

    return predicted_emotion

# Пример использования
test_file_path = "C:/Users/Михаил/Desktop/Все/Учеба/Запись1.wav"
result = predict_emotion(test_file_path)
print(f"Predicted Emotion: {result}")
