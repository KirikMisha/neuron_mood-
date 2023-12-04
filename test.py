import numpy as np
from keras.models import load_model

from read_audio import label_encoder, extract_features


def predict_emotion(audio_file_path, model_path="voice_emotion_model.h5"):
    # Загрузка обученной модели
    model = load_model(model_path)

    # Извлечение признаков из тестового файла
    test_features = extract_features(audio_file_path)

    # Прогнозирование эмоции с использованием обученной модели
    predicted_probs = model.predict(np.array([test_features]))

    # Получение предсказанного класса
    predicted_label = np.argmax(predicted_probs)

    # Обратное преобразование числовой метки в текстовую эмоцию
    predicted_emotion = label_encoder.inverse_transform([predicted_label])[0]

    return predicted_emotion

# Замените путь на ваш голосовой файл
test_file_path = "C:/Users/Михаил/Desktop/Все/Учеба/Запись1.wav"

# Вызов функции для прогнозирования эмоции
result = predict_emotion(test_file_path)

# Вывод результата
print(f"Predicted Emotion: {result}")
