# predict.py

import pandas as pd
import joblib
import argparse
import warnings

# Игнорируем будущие предупреждения от scikit-learn, чтобы вывод был чище
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def make_prediction(input_data):
    """
    Загружает обученную модель и делает предсказание.
    
    Args:
        input_data (dict): Словарь с признаками для предсказания.
    
    Returns:
        tuple: (прогноз класса, вероятность паводка)
    """
    # --- 1. Загрузка модели ---
    model_path = 'logreg_flood_model.joblib'
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Ошибка: Файл модели '{model_path}' не найден.")
        print("Пожалуйста, запустите ноутбук, чтобы обучить и сохранить модель.")
        return None, None

    # --- 2. Подготовка данных ---
    features = [
        'season_avg_temp', 'temp_min', 'temp_max', 'season_total_precip', 
        'precip_max', 'season_total_snow', 'season_max_snowdepth', 
        'windspeed_mean', 'humidity_mean'
    ]
    
    df = pd.DataFrame([input_data], columns=features)
    
    # --- 3. Предсказание ---
    prediction_class = model.predict(df)[0]
    prediction_proba = model.predict_proba(df)[0][1] # Вероятность класса "1"
    
    return prediction_class, prediction_proba


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Предсказание паводков на основе погодных данных.")
    parser.add_argument("--snow_depth", type=float, required=True, help="Макс. глубина снега (season_max_snowdepth). Пример: 40.0")
    parser.add_argument("--total_snow", type=float, required=True, help="Общее кол-во снега (season_total_snow). Пример: 350.0")
    parser.add_argument("--max_precip", type=float, required=True, help="Макс. суточные осадки (precip_max). Пример: 15.0")
    
    args = parser.parse_args()

    sample_data = {
        'season_avg_temp': -8.0, 'temp_min': -30.0, 'temp_max': 18.0,
        'season_total_precip': 130.0, 'precip_max': args.max_precip,
        'season_total_snow': args.total_snow, 'season_max_snowdepth': args.snow_depth,
        'windspeed_mean': 25.0, 'humidity_mean': 80.0
    }
    
    print("\n--- Входные данные для прогноза ---")
    print(pd.DataFrame([sample_data]))
    
    pred_class, pred_proba = make_prediction(sample_data)
    
    if pred_class is not None:
        print("\n--- РЕЗУЛЬТАТ ПРОГНОЗА ---")
        print(f"Вероятность паводка: {pred_proba:.2%}")
        
        if pred_class == 1:
            print("Прогноз: ВЫСОКИЙ РИСК ПАВОДКА")
        else:
            print("Прогноз: Низкий риск паводка")