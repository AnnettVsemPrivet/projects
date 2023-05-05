import numpy as np
import pandas as pd
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


def data_division(
    data: pd.DataFrame,
    column_value: str,
    period: int,
    step: int,
    rolling_period: int,
    predict_period: int,
    train_sample: int,
    column_time: str = 'begin',
    seed: int = 0,
) -> np.ndarray:
    """Разбивает данные строго внутри дня, по {period} минут, с шагом {step}.
    Потом достает {train_sample} рандомных временных рядов и стандартизирует
    каждый в отдельности.

    Функция используется отдельно для подбора параметров разбиения.

    Args:
        data (DataFrame): Датафрейм, содержащий столбцы {column_value} и {column_time}.
        
        column_value (str): Столбец, значения из которого обрабатываем.
        
        period (int): Количество свечей в паттерне.
        
        step (int): Сдвиг с которым создаются новые элементы выборки
        (для последовательного разбиения step=period).

        rolling_period (int): Размер окна для скользящего среднего
        (для отсутствия усреднения rolling_period=1).
        
        predict_period (int): Размер периода после основного временного ряда, 
        который стоит сохранять отдельно, чтобы позже проверить возможность 
        прогнозировать по конкретному кластеру. 
        (если не нужно, можно поставить predict_period=0)

        train_sample (int): Длина подвыборки из исходного датасета
        (без повтора).

        column_time (str, optional): Столбец с переменной времени
        (тип datetime или строка, которую можно преобразовать в datetime). Defaults to 'begin'.
        
        seed (int, optional): Значение для инициализации случайных чисел. Defaults to 0.

    Returns: array_cluster, array_predict, scaler
        array_cluster: Рандомная выборка из стандартизированных
        временных рядов заданного размера.
        array_predict: Та же рандомная выборка, но увеличенная на predict_period.
        scaler: Скалер, который потом пригодится чтобы отмасштабировать данные в обратную сторону.
    """

    try:
        data[column_time] = pd.to_datetime(data[column_time])
        data["new_col"] = data[column_value].rolling(rolling_period).mean()
        data.dropna(axis=0, inplace=True)
        df_new = data[[column_time, "new_col"]]
        df_grouped = df_new.groupby([df_new[column_time].dt.date])
        full_range = []

        for dates in list(df_grouped.groups.keys()):
            df_day = df_grouped.get_group(dates)
            my_list = list(df_day["new_col"])
            composite_list = [
                my_list[x : x + period + predict_period]
                for x in range(0, len(my_list) - period - predict_period + 1, step)
            ]
            full_range += composite_list

        new_df = pd.DataFrame(full_range)
        print(f"Размер всей выборки: {new_df.shape}")
    except Exception as ex:
        print(f"Ошибка при формировании датасета: {ex}")

    # Извлечение части датасета и нормализация.
    np.random.seed(seed)
    try:
        new_df = new_df.iloc[
            list(
                np.random.choice(
                    range(len(new_df)), train_sample, replace=False
                )
            )
        ]
        scaler = TimeSeriesScalerMeanVariance()
        df_norm = scaler.fit_transform(new_df.values)
        print(f"Размер итоговой выборки: {df_norm.shape}")

        array_predict = pd.DataFrame([x.flatten().tolist() for x in df_norm])
        array_cluster = np.array([x.flatten()[:period].reshape(period,1).tolist() for x in df_norm])

        return array_cluster, array_predict, scaler

    except Exception as ex:
        print(f"Ошибка при выборе и стандартизации: {ex}")
