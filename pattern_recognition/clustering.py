import time
import numpy as np
import pandas as pd
import optuna
from tslearn.clustering import KShape
import matplotlib.pylab as plt


def std(
    model: object,
    data: np.ndarray,
    y_pred: np.ndarray,
) -> list:
    """Рассчитывает среднеквадратичное отклонение от центроидов.

    Args:
        model (object): Модель кластеризации.
        data (np.ndarray): Стандартизированный датасет.
        y_pred (np.ndarray): Предсказанные кластеры.

    Returns:
        std_cl (list): Список СКО.
    """

    all_cl = model.cluster_centers_.shape[0]
    std_cl = []
    try:
        for i in range(all_cl):
            centr = model.cluster_centers_[i].ravel()
            all_ts = []
            for j in range(len(data[y_pred == i])):
                ts = data[y_pred == i][j].ravel()
                all_ts += [list(ts)]
            all_avg = np.array(pd.DataFrame(all_ts).mean(axis=0))
            diff = (centr - all_avg) ** 2
            std = (np.sum(diff) / (len(diff) - 1)) ** (1 / 2)
            std_cl.append(std)

        return std_cl

    except Exception as ex:
        
        print(f"Ошибка при расчете СКО: {ex}")
        print(f"Кластер {i}, Строка {j}")


def metric_std(
    model: object,
    data: np.ndarray,
    y_pred: np.ndarray,
    best_cl: int = 10,
) -> float:
    """Рассчитывает среднее среднеквадратичное отклонение от центроидов по {best_cl} 
    лучшим кластерам (зависит от числа кластеров, которое минимально хотим найти).

    Args:
        model (object): Модель кластеризации.
        data (np.ndarray): Стандартизированный датасет.
        y_pred (np.ndarray): Предсказанные кластеры.
        best_cl (int, optional): Количество кластеров для расчета метрики. Defaults to 10.

    Returns:
        mean_std (float): Среднее СКО на лучших кластерах.
    """
    list_std = std(model, data, y_pred)
    mean_std = np.mean(np.sort(list_std)[:best_cl])

    return mean_std


def indices_std(
    model: object,
    data: np.ndarray,
    y_pred: np.ndarray,
    best_cl: int = 10,
) -> list:
    """Рассчитывает среднее среднеквадратичное отклонение от центроидов по {best_cl} 
    лучшим кластерам (зависит от числа кластеров, которое минимально хотим найти).

    Args:
        model (object): Модель кластеризации.
        data (np.ndarray): Стандартизированный датасет.
        y_pred (np.ndarray): Предсказанные кластеры.
        best_cl (int, optional): Количество кластеров для расчета метрики. Defaults to 10.

    Returns:
        idx (list): Список индексов кластеров с лучшими СКО.
    """
    list_std = std(model, data, y_pred)
    idx = list(np.argsort(list_std)[:best_cl])

    return idx


def optimize(
    data: np.ndarray,
    all_clusters_min_max_step: tuple,
    best_clusters: int = 10,
    n_trials: int = 10,
    seed: int = 0,
) -> dict:
    """Подбирает параметры для датасета и модели.

    Args:
        data (np.ndarray): Стандартизированный датасет.
        
        all_clusters_min_max_step (tuple): Число кластеров, которые подбирает модель (ОТ, ДО, ШАГ).
        
        best_clusters (int): Число хороших кластеров, которые ищет модель. Defaults to 10.
        
        n_trials (int, optional): Количество проходов оптимизации. Defaults to 10.
        (если хочется пройти все варианты, то надо ставить хотя бы число равное кол-ву сочетаний *4, 
        но обычно хватает кол-ва сочетаний / 2)
        
        seed (int, optional): Значение для инициализации случайных чисел. Defaults to 0.
       
    Returns:
        best_params (dict): Подобранные параметры (общее число кластеров).

    """

    try:
        param_history=[]
        iteration=[1] # в objective можно извне передать только list
        def objective(trial):
            ts = time.time()

            n_clusters_min = all_clusters_min_max_step[0]
            n_clusters_max = all_clusters_min_max_step[1]
            n_clusters_step = all_clusters_min_max_step[2]

            n_clusters = trial.suggest_int("n_clusters", n_clusters_min, n_clusters_max,
                                           step=n_clusters_step)
            print(f'n_clusters = {n_clusters}:')

            # test for repeated params
            # we use tuple because that's easier to add more params this way
            if (n_clusters) in param_history:
                print(f'Итерация {np.sum(iteration)}/{n_trials} завершена'+ \
                      'в связи с повтором параметров')
                iteration.append(1)
                raise optuna.exceptions.TrialPruned()
            param_history.append((n_clusters))

            np.random.seed(seed)
            try:
                ks = KShape(
                    n_clusters=n_clusters,
                    max_iter=30,
                    n_init=2,
                    random_state=seed,
                    verbose=False,
                )

                y_pred = ks.fit_predict(data)

                score = metric_std(ks, data, y_pred, best_clusters)

                sec = time.time() - ts
                print(f'Итерация {np.sum(iteration)}/{n_trials} завершена, '+ \
                      f'обработка {n_clusters} кластеров заняла {np.round(sec/60,2)} минут')
                iteration.append(1)

                return score

            except EmptyClusterError:
                print(f'Итерация {np.sum(iteration)}/{n_trials} завершена из-за слишком '+ \
                      'большого числа заданных кластеров')
                iteration.append(1)
                raise optuna.exceptions.TrialPruned()

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # визуализация сходимости
        fig = optuna.visualization.plot_optimization_history(study)
        fig.show()

        print(f"Лучшие параметры: {study.best_params}")

        return study.best_params

    except Exception as ex:
        print(f"Ошибка при выборе параметров: {ex}")


class model_kshape(object):
    """Создает модель кластеризации KShape по заданным параметрам

    Methods:
        train(data): обучает модель по np.ndarray data
        predict(data): прогнозирует кластеры исходя из обученных центров
        visualize(n_best): выводит n лучших кластеров
        visualize_with_predict(df, data, n_best): выводит n лучших кластеров и все продолжения их
            временных рядов на {predict_period} вперед (период задается при формировании датасета)
        save(name): сохраняет в формате .hdf5 с указанным названием
    """

    def __init__(
        self,
        n_clusters: int,
        seed: int = 0,
    ):
        """Устанавливает все необходимые атрибуты для объекта model_kshape.

        Args:
            n_clusters (int): Число кластеров, которые подбирает модель.

            seed (int, optional): Значение для инициализации случайных чисел. Defaults to 0.
        """ 
        self.n_clusters = n_clusters
        self.seed = seed

    def train(self, data: np.ndarray):
        self.model = KShape(n_clusters=self.n_clusters, verbose=False, n_init=2,
                            random_state=self.seed)
        self.model.fit(data)

    def predict(self, data: np.ndarray):
        y_pred = self.model.predict(data)
        return y_pred

    def visualize(self, data: np.ndarray, n_best: int = 10):
        y_pred = self.model.predict(data)
        best_idx = indices_std(self.model, data, y_pred, n_best)
        list_std = std(self.model, data, y_pred)
        # длина временного ряда
        sz = data.shape[1]
        # число кластеров
        nc = len(best_idx)

        plt.figure(figsize=(12, nc * 4))
        for (yi, cl) in enumerate(best_idx):
            plt.subplot(nc, 1, 1 + yi)
            for xx in data[y_pred == cl]:
                plt.plot(xx.ravel(), "k-", alpha=0.2)
            plt.plot(self.model.cluster_centers_[cl].ravel(), "r-")
            plt.xlim(0, sz)
            plt.ylim(-4, 4)
            plt.title(
                f"Cluster {str(cl)} has {str(data[y_pred == cl].shape[0])} "+ \
                    "timeseries and std is %.2f"%list_std[cl]
            )

        plt.tight_layout()
        plt.show()

    def visualize_with_predict(self, df: pd.DataFrame, data: np.ndarray, n_best: int = 10):
        y_pred = self.model.predict(data)
        best_idx = indices_std(self.model, data, y_pred, n_best)
        list_std = std(self.model, data, y_pred)
        # длина временного ряда
        sz = df.shape[1]
        # число кластеров
        nc = len(best_idx)

        plt.figure(figsize=(12, nc * 4))
        for (yi, cl) in enumerate(best_idx):
            plt.subplot(nc, 1, 1 + yi)
            cl_df = df[y_pred == cl].reset_index(drop=True)
            for xx in range(len(cl_df)):
                plt.plot(cl_df.iloc[xx,:], "k-", alpha=0.2)
            plt.plot(self.model.cluster_centers_[cl].ravel(), "r-")
            plt.xlim(0, sz)
            plt.ylim(-4, 4)
            plt.title(
                f"Cluster {str(cl)} has {str(df[y_pred == cl].shape[0])} timeseries "+ \
                    "and std is %.2f"%list_std[cl]
            )

        plt.tight_layout()
        plt.show()

    def save(self, name: str):
        self.model.to_hdf5(str(name)+'.hdf5')
