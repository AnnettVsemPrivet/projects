import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import time
import os
import fnmatch
import copy
import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import optuna
# мьютит вывод промежуточных результатов в optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


# функция для выбора подходящих для прогнозирования паттернов
def best_for_predict(
    dataset: pd.DataFrame,
    model: object,
    clusters: np.ndarray,
    best_cl_from_cl: list,
    predict_period: int,
    best_cl_for_pr: int
) -> np.ndarray:
    """Выдает список лучших кластеров для прогнозирования,
    исходя из отклонения от среднего.

    Args:
        dataset: pd.DataFrame - датафрейм, в котором объединены
        временные ряды по периодам паттерна + прогноза.

        model: object - модель кластеризации KShape.

        clusters: np.ndarray - предсказанные метки кластеров.

        best_cl_from_cl: list - число кластеров, по которым будем
        смотреть качество периода паттерна.

        predict_period: int - прогнозный период.

        best_cl_for_pr: int - число кластеров, которые хотим получить.

    Returns:
        cluster_inds: np.ndarray - отсортированные номера кластеров
        от наиболее однородного на прогнозном периоде до наименее.
    """

    # все кластеры
    all_cl = model.cluster_centers_.shape[0]

    # отбираем {best_cl_from_cl} лучших по периоду паттерна
    all_std = []
    for cl in range(all_cl):
        cl_df = dataset[clusters == cl].iloc[:, :-predict_period] \
            .reset_index(drop=True)
        centr = list(model.cluster_centers_[cl].ravel())
        all_std.append((((cl_df - centr) ** 2).sum() / (len(
            cl_df)) ** .5).mean())
    cluster_inds = np.array(range(all_cl))[np.argsort(
        all_std)[:best_cl_from_cl]]

    best_cl = copy.deepcopy(list(cluster_inds))

    # отбираем {best_cl_for_pr} лучших по периоду прогноза
    # делаем вывод что такие кластеры ведут себя наиболее предсказуемо
    all_std = []
    for cl in best_cl:
        cl_df = dataset[clusters == cl].iloc[:, -predict_period:] \
            .reset_index(drop=True)
        centr = list(cl_df.mean())
        all_std.append((((cl_df - centr)**2).sum() / (len(cl_df))**.5).mean())
    cluster_inds = np.array(best_cl)[np.argsort(all_std)[:best_cl_for_pr]]

    return cluster_inds


def split_n_bootstrap(
        predict_period: int,
        dataframe: pd.DataFrame,
        predicted_clusters: np.ndarray,
        cluster: int,
        test_percent: float,
        bootstrap_times: int,
        seed: int
):
    """Разбивает на X и y, потом на test и train.
    Для train применяет бутстрап.

    Args:
        predict_period: int - период прогнозирования,

        dataframe: pd.DataFrame - датафрейм, в котором объединены
        временные ряды по периодам паттерна + прогноза.

        predicted_clusters: np.ndarray - метки кластеров.

        cluster: int - кластер, строки по которому надо взять.

        test_percent: float - доля тестовой выборки.

        bootstrap_times: int - во сколько раз увеличить выборку,
        с помощью бутстрапа.

        seed: int - значение для инициализации случайных чисел.

    Returns:
        X_train, y_train, X_test, y_test: np.ndarray - выборки для обучения
        модели.
    """

    # отделяем датасет с кластером
    nn_df = dataframe[predicted_clusters == cluster].reset_index(drop=True)
    # отделяем тест перед бутстрэпом, т.к. в тесте не должно быть повторов
    nn_df, df_test = train_test_split(np.array(nn_df), test_size=test_percent,
                                      random_state=seed)
    nn_df = pd.DataFrame(nn_df).reset_index(drop=True)
    df_test = pd.DataFrame(df_test).reset_index(drop=True)

    # bootstrap part
    np.random.seed(seed)
    # набираем индексы с возвращением чтобы увеличить выборку
    lst = np.random.choice(range(len(nn_df)), len(nn_df) * bootstrap_times,
                           replace=True)
    # увеличиваем выборку
    nn_df = nn_df.iloc[lst, :].reset_index(drop=True)

    # splitting the data into test, validation, and train sets
    X_train, y_train = np.array(nn_df.iloc[:, :-predict_period]), \
        np.array(nn_df.iloc[:, -predict_period:])
    X_test, y_test = np.array(df_test.iloc[:, :-predict_period]), \
        np.array(df_test.iloc[:, -predict_period:])

    return X_train, y_train, X_test, y_test


# plot samples for manual check
def visualize_split(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
):
    """Выводит графики train и test выборок.

    Args:
        X_train: np.ndarray - паттерны для обучения.

        y_train: np.ndarray - прогнозные ряды для обучения.

        X_test: np.ndarray - паттерны для теста.

        y_test: np.ndarray - прогнозные ряды для теста.

    Returns:
        None - рисует два графика.
    """

    x_lim = X_train.shape[1] + y_train.shape[1]

    plt.figure(figsize=(10, 4))
    for i in range(X_train.shape[0]):
        label = '_nolegend_' if i != 1 else 'train sample'
        plt.plot(pd.concat([pd.DataFrame(X_train[i]),
                            pd.DataFrame(y_train[i])],
                           axis=0).reset_index(drop=True),
                 "-", alpha=0.2, c='b', label=label)
    plt.xlim(0, x_lim)
    plt.ylim(-4, 4)
    plt.axvline(x=X_train.shape[1], color='r',
                linestyle="--", label='prediction horizon')
    plt.title('training set')
    plt.legend(loc='upper left')
    plt.show()

    plt.figure(figsize=(10, 4))
    for i in range(X_test.shape[0]):
        label = '_nolegend_' if i != 1 else 'test sample'
        plt.plot(pd.concat([pd.DataFrame(X_test[i]),
                            pd.DataFrame(y_test[i])],
                           axis=0).reset_index(drop=True),
                 "-", alpha=0.2, c='c', label=label)
    plt.xlim(0, x_lim)
    plt.ylim(-4, 4)
    plt.axvline(x=X_test.shape[1], color='r',
                linestyle="--", label='prediction horizon')
    plt.title('testing set')
    plt.legend(loc='upper left')
    plt.show()


# error metrics
def quality_check(
        predictions: np.ndarray,
        y_test: np.ndarray
):
    """Выводит принты R2 и MSE.

    Args:
        predictions: np.ndarray - спрогнозированные ряды.

        y_test: np.ndarray - прогнозные ряды для теста.

    Returns:
        None - выводит 2 принта.
    """

    print('avg R2: %.2f, avg MSE: %.2f' %
          tuple(np.mean(pd.DataFrame([[r2_score(y_true, y_predict),
                                       mean_squared_error(y_true,
                                                          y_predict)]
                                      for (y_true, y_predict)
                                      in zip(y_test, predictions)]),
                        axis=0)))

    print('med R2: %.2f, med MSE: %.2f' %
          tuple(np.median(pd.DataFrame([[r2_score(y_true, y_predict),
                                         mean_squared_error(y_true,
                                                            y_predict)]
                                        for (y_true, y_predict)
                                        in zip(y_test, predictions)]),
                          axis=0)))


# random predictions from test sample
def random_predictions(predictions: np.ndarray,
                       pi: np.ndarray,
                       y_test: np.ndarray,
                       n: int
                       ):
    """Выводит графики {n} рандомных прогнозов, чтобы лучше рассмотреть.

    Args:
        predictions: np.ndarray - спрогнозированные ряды.

        pi: np.ndarray - дов. интервал (2 ско для каждой точки прогноза).

        y_test: np.ndarray - прогнозные ряды для теста.

        n: int - число рандомных прогнозов, которое вывести.

    Returns:
        None - выводит 1 график.
    """

    choices = list(np.random.choice(range(len(predictions)), n, replace=False))

    for i in choices:
        p = plt.plot(y_test[i])
        color = p[0].get_color()
        plt.plot(predictions[i], ':', color=color, label='_nolegend_')
        plt.fill_between(p[0].get_xdata(), (predictions[i] - pi),
                         (predictions[i] + pi), color=color,
                         alpha=.2, label='_nolegend_')
    plt.legend(choices, loc='upper left')
    plt.title('random test samples with predictions')
    plt.show()


# first 9 examples from test sample
def first_9_predictions(
        predictions: np.ndarray,
        pi: np.ndarray,
        y_test: np.ndarray,
        why_r2: int
):
    """Выводит графики 9 первых прогнозов с метриками.

    Args:
        predictions: np.ndarray - спрогнозированные ряды.

        pi: np.ndarray - дов. интервал (2 ско для каждой точки прогноза).

        y_test: np.ndarray - прогнозные ряды для теста.

        why_r2: int - если ставим 1, то по графикам становитя лучше понятно,
        почему R2 низкий.

    Returns:
        None - выводит 1 график.
    """

    fig, axs = plt.subplots(3, 3, figsize=(10, 10), sharey=not why_r2)
    for i in range(9):
        axs[int(i / 3), i % 3].plot(y_test[i], color='cyan')
        ax = axs[int(i / 3), i % 3].plot(predictions[i], '--', color='b')
        axs[int(i / 3), i % 3].fill_between(ax[0].get_xdata(),
                                            (predictions[i] - pi),
                                            (predictions[i] + pi),
                                            color='b', alpha=.2,
                                            label='_nolegend_')
        axs[int(i / 3), i % 3].title.set_text('R2: %.2f, MSE: %.3f' %
                                              (r2_score(y_test[i],
                                                        predictions[i]),
                                               mean_squared_error(y_test[i],
                                                                  predictions[
                                                                      i])))


class LSTM(nn.Module):
    "Класс нейросети с LSTM-слоями"

    def __init__(self,
                 output_size: int,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int
                 ):
        """Инициализация нейросети, выставляем нули на параметрах.

        Args:
            output_size: int - выходной слой (прогнозный период),

            input_size: int - входной слой (период паттерна),

            hidden_size: int - скрытый слой (размер LSTM-слоев),

            num_layers: int - число слоев LSTM

        Returns:
            model
        """

        super(LSTM, self).__init__()
        self.output_size = output_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)  # lstm
        self.fc_1 = nn.Linear(hidden_size, output_size)  # fully connected
        self.relu = nn.ReLU()  # activation function

        self.pi = nn.Parameter(torch.zeros(output_size))  # prediction interval

    def forward(self,
                x: torch.Tensor
                ):
        """Прямой проход нейросети (по всем функциям активации).

        Args:
            x: torch.Tensor - данные для обучения в виде тензора.

        Returns:
            результат последней функции активации и 2 СКО
            для доверительных интервалов.
        """

        h_0 = Variable(torch.zeros(self.num_layers, x.size(0),
                                   self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0),
                                   self.hidden_size))  # cell state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn[-1, :, :].view(-1, self.hidden_size)  # reshaping for next
        out = self.relu(hn)
        out = self.fc_1(out)  # Final Output
        return out, self.pi.data.numpy()

    def renew_pi(self,
                 outputs: torch.Tensor,
                 y_test_tensors: torch.Tensor
                 ):
        """Задаем отклонения на каждой точке для дов. интервалов в будущем.
        Надо запускать после обучения.

        Args:
            outputs: torch.Tensor - прогноз в виде тензора,

            y_test_tensors: torch.Tensor - прогнозные данные для теста
            в виде тензора.

        Returns:
            None.
        """

        predictions = outputs.data.numpy()
        values = y_test_tensors.data.numpy()
        pi = np.mean((predictions - values) ** 2, axis=0) ** .5 * 2
        self.pi = nn.Parameter(torch.from_numpy(pi).float())

    def load(self,
             path: str
             ):
        """Загружаем модель из местоположения файла.

        Args:
            path: str - весь путь к модели, включая название и тип файла.

        Returns:
            None.
        """
        self.load_state_dict(torch.load(path))

    def loaded_predict(self,
                       X_test: np.ndarray
                       ):
        """Прогнозируем используя загруженную модель.

        Args:
            X_test: np.ndarray - временные ряды паттернов для прогноза.

        Returns:
            predictions, pi - прогноз + дов. интервал по каждой точке прогноза.
        """
        X_test_tensors = Variable(torch.Tensor(X_test))
        X_test_tensors_final = torch.reshape(X_test_tensors,
                                             (X_test_tensors.shape[0], 1,
                                              X_test_tensors.shape[1]))
        train_predict, pi = self.forward(X_test_tensors_final)
        predictions = train_predict.data.numpy()
        return predictions, pi


class train_LSTM():
    "Класс для обучения нейросети."
    def __init__(self,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_test: np.ndarray,
                 y_test: np.ndarray,
                 lstm_model: object,
                 num_epochs: int,
                 learning_rate: float,
                 weight_decay: float,
                 if_print: int
                 ):
        """Инициализация и проведение обучения нейросети.

        Args:
            X_train: np.ndarray - паттерны для обучения.

            y_train: np.ndarray - прогнозные ряды для обучения.

            X_test: np.ndarray - паттерны для теста.

            y_test: np.ndarray - прогнозные ряды для теста.

            lstm_model: object - модель класса LSTM.

            num_epochs: int - число эпох (число проходов по всей выборке).

            learning_rate: float - коэффициент/скорость обучения.

            weight_decay: float - регуляризация весов.

            if_print: int - если 1, то будет показывать принты обучения.

        Returns:
            trained model.
        """

        X_train_tensors = Variable(torch.Tensor(X_train))
        X_test_tensors = Variable(torch.Tensor(X_test))
        y_train_tensors = Variable(torch.Tensor(y_train))
        y_test_tensors = Variable(torch.Tensor(y_test))
        self.model = lstm_model

        # reshaping to (batch_size, sequence_size, input_size)
        X_train_tensors_final = torch.reshape(X_train_tensors,
                                              (X_train_tensors.shape[0], 1,
                                               X_train_tensors.shape[1]))
        X_test_tensors_final = torch.reshape(X_test_tensors,
                                             (X_test_tensors.shape[0], 1,
                                              X_test_tensors.shape[1]))

        self.X_train = X_train_tensors_final
        self.y_train = y_train_tensors

        self.criterion = torch.nn.MSELoss(reduction="mean")
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate,
                                          weight_decay=weight_decay)

        # training
        train_loss = []
        test_loss = []
        for epoch in range(num_epochs):
            outputs, _ = self.model.forward(self.X_train)
            self.optimizer.zero_grad()  # calculate the gradient (setting to 0)
            loss = self.criterion(outputs, self.y_train)  # obtain loss func
            loss.backward()  # calculates the loss of the loss function
            self.optimizer.step()  # improve from loss, i.e backprop

            # print loss function results
            if if_print == 1 and epoch % 100 == 0:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

            train_loss.append(loss.item())
            outputs, _ = self.model.forward(X_test_tensors_final)
            loss = self.criterion(outputs, y_test_tensors)
            test_loss.append(loss.item())

        self.train_loss = train_loss
        self.test_loss = test_loss

        self.model.renew_pi(outputs, y_test_tensors)

    def plot_losses(self):
        """После обучения выдает график уменьшения ошибки
        на обучающей и тестовой выборках.
        """

        plt.style.use('ggplot')
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_loss, label="Training loss")
        plt.plot(self.test_loss, label="Testing loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()

    def score(self):
        "После обучения выдает актуальную ошибку по тестовой выборке."

        return self.test_loss[-1]

    # only forward (=prediction)
    def predict(self,
                X_test: np.ndarray
                ):
        """Прогнозируем используя обученную модель.

        Args:
            X_test: np.ndarray - временные ряды паттернов для прогноза.

        Returns:
            predictions, pi - прогноз + дов. интервал по каждой точке прогноза.
        """

        X_test_tensors = Variable(torch.Tensor(X_test))
        X_test_tensors_final = torch.reshape(X_test_tensors,
                                             (X_test_tensors.shape[0], 1,
                                              X_test_tensors.shape[1]))
        train_predict, pi = self.model.forward(X_test_tensors_final)
        predictions = train_predict.data.numpy()
        return predictions, pi

    def save(self,
             path: str
             ):
        """Сохраняем обученную модель в папке path.
        Можно также в конце дописать то, что надо добавить в название,
        например номер кластера.

        Args:
            path: str - путь к папке.

        Returns:
            None.
        """
        safe = self.model.state_dict()
        full_path = path + \
            'output_%s_input_%s_hidden_%s_layers_%s_ci_%.2f_time_%s.pth' % \
            (str(self.model.output_size), str(self.model.input_size),
             str(self.model.hidden_size), str(self.model.num_layers),
             np.mean(self.model.pi.data.numpy()),
             str(int(datetime.datetime.utcnow().timestamp())))
        torch.save(safe, full_path)


def optimize(X_train: np.ndarray,
             y_train: np.ndarray,
             X_test: np.ndarray,
             y_test: np.ndarray,
             output_size: int,
             input_size: int,
             num_epochs: int,
             params: dict,
             n_trials: int,
             seed: int
             ):
    """Функция поиска оптимальных параметров нейросети.

    Args:
        X_train: np.ndarray - паттерны для обучения.

        y_train: np.ndarray - прогнозные ряды для обучения.

        X_test: np.ndarray - паттерны для теста.

        y_test: np.ndarray - прогнозные ряды для теста.

        output_size: int - выходной слой (прогнозный период).

        input_size: int - входной слой (период паттерна).

        num_epochs: int - число эпох (число проходов по всей выборке).

        params: dict - параметры для оптимизации в виде словаря.
        Для каждого параметра нужно указать:
        min_value, max_value, step, denominator.
        Последнее - это то, на что делим каждое число для данного параметра.

        n_trials: int - число рандомных подборов параметров в рамках
        оптимизации. Т.к. дублирующиеся параметры засчитываются тоже,
        стоит выбирать число с запасом.

        seed: int - значение для инициализации случайных чисел.

    Returns:
        dict - лучшие параметры.
    """

    param_history = []
    iteration = [1]  # в objective можно извне передать только list

    def objective(trial):
        hidden_size = trial.suggest_int('hidden_size',
                                        params['hidden_size'][0],
                                        params['hidden_size'][1],
                                        step=params['hidden_size'][2]) * \
            params['hidden_size'][3]

        num_layers = trial.suggest_int('num_layers',
                                       params['num_layers'][0],
                                       params['num_layers'][1],
                                       step=params['num_layers'][2]) * \
            params['num_layers'][3]

        learning_rate = trial.suggest_int('learning_rate',
                                          params['learning_rate'][0],
                                          params['learning_rate'][1],
                                          step=params['learning_rate'][2]) * \
            params['learning_rate'][3]

        weight_decay = trial.suggest_int('weight_decay',
                                         params['weight_decay'][0],
                                         params['weight_decay'][1],
                                         step=params['weight_decay'][2]) * \
            params['weight_decay'][3]

        # test for repeated params - helps to save a lot of time!
        # because optuna can return to the same params many times
        # we use tuple because that's easier to add more params this way
        if (hidden_size, num_layers, learning_rate, weight_decay) in \
           param_history:
            # print(f'Итерация {np.sum(iteration)} / {n_trials} завершена'
            #      + 'в связи с повтором параметров')
            iteration.append(1)
            raise optuna.exceptions.TrialPruned()
        param_history.append((hidden_size, num_layers,
                              learning_rate, weight_decay))

        np.random.seed(seed)

        lstm1 = LSTM(output_size=output_size, input_size=input_size,
                     hidden_size=hidden_size, num_layers=num_layers)

        trained_model = train_LSTM(X_train, y_train, X_test, y_test,
                                   lstm1, num_epochs=num_epochs,
                                   learning_rate=learning_rate,
                                   weight_decay=weight_decay, if_print=0)

#         sec = time.time() - ts
#         print(f'Итерация {np.sum(iteration)} / {n_trials} завершена, '
#               + f'обработка {hidden_size} скрытых нейронов заняла '
#               + f'{np.round(sec/60,2)} минут')
        iteration.append(1)

        return trained_model.score()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

#     # визуализация сходимости
#     fig = optuna.visualization.plot_optimization_history(study)
#     fig.show()

    best_prm = copy.deepcopy(study.best_params)

    best_prm['hidden_size'] = best_prm['hidden_size'] * \
        params['hidden_size'][3]

    best_prm['num_layers'] = best_prm['num_layers'] * \
        params['num_layers'][3]

    best_prm['learning_rate'] = best_prm['learning_rate'] * \
        params['learning_rate'][3]

    best_prm['weight_decay'] = best_prm['weight_decay'] * \
        params['weight_decay'][3]

    return best_prm


def find_model(pattern: str,
               path: str
               ):
    "Находим в папке {path} все файлы подходящие под RegEx паттерн {pattern}"

    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def get_nn_models(pattern: str = 'NN_cluster*.pth',
                  path: str = 'models/'
                  ):
    """Находим в папке {path} все файлы подходящие под RegEx паттерн {pattern},
    создаем df со всеми параметрами этих моделей (берем из названия файла)."""
    first_let = len(path)

    all_models = pd.DataFrame(find_model(pattern, path), columns=['path'])

    all_models['cluster'] = \
        all_models.apply(lambda x: int(x['path'][first_let:-4].split('_')[2]), axis=1)

    all_models['output'] = \
        all_models.apply(lambda x: int(x['path'][first_let:-4].split('_')[4]), axis=1)

    all_models['input'] = \
        all_models.apply(lambda x: int(x['path'][first_let:-4].split('_')[6]), axis=1)

    all_models['hidden'] = \
        all_models.apply(lambda x: int(x['path'][first_let:-4].split('_')[8]), axis=1)

    all_models['layers'] = \
        all_models.apply(lambda x: int(x['path'][first_let:-4].split('_')[10]),
                         axis=1)

    all_models['ci'] = \
        all_models.apply(lambda x: float(x['path'][first_let:-4].split('_')[12]),
                         axis=1)

    all_models['datetime'] = \
        all_models.apply(lambda x:
                         datetime.datetime.fromtimestamp(int(x['path']
                                                             [first_let:-4].split(
                                                         '_')[14])), axis=1)

    return all_models
