### english
The main purpose of this project is to predict stock market prices by identifying the current pattern in which price of the asset behaves. 
There are several python modules to solve this task: [preprocessing](https://github.com/AnnettVsemPrivet/projects/blob/main/pattern_recognition/py_modules/preprocessing.py), [clustering](https://github.com/AnnettVsemPrivet/projects/blob/main/pattern_recognition/py_modules/clustering.py) and [predicting](https://github.com/AnnettVsemPrivet/projects/blob/main/pattern_recognition/py_modules/predicting.py).
To see the example of using all these functions you can look at the file [clustering_n_predicting](https://github.com/AnnettVsemPrivet/projects/blob/main/pattern_recognition/clustering_n_predicting.ipynb).
Here we take minute-by-minute data on the shares of Sberbank from the year 2021, then take time periods by 30 minutes. We train KShape model on this data using optuna library to optimize the number of clusters. After that, at first we go in details on building Neural Net model to predict the 6th cluster, and later we evaluate automatically which patterns will be the most efficient for prediction and build NN models for all of them (using LSTM from torch library because we deal with time series). For creating a model we also use optuna to decide on such parameters as: learning_rate, weight_decay, number of hidden layers, number of neurons in every hidden layer. In the end you can see the results of prediction which are valued by R2 and MSE. 

### russian
Задача - выделение паттернов на биржевом рынке и прогноз временного отрезка следующего за паттерном. 
Для решения задачи составлены следующие модули:
- [preprocessing](https://github.com/AnnettVsemPrivet/projects/blob/main/pattern_recognition/py_modules/preprocessing.py): используется для подготовки обучающей выборки
- [clustering](https://github.com/AnnettVsemPrivet/projects/blob/main/pattern_recognition/py_modules/clustering.py): используется для обучения и сохранения модели кластеризации KShape
- [predicting](https://github.com/AnnettVsemPrivet/projects/blob/main/pattern_recognition/py_modules/predicting.py): используется для обучения и предсказания модели прогнозирования LSTM

В качестве примера можно посмотреть ноутбук [clustering_n_predicting](https://github.com/AnnettVsemPrivet/projects/blob/main/pattern_recognition/clustering_n_predicting.ipynb), 
в котором берутся поминутные данные по акциям Сбербанка за 2021г, из датасета выделяются 30-минутные отрезки. 
По 30-минутным периодам обучается модель кластеризации ks_trained_4.hdf5, число кластеров которой было оптимизировано с помощью optuna. Затем через данную модель пропускается более обширная выборка для предсказания кластеров.
Выбираем любой кластер, и по собранным обучающим данным тренируем модель прогнозирования. В конце можно посмотреть на результаты прогноза по R2 и MSE.
