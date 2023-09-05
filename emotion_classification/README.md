## english

The main goal of this project is to create a telegram bot for psychological help by combining a model for text generation to answer and a model for emotion classification from text messages. The [final bot](http://t.me/ChatWithCareBot) is running inside a Docker Container, which is built with [Dockerfile](https://github.com/AnnettVsemPrivet/projects/blob/main/emotion_classification/Dockerfile), on a remote server. This bot not only provides help but also can show user's emotion range from the last 30 days. It's [using](https://github.com/AnnettVsemPrivet/projects/blob/main/emotion_classification/py_modules/async_tg_bot.py) asynchronous library aiogram and the models: ChatGPT 3.5 turbo (OpenAI), [michellejieli/emotion_text_classifier](https://huggingface.co/michellejieli/emotion_text_classifier) (huggingface). 

While working on the project I've tried to train new classification model, and used Kaggle [dataset](https://www.kaggle.com/datasets/parulpandey/emotion-dataset) with text emotions, then pycaret and MLflow + RandomizedSearchCV to choose between models, you can see details in the notebook [emotion_classification](https://github.com/AnnettVsemPrivet/projects/blob/main/emotion_classification/emotion_classification.ipynb). The final model [LinearDiscriminantAnalysis](https://github.com/AnnettVsemPrivet/projects/blob/main/emotion_classification/models/LDA_clf.joblib) got an average F1-score 0.87 on validation data and has the following parameters:
 - 'TfidfVectorizer__stop_words': 'english'
 - 'TfidfVectorizer__ngram_range': (1, 2)
 - 'TfidfVectorizer__min_df': 10
 - 'TfidfVectorizer__max_df': 0.9
 - 'TruncatedSVD__n_components': 1000
 - 'classifier__tol': 0.0006

However, despite having good quality, this model can't apprehend not so obvious negative emotions in comparison to a huggingface model which I think comes from much samller training dataset.

The future of this project is to test a hypothesis that an improvement in mood by the end of conversation with bot has a positive impact on user's Retention Rate.

## russian

Главная задача данного проекта - создать телеграмм-бота для психологической поддержки, объединив модель генерации текста для ответов и модель классификации эмоций по текстовым сообщениям. [Итоговый бот](http://t.me/ChatWithCareBot) запущен с помощью [docker-файла](https://github.com/AnnettVsemPrivet/projects/blob/main/emotion_classification/Dockerfile) на удаленном сервере и помимо выдачи ответов также предоставляет статистику по эмоциям за последние 30 дней общения с юзером. Для [работы бота](https://github.com/AnnettVsemPrivet/projects/blob/main/emotion_classification/py_modules/async_tg_bot.py) используется асинхронная библиотека aiogram, в качестве моделей взяты ChatGPT 3.5 turbo (OpenAI) и модель классификации [michellejieli/emotion_text_classifier](https://huggingface.co/michellejieli/emotion_text_classifier) из huggingface. 

В рамках реализации проекта была совершена попытка построить свою модель классификации эмоций по тексту, для обучения использовался [датасет](https://www.kaggle.com/datasets/parulpandey/emotion-dataset) из kaggle, а для выбора модели были использованы pycaret и MLflow + RandomizedSearchCV, подробности экспериментов можно посмотреть в файле [emotion_classification](https://github.com/AnnettVsemPrivet/projects/blob/main/emotion_classification/emotion_classification.ipynb). Итоговая модель [LinearDiscriminantAnalysis](https://github.com/AnnettVsemPrivet/projects/blob/main/emotion_classification/models/LDA_clf.joblib) получила средний F1-score по валидационной выборке 0.87 и имеет следующие параметры:
 - 'TfidfVectorizer__stop_words': 'english'
 - 'TfidfVectorizer__ngram_range': (1, 2)
 - 'TfidfVectorizer__min_df': 10
 - 'TfidfVectorizer__max_df': 0.9
 - 'TruncatedSVD__n_components': 1000
 - 'classifier__tol': 0.0006

Однако, несмотря на хорошие показатели по всем классам, модель плохо справляется с определением неочевидных негативных эмоций по сравнению с моделью с huggingface, скорее всего из-за недостаточно обширной обучающей выборки, поэтому было принято решение использовать huggingface. 

Дальнейшие перспективы - проверить гипотезу о том, насколько влияет улучшение настроения к концу диалога с ботом на Retention Rate пользователей.


