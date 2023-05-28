Главная задача данного проекта - обучить модель многоклассовой классификации для определения главной эмоции сообщения, 
чтобы применить эту классификацию сверху обработки входящих сообщений от пользователей [тг-бота](http://t.me/ChatWithCareBot) для психологической поддержки.
Это позволит не только аккумулировать информацию о настроении пользователей и использовать эти данные как дополнительную статистику, 
доступную пользователям о самих себе, но также выдвигается гипотеза о возможности предсказывать RR по изменению эмоционального фона по итогу общения с ботом. 
Бот работает на основе модели ChatGPT 3.5 turbo (OpenAI). Бот может запускаться из [двух разных модулей](https://github.com/AnnettVsemPrivet/projects/tree/main/emotion_classification/py_modules), однако если пользователей много, 
то рекомендуется использовать асинхронную версию.

В качестве данных для обучения был взят [датасет](https://www.kaggle.com/datasets/parulpandey/emotion-dataset) с 6 эмоциями:
 - sadness
 - joy
 - love
 - anger
 - fear
 - surprise

С помощью библиотеки pycaret ([раздел baseline](https://github.com/AnnettVsemPrivet/projects/blob/main/emotion_classification/emotion_classification.ipynb))
были выбраны 2 главные модели, демонстрирующие лучшее качество по кросс-валидации на F1 и ROC-AUC:
 - LinearDiscriminantAnalysis
 - LogisticRegression

Используя MLflow и RandomizedSearchCV ([раздел MLflow](https://github.com/AnnettVsemPrivet/projects/blob/main/emotion_classification/emotion_classification.ipynb)), 
были подобраны оптимальные параметры для каждой модели и выбрана лучшая:
 - LinearDiscriminantAnalysis, с параметрами:
   - 'vectorizer__stop_words': 'english'
   - 'vectorizer__ngram_range': (1, 2)
   - 'vectorizer__min_df': 10
   - 'vectorizer__max_df': 0.9
   - 'transformer__n_components': 1000
   - 'classifier__tol': 0.0006000000000000001
 - итоговое качество по валидационной выборке F1 score: 0.87
