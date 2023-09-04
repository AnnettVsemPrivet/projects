### english 

The main task of this project is to create a binary classifier to 
identify if the tone of review is negative or positive. 

The data is already labeled and we can see the imbalance. There're different techniques to handle it but here we use oversampling. Then we choose between different linear, Bayes, decision tree, gradient boosting models, using RandomizedSearchCV with 5 folds. The final model is LinearSVC with hinge loss function and C=1.6 and with Tfidf Vectorizer without stop_words, with ngrams up to 3 words, and maximum occurrence of a word to turn into variable - 90%.

### russian

Главная задача этого проекта - обучить модель бинарной классификации для определения тона отзыва (позитивный/негативный).

По данным сразу можно заметить дисбаланс классов. Есть разные способы справиться с этой проблемой, здесь мы используем дублирование объектов наименее представленного класса. Далее мы выбираем с помощью RandomizedSearchCV с 5 фолдами между бейзлайн моделями разных типов: линейные, байесовские, с деревьями решений и с градиентным бустингом. Итоговая модель - LinearSVC с функцией потерь hinge, коэффициентом С=1.6, Tfidf-векторизацией для преобразования слов в ngrams до 3х слов, и слова должны встречаться не более чем в 90% элементов выборки.
