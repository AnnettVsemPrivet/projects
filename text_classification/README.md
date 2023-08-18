The main task of this project is to create a binary classifier to 
identify if the tone of review is negative or positive. 

The data is already labeled and we can see the imbalance. There're different techniques to handle it but here we use oversampling. Then we choose between different linear, Bayes, decision tree, gradient boosting models, using RandomizedSearchCV with 5 folds. The final model is LinearSVC with hinge loss function and C=1.6 and with Tfidf Vectorizer without stop_words, with ngrams up to 3 words, and maximum occurrence of a word to turn into variable - 90%.
