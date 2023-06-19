# Sentiment-Classification-Movie_reviews
As part of the Kaggle Competition, a Movie Reviews Dataset was used to build a Machine Learning model to
predict the sentiments of movie reviews. It is a binary classification problem (positive or negative sentiment).
Multiple models were trained on a training dataset and accessed to derive the best model. The best model was
used to predict on a test dataset to be submitted in Kaggle for assessment.

## Dataset
The training dataset has 24995 records with equal number of positive and negative sentiments (balanced
dataset). The Test dataset has 25865 records with no labels on which we need to use our model to predict the
labels. The maximum length of the length of the reviews is ~1000 words and the mean length is ~400 words.
![image](https://github.com/j0519740/Sentiment-Classification-Movie_reviews/assets/99134168/5ff6a9ee-f206-4c05-a12a-9eb1a02777f7)
![image](https://github.com/j0519740/Sentiment-Classification-Movie_reviews/assets/99134168/6ddbcf94-8f2e-4741-a41f-1821c585067d)


## Data Preprocessing
For data preprocessing, TextHero package was used to preprocess the text to remove URLS, HTML tags, digits,
punctuations, whitespace, mentions (@), hashtags and emails in addition lowercasing the text.

## Feature Engineering
The Roberta-Large model is used to extract features with dimensions of 1024 from the pre-processed dataset. It
is a pre-trained Mask-Language -Model that was pre-trained on a huge number of raw-texts and has learnt
bidirectional representation of sentences. Therefore, the model can produce features that represent an inner
representation of the English language and context in a sentence. Using these reviews pre-processed text as input,
features of dimensions 1024 were extracted using this Roberta-Large model to be used input for our models. 

## Model Validation
The train dataset was split into 80% for training and 20% for validation. Accuracy was used to check the
trained model’s performance on the validation dataset.

## Models
The extracted features from the Roberta-Large model were used as input to various models that were tested. As a
baseline model, the logistic regression model was used and the validation accuracy was 0.88. Using this model as
a baseline, multiple neural network models were assessed to deduce the best model to be submitted in the Kaggle
Competition. The results of the multiple models accessed is shown in Table 1. For the Neural Network models,
the loss function used was cross entropy and the optimiser used was momentum based Adam. The learning rate
and dropout rate were assessed to test the model on accuracy to ensure that the model does not overfit. 250 epochs
for training were used for the models 1-7 with early stopping when the validation loss stabilized for 5 epochs. 2
epochs were used for model 8 and model 9.

![image](https://github.com/j0519740/Sentiment-Classification-Movie_reviews/assets/99134168/198306d0-7aec-4a56-9906-f4db0006dcd8)

## Conclusion - Best Models (Model 8 and Model 9)
The last two models assessed was using the BERT model where the model was fine-tuned using our dataset.
The preprocessed reviews text were tokenized using the BERT tokenizer with a maximum length of 512.
The output of the BERT encoder model of dimensions 768 are pooled to a linear head output layer of 2 for
our binary classification. A Softmax function was used in the output head layer for our binary classification
task. Since the model is very large and pre-trained, an epoch size of 2 was sufficient to train the model. A
learning rate of 2 x e-5 was utilized for the training. A validation accuracy of 0.9242 was achieved with this
model which was the best model among all the models assessed thus far. This model (model 8) was used to
predict the sentiments for the test dataset and submitted in Kaggle. The test accuracy from the public
leaderboard was 0.92737 which is the best test accuracy that has been achieved thus far.
As a test to check if pre-processing the reviews text made reviews lose some context that could help in
predicting the sentiments, the same model was trained using the raw text of the reviews which were tokenized
as input (Model 9). The fine-tuned model with these tokenized raw reviews as input gave a better validation
accuracy of 0.9276. This fine-tuned model was also used to make predictions on the test dataset and
submitted in Kaggle. Interestingly, it shows that for the fine-tuning of the BERT model, pre-processing of
the reviews removes some of the context of the reviews that may help in better predicting the sentiments of
these reviews. This is the best model that predicts the sentiments of the movie reviews with a test accuracy
of 0.9356 after submission in Kaggle.

For further discussions on other models assessed, it can be found in the report in the Repo.

