# from sre_parse import State
# from tabnanny import verbose
import numpy as np
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def buildMLP(train_features, test_features, train_targets, test_targets, num_neurons):

    classifier = MLPClassifier(hidden_layer_sizes=num_neurons, max_iter=2000, activation='tanh',
                               solver='adam', random_state=762, learning_rate='invscaling', verbose=True)
    classifier.fit(train_features, train_targets)

    # Przewidywanie na podstawie MLP
    predictions = classifier.predict(test_features)
    # Obliczanie wyniku
    score = np.round(metrics.accuracy_score(test_targets, predictions), 2)

    print('Mean accuracy of predictions: ' + str(score))
    print('Test targets: ' + str(test_targets))
    print('Predictions: ' + str(predictions))


corpus = [
    'We enjoyed our stay so much! Everything was perfect.',
    'Goin to think twice before staying here again. The wifi was bad and rooms were small.',
    'The perfect place to relax and recharge.',
    'The pictures were misleading, so I was expecting the common areas to be bigger. But the service was good.',
    'There were no clean linens when I got to my room and the breakfast option were not that many.',
    'Was expecting it to be a bit far from historical downtown, but it was almost impossible to drive through those narrow roads.',
    'I thought that waking up with the chickens was fun, but I was wrong.',
    'Great place for a quick gateway from the city. Everyone is friendly and polite.',
    'Unfortunately it was raining during our stay, and there weren\'t many options for indoors activities. Everything was great, but there was literally no other options besides being in the rain',
    'The town festival was postponed, so the area was a complete ghost town. We were the only guests. Not the experience I was looking for.',
    'We had a lovely time. it\'s a fantastic place to go with the children, the loved all the animals.',
    'A little bit off the beaten track, but completly worth it. You can hear the birds sing in the morning anad the you are greeted with the biggest, sincerest smiles from the owners. Loved it!',
    'It was good to be outside in the country, visiting old town. Everythin  was prepared to the upmost detail',
    'Staff was friendly. Going to come back for sure.',
    'They didn\'t have enough staff for the amount of guests. It took some time to get our breakfast and we had to wait 20 minutes to get more information about old town.',
    'The pictures looked way diffrent.',
    'Best weekend in the countryside I\'ve ever had.',
    'Terrible. Slow staff, slow town. Only good thing was being surrounded by nature.',
    'Not as clean as advertised, Found some cobweb in the corner of the room.',
    'It was a peaceful gateaway in the countryside.',
    'Everyone was nice. Had a goodtime.',
    'The kid loved running around in nature, we loved the old town. Definitely going back.',
    'Had worse experiences.',
    'Suprised this was much diffrent than what was on the website.',
    'Not that mindblowing.',
    'Best trip ever!'
]
# 0: negative, 1: positive
targets = [1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1,
           1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1]

# Rozłączenie danych na kilka zmiennych
# train_features = wszystkie dane z corpus, test_features = losowo wybrane dane z corpus do testu
# train targets = wszystkie targety output, test_targets = targety dla testowanych danych z corpus
train_features, test_features, train_targets, test_targets = train_test_split(
    corpus, targets, test_size=0.2, random_state=233)


num_neurons = 14

vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, norm='l1')

print(test_features)
# wektoryzacja czyli tłumaczenie na komputerowy
train_features = vectorizer.fit_transform(train_features)
test_features = vectorizer.transform(test_features)

buildMLP(train_features, test_features,
         train_targets, test_targets, num_neurons)
