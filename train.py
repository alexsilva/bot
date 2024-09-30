import json
import pickle
import nltk
import random
import numpy as np

from nltk.stem import WordNetLemmatizer
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

nltk.download('punkt_tab')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# inicializaremos nossa lista de palavras, classes, documentos e
# definimos quais palavras serão ignoradas
words = []
documents = []
intents = json.loads(open('intents.json', encoding='utf-8').read())
# adicionamos as tags em nossa lista de classes
classes = [i['tag'] for i in intents['intents']]
ignore_words = ["!", "@", "#", "$", "%", "*", "?"]

# é feita a leitura do arquivo intents.json e transformado em json
intents = json.loads(open('intents.json', encoding='utf-8').read())

# percorremos nosso array de objetos
for intent in intents['intents']:
	for pattern in intent['patterns']:
		# com ajuda no nltk fazemos aqui a tokenizaçao dos patterns
		# e adicionamos na lista de palavras
		word = nltk.word_tokenize(pattern)
		words.extend(word)

		# adiciona aos documentos para identificarmos a tag para a mesma
		documents.append((word, intent['tag']))

# lematizamos as palavras ignorando os palavras da lista ignore_words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]

# classificamos nossas listas
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# ======== Aprendizado profundo  ========

# salva as palavras e classes nos arquivos pkl
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Initialize training data
training = []
output_empty = [0] * len(classes)

for document in documents:
	# Initialize the bag of words with zeros
	bag = np.zeros(len(words))

	pattern_words = document[0]
	pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

	for word in pattern_words:
		if word in words:
			bag[words.index(word)] = 1

	output_row = list(output_empty)
	output_row[classes.index(document[1])] = 1

	training.append([bag, output_row])

# Shuffle the training data
random.shuffle(training)

# Separate the features (x) and labels (y)
x = np.array([row[0] for row in training])
y = np.array([row[1] for row in training])

# Cria nosso modelo com 3 camadas.
# Primeira camada de 128 neurônios,
# segunda camada de 64 neurônios e terceira camada de saída
# contém número de neurônios igual ao número de intenções para prever a intenção de saída com softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(x[0]),), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y[0]), activation='softmax'))

# O modelo é compilado com descida de gradiente estocástica
# com gradiente acelerado de Nesterov.
# A ideia da otimização do Momentum de Nesterov, ou Nesterov Accelerated Gradient (NAG),
# é medir o gradiente da função de custo não na posição local,
# mas ligeiramente à frente na direção do momentum.
# A única diferença entre a otimização de Momentum é que o gradiente é medido em θ + βm em vez de em θ.
sgd = SGD(momentum=0.9, nesterov=True, weight_decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# ajustamos e salvamos o modelo
m = model.fit(np.array(x), np.array(y), epochs=200, batch_size=5, verbose=1)
model.save('model.h5', m)
