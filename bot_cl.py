import json
from keras.models import load_model

from extract import class_prediction, get_response

# extraimos o modelo usando o keras
model = load_model('model.h5')

# carregamos nossas intenções
intents = json.loads(open('intents.json', encoding='utf-8').read())


def chatbot_response(text):
	"""
		Resposta do bot
	"""
	ints = class_prediction(text, model)
	res = get_response(ints, intents)
	return res


while True:
	print('=' * 25)
	text = input('Informe uma mensagem: ')
	if text == '/exit':
		break
	response = chatbot_response(text)
	print(response)

