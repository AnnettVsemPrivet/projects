print('Bot started')
# cd jupyter\projects\emotion_classification
# python tg_bot.py


# импорт


import datetime
import time
import pandas as pd
import json

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.chat_models import ChatOpenAI

import telebot

import sys
sys.path.append('../../../jupyter/')
from CREDS import *


# запуск бота


# модель
all_convos = {}
bot = telebot.TeleBot(token())
API_O = api()
#['gpt-3.5-turbo','text-davinci-003','text-davinci-002','code-davinci-002']
MODEL = 'gpt-3.5-turbo'
# Summary of prompts to consider (min_value=3,max_value=1000)
K = 1000

llm = ChatOpenAI(temperature=0,
            openai_api_key=API_O, 
            model_name=MODEL, 
            verbose=False) 

# сегодняшняя метка для обозначения диалогов
timestamp = int(time.mktime(datetime.datetime.utcnow().date().timetuple()))

# информация о юзерах (айди, имя, фамилия, ник)
try:
    with open('../bot_data/arr_users.json') as user_file:
        dict_users = json.load(user_file)
except:
    dict_users = {}

# информация о диалогах (уникальный номер, отправленные и полученные сообщения, время сообщений)
# уникальные номера разговоров - берутся по числовому номеру дня + айди юзера
try:
    with open('../bot_data/arr_discs.json') as user_file:
        dict_discs = json.load(user_file)
except:
    dict_discs = {}


# кто-то пишет старт


@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.from_user.id,"Howdy, how are you doing? ✌️ ")
    bot.send_video(message.from_user.id, video=open('../video.mp4', 'rb'), supports_streaming=True)
    all_convos[str(message.from_user.id)] = ConversationChain(
                                        llm=llm, 
                                        prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
                                        memory=ConversationEntityMemory(llm=llm, k=K)
                                        ) 
    # заносим информацию в таблицу о юзерах
    try:
        dict_users[str(message.from_user.id)]
    except:
        dict_users[str(message.from_user.id)] = [message.from_user.first_name,
                                                 message.from_user.last_name,
                                                 message.from_user.username]

        with open('../bot_data/arr_users.json', 'w') as json_file:
            json.dump(dict_users, json_file)

    # создаем пустое поле диалога за этот день
    dict_discs[str(message.from_user.id)+'_'+str(timestamp)] = []

    # заносим информацию в таблицу о диалогах
    with open('../bot_data/arr_discs.json', 'w') as json_file:
        json.dump(dict_discs, json_file)


# кто-то продолжает разговор


@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    bot.send_chat_action(message.from_user.id, 'typing')
    
    # обрабатываем сообщение
    try:
        all_convos[str(message.from_user.id)]
    except:
        all_convos[str(message.from_user.id)] = ConversationChain(
                                    llm=llm, 
                                    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
                                    memory=ConversationEntityMemory(llm=llm, k=K)
                                    )
        
    time_now = int(datetime.datetime.utcnow().timestamp())
    answer = all_convos[str(message.from_user.id)].run(input=message.text.lower())
    bot.send_message(message.from_user.id, answer)

    # заносим информацию в таблицу о диалогах
    try:
        dict_discs[str(message.from_user.id)+'_'+str(timestamp)]
    except: 
        dict_discs[str(message.from_user.id)+'_'+str(timestamp)] = []

    dict_discs[str(message.from_user.id)+'_'+str(timestamp)].append([message.text, answer, time_now])
    # заносим информацию в таблицу о диалогах
    with open('../bot_data/arr_discs.json', 'w') as json_file:
        json.dump(dict_discs, json_file)


bot.polling(none_stop=True)

