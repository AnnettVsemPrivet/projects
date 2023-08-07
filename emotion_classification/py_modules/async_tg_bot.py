#!/usr/bin/env python
# -*- coding: utf-8 -*- 

print('Bot started')
# cd jupyter\projects\emotion_classification\py_modules
# python tg_bot_2.py


# импорт


import datetime
import time
import pandas as pd
import json
import numpy as np

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.chat_models import ChatOpenAI

from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
import asyncio

#import sys
#sys.path.append('../../../../jupyter/')
from CREDS import *

import itertools
# запуск бота


# модель
all_convos = {}
#mytuple = tuple([5]*5+[10]*5)
#myit = iter(mytuple)
#myit = itertools.cycle([0]*3+[1]*3+[2]*3)
bot = Bot(token=token())
dp = Dispatcher(bot)
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
    with open('../bot_data/arr_users.json', 'r+') as user_file:
        dict_users = json.load(user_file)
except:
    dict_users = {}

# информация о диалогах (уникальный номер, отправленные и полученные сообщения, время сообщений)
# уникальные номера разговоров - берутся по числовому номеру дня + айди юзера
try:
    with open('../bot_data/arr_discs.json', 'r+') as user_file:
        dict_discs = json.load(user_file)
except:
    dict_discs = {}

# отлов ошибок
try:
    with open('../bot_data/arr_mess.json', 'r+') as user_file:
        dict_mess = json.load(user_file)
except:
    dict_mess = {}


# кто-то пишет старт


@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    #await message.reply("Howdy, how are you doing? ✌️ ")
    await bot.send_message(message.from_user.id, "Howdy, how are you doing? ✌️ ")
    #await bot.send_chat_action(msg.from_user.id, action=upload_video)
    await bot.send_video(message.from_user.id, video=open('../video.mp4', 'rb'), supports_streaming=True)
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


@dp.message_handler()
async def echo_message(message: types.Message):

    # отлов ошибок
    try:
        with open('../bot_data/arr_mess.json', 'r+') as user_file:
            dict_mess = json.load(user_file)
    except:
        dict_mess = {}

    # отлов ошибок 2
    try:
        with open('../bot_data/arr_mess_all.json', 'r+') as user_file:
            dict_mess_all = json.load(user_file)
    except:
        dict_mess_all = {}

    do_it = 0
    time_now = int(datetime.datetime.utcnow().timestamp())

    # заносим информацию в таблицу об ошибках
    try:
        dict_mess[str(message.from_user.id)+'_'+str(timestamp)]
    except:
        dict_mess[str(message.from_user.id)+'_'+str(timestamp)] = []

    dict_mess[str(message.from_user.id)+'_'+str(timestamp)].append([message.text, time_now])

    # заносим информацию в таблицу об ошибках_2
    try:
        dict_mess_all[str(timestamp)]
    except:
        dict_mess_all[str(timestamp)] = []

    dict_mess_all[str(timestamp)].append([message.text, time_now])

    # заносим информацию в таблицу об ошибках
    with open('../bot_data/arr_mess.json', 'w') as json_file:
        json.dump(dict_mess, json_file)

    with open('../bot_data/arr_mess.json', 'r+') as user_file:
        dict_mess = json.load(user_file)

    all_timestamps = [x[1] for x in dict_mess[str(message.from_user.id)+'_'+str(timestamp)]]
    if len(all_timestamps) > 1:
        if all_timestamps[-1] - all_timestamps[-2] < 40:
            do_it = 0
        else:
            do_it = 1
    else:
        do_it = 1

    # заносим информацию в таблицу об ошибках_2
    with open('../bot_data/arr_mess_all.json', 'w') as json_file:
        json.dump(dict_mess_all, json_file)

    with open('../bot_data/arr_mess_all.json', 'r+') as user_file:
        dict_mess_all = json.load(user_file)

    all_timestamps = [x[1] for x in dict_mess_all[str(timestamp)]]
    if len(all_timestamps) > 1:
        if all_timestamps[-1] - all_timestamps[-2] < 40:
            if len(all_timestamps) > 2:
                if all_timestamps[-1] - all_timestamps[-3] < 40:
                    myit = 2
                else:
                    myit = 1
            else:
                myit = 1
        else:
            myit = 0
    else:
        myit = 0

    if do_it == 1:
        # если сообщение было недавно и пока без ответа, то не обрабатываем моделью
        
        if myit in [0,1,2]:
            print('0__'+str(message.from_user.id))
            await bot.send_chat_action(message.from_user.id, action=types.ChatActions.TYPING)
            await asyncio.sleep(5)
            await bot.send_chat_action(message.from_user.id, action=types.ChatActions.TYPING)
            await asyncio.sleep(5)
            await bot.send_chat_action(message.from_user.id, action=types.ChatActions.TYPING)
            await asyncio.sleep(5)
            await bot.send_chat_action(message.from_user.id, action=types.ChatActions.TYPING)
            await asyncio.sleep(5)
            await bot.send_chat_action(message.from_user.id, action=types.ChatActions.TYPING)
            await asyncio.sleep(5)
            await bot.send_chat_action(message.from_user.id, action=types.ChatActions.TYPING)
            await asyncio.sleep(5)
            await bot.send_chat_action(message.from_user.id, action=types.ChatActions.TYPING)
            await asyncio.sleep(5)
            await bot.send_chat_action(message.from_user.id, action=types.ChatActions.TYPING)
            await asyncio.sleep(5)
            await bot.send_chat_action(message.from_user.id, action=types.ChatActions.TYPING)

        if myit in [1,2]:
            print('1__'+str(message.from_user.id))
            await asyncio.sleep(5)
            await bot.send_chat_action(message.from_user.id, action=types.ChatActions.TYPING)
            await asyncio.sleep(5)
            await bot.send_chat_action(message.from_user.id, action=types.ChatActions.TYPING)
            await asyncio.sleep(5)
            await bot.send_chat_action(message.from_user.id, action=types.ChatActions.TYPING)
            await asyncio.sleep(5)
            await bot.send_chat_action(message.from_user.id, action=types.ChatActions.TYPING)
            await asyncio.sleep(5)
            await bot.send_chat_action(message.from_user.id, action=types.ChatActions.TYPING)
            await asyncio.sleep(5)
            await bot.send_chat_action(message.from_user.id, action=types.ChatActions.TYPING)
            await asyncio.sleep(5)
            await bot.send_chat_action(message.from_user.id, action=types.ChatActions.TYPING)
            await asyncio.sleep(5)
            await bot.send_chat_action(message.from_user.id, action=types.ChatActions.TYPING)

        if myit == 2:
            print('2__'+str(message.from_user.id))
            await asyncio.sleep(5)
            await bot.send_chat_action(message.from_user.id, action=types.ChatActions.TYPING)
            await asyncio.sleep(5)
            await bot.send_chat_action(message.from_user.id, action=types.ChatActions.TYPING)
            await asyncio.sleep(5)
            await bot.send_chat_action(message.from_user.id, action=types.ChatActions.TYPING)
            await asyncio.sleep(5)
            await bot.send_chat_action(message.from_user.id, action=types.ChatActions.TYPING)
            await asyncio.sleep(5)
            await bot.send_chat_action(message.from_user.id, action=types.ChatActions.TYPING)
            await asyncio.sleep(5)
            await bot.send_chat_action(message.from_user.id, action=types.ChatActions.TYPING)
            await asyncio.sleep(5)
            await bot.send_chat_action(message.from_user.id, action=types.ChatActions.TYPING)
            await asyncio.sleep(5)
            await bot.send_chat_action(message.from_user.id, action=types.ChatActions.TYPING)

        # обрабатываем сообщение
        try:
            all_convos[str(message.from_user.id)]
        except:
            all_convos[str(message.from_user.id)] = ConversationChain(
                                        llm=llm, 
                                        prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
                                        memory=ConversationEntityMemory(llm=llm, k=K)
                                        )
        answer = all_convos[str(message.from_user.id)].run(input=message.text.lower())
        print(message.text)
        # await bot.send_message(message.from_user.id, answer)
        # await message.reply(answer)
        await bot.send_message(message.from_user.id, answer, reply_to_message_id=message.message_id)

        # заносим информацию в таблицу о диалогах
        try:
            dict_discs[str(message.from_user.id)+'_'+str(timestamp)]
        except: 
            dict_discs[str(message.from_user.id)+'_'+str(timestamp)] = []

        dict_discs[str(message.from_user.id)+'_'+str(timestamp)].append([message.text, answer, time_now])
        # заносим информацию в таблицу о диалогах
        with open('../bot_data/arr_discs.json', 'w') as json_file:
            json.dump(dict_discs, json_file)


if __name__ == '__main__':
    executor.start_polling(dp)
