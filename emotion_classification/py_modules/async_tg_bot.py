#!/usr/bin/env python
# -*- coding: utf-8 -*- 

print('Bot started')


# импорт


import datetime
import pandas as pd
import json
import numpy as np
import copy
import sqlite3
from transformers import pipeline

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.schema.messages import messages_to_dict, messages_from_dict
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory

from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor

from cccalendar import draw_colour_calendar
from pyplutchik import plutchik
from sqlite import CorrectedSQLiteEntityStore

import warnings
warnings.filterwarnings("ignore")

import os
BOT_TOKEN = os.environ.get('BOT_TOKEN')
OPENAI_API = os.environ.get('OPENAI_API_KEY')

def timer(sec):
    time_start = datetime.datetime.now()
    time_end = time_start + datetime.timedelta(seconds = sec)
    while True: 
        if datetime.datetime.now() >= time_end:
            break

def normalize(d, target=1.0):
    raw = max(d.values())
    if raw==0:
        factor = 0
    else:
        factor = target/raw
    return {key:value*factor for key,value in d.items()}


# загрузка модели


bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)
#['gpt-3.5-turbo','text-davinci-003','text-davinci-002','code-davinci-002']
MODEL = 'gpt-3.5-turbo'
# Summary of prompts to consider (min_value=3, max_value=1000)
K = 1000
# List of all timestamps when query was sent
all_timestamps = []
# Time to wait between queries in seconds
Needed_time = 45
# Daily limit for queries is 200 per day!

llm = ChatOpenAI(temperature=0,
            openai_api_key=OPENAI_API, 
            model_name=MODEL, 
            verbose=False) 

colour_map = {
    'surprise': 'darkorange',
    'joy': 'gold',
    'sadness': 'forestgreen',
    'fear': 'skyblue', 
    'disgust': 'slateblue',
    'anger': 'orangered'
}

loaded_model = pipeline("text-classification", model="michellejieli/emotion_text_classifier")

# информация о юзерах (айди юзера, имя, фамилия, ник)
try:
    with open('../bot_data/arr_users.json') as user_file:
        dict_users = json.load(user_file)
except:
    dict_users = {}

# информация о диалогах - вопрос+ответ (айди юзера, отправленное и полученное юзером сообщение, время, эмоция)
try:
    with open('../bot_data/arr_discs.json') as user_file:
        dict_discs = json.load(user_file)
except:
    dict_discs = {}

# информация о всех сообщениях (айди юзера, отправленное или полученное юзером сообщение, время)
try:
    with open('../bot_data/arr_mess.json') as user_file:
        dict_mess = json.load(user_file)
except:
    dict_mess = {}

# информация для бэкапа памяти моделей (айди юзера, история сообщений)
try:
    with open('../bot_data/arr_convos.json') as user_file:
        memory_lain = json.load(user_file)

    all_convos = {}
    for key in memory_lain.keys():
        all_convos[key] = ConversationChain(
                                            llm=llm, 
                                            prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
                                            memory=ConversationEntityMemory(llm=llm, k=K, 
                                                chat_memory=ChatMessageHistory(messages=messages_from_dict(memory_lain[key])),
                                                entity_store=CorrectedSQLiteEntityStore(db_file='../bot_data/entities.db', table_name='user_'+key))
                                            ) 
except:
    memory_lain = {}
    all_convos = {}


# кто-то пишет старт


@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    await bot.send_message(message.from_user.id, "Howdy, how are you doing? ✌️ ")
    await bot.send_video(message.from_user.id, video=open('../video.mp4', 'rb'), supports_streaming=True)

    # начинаем диалог заново (обновляем историю сообщений, даже если она уже была)
    try:
        con = sqlite3.connect('../bot_data/entities.db')
        con.cursor().execute("DROP TABLE user_"+str(message.from_user.id)+"_default;")
        con.close()
    except:
        pass

    all_convos[str(message.from_user.id)] = ConversationChain(
                                        llm=llm, 
                                        prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
                                        memory=ConversationEntityMemory(llm=llm, k=K,
                                        entity_store=CorrectedSQLiteEntityStore(db_file='../bot_data/entities.db', table_name='user_'+str(message.from_user.id)))
                                        ) 

    # заносим информацию в таблицу с бэкапом памяти моделей
    memory_lain[str(message.from_user.id)] = messages_to_dict(all_convos[str(message.from_user.id)].memory.buffer)

    with open('../bot_data/arr_convos.json', 'w') as json_file:
        json.dump(memory_lain, json_file)

    # заносим информацию в таблицу о юзерах
    try:
        dict_users[str(message.from_user.id)]
    except:
        dict_users[str(message.from_user.id)] = [message.from_user.first_name,
                                                 message.from_user.last_name,
                                                 message.from_user.username]

        with open('../bot_data/arr_users.json', 'w') as json_file:
            json.dump(dict_users, json_file)

    # создаем пустое поле диалога
    dict_discs[str(message.from_user.id)] = []
    dict_mess[str(message.from_user.id)] = []

    # заносим информацию в таблицу о диалогах (это дает возможность стереть свой диалог с ботом при желании)
    with open('../bot_data/arr_discs.json', 'w') as json_file:
        json.dump(dict_discs, json_file)

    with open('../bot_data/arr_mess.json', 'w') as json_file:
        json.dump(dict_mess, json_file)


# кто-то хочет посмотреть эмоции


@dp.message_handler(commands=['emotions'])
async def cmd_emotions(message: types.Message):
    keyboard = types.InlineKeyboardMarkup()
    btn_1 = types.InlineKeyboardButton(text="Today", callback_data="btn_1")
    btn_2 = types.InlineKeyboardButton(text="Last Week", callback_data="btn_2")
    btn_3 = types.InlineKeyboardButton(text="Last Month", callback_data="btn_3")
    btn_4 = types.InlineKeyboardButton(text="Calendar", callback_data="btn_4")
    keyboard.add(btn_1, btn_2, btn_3, btn_4)
    await message.answer("Choose time period:", reply_markup=keyboard)

# за сегодня
@dp.callback_query_handler(text="btn_1")
async def send_today_emotions(call: types.CallbackQuery):

    try:
        with open('../bot_data/arr_discs.json') as user_file:
            dict_discs = json.load(user_file)
    except:
        dict_discs = {}

    all_me = dict_discs[str(call.from_user.id)]
    df = pd.DataFrame([(datetime.datetime.fromtimestamp(x[-2]).date(), x[-1]) for x in all_me], columns=['date','emotion']).query('emotion!="neutral" and emotion!="model_error"')

    new_df = df.query('date == datetime.datetime.now().date()')
    title = 'Today'

    new_emotions = new_df.groupby(['emotion']).count().to_dict()['date']

    emotion_wheel = {
        'joy': 0,
        'disgust': 0,
        'fear': 0,
        'surprise': 0,
        'sadness': 0,
        'anger': 0
        }

    emotion_wheel.update(new_emotions)

    plutchik(normalize(emotion_wheel), title = title, title_size = 18)
    await bot.send_photo(call.from_user.id, photo=open('../bot_data/photo.png', 'rb'))

# за неделю
@dp.callback_query_handler(text="btn_2")
async def send_week_emotions(call: types.CallbackQuery):

    try:
        with open('../bot_data/arr_discs.json') as user_file:
            dict_discs = json.load(user_file)
    except:
        dict_discs = {}

    all_me = dict_discs[str(call.from_user.id)]
    df = pd.DataFrame([(datetime.datetime.fromtimestamp(x[-2]).date(), x[-1]) for x in all_me], columns=['date','emotion']).query('emotion!="neutral" and emotion!="model_error"')

    new_df = df.loc[(df['date'] >= datetime.datetime.now().date()-datetime.timedelta(days = 7))]
    title = 'From '+str(datetime.datetime.now().date()-datetime.timedelta(days = 7))+'\nTo '+str(datetime.datetime.now().date())

    new_emotions = new_df.groupby(['emotion']).count().to_dict()['date']

    emotion_wheel = {
        'joy': 0,
        'disgust': 0,
        'fear': 0,
        'surprise': 0,
        'sadness': 0,
        'anger': 0
        }

    emotion_wheel.update(new_emotions)

    plutchik(normalize(emotion_wheel), title = title, title_size = 18)
    await bot.send_photo(call.from_user.id, photo=open('../bot_data/photo.png', 'rb'))

# за месяц
@dp.callback_query_handler(text="btn_3")
async def send_month_emotions(call: types.CallbackQuery):

    try:
        with open('../bot_data/arr_discs.json') as user_file:
            dict_discs = json.load(user_file)
    except:
        dict_discs = {}

    all_me = dict_discs[str(call.from_user.id)]
    df = pd.DataFrame([(datetime.datetime.fromtimestamp(x[-2]).date(), x[-1]) for x in all_me], columns=['date','emotion']).query('emotion!="neutral" and emotion!="model_error"')

    new_df = df.loc[(df['date'] >= datetime.datetime.now().date()-datetime.timedelta(days = 30))]
    title = 'From '+str(datetime.datetime.now().date()-datetime.timedelta(days = 30))+'\nTo '+str(datetime.datetime.now().date())

    new_emotions = new_df.groupby(['emotion']).count().to_dict()['date']

    emotion_wheel = {
        'joy': 0,
        'disgust': 0,
        'fear': 0,
        'surprise': 0,
        'sadness': 0,
        'anger': 0
        }

    emotion_wheel.update(new_emotions)

    plutchik(normalize(emotion_wheel), title = title, title_size = 18)
    await bot.send_photo(call.from_user.id, photo=open('../bot_data/photo.png', 'rb'))

# за календарный месяц
@dp.callback_query_handler(text="btn_4")
async def send_calendar_emotions(call: types.CallbackQuery):

    try:
        with open('../bot_data/arr_discs.json') as user_file:
            dict_discs = json.load(user_file)
    except:
        dict_discs = {}

    all_me = dict_discs[str(call.from_user.id)]
    df = pd.DataFrame([(datetime.datetime.fromtimestamp(x[-2]).date(), x[-1]) for x in all_me], columns=['date','emotion']).query('emotion!="neutral" and emotion!="model_error"')

    # for month + year
    new_df = df.loc[(pd.to_datetime(df['date']).dt.month==datetime.datetime.now().date().month) & (pd.to_datetime(df['date']).dt.year==datetime.datetime.now().date().year)]
    new_df = new_df.groupby('date').agg(pd.Series.mode).explode('emotion')
    new_df.reset_index(inplace=True)
    new_series = pd.Series()
    for i in new_df.index:
        new_series = new_series._append(pd.Series({new_df.loc[i]['date']: new_df.loc[i]['emotion']}))
    if len(new_series)==0:
        await call.message.answer("Please try again after sending any text messages to me!")
    else:
        draw_colour_calendar(new_series, colour_map, date_colour='lightgray')
        await bot.send_photo(call.from_user.id, photo=open('../bot_data/photo.png', 'rb'))


# кто-то продолжает разговор


@dp.message_handler()
async def echo_message(message: types.Message):

    # заносим информацию в таблицу сообщений
    try:
        dict_mess[str(message.from_user.id)]
    except:
        dict_mess[str(message.from_user.id)] = []

    # проверяем наличие уже обрабатываемых моделью сообщений от юзера
    do_it = 0
    time_now = int(message.date.timestamp())
    if len(dict_mess[str(message.from_user.id)]) > 0:
        # если юзер отвечает на сообщение бота, то автоматически берем в работу
        if len(dict_mess[str(message.from_user.id)][-1]) == 3:
            do_it = 1
        # а если до данного сообщения только сообщения от юзера а не от бота, 
        # то отвечаем только если между ними интервал больше 45 секунд 
        # чтобы модель успела еще кому-то ответить, а не общалась с одним юзером, от которого много идущих подряд сообщений
        else:
            if time_now - dict_mess[str(message.from_user.id)][-1][-1] < Needed_time:
                do_it = 0
            else:
                do_it = 1
    else:
        do_it = 1

    # заносим информацию в таблицу сообщений
    dict_mess[str(message.from_user.id)].append([message.text, time_now])

    with open('../bot_data/arr_mess.json', 'w') as json_file:
        json.dump(dict_mess, json_file)


    if do_it == 1:

        # смотрим есть ли эмоция с уверенностью классификатора больше 0.5
        emotion=''
        try:
            emotion = loaded_model(message.text.lower())[0]['label']
            prob = loaded_model(message.text.lower())[0]['score']

            if prob < 0.5:
                emotion='neutral'
            else:
                pass
        except:
            emotion='model_error'

        # смотрим сколько времени надо подождать перед запуском модели
        real_time_now = int(datetime.datetime.utcnow().timestamp())
        diff = copy.deepcopy(Needed_time)
        if len(all_timestamps) > 0:
            diff = real_time_now - all_timestamps[-1]
            if diff > Needed_time:
                diff = copy.deepcopy(Needed_time)

        timer(Needed_time-diff)

        # обрабатываем сообщение
        try:
            all_convos[str(message.from_user.id)]
        except:
            all_convos[str(message.from_user.id)] = ConversationChain(
                                        llm=llm, 
                                        prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
                                        memory=ConversationEntityMemory(llm=llm, k=K,
                                        entity_store=CorrectedSQLiteEntityStore(db_file='../bot_data/entities.db', table_name='user_'+str(message.from_user.id)))
                                        )

        real_time_now = int(datetime.datetime.utcnow().timestamp())
        answer = all_convos[str(message.from_user.id)].run(input=message.text.lower())

        # записываем время запуска модели, чтобы перед следующим запуском прошло необходимое время
        all_timestamps.append(real_time_now)

        await bot.send_message(message.from_user.id, answer, reply_to_message_id=message.message_id)

        real_time_now = int(datetime.datetime.utcnow().timestamp())

        # записываем примерное время ответа бота (т.к. ответ бота лежит в await, то 
        # при нескольких сообщениях подряд в очереди - реальное время может различаться, но на эту строчку они придут одновременно)
        # заносим информацию в таблицу сообщений
        dict_mess[str(message.from_user.id)].append(['bot', answer, real_time_now])

        with open('../bot_data/arr_mess.json', 'w') as json_file:
            json.dump(dict_mess, json_file)

        # заносим информацию в таблицу о диалогах
        try:
            dict_discs[str(message.from_user.id)]
        except: 
            dict_discs[str(message.from_user.id)] = []

        dict_discs[str(message.from_user.id)].append([message.text, answer, time_now, emotion])

        with open('../bot_data/arr_discs.json', 'w') as json_file:
            json.dump(dict_discs, json_file)

        # заносим информацию в таблицу с бэкапом памяти моделей
        memory_lain[str(message.from_user.id)] = messages_to_dict(all_convos[str(message.from_user.id)].memory.buffer)

        with open('../bot_data/arr_convos.json', 'w') as json_file:
            json.dump(memory_lain, json_file)


# запуск бота
if __name__ == '__main__':
    executor.start_polling(dp)
