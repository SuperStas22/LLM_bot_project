import os
import json
import openai
import logging
import time
from langchain.schema import SystemMessage
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# 1. Инициализация: загрузка переменных, настройка логов, создание бота
load_dotenv()  # загружаем .env, чтобы получить ключи
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE", "https://api.essayai.ru/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
BRAND_NAME = os.getenv("BRAND_NAME", "Shoply")

FAQ_PATH = "data/faq.json"
ORDERS_PATH = "data/orders.json"
LOGS_DIR = "logs"

CONTEXT_SIZE = 8 # на всякий случай

## Загрузка файлоы
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
faq = load_json(FAQ_PATH)
orders = load_json(ORDERS_PATH)

# Настраиваем логирование в файл
logging.basicConfig(
    filename=f"logs/session_{int(time.time())}.jsonl", level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logging.info("=== New session ===")

# смотрим есть ли ответ в faq
def find_answear(user_text):
    for item in faq:
        if item['q'].lower() in user_text.lower() or user_text.lower() in item['q'].lower():
            return item['a']
    return None
    
# возращаем инфу по заказу
def get_oreder(order_id):
    order = orders.get(order_id)
    if not order:
        return None

    if order.get('status') == "in_transit":
        return (
            f"Статус заказа {order_id}: в пути, доставка ожидается через {order['eta_days']}"
            f"дней службой {order['carrier']}"
        )
    elif order.get("status") == "delivered":
        date = order.get("delivered_at", "неизвестно когда")
        return f"Статус заказа {order_id}: Доставлен {date}."
    elif order.get("status") == "processing":
        note = order.get("note", "")
        return f"Статус заказа {order_id}: В обработке. {note}".strip()
    else:
        return f"Статус заказа {order_id}: {order.get('status', 'неизвестен')}."

# считаем токены
def count_tokens(usage):
    return usage.get('total_tokens', 0)

def chat_loop():
    print(f"Привет! Я бот поддержки {BRAND_NAME}. Для выхода напишите «выход».\n")

    llm = ChatOpenAI(model_name="gpt-5", temperature=1, request_timeout=25)
    memory = ConversationBufferMemory(k=CONTEXT_SIZE)
    conversation = ConversationChain(llm=llm, memory=memory)
    memory.chat_memory.add_message(
         SystemMessage(content="Ты — полезный, вежливый и точный ассистент-консультант в комапнии по доставке товаров, если к тебе обращаются, то скорее всего не правильно указали номер заказа или такого вопроса не было в faq.")
    )

    while True:
        try:
            user_text = input("Введите ваш запрос: ").strip()
        except KeyboardInterrupt:
            print("\n[Прерывание пользователем]")
            break 

        if user_text == "":
            continue  # пустой ввод - пропускаем
        # Проверка команды выхода
        if user_text.lower() in ("выход", "quit", "exit"):
            print("Бот: До свидания!")
            logging.info("User initiated exit. Session ended.")
            break

        logging.info(f"User: {user_text}")

        ## начало обработки запроса
        if user_text.lower().startswith('/order'):
            order_id = user_text.split()[1].strip()
            status = get_oreder(order_id)
            if status is None:
                bot_answer = f"Извините, заказ с ID {order_id} не найден."
            else:
                bot_answer = status

        else:
            answer = find_answear(user_text)
            if answer is not None:
                bot_answer = answer
            else:
                try:
                    bot_answer = conversation.predict(input=user_text)
                except Exception as e:
                    # Логируем и выводим ошибку, продолжаем чат
                    logging.error(f"Error: {e}")
                    print(f"Бот: [Ошибка] {e}")
                    continue

        bot_answer = bot_answer.strip()
        logging.info(f"Bot: {bot_answer}")
        print(f"Бот: {bot_answer}")

chat_loop()