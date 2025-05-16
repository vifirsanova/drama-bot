import os, argparse, re
from dotenv import load_dotenv
from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM
import torch

load_dotenv()

TG_API_KEY = os.getenv("BOT_TOKEN")

parser = argparse.ArgumentParser(description="Инструмент для генерации задач на развитие навыков креативного письма")
parser.add_argument("--prompt", type=str, required=True, help="Промпт пользователя с запросом на генерацию задания")
args = parser.parse_args()
prompt = args.prompt

model_id = "google/gemma-3-1b-it"

# Применяем квантизацию: загружаем модель меньшей размерности
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Инициализация модели из HuggingFace: загружается локально на наше устройство
# Это значит, что она не использует сторонние сервисы, а все вычисления выполняются у нас
model = Gemma3ForCausalLM.from_pretrained(
    model_id, quantization_config=quantization_config
).eval()

# Токенизация тоже производится локально, т.е. на нашем устройстве
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Подгрузка промптов с файла
with open('prompts/generate_task.txt') as f:
    system_prompt = f.read()

# Системные роли удобнее подгружать из отдельного файла
messages = [
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt},]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt},]
        },
    ],
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=1024)

outputs = tokenizer.batch_decode(outputs)

# Добавляем парсинг ответа модели
def extract_between_tokens(text):
    # Using regular expression to find all occurrences between the tokens
    pattern = r'<start_of_turn>(.*?)<end_of_turn>'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

extracted_data = extract_between_tokens(outputs)

for i, data in enumerate(extracted_data, 1):
    print(f"Extracted {i}: {data.strip()}")