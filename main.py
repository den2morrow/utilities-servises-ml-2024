#@title Download librarys, login HuggingFace, Ignore warnings
# !pip install sweetviz transformers langchain huggingface_hub python-dotenv -q
#########################


import os
import re
import csv
from random import randint
from time import sleep, time
from tqdm import tqdm

import pandas as pd
import requests
from dotenv import load_dotenv
from transformers import pipeline

#@title Get Google Secrets Keys
from google.colab import userdata
##############################

#@title Login in HuggingFace
from huggingface_hub import login
############################

#@title Ignore warnings
import warnings
#######################


load_dotenv()
login(os.getenv('HUG_FACE_API_KEY'))   # login(userdata.get('hugface'))
warnings.filterwarnings('ignore')


class AddressExtractor:
    def __init__(
        self,
        api_key: str,
        addresses: str = "/content/drive/MyDrive/saved_datasets/VolgaIT_2024_semifinal/volgait2024-semifinal-addresses.csv",
        tasks: str = "/content/drive/MyDrive/saved_datasets/VolgaIT_2024_semifinal/volgait2024-semifinal-task.csv",
    ) -> None:
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.addresses = pd.read_csv(addresses, delimiter=";")
        self.tasks = pd.read_csv(tasks, delimiter=";")

    def get_postprocess_text(self, text: str) -> str:
        """Preprocess the output text for address extraction."""
        text = text.lower()
        text = re.sub(r"\bпос\.?\b", "поселок", text)  # "пос." or "пос"
        text = re.sub(r"\bп\.?\b", "поселок", text)  # "п." or "п"
        text = re.sub(r"\bул\.?\b", "улица", text)  # "ул." or "ул"
        text = re.sub(r"\bпер\.?\b", "переулок", text)  # "пер." or "пер"
        text = re.sub(r"\bобл\.?\b", "область", text)  # "обл." or "обл"
        return text

    def query_local_model(self, prompt: list[dict[str, str]]) -> dict:
        """Query the local model with the given prompt."""
        # Load the model
        # pipe = pipeline(
        #     "text-generation",
        #     model="Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct-abliterated",
        # )
        pipe = pipeline("text-generation",
                        model="Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct",
                        device_map='auto',
                        do_sample=False,
                        # temperature=0.5,
                        # top_k=50
                        # top_p=0.95,
                        )

        response = pipe(prompt, max_new_tokens=512)

        return response[0]["generated_text"][2]['content']

    def query_groq(self, prompt: list[dict[str, str]]) -> dict:
        """Query the Groq API with the given prompt."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "llama-3.1-70b-versatile",
            "messages": prompt,
            "max_tokens": 500,
        }

        response = requests.post(self.base_url, headers=headers, json=data, verify=False)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")

    def create_prompt(self, comment: str) -> list[dict[str, str]]:
        """Create a prompt for the API from the given comment."""
        # ул пер с (тер. СНТ) пр-кт ш проезд (им <Имя> п) д п тер.
        template = """Ты — система извлечения информации для адресов.
                          Твоя задача — извлечь все упомянутые адреса из данного текста и вернуть их в формате:
                          тип_адреса (ул, пр-кт, ш, пер. и т.д.), номер дома и запятую, корпус_номеркорпуса (только если он есть в принципе).
                          Все адреса должны быть строго в сокращенном виде, как указано ниже.

                          Вот несколько важных указаний:
                          1. Обрати внимание на общепринятые сокращения, такие как "ул" для "улица", "д" для "дом", "к" или "корп" для "корпус", "пер" для "переулок", "пр-кт" для "проспект", "ш" для "шоссе", "проезд" для "проезд".
                          2. Все сокращения должны оставаться в виде сокращений, как они указаны в тексте.
                          3. Если в тексте упоминаются несколько домов, извлеки все адреса.
                          4. Используй форматирование: каждый адрес должен быть на новой строке.
                          5. Будь внимателен к пробелам и пунктуации.
                          6. В ответе пиши только адреса без своих комментариев.
                          7. Если написано число1-число2, то это все числа от числа1 до числа2 должны быть.
                          8. Если нет упоминаний адресов, то ВООБЩЕ ничего не пиши
                          9. Город, область, край, страну не указывай

                          Пример:
                          Текст: "из п/з Д=100, без ХВС К. Либкнехта 21, 23, 23а 25"
                          Ожидаемый ответ:
                          Карла Либкнехта ул, 21
                          Карла Либкнехта ул, 23
                          Карла Либкнехта ул, 23А
                          Карла Либкнехта ул, 25

                          Текст: "ремонт пожарного гидранта, без ХВС Московское шоссе 1ж (корп. 1,2,3)"
                          Ожидаемый ответ:
                          Московское ш, 1Ж к. 1
                          Московское ш, 1Ж к. 2
                          Московское ш, 1Ж к. 3

                          Текст: "Московский пр-кт 15, 17 и 19, а также пр. Ленина 5"
                          Ожидаемый ответ:
                          Московский пр-кт, 15
                          Московский пр-кт, 17
                          Московский пр-кт, 19
                          Пр. Ленина, 5

                          Теперь, пожалуйста, извлеки адреса из следующего комментария: {comment}"""

        messages = [
            {"role": "system", "content": template},
            {"role": "user", "content": comment},
        ]

        return messages

    def extract_and_save_uuids(self, text: str) -> str:
        uuids = []
        ext_addresses = self.extract_addresses_from_comment(text)

        for address in ext_addresses.split("\n"):  # was tqdm(...)
            try:
                address_type, number = address.split(",")
                matches_houses = self.find_match_house_uuid(
                    self.addresses, address_type, number
                )
                if matches_houses != "":
                    uuids.append(matches_houses)
            except ValueError:
                continue

        with open('./data/volgait2024-semifinal-house-uuids.csv', mode='a',
                newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')

                # Записываем заголовки столбцов, если файл пустой
                if csvfile.tell() == 0:
                    writer.writerow(["house_uuid"])
                
                writer.writerow([",".join(uuids) + '\n'])
        
        return ",".join(uuids)

    def extract_addresses_from_comment(self, comment: str) -> str:
        """Extract addresses from a given comment."""
        sleep(randint(2, 5))
        prompt = self.create_prompt(comment)
        # response = self.query_groq(prompt)
        response = self.query_local_model(prompt)

        return response

    def find_match_house_uuid(self, df: pd.DataFrame, street: str, house_number: str):
        # Убираем лишние пробелы и приводим к нижнему регистру
        street = street.strip().lower()

        # Фильтрация DataFrame с использованием регулярных выражений
        mask = df["house_full_address"].str.contains(
            r"\b" + re.escape(street), case=False
        ) & df["house_full_address"].str.contains(
            r"(?<!\d)" + re.escape(house_number) + r"(?![\w\S])",
            case=False,  # Поиск номера как отдельного слова
        )

        # Получение house_uuid по маске
        matching_houses = df.loc[mask, "house_uuid"].tolist()

        return ",".join(matching_houses)



def main():

    tqdm.pandas()

    api_groq = os.getenv("API_GROQ", "")
    api_groq = api_groq if api_groq != "" else userdata.get('qroq')
    extractor = AddressExtractor(
        api_key=api_groq,
        addresses='/content/drive/MyDrive/saved_datasets/VolgaIT_2024_semifinal/volgait2024-semifinal-addresses.csv',
        tasks='/content/drive/MyDrive/saved_datasets/VolgaIT_2024_semifinal/volgait2024-semifinal-task.csv')

    extractor.tasks.iloc[0:]["comment"].progress_apply(
        extractor.extract_and_save_uuids
    )

    shutdown_id_col = extractor.tasks[['shutdown_id']]
    house_uuid_col = pd.read_csv('./data/volgait2024-semifinal-house-uuids.csv',
                                 delimiter=';',
                                 encoding='utf-8')
    result = pd.concat([shutdown_id_col, house_uuid_col], ignore_index=True)
    result.to_csv('./data/volgait2024-semifinal-result.csv', encoding='utf-8', index=False)


if __name__ == "__main__":
    start_time = time()
    main()
    print(f"Work time: {time() - start_time}")