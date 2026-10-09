"""Когнитивный цикл: экстрактор → монолог → ответ → критик."""

import asyncio
from openai import AsyncOpenAI
from config.settings import OLLAMA_API, MODEL_BRAIN, MODEL_EXTRACTOR, NUM_CTX

client = AsyncOpenAI(base_url=OLLAMA_API, api_key="ollama")


async def extract_facts(text: str) -> str:
    """Экстрактор фактов на 1.7b."""
    r = await client.chat.completions.create(
        model=MODEL_EXTRACTOR,
        messages=[
            {"role": "system", "content": "Извлеки структурированные факты из текста."},
            {"role": "user", "content": text},
        ],
        extra_body={"num_ctx": 8192, "num_predict": 256, "think": False},
    )
    return r.choices[0].message.content


async def monologue(facts: str, memories: list, user_msg: str) -> str:
    """Внутренний монолог мозга."""
    r = await client.chat.completions.create(
        model=MODEL_BRAIN,
        messages=[
            {"role": "system", "content": "Ты ассистент София. Подумай вслух для себя."},
            {"role": "user", "content": f"Факты: {facts}\nВоспоминания: {memories}\nВопрос: {user_msg}"},
        ],
        extra_body={"num_ctx": NUM_CTX, "num_predict": 256, "think": True},
    )
    return r.choices[0].message.content


async def generate_response(monologue_text: str, user_msg: str):
    """Генерация ответа с стримингом."""
    stream = await client.chat.completions.create(
        model=MODEL_BRAIN,
        messages=[
            {"role": "system", "content": "Ответь пользователю. Твой внутренний монолог ниже."},
            {"role": "assistant", "content": monologue_text},
            {"role": "user", "content": user_msg},
        ],
        extra_body={"num_ctx": NUM_CTX, "num_predict": 512, "temperature": 0.7},
        stream=True,
    )
    async for chunk in stream:
        content = chunk.choices[0].delta.content or ""
        if content:
            yield content


async def critic(response: str, facts: str) -> str:
    """Критик: найди ошибки и противоречия."""
    r = await client.chat.completions.create(
        model=MODEL_BRAIN,
        messages=[
            {"role": "system", "content": "Ты критик. Найди ошибки, противоречия, неуверенности."},
            {"role": "user", "content": f"Факты: {facts}\nОтвет: {response}"},
        ],
        extra_body={"num_ctx": NUM_CTX, "num_predict": 256, "think": True},
    )
    return r.choices[0].message.content
