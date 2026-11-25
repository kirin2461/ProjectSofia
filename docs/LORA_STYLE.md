# Стилевая LoRA: отдельный шаг, не часть exe

Зачем: RAG даёт **знания** (факты из книг), а LoRA — **манеру**: обращение на
«ты», спокойный тон без «Отличный вопрос!», короткие абзацы, русская
терминология. Это разные задачи, и мешать их в одну сборку нельзя.

Почему обучение не внутри `Sofia.exe`:

- QLoRA даже на 4B требует ~8–10 ГБ VRAM и часы; у пользователя в это время
  должна работать сама София, а не веса;
- обучение — редкое действие (раз в несколько месяцев), а `.exe` запускается
  каждый день;
- GGUF-адаптер — отдельный файл на 50–200 МБ; его проще распространять
  отдельно от 9-ГБ сборки.

Итог: тренируете один раз на своей машине, получаете `style-lora.gguf`,
кладёте рядом с моделью — `llama-server` подхватывает его флагом `--lora`.

## 1. Датасет

Нужны пары «вопрос → ответ в нужном тоне». Готовых нет — их собирает скрипт
из ваших же книг:

```
.venv/Scripts/python.exe scripts/build_style_dataset.py --limit 200 --dry-run
.venv/Scripts/python.exe scripts/build_style_dataset.py --limit 200 \
    --out data/lora/style.jsonl
```

Что делает: берёт фрагменты из индекса (`library__e5s`), просит модель
сформулировать по фрагменту один вопрос и ответить на него в заданном стиле,
отбрасывает брак (не JSON, слишком короткий/длинный ответ, повторы) и пишет
JSONL в формате `messages` — его понимают и `trl`, и chat-шаблоны.

Требуется запущенный `llama-server` (генерация идёт через него же). 200
фрагментов ≈ 200 запросов ≈ 15–25 минут на 4B.

Качество датасета важнее размера: 200 аккуратных пар работают лучше 2000
случайных. Проверьте глазами первые двадцать строк — если модель отвечает
«Конечно! Вот...», добавьте в `PROMPT` скрипта запрет явно, а не надейтесь,
что обучение это исправит.

## 2. Обучение (QLoRA, вне сборки)

Отдельное окружение, чтобы не ломать рабочее:

```powershell
python -m venv .venv-train
.venv-train\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cu128 torch
pip install transformers peft trl datasets accelerate bitsandbytes
```

Скрипт обучения (`train_style_lora.py`, пишется по мере надобности) — обычный
`SFTTrainer`:

```python
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

ds = load_dataset("json", data_files="data/lora/style.jsonl", split="train")
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                 task_type="CAUSAL_LM")
trainer = SFTTrainer(
    model="<локальная копия базовой модели в HF-формате>",
    train_dataset=ds,
    peft_config=cfg,
    args=SFTConfig(output_dir="data/lora/out", num_train_epochs=2,
                   per_device_train_batch_size=1, gradient_accumulation_steps=8,
                   learning_rate=1e-4, max_length=1024,
                   gradient_checkpointing=True, bf16=True),
)
trainer.train()
trainer.save_model("data/lora/out")
```

Ориентиры: 4B, r=16, 2 эпохи, 200–500 пар — 30–90 минут на 12 ГБ VRAM. Больше
эпох — модель начнёт цитировать книги дословно, это видно по ответам.

## 3. Конвертация в GGUF

```powershell
python convert_lora_to_gguf.py data/lora/out --outfile style-lora.gguf --outtype f16
```

(`convert_lora_to_gguf.py` — из репозитория llama.cpp.)

## 4. Подключение

`llama-server` запускается лаунчером; адаптер добавляется флагом:

```
llama-server ... --lora style-lora.gguf
```

Проверка: одинаковый вопрос с `--lora` и без. Стиль должен меняться, а факты —
нет. Если модель начала путать числа и формулы — переобучение, уменьшайте
эпохи или `r`.

## 5. Как это связано с остальным

| Что | Откуда знания | Откуда стиль |
|---|---|---|
| Ответ без книг | веса базовой модели | LoRA (если подключена) |
| Ответ по книге | RAG: фрагмент в промпте + цитата | LoRA |
| Память о вас | эпизоды в Chroma, сводка диалога | LoRA |

LoRA не заменяет RAG и не «запоминает» книги: адаптер на 200 парах не удержит
содержание учебника, а поиск по фрагментам удерживает.

## 6. Чего ждать не стоит

- **Новых знаний.** LoRA на стилевых парах учит форму, не факты.
- **Идеального русского.** Акцент и кальки базовой модели ослабнут, но не
  исчезнут; 4B остаётся 4B.
- **Работы «из коробки».** Нужен HF-формат базовой модели (не GGUF) —
  конвертация GGUF → HF для обучения отдельная возня.
