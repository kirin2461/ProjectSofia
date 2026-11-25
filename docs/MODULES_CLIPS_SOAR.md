# Продукционные системы Софии: CLIPS и Soar

Обе системы настоящие — это не имитация и не регулярные выражения.
Если пакет недоступен (например, запуск не из `dist/`), код честно
переключается на резерв и сообщает об этом в `/state`.

## CLIPS — предохранитель ввода и вывода

| Что | Где |
|---|---|
| Движок (clipspy 1.x) | `core/safety_engine.py`, класс `ClipsBackend` |
| Правила | `core/clips_rules/safety.clp` (21 правило) |
| Резерв | те же проверки регулярками в `SafetyEngine._RULES` |

Как устроено. Python режет текст на токены (`tokenize()`) и утверждает
факты вида

```clips
(token (pos 3) (kind word) (text "cvv") (dot no) (digits 0))
```

Правила CLIPS работают по соседям и типам токенов, а не по подстрокам,
поэтому «Покажи карту мира», «карта памяти» и «Что такое CVV?» проходят,
а «мой CVV 123» или «номер карты 4276…» — блокируются. Вердикт правила
пишет фактом:

```clips
(verdict (action block|warn) (rule "financial_cvv") (reason "…") (confidence 0.9))
```

Движок берёт самый строгий вердикт; затем `SafetyEngine` сравнивает его
с результатом регулярных правил и оставляет более строгий (при равенстве
побеждает CLIPS, потому что он объясняет причину).

Проверить вручную:

```powershell
.venv\Scripts\python.exe -c "from core.safety_engine import SafetyEngine; s=SafetyEngine(); print(s.clips.rule_count, s.check_input('мой CVV 123'))"
```

## Soar — выбор стратегии ответа и обучение

| Что | Где |
|---|---|
| Мост (soar-sml 9.6) | `core/soar_bridge.py`, классы `NativeSoar`, `SOARBridge` |
| Правила | `core/soar_rules/sofia.soar` (10 продукций) |
| Выученные чанки | `data/soar_chunks.soar` |
| Резерв | `SimulatedSoar` (те же операторы, обучение по счётчику) |

Каждая реплика — один такт агента `sofia`. На input-link кладётся узел
`^message` с признаками (`greeting`, `question`, `vague`, `context`),
правила предлагают операторы, агент выбирает один и пишет команду на
output-link:

```
op_greet              — поздороваться и предложить помощь
op_answer_question    — отвечать по существу
op_ask_clarification  — задать уточняющий вопрос
```

Неоднозначное сообщение без контекста («как это?») уводит агента в
tie-импасс: два правила предлагают разные операторы. Правило разрешения
импасса выбирает уточнение, и встроенный чанкинг Soar компилирует вывод
в новое правило `chunk*sofia*…`. Чанки сохраняются после одобрения
критиком и загружаются при следующем запуске — то есть София действительно
запоминает принятые решения, а не «делает вид».

Решение Soar попадает в монолог как подсказка (`strategy`), а не как
жёсткое правило: LLM может ответить иначе, если факты требуют.

### Гигиена рабочей памяти

- `init-soar` между репликами не используется: он оставляет старый
  `^message` и ломает разрешение импасса. Вместо него предыдущий `^message`
  уничтожается на input-link.
- Узлы output-link вручную не удаляются — это роняет движок (segfault).
  Вместо этого раз в 250 реплик агент пересоздаётся с перезагрузкой правил
  и чанков (`RESTART_AFTER`).
- Kernel создаётся в `startup()` приложения, то есть в потоке uvicorn:
  SML привязан к потоку создания.

Проверить вручную:

```powershell
.venv\Scripts\python.exe -c "from core.soar_bridge import SOARBridge; s=SOARBridge(); print(s.status()); print(s.step('как это?').operator)"
```

## Что видно в интерфейсе

`/state` отдаёт два блока, и боковая панель показывает их живьём:

```json
"clips": {"active": true, "rules": 21},
"soar":  {"engine": "soar", "active": true, "rules": 10, "chunks": 1,
          "learned": ["chunk*sofia*tie*prefer*clarification*t1-1"]}
```

`engine: "simulated"` или `active: false` означает, что настоящий движок
не поднялся — это видно сразу, без чтения логов.

## Выключатели и диагностика

В `.env` рядом с `Sofia.exe` (или в `SOFIA_HOME`):

| Переменная | Что делает |
|---|---|
| `SOAR_ENABLED=false` | не поднимать Soar (стратегия — по умолчанию) |
| `TTS_ENABLED=false` | не озвучивать ответы |
| `SOFIA_LOG_LEVEL=DEBUG` | писать в `logs/sofia.log` каждый шаг реплики (`шаг safety in`, `шаг soar`, `шаг tts start`…) |

Если процесс падает в нативном коде, `faulthandler` пишет стек Python
в `logs/sofia-fatal.log`.

> Сборка: в `sofia.spec` нельзя исключать `unittest`. Его импортирует
> `torch.utils._config_module`. Без него `torch.distributed` инициализируется
> наполовину, и при первой озвучке exe падает с кодом `0xC0000409`
> («GradBucket already defined»). Это проверяет `tests/unit/test_spec.py`.
