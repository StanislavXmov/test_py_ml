# test_py_ml

Репозиторий с небольшими ML-экспериментами. Основной интерактивный проект — **крестики-нолики с ИИ**, обученным методом Deep Q-Learning (DQN), и веб-интерфейс для игры против него.

## Крестики-нолики (tictactoe)

Модуль `tictactoe/` содержит нейросеть `QNet` — простую MLP (9 → 128 → 128 → 9), которая оценивает Q-значения для каждой клетки поля.

### Как это работает

- **Обучение** (`tictactoe/train.py`) — агент играет сам с собой (self-play) с ε-greedy exploration и replay buffer. Модель сохраняется в `tictactoe_model.pth`.
- **Игра в консоли** (`tictactoe/play.py`) — загружает обученную модель и позволяет сыграть против ИИ в терминале (код игрового цикла закомментирован, можно раскомментировать).
- **Экспорт весов** (`tictactoe/export_weights.py`) — конвертирует PyTorch-веса в JSON для фронтенда (`frontend/public/weights.json`).

### Запуск (Python)

Зависимости управляются через [uv](https://docs.astral.sh/uv/). Из корня репозитория:

```bash
# Установить зависимости
uv sync

# Обучить модель (~20 000 эпизодов, сохранит tictactoe_model.pth в корень)
uv run python tictactoe/train.py

# Опционально: указать устройство (по умолчанию cuda, если доступна)
uv run python tictactoe/train.py --device cpu

# Экспортировать веса для фронтенда
uv run python tictactoe/export_weights.py

# Консольная игра (после обучения)
cd tictactoe && uv run python play.py
```

> **Примечание:** `export_weights.py` читает `tictactoe_model.pth` из корня репозитория. Если вы обучали модель из папки `tictactoe/`, скопируйте файл в корень или переобучите из корня.

## Frontend

Веб-приложение на **React + TypeScript + Vite** в папке `frontend/`. Игра полностью работает на клиенте: веса нейросети загружаются из `public/weights.json`, inference выполняется в браузере (файл `src/ai/qnet.ts` повторяет архитектуру Python-модели).

- Вы играете за **X**, ИИ — за **O**
- После вашего хода ИИ автоматически выбирает лучший ход по Q-значениям
- Кнопка «Новая игра» сбрасывает поле

### Запуск (frontend)

```bash
cd frontend

# Установить зависимости
npm install

# Dev-сервер (обычно http://localhost:5173)
npm run dev

# Сборка для продакшена
npm run build

# Просмотр production-сборки
npm run preview
```

Перед первым запуском убедитесь, что `frontend/public/weights.json` существует. Если файла нет — выполните экспорт весов (см. выше).

### Проверка паритета Python ↔ TypeScript

Скрипт сравнивает выходы Python-модели и JS-реализации на одних и тех же позициях:

```bash
cd frontend
npm run test:parity
```

## Типичный workflow

1. `uv sync` — установить Python-зависимости
2. `uv run python tictactoe/train.py` — обучить модель
3. `uv run python tictactoe/export_weights.py` — экспортировать веса
4. `cd frontend && npm install && npm run dev` — запустить игру в браузере

## Структура проекта

```
test_py_ml/
├── tictactoe/
│   ├── model.py          # QNet — архитектура нейросети
│   ├── train.py          # DQN-обучение
│   ├── play.py           # игра в консоли
│   └── export_weights.py # экспорт весов в JSON
├── frontend/
│   ├── src/
│   │   ├── ai/qnet.ts    # inference в браузере
│   │   ├── game/board.ts # логика поля и победителя
│   │   └── components/   # React-компоненты
│   └── public/weights.json
├── tictactoe_model.pth   # обученные веса (PyTorch)
└── pyproject.toml        # Python-зависимости (uv)
```
