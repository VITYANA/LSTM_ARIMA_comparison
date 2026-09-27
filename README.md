# Сравнение ARIMA, LSTM и ARIMA-LSTM

Учебный исследовательский проект по сравнению моделей ARIMA, LSTM и гибридной
ARIMA-LSTM в задаче однодневного прогнозирования доходности финансовых активов.

## Исследовательский вопрос

Даёт ли моделирование нелинейной структуры остатков ARIMA с помощью LSTM
статистически и практически значимое улучшение прогноза доходности следующего
торгового дня?

## Предварительный дизайн

- Активы: `SPY`, `AAPL`, `JPM`, `XOM`.
- Дополнительный стресс-тест: `TSLA`.
- Период исследования: 3 января 2006 года — 31 декабря 2025 года.
- Основная целевая переменная: логарифмическая доходность следующего торгового дня.
- Дополнительные цели: восстановленная цена и направление движения.
- Сравниваемые подходы: наивный прогноз, ARIMA, LSTM и ARIMA-LSTM.
- Проверка качества: expanding-window backtest без перемешивания наблюдений.

## Основные гипотезы

1. Хотя бы одна обучаемая модель превосходит наивный прогноз.
2. Гибрид ARIMA-LSTM точнее отдельных ARIMA и LSTM.
3. Ранжирование моделей зависит от режима волатильности.
4. Результаты устойчивы между активами и случайными инициализациями LSTM.

## Метрики

Для доходности будут рассчитаны MAE и RMSE в базисных пунктах, относительная
MAE, вневыборочный R² и доля правильно предсказанных направлений движения.
Различия между моделями будут дополнительно проверены тестом Диболда—Мариано
и блочным bootstrap.

## Данные

Основной источник — Yahoo Finance через `yfinance`. Резервный источник — Tiingo
EOD API. Исходные снимки данных будут сохраняться вместе с датой загрузки,
параметрами запроса и контрольными суммами SHA-256.

Рыночные данные не включаются в Git-репозиторий и не покрываются лицензией
исходного кода.

## Структура проекта

```text
.
├── analysis/
│   ├── exploration.py         # статистика, корреляции и rolling volatility
│   ├── loading.py             # проверяемая загрузка сырого снимка
│   ├── preparation.py         # цель следующего дня и временные выборки
│   ├── quality.py             # контроль качества и доходности
│   └── stationarity.py        # ACF/PACF, Ljung–Box, ADF и KPSS
├── data/
│   ├── data.py                # загрузка и фиксация сырых данных
│   ├── prepare.py             # обработанный снимок и его манифест
│   ├── raw/                   # локальные исходные снимки
│   └── processed/             # локальный модельный набор и манифест
├── evaluation/
│   ├── contracts.py           # общий контракт таблицы прогнозов
│   ├── loading.py             # проверяемая загрузка модельного набора
│   ├── metrics.py             # относительные и pooled-метрики
│   ├── pipeline.py            # стандартизация прогнозов моделей
│   ├── selection.py           # выбор порядка ARIMA на validation
│   └── walk_forward.py        # ежедневный expanding-window backtest
├── models/
│   ├── arima.py               # поиск, обучение и BIC shortlist ARIMA
│   ├── interfaces.py          # структурный интерфейс ForecastModel
│   └── naive.py               # нулевой прогноз доходности
├── notebooks/
│   ├── 01_data_quality.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_stationarity_analysis.ipynb
│   ├── 04_evaluation_baseline.ipynb
│   └── 05_arima_validation.ipynb
├── docs/
│   ├── arima_protocol.md      # train shortlist и validation selection
│   ├── evaluation_protocol.md # критерии этапа оценки
│   └── research_protocol.md   # протокол эксперимента
└── tests/
    ├── analysis/
    ├── data/
    ├── evaluation/
    │   ├── test_selection.py
    │   └── test_walk_forward.py
    └── models/
        └── test_arima.py      # структура тестов повторяет рабочие модули
```

## Установка зависимостей

Проект использует Python 3.11 и Poetry:

```bash
poetry install --with dev
```

Создать модельный набор из проверенного сырого снимка:

```bash
poetry run python -m data.prepare
```

Команда создаёт локальный файл
`data/processed/next_trading_day_returns.csv`, который игнорируется Git.
Отслеживаемый файл
`data/processed/next_trading_day_returns.manifest.json` фиксирует параметры
подготовки, число строк, размер и SHA-256 набора данных.

Запустить JupyterLab из корня проекта:

```bash
poetry run jupyter lab
```

Ноутбуки `01_data_quality.ipynb`, `02_exploratory_analysis.ipynb` и
`03_stationarity_analysis.ipynb` содержат выполненный аудит данных, EDA и
диагностику стационарности. Они читают снимок через манифест и не изменяют
сырые данные.

## Воспроизведение наивного baseline

Перед запуском должен существовать обработанный снимок
`data/processed/next_trading_day_returns.csv` и соответствующий манифест. Если
снимок отсутствует, создайте его командой `poetry run python -m data.prepare`.

Выполнить notebook и сохранить результаты:

```bash
poetry run jupyter execute notebooks/04_evaluation_baseline.ipynb --inplace
```

Notebook рассчитывает baseline отдельно для train и validation. Final test не
передаётся модели и не отображается в результатах.

## Воспроизведение выбора ARIMA

Перед запуском должен существовать проверяемый обработанный снимок
`data/processed/next_trading_day_returns.csv` и соответствующий манифест. При
необходимости создайте их командой `poetry run python -m data.prepare`.

Выполнить train-поиск, ежедневный validation backtest и сохранить результаты:

```bash
poetry run jupyter execute notebooks/05_arima_validation.ipynb --inplace
```

Notebook проверяет 36 порядков `ARIMA(p, 0, q)` для каждого основного тикера,
формирует BIC top-3 и выбирает окончательный порядок по validation MAE. В
сохранённом запуске выбраны `ARIMA(0,0,0)` для AAPL, `ARIMA(0,0,1)` для JPM,
`ARIMA(0,0,2)` для SPY и `ARIMA(2,0,1)` для XOM. На объединённой validation
ARIMA получила MAE `84.314` б.п. против `84.101` б.п. у `naive_zero`. Эти
значения относятся к model selection; final test остаётся закрытым.

## Проверки качества

Локально выполняются те же проверки, что и в GitHub Actions:

```bash
poetry check --lock
poetry run black --check .
poetry run ruff check .
poetry run mypy
poetry run pytest
poetry run pre-commit run --all-files
```

## Git-хуки

Установить хуки после `poetry install`:

```bash
poetry run pre-commit install
```

Запустить все хуки вручную:

```bash
poetry run pre-commit run --all-files
```

## Статус

Подготовлены воспроизводимый конвейер данных, модельный набор, EDA, диагностика
стационарности, evaluation pipeline, наивный baseline и выполненный
двухступенчатый выбор ARIMA на validation. Порядки ARIMA зафиксированы, а final
test остаётся закрытым. LSTM и ARIMA-LSTM выполняются на следующих этапах.

## Лицензия

Исходный код распространяется по лицензии MIT. См. файл [LICENSE](LICENSE).
