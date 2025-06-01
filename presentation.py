import streamlit as st
import reveal_slides as rs
from streamlit.components.v1 import components


def presentation_page():
    st.title("Презентация проекта")

    # Содержание презентации в формате Markdown
    presentation_markdown = """
# Прогнозирование отказов оборудования
---
## Содержание
1. Введение
2. Описание датасета
3. Предобработка данных
4. Модели машинного обучения
5. Результаты обучения
6. Streamlit-приложение
7. Технологический стек
8. Заключение

---

## Введение
- **Задача**: Бинарная классификация отказов промышленного оборудования
- **Цель**: Разработка модели для предсказания отказов с точностью >95%
- **Актуальность**:
  - Снижение простоев оборудования
  - Оптимизация затрат на обслуживание
  - Предотвращение аварийных ситуаций
- **Целевая переменная**: 
  - `0` - отказ не произошел
  - `1` - отказ оборудования

---

## Описание датасета
- **Источник**: AI4I 2020 Predictive Maintenance Dataset (UCI)
- **Объем**: 10,000 записей, 14 признаков
- **Основные признаки**:
  - `Type`: Тип продукта (L/M/H)
  - `Air temperature`: Температура окружающей среды (K)
  - `Process temperature`: Рабочая температура (K)
  - `Rotational speed`: Скорость вращения (rpm)
  - `Torque`: Крутящий момент (Nm)
  - `Tool wear`: Износ инструмента (min)
- **Целевая переменная**: `Machine failure` (0/1)

---

## Предобработка данных
1. **Стандартизация названий столбцов**
2. **Кодирование категориальных признаков**
3. **Масштабирование числовых признаков**
4. **Feature Engineering**:
    - `Power = Rotational speed × Torque`
    - `Temp_diff = Process temp - Air temp`
    - `Wear_Torque = Tool wear × Torque`

---

## Модели машинного обучения
| Модель | Описание | Преимущества |
|--------|----------|-------------|
| **Logistic Regression** | Линейная модель | Простота, интерпретируемость |
| **Random Forest** | Ансамбль деревьев | Высокая точность, устойчивость к переобучению |
| **XGBoost** | Градиентный бустинг | Лучшая производительность на сложных данных |
| **SVM** | Метод опорных векторов | Эффективность на данных высокой размерности |

---

## Результаты обучения
| Модель          | Accuracy | ROC-AUC |
|-----------------|----------|---------|
| Random Forest   | 0.96     | 0.98    |
| XGBoost         | 0.95     | 0.97    |
| Logistic Reg.   | 0.91     | 0.94    |
| SVM             | 0.93     | 0.95    |

### Лучшая модель: Random Forest
  """

    with st.sidebar:
        st.header("Настройки презентации")
        theme = st.selectbox("Тема", ["black", "white", "league", "beige",
            "sky", "night", "serif", "simple", "solarized"])
        height = st.number_input("Высота слайдов", value=500)
        transition = st.selectbox("Переход", ["slide", "convex", "concave",
            "zoom", "none"])
        plugins = st.multiselect("Плагины", ["highlight", "katex",
            "mathjax2", "mathjax3", "notes", "search", "zoom"], [])

    rs.slides(
        presentation_markdown,
        height=height,
        theme=theme,
        config={
            "transition": transition,
            "plugins": plugins,
        },
        markdown_props={"data-separator": "^---$", "data-separator-vertical": "^--$"},
    )
