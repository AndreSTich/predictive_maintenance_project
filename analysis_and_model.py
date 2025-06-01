import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import GridSearchCV


# Инициализация состояния сеанса
def init_session_state():
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'model_type' not in st.session_state:
        st.session_state.model_type = "Random Forest"
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    if 'le' not in st.session_state:
        st.session_state.le = None
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'feature_engineered' not in st.session_state:
        st.session_state.feature_engineered = False


def standardize_column_names(df):
    """Приведение названий столбцов к стандартному формату"""
    new_columns = []
    for col in df.columns:
        col = re.sub(r'[\[\]]', '', col)
        col = re.sub(r'\s+', ' ', col).strip()
        col_lower = col.lower()

        if 'air' in col_lower and 'temp' in col_lower:
            new_columns.append('Air temperature')
        elif 'process' in col_lower and 'temp' in col_lower:
            new_columns.append('Process temperature')
        elif 'rotation' in col_lower and 'speed' in col_lower:
            new_columns.append('Rotational speed')
        elif 'torque' in col_lower:
            new_columns.append('Torque')
        elif 'tool' in col_lower and 'wear' in col_lower:
            new_columns.append('Tool wear')
        elif 'type' in col_lower:
            new_columns.append('Type')
        elif 'machine' in col_lower and 'failure' in col_lower:
            new_columns.append('Machine failure')
        elif 'udi' in col_lower:
            new_columns.append('UDI')
        elif 'product' in col_lower and 'id' in col_lower:
            new_columns.append('Product ID')
        else:
            new_columns.append(col)

    df.columns = new_columns
    return df


def add_engineered_features(df):
    """Добавление новых признаков на основе предметной области"""
    # Мощность = Скорость вращения * Крутящий момент
    df['Power'] = df['Rotational speed'] * df['Torque']

    # Разница температур
    df['Temp_diff'] = df['Process temperature'] - df['Air temperature']

    # Произведение износа и крутящего момента (для определения OSF)
    df['Wear_Torque'] = df['Tool wear'] * df['Torque']

    # Пороговые признаки для типов отказов
    df['TWF_risk'] = ((df['Tool wear'] >= 200) & (df['Tool wear'] <= 240)).astype(int)
    df['HDF_risk'] = ((df['Temp_diff'] < 8.6) & (df['Rotational speed'] < 1380)).astype(int)
    df['PWF_risk'] = ((df['Power'] < 3500) | (df['Power'] > 9000)).astype(int)

    # Пороговые значения OSF в зависимости от типа продукта
    df['OSF_threshold'] = df['Type'].map({'L': 11000, 'M': 12000, 'H': 13000})
    df['OSF_risk'] = (df['Wear_Torque'] > df['OSF_threshold']).astype(int)

    return df


def analysis_and_model_page():
    init_session_state()
    st.title("📊 Анализ данных и модель")

    # Загрузка данных
    st.header("1. Загрузка данных")
    uploaded_file = st.file_uploader("Загрузите CSV-файл", type="csv")

    if uploaded_file is not None:
        try:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.session_state.data = standardize_column_names(st.session_state.data)
            st.session_state.data_loaded = True
            st.success(f"Файл {uploaded_file.name} успешно загружен!")
        except Exception as e:
            st.error(f"Ошибка при чтении файла: {e}")

    if st.button("Использовать встроенный датасет"):
        try:
            dataset = fetch_ucirepo(id=601)
            st.session_state.data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
            st.session_state.data = standardize_column_names(st.session_state.data)
            st.session_state.data_loaded = True
            st.success("Данные UCI успешно загружены!")
        except Exception as e:
            st.error(f"Ошибка загрузки данных: {e}")

    if not st.session_state.data_loaded:
        st.info("Загрузите данные или используйте встроенный датасет")
        return

    # Отладочная информация
    st.subheader("Структура данных после обработки")
    st.write(f"Количество строк: {st.session_state.data.shape[0]}, столбцов: {st.session_state.data.shape[1]}")
    st.write("Первые 3 строки данных:")
    st.dataframe(st.session_state.data.head(3))

    # Проверка наличия необходимых столбцов
    required_columns = {
        'Type',
        'Air temperature',
        'Process temperature',
        'Rotational speed',
        'Torque',
        'Tool wear',
        'Machine failure'
    }

    if not required_columns.issubset(st.session_state.data.columns):
        missing = required_columns - set(st.session_state.data.columns)
        st.error(f"В данных отсутствуют необходимые столбцы: {', '.join(missing)}")
        st.warning("Пожалуйста, убедитесь, что в данных есть следующие столбцы:")
        st.write(list(required_columns))
        return

    # Добавление новых признаков
    if not st.session_state.feature_engineered:
        st.session_state.data = add_engineered_features(st.session_state.data)
        st.session_state.feature_engineered = True
        st.success("Добавлены новые признаки на основе предметной области!")

    # Предобработка
    st.header("2. Предобработка данных")
    data_cleaned = st.session_state.data.drop(
        columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'],
        errors='ignore'
    )

    if st.session_state.le is None:
        st.session_state.le = LabelEncoder()
        data_cleaned['Type'] = st.session_state.le.fit_transform(data_cleaned['Type'])
    else:
        data_cleaned['Type'] = st.session_state.le.transform(data_cleaned['Type'])

    # Масштабирование числовых признаков
    num_cols = [
        'Air temperature',
        'Process temperature',
        'Rotational speed',
        'Torque',
        'Tool wear',
        'Power',
        'Temp_diff',
        'Wear_Torque'
    ]

    # Только существующие столбцы
    num_cols = [col for col in num_cols if col in data_cleaned.columns]

    if st.session_state.scaler is None:
        st.session_state.scaler = StandardScaler()
        data_cleaned[num_cols] = st.session_state.scaler.fit_transform(data_cleaned[num_cols])
    else:
        data_cleaned[num_cols] = st.session_state.scaler.transform(data_cleaned[num_cols])

    # Разделение данных
    X = data_cleaned.drop(columns=['Machine failure'])
    y = data_cleaned['Machine failure']

    if st.session_state.X_train is None:
        (
            st.session_state.X_train,
            st.session_state.X_test,
            st.session_state.y_train,
            st.session_state.y_test
        ) = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        st.info("Данные разделены на обучающую и тестовую выборки")

    # Обучение моделей
    st.header("3. Обучение моделей")
    model_type = st.selectbox(
        "Выберите модель",
        ["Logistic Regression", "Random Forest", "XGBoost", "SVM"],
        key="model_select"
    )

    if st.button("Обучить модель") or st.session_state.model_trained:
        if not st.session_state.model_trained or model_type != st.session_state.model_type:
            # Создаем новую модель только если тип изменился или модель еще не обучена
            if model_type == "Logistic Regression":
                model = LogisticRegression(max_iter=1000, class_weight='balanced')
            elif model_type == "Random Forest":
                # Улучшенная модель Random Forest
                model = RandomForestClassifier(
                    n_estimators=500,
                    max_depth=12,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                )
            elif model_type == "XGBoost":
                model = XGBClassifier(
                    n_estimators=500,
                    max_depth=7,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=9,  # Учет дисбаланса классов (10% отказов)
                    random_state=42
                )
            else:
                model = SVC(probability=True, class_weight='balanced')

            # Обучение с индикатором прогресса
            with st.spinner(f"Обучение модели {model_type}..."):
                model.fit(st.session_state.X_train, st.session_state.y_train)

            st.session_state.model = model
            st.session_state.model_type = model_type
            st.session_state.model_trained = True
            st.success(f"Модель {model_type} успешно обучена!")
        else:
            model = st.session_state.model
            st.info(f"Используется ранее обученная модель {model_type}")

        # Оценка
        y_pred = model.predict(st.session_state.X_test)
        y_proba = model.predict_proba(st.session_state.X_test)[:, 1] if hasattr(model, "predict_proba") else None

        st.subheader("Результаты оценки")
        col1, col2 = st.columns(2)
        accuracy = accuracy_score(st.session_state.y_test, y_pred)
        col1.metric("Accuracy", f"{accuracy:.4f}")

        if y_proba is not None:
            roc_auc = roc_auc_score(st.session_state.y_test, y_proba)
            col2.metric("ROC-AUC", f"{roc_auc:.4f}")
        else:
            col2.metric("ROC-AUC", "N/A")

        # Confusion Matrix
        st.subheader("Матрица ошибок")
        fig, ax = plt.subplots()
        sns.heatmap(
            confusion_matrix(st.session_state.y_test, y_pred),
            annot=True, fmt='d', cmap='Blues',
            ax=ax,
            xticklabels=['Нет отказа', 'Отказ'],
            yticklabels=['Нет отказа', 'Отказ']
        )
        ax.set_xlabel('Предсказание')
        ax.set_ylabel('Факт')
        st.pyplot(fig)

        # ROC Curve
        if y_proba is not None:
            st.subheader("ROC-кривая")
            fpr, tpr, _ = roc_curve(st.session_state.y_test, y_proba)
            plt.figure()
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            st.pyplot(plt)

            # Рассчет точности для положительного класса
            precision = confusion_matrix(st.session_state.y_test, y_pred)[1, 1] / (
                        confusion_matrix(st.session_state.y_test, y_pred)[1, 1] +
                        confusion_matrix(st.session_state.y_test, y_pred)[0, 1])
            recall = confusion_matrix(st.session_state.y_test, y_pred)[1, 1] / (
                        confusion_matrix(st.session_state.y_test, y_pred)[1, 1] +
                        confusion_matrix(st.session_state.y_test, y_pred)[1, 0])
            st.info(f"Precision (точность): {precision:.4f} | Recall (полнота): {recall:.4f}")

    # Предсказание
    st.header("4. Предсказание")
    with st.form("prediction_form"):
        st.write("### Введите параметры оборудования:")

        col1, col2 = st.columns(2)
        air_temp = col1.number_input("Температура воздуха [K]", 295.0, 305.0, 300.0)
        process_temp = col2.number_input("Рабочая температура [K]", 305.0, 315.0, 310.0)

        col1, col2 = st.columns(2)
        rotational_speed = col1.number_input("Скорость вращения [rpm]", 1000, 3000, 1500)
        torque = col2.number_input("Крутящий момент [Nm]", 10.0, 100.0, 40.0)

        col1, col2 = st.columns(2)
        tool_wear = col1.number_input("Износ инструмента [min]", 0, 300, 50)
        product_type = col2.selectbox("Тип продукта", ["L", "M", "H"])

        submit_button = st.form_submit_button("Предсказать вероятность отказа")

        if submit_button:
            input_data = pd.DataFrame({
                'Type': [product_type],
                'Air temperature': [air_temp],
                'Process temperature': [process_temp],
                'Rotational speed': [rotational_speed],
                'Torque': [torque],
                'Tool wear': [tool_wear]
            })

            # Добавляем новые признаки
            input_data = add_engineered_features(input_data)

            # Преобразование типа продукта
            input_data['Type'] = st.session_state.le.transform(input_data['Type'])

            # Масштабирование - только существующие столбцы
            input_data_cols = [col for col in num_cols if col in input_data.columns]
            input_data[input_data_cols] = st.session_state.scaler.transform(input_data[input_data_cols])

            # Проверка наличия модели
            if st.session_state.model is None:
                st.warning("Модель не обучена! Используется улучшенная модель Random Forest")
                model = RandomForestClassifier(
                    n_estimators=500,
                    max_depth=12,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight='balanced',
                    random_state=42
                )
                model.fit(st.session_state.X_train, st.session_state.y_train)
                st.session_state.model = model
                st.session_state.model_type = "Random Forest"
                st.session_state.model_trained = True
            else:
                model = st.session_state.model

            prediction = model.predict(input_data)[0]

            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(input_data)[0][1]
            else:
                probability = prediction

            # Визуализация результата
            st.subheader("Результат предсказания")

            # Рассчет рисков по правилам
            power = rotational_speed * torque
            temp_diff = process_temp - air_temp
            wear_torque = tool_wear * torque

            # Пороговые значения
            osf_threshold = {'L': 11000, 'M': 12000, 'H': 13000}[product_type]

            # Проверка условий отказов
            twf_risk = 200 <= tool_wear <= 240
            hdf_risk = (temp_diff < 8.6) and (rotational_speed < 1380)
            pwf_risk = (power < 3500) or (power > 9000)
            osf_risk = wear_torque > osf_threshold

            # Отображение информации о рисках
            st.info("### Анализ рисков по правилам:")
            cols = st.columns(4)
            cols[0].metric("TWF Риск", "✅" if not twf_risk else "❌", "Износ инструмента")
            cols[1].metric("HDF Риск", "✅" if not hdf_risk else "❌", "Теплоотвод")
            cols[2].metric("PWF Риск", "✅" if not pwf_risk else "❌", "Мощность")
            cols[3].metric("OSF Риск", "✅" if not osf_risk else "❌", "Перегрузка")

            # Основной результат
            if prediction == 1 or any([twf_risk, hdf_risk, pwf_risk, osf_risk]):
                st.error(f"❌ ВЫСОКАЯ ВЕРОЯТНОСТЬ ОТКАЗА: {probability:.2%}")
                st.progress(float(probability))
                if twf_risk:
                    st.warning("⚠️ Риск отказа из-за износа инструмента (TWF)")
                if hdf_risk:
                    st.warning("⚠️ Риск отказа из-за недостаточного теплоотвода (HDF)")
                if pwf_risk:
                    st.warning(f"⚠️ Риск отказа из-за проблем с мощностью (PWF) - мощность: {power:.1f} Вт")
                if osf_risk:
                    st.warning(
                        f"⚠️ Риск отказа из-за перегрузки (OSF) - произведение износа и момента: {wear_torque:.0f} > {osf_threshold}")
            else:
                st.success(f"✅ Низкая вероятность отказа: {probability:.2%}")
                st.progress(float(probability))

            # Дополнительная информация
            st.info(f"Модель: {st.session_state.model_type}")
            st.info(f"Вероятность отказа: {probability:.4f}")
            st.info(f"Расчетная мощность: {power:.1f} Вт, Разница температур: {temp_diff:.1f} K")