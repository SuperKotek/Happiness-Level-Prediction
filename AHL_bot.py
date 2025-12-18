import logging
import os
import traceback
import tempfile
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    ApplicationBuilder, ContextTypes, CommandHandler,
    MessageHandler, filters, ConversationHandler
)

BOT_TOKEN = "8432360038:AAFfuiuTYCUcKHckjBkIjBHKsNNoLjPyuyk"

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Глобальное хранилище данных по пользователям
user_data = {}

TARGET_COLUMN = "Happiness Score"
NUMERIC_COLUMNS = [
    "Happiness Score", "Standard Error", "Economy", "Family", "Health",
    "Freedom", "Trust Government", "Generosity", "Dystopia Residual"
]

# Состояния
AWAITING_MODEL_CHOICE, = range(1)

MODEL_NAMES = [
    "Линейная регрессия",
    "Нейронная сеть (MLP)",
    "Дерево решений",
    "Случайный лес"
]


# Вспомогательные функции 
def handle_missing_and_outliers(df):
    report = []
    df = df.copy()
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        series = df[col].copy()
        n_missing = series.isna().sum()
        if n_missing > 0:
            mean_val = series.mean()
            df[col] = df[col].fillna(mean_val)
            report.append(f"{col}: заполнено {n_missing} пропусков средним ({mean_val:.3f})")

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_low = series < lower_bound
        outliers_high = series > upper_bound
        n_outliers = outliers_low.sum() + outliers_high.sum()

        if n_outliers > 0:
            df.loc[outliers_low, col] = lower_bound
            df.loc[outliers_high, col] = upper_bound
            report.append(f"{col}: обработано {n_outliers} выбросов")
    return df, report


def create_overall_r2_plot(results):
    plt.close('all')
    model_names = list(results.keys())
    r2_scores = [results[name]["r2"] for name in model_names]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#667eea", "#764ba2", "#f093fb", "#6fcf97"]
    bars = ax.bar(model_names, r2_scores, color=colors)
    ax.set_title('Сравнение R² по моделям', fontsize=14, weight='bold')
    ax.set_ylabel('Коэффициент детерминации (R²)')
    ax.set_ylim(0, 1.05)
    for bar, score in zip(bars, r2_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{score:.3f}", ha='center', va='bottom', fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        plt.tight_layout()
        plt.savefig(f.name, dpi=150)
        plt.close(fig)
        return f.name


def create_prediction_vs_actual_plot(y_true, y_pred, model_name):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.7, color="#667eea", edgecolors="white", linewidth=0.5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label="Идеальное предсказание")
    ax.set_xlabel('Фактический балл счастья')
    ax.set_ylabel('Предсказанный балл счастья')
    ax.set_title(f'Предсказание vs Реальность\n{model_name}', fontsize=13, weight='bold')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        plt.tight_layout()
        plt.savefig(f.name, dpi=150)
        plt.close(fig)
        return f.name


def train_and_evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Линейная регрессия": LinearRegression(),
        "Нейронная сеть (MLP)": MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
        "Дерево решений": DecisionTreeRegressor(random_state=42),
        "Случайный лес": RandomForestRegressor(n_estimators=50, random_state=42)
    }

    results = {}
    for name, model in models.items():
        if name in ["Линейная регрессия", "Нейронная сеть (MLP)"]:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        results[name] = {
            "r2": float(r2_score(y_test, y_pred)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "mse": float(mean_squared_error(y_test, y_pred)),
            "y_pred": y_pred
        }

    return results, scaler, X_test, y_test


# Обработчики

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "Привет! Я бот для анализа уровня счастья.\n\n"
        "Чтобы начать, отправьте Excel-файл (.xlsx), содержащий следующие столбцы:\n"
        + ", ".join(NUMERIC_COLUMNS) + "\n\n"
        "После обработки доступны команды:\n"
        "/models — сравнение моделей по R²\n"
        "/model — выбрать модель для детального просмотра"
    )
    await update.message.reply_text(text)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "Инструкция:\n\n"
        "1. Отправьте файл .xlsx с нужными столбцами.\n"
        "2. Бот выполнит:\n"
        "   - Заполнение пропусков средним\n"
        "   - Коррекцию выбросов (метод IQR)\n"
        "   - Обучение 4 регрессионных моделей\n"
        "3. Используйте команды:\n"
        "   /models — график сравнения R²\n"
        "   /model — интерактивный выбор модели"
    )
    await update.message.reply_text(text)


async def receive_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    document = update.message.document

    if not (document and document.file_name and document.file_name.endswith('.xlsx')):
        await update.message.reply_text("Пожалуйста, отправьте файл в формате .xlsx")
        return

    try:
        await update.message.reply_text("Скачиваю файл...")
        file = await context.bot.get_file(document.file_id)
        
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            await file.download_to_drive(tmp.name)
            file_path = tmp.name

        await update.message.reply_text("Обрабатываю данные...")

        df = pd.read_excel(file_path)
        missing_cols = [c for c in NUMERIC_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Отсутствуют столбцы: {missing_cols}")

        df = df[NUMERIC_COLUMNS].copy()
        df_cleaned, report = handle_missing_and_outliers(df)

        y = df_cleaned[TARGET_COLUMN]
        X = df_cleaned.drop(columns=[TARGET_COLUMN])

        await update.message.reply_text("Обучаю модели...")
        results, scaler, X_test, y_test = train_and_evaluate_models(X, y)

        user_data[user_id] = {
            'results': results,
            'X_test': X_test,
            'y_test': y_test,
            'df_original': df,
            'df_cleaned': df_cleaned,
            'preprocess_report': report
        }

        os.unlink(file_path)

        orig_n = len(df)
        clean_n = len(df_cleaned)
        report_text = f"Обработка завершена.\nИсходно: {orig_n} строк. После обработки: {clean_n} строк.\n"
        if report:
            report_text += "Выполнено:\n" + "\n".join(f"- {r}" for r in report)
        else:
            report_text += "Пропусков и выбросов не обнаружено."

        await update.message.reply_text(report_text)

        stats_lines = []
        for col in df_cleaned.columns:
            s = df_cleaned[col]
            stats_lines.append(f"- {col}: среднее = {s.mean():.3f}, std = {s.std():.3f}")
        await update.message.reply_text("Описательная статистика:\n" + "\n".join(stats_lines))

        await update.message.reply_text(
            "Готово! Доступные команды:\n"
            "/models — сравнить модели\n"
            "/model — выбрать модель для анализа"
        )

    except Exception as e:
        error_msg = f"Ошибка: {str(e)}"
        logger.error(f"Ошибка: {e}\n{traceback.format_exc()}")
        await update.message.reply_text(error_msg[:300])


async def show_models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in user_data:
        await update.message.reply_text("Сначала загрузите файл .xlsx")
        return

    results = user_data[user_id]['results']
    try:
        plot_path = create_overall_r2_plot(results)
        await update.message.reply_photo(
            photo=open(plot_path, 'rb'),
            caption="Сравнение моделей по коэффициенту детерминации (R²)"
        )
        os.unlink(plot_path)
    except Exception as e:
        await update.message.reply_text(f"Не удалось построить график: {e}")


async def model_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in user_data:
        await update.message.reply_text("Сначала загрузите файл .xlsx")
        return ConversationHandler.END

    options = [[str(i+1) + ". " + MODEL_NAMES[i]] for i in range(len(MODEL_NAMES))]
    reply_markup = ReplyKeyboardMarkup(
        options,
        one_time_keyboard=True,
        resize_keyboard=True,
        input_field_placeholder="Выберите номер модели"
    )
    await update.message.reply_text(
        "Выберите модель для подробного анализа:\n"
        "1. Линейная регрессия\n"
        "2. Нейронная сеть (MLP)\n"
        "3. Дерево решений\n"
        "4. Случайный лес",
        reply_markup=reply_markup
    )
    return AWAITING_MODEL_CHOICE


async def model_choice_received(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text.strip()

    try:
        choice = int(text.split('.')[0])
        if choice < 1 or choice > 4:
            raise ValueError()
        model_name = MODEL_NAMES[choice - 1]
    except:
        await update.message.reply_text(
            "Некорректный ввод. Пожалуйста, введите цифру от 1 до 4.",
            reply_markup=ReplyKeyboardRemove()
        )
        return AWAITING_MODEL_CHOICE

    await update.message.reply_text(
        f"Анализ модели: {model_name}",
        reply_markup=ReplyKeyboardRemove()
    )

    data = user_data[user_id]['results'][model_name]
    y_test = user_data[user_id]['y_test']
    y_pred = data['y_pred']
    rmse = np.sqrt(data['mse'])

    metrics_text = (
        f"Модель: {model_name}\n\n"
        f"Коэффициент детерминации (R²): {data['r2']:.4f}\n"
        f"Средняя абсолютная ошибка (MAE): {data['mae']:.4f}\n"
        f"Среднеквадратичная ошибка (MSE): {data['mse']:.4f}\n"
        f"Корень из MSE (RMSE): {rmse:.4f}"
    )
    await update.message.reply_text(metrics_text)

    try:
        plot_path = create_prediction_vs_actual_plot(y_test, y_pred, model_name)
        await update.message.reply_photo(photo=open(plot_path, 'rb'))
        os.unlink(plot_path)
    except Exception as e:
        await update.message.reply_text(f"Не удалось построить график предсказания: {e}")

    return ConversationHandler.END


async def model_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Выбор модели отменён.", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END


# Основной запуск

def main():
    application = ApplicationBuilder().token(BOT_TOKEN).build()

    model_conv = ConversationHandler(
        entry_points=[CommandHandler('model', model_start)],
        states={
            AWAITING_MODEL_CHOICE: [MessageHandler(filters.TEXT & ~filters.COMMAND, model_choice_received)]
        },
        fallbacks=[CommandHandler('cancel', model_cancel)]
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("models", show_models))
    application.add_handler(model_conv)
    application.add_handler(MessageHandler(filters.Document.ALL, receive_file))

    logger.info("Бот запущен.")
    application.run_polling()


if __name__ == '__main__':
    main()