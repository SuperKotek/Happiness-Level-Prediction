import flet as ft
import pandas as pd
import numpy as np
import traceback
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os

df_original = None
X_processed = None
y_processed = None
model_results = None
X_train, X_test, y_train, y_test = None, None, None, None
scaler = None

TARGET_COLUMN = "Happiness Score"
NUMERIC_COLUMNS = [
    "Happiness Score", "Standard Error", "Economy", "Family", "Health",
    "Freedom", "Trust Government", "Generosity", "Dystopia Residual"
]


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


def preprocess_data(file_path):
    global df_original
    try:
        df = pd.read_excel(file_path)
        df_original = df.copy()

        missing_cols = [col for col in NUMERIC_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Отсутствуют столбцы: {missing_cols}")

        df = df[NUMERIC_COLUMNS].copy()
        df_cleaned, report = handle_missing_and_outliers(df)

        y = df_cleaned[TARGET_COLUMN]
        X = df_cleaned.drop(columns=[TARGET_COLUMN])

        return X, y, df_cleaned, report
    except Exception as e:
        print(f"Ошибка при предобработке: {e}")
        traceback.print_exc()
        return None, None, None, None


def train_and_evaluate_models(X, y):
    global X_train, X_test, y_train, y_test, scaler
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Линейная регрессия": LinearRegression(),
        "Нейронная сеть (MLP)": MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
        "Дерево решений": DecisionTreeRegressor(random_state=42),
        "Случайный лес": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    results = {}
    for name, model in models.items():
        if name == "Линейная регрессия" or name == "Нейронная сеть (MLP)":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "_")
        joblib.dump(model, f'{safe_name}_model.joblib')

        results[name] = {
            "model": model,
            "r2": r2_score(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred),
            "y_pred": y_pred
        }

    joblib.dump(scaler, 'scaler.joblib')
    final_features = X.columns.tolist()
    with open('final_features.json', 'w', encoding='utf-8') as f:
        json.dump(final_features, f, ensure_ascii=False, indent=4)

    return results


def create_overall_r2_plot(results):
    try:
        plt.close('all')
        model_names = list(results.keys())
        r2_scores = [results[name]["r2"] for name in model_names]
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#667eea", "#764ba2", "#f093fb", "#6fcf97"]
        bars = ax.bar(model_names, r2_scores, color=colors)
        ax.set_title('Сравнение R² по моделям', fontsize=16, weight='bold')
        ax.set_ylabel('Коэффициент детерминации (R²)', fontsize=12)
        ax.set_ylim(0, 1.05)
        for bar, score in zip(bars, r2_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{score:.3f}", ha='center', va='bottom', fontweight='bold')
        plt.xticks(rotation=15, ha='right')
        temp_filename = "temp_overall_r2.png"
        plt.tight_layout()
        plt.savefig(temp_filename, dpi=150)
        plt.close(fig)
        return temp_filename
    except Exception as e:
        print(f"Ошибка при создании R² графика: {e}")
        plt.close('all')
        return None


def create_prediction_vs_actual_plot(y_true, y_pred, model_name):
    try:
        plt.close('all')
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_true, y_pred, alpha=0.7, color="#667eea", edgecolors="white", linewidth=0.5)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label="Идеальное предсказание")
        ax.set_xlabel('Фактический балл счастья', fontsize=11)
        ax.set_ylabel('Предсказанный балл счастья', fontsize=11)
        ax.set_title(f'Предсказание vs Реальность\n{model_name}', fontsize=13, weight='bold')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "_")
        filename = f"pred_vs_actual_{safe_name}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close(fig)
        return filename
    except Exception as e:
        print(f"Ошибка при создании графика: {e}")
        plt.close('all')
        return None


def generate_preprocessing_report(df_original, X_processed, report_lines):
    original_shape = df_original.shape
    return ft.Column([
        ft.Text("Отчёт о предобработке данных", size=24, weight=ft.FontWeight.BOLD, color=ft.Colors.INDIGO_800),
        ft.Divider(color=ft.Colors.INDIGO_200),
        ft.Text(f"Исходный размер: {original_shape[0]} строк × {original_shape[1]} столбцов", size=15),
        ft.Text(f"После обработки: {X_processed.shape[0]} строк (все строки сохранены)", size=15),
        ft.Divider(),
        ft.Text("Выполнено:", weight=ft.FontWeight.W_600, size=15),
        ft.Column([ft.Text(f"• {line}", size=14) for line in report_lines] if report_lines else [ft.Text("• Пропусков и выбросов не обнаружено.", size=14)]),
    ], spacing=12, width=600)


def generate_statistics_report(df):
    if df is None or df.empty:
        return ft.Text("Нет данных для статистики.", color=ft.Colors.GREY_600, size=16)

    header = ft.Container(
        content=ft.Row([
            ft.Container(ft.Text("Признак", weight=ft.FontWeight.BOLD, size=13), width=180, padding=ft.padding.only(left=12)),
            ft.Container(ft.Text("Среднее", weight=ft.FontWeight.BOLD, size=13, text_align=ft.TextAlign.RIGHT), width=100),
            ft.Container(ft.Text("Стд", weight=ft.FontWeight.BOLD, size=13, text_align=ft.TextAlign.RIGHT), width=80),
            ft.Container(ft.Text("Мин", weight=ft.FontWeight.BOLD, size=13, text_align=ft.TextAlign.RIGHT), width=80),
            ft.Container(ft.Text("Макс", weight=ft.FontWeight.BOLD, size=13, text_align=ft.TextAlign.RIGHT), width=80),
        ], spacing=8),
        bgcolor=ft.Colors.INDIGO_50,
        padding=ft.padding.symmetric(vertical=10),
        border_radius=ft.border_radius.only(top_left=8, top_right=8),
    )

    rows = []
    for col in df.columns:
        s = df[col]
        row = ft.Container(
            content=ft.Row([
                ft.Container(ft.Text(col, size=13), width=180, padding=ft.padding.only(left=12)),
                ft.Container(ft.Text(f"{s.mean():.3f}", size=13, text_align=ft.TextAlign.RIGHT), width=100),
                ft.Container(ft.Text(f"{s.std():.3f}", size=13, text_align=ft.TextAlign.RIGHT), width=80),
                ft.Container(ft.Text(f"{s.min():.3f}", size=13, text_align=ft.TextAlign.RIGHT), width=80),
                ft.Container(ft.Text(f"{s.max():.3f}", size=13, text_align=ft.TextAlign.RIGHT), width=80),
            ], spacing=8, alignment=ft.MainAxisAlignment.START),
            padding=ft.padding.symmetric(vertical=8),
            bgcolor=ft.Colors.WHITE,
            border=ft.border.only(bottom=ft.border.BorderSide(0.5, ft.Colors.OUTLINE_VARIANT)),
        )
        rows.append(row)

    table_body = ft.Column(
        controls=rows,
        spacing=0,
        scroll=ft.ScrollMode.ALWAYS,
        height=320,
    )

    table_container = ft.Container(
        content=ft.Column([header, table_body], spacing=0),
        border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
        border_radius=8,
        clip_behavior=ft.ClipBehavior.HARD_EDGE,
        height=370,
    )

    return ft.Column([
        ft.Text("Описательная статистика", size=20, weight=ft.FontWeight.BOLD, color=ft.Colors.INDIGO_800),
        table_container
    ], spacing=15)


def generate_raw_data_table(df):
    if df is None or df.empty:
        return ft.Text("Данные отсутствуют", color=ft.Colors.RED, size=16)

    col_width = 114

    header_cells = [
        ft.Container(
            ft.Text(str(col), weight=ft.FontWeight.BOLD, size=12, text_align=ft.TextAlign.CENTER),
            width=col_width,
            padding=8,
            bgcolor=ft.Colors.INDIGO_100,
            alignment=ft.alignment.center,
            border=ft.border.only(right=ft.border.BorderSide(0.5, ft.Colors.OUTLINE_VARIANT)),
        )
        for col in df.columns
    ]
    header_row = ft.Container(
        content=ft.Row(header_cells, spacing=0),
        bgcolor=ft.Colors.INDIGO_100,
        clip_behavior=ft.ClipBehavior.HARD_EDGE,
        height=40,
    )

    data_rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        cells = []
        for val in row:
            text = f"{val:.4f}" if isinstance(val, (float, int)) else str(val)
            cells.append(
                ft.Container(
                    ft.Text(text, size=12, text_align=ft.TextAlign.RIGHT),
                    width=col_width,
                    padding=ft.padding.symmetric(horizontal=6, vertical=4),
                    alignment=ft.alignment.center_right,
                    border=ft.border.only(right=ft.border.BorderSide(0.5, ft.Colors.OUTLINE_VARIANT)),
                )
            )
        row_container = ft.Container(
            content=ft.Row(cells, spacing=0),
            bgcolor=ft.Colors.WHITE if i % 2 == 0 else ft.Colors.GREY_50,
            height=32,
            border=ft.border.only(bottom=ft.border.BorderSide(0.5, ft.Colors.OUTLINE_VARIANT)),
        )
        data_rows.append(row_container)

    table_content = ft.Column(
        controls=[header_row] + data_rows,
        spacing=0,
        tight=True,
    )

    scrollable_table = ft.Container(
        content=ft.ListView(
            controls=[table_content],
            horizontal=True,
            auto_scroll=False,
        ),
        border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
        border_radius=8,
        clip_behavior=ft.ClipBehavior.HARD_EDGE,
        height=5500,  # ✅ точная высота для 170 строк + запас: 40 + 170*32 + 20 = 5500
    )

    return ft.Column([
        ft.Text(
            f"Исходные данные ({len(df)} строк × {len(df.columns)} столбцов)",
            size=20,
            weight=ft.FontWeight.BOLD,
            color=ft.Colors.INDIGO_800
        ),
        scrollable_table
    ], spacing=15)


def main(page: ft.Page):
    global df_original, X_processed, y_processed, model_results

    page.title = "Анализ уровня счастья"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 25
    page.scroll = ft.ScrollMode.AUTO

    page.theme = ft.Theme(
        color_scheme=ft.ColorScheme(
            primary=ft.Colors.INDIGO,
            secondary=ft.Colors.TEAL,
            surface=ft.Colors.WHITE,
            background=ft.Colors.GREY_50,
        )
    )

    file_path_text = ft.Text("Файл не выбран", color=ft.Colors.INDIGO_700, size=14)
    process_button = ft.ElevatedButton(
        "Загрузить и обработать данные",
        icon=ft.Icons.AUTO_GRAPH_OUTLINED,
        disabled=True,
        width=280,
        height=50,
        style=ft.ButtonStyle(
            color=ft.Colors.WHITE,
            bgcolor=ft.Colors.INDIGO,
            padding=ft.padding.symmetric(horizontal=20, vertical=12),
            shape=ft.RoundedRectangleBorder(radius=10),
            elevation=2,
        )
    )
    progress_ring = ft.ProgressRing(visible=False, color=ft.Colors.INDIGO)
    status_text = ft.Text("Выберите Excel-файл с данными", italic=True, color=ft.Colors.GREY_700, size=14)

    file_picker = ft.FilePicker(on_result=lambda e: on_file_pick(e))
    save_json_picker = ft.FilePicker(on_result=lambda e: on_save_json(e))
    page.overlay.extend([file_picker, save_json_picker])

    selected_file_path = None

    model_dropdown = ft.Dropdown(
        label="Выберите модель",
        width=320,
        dense=True,
        text_size=14,
        label_style=ft.TextStyle(size=14),
        disabled=True,
        options=[]
    )
    save_json_button = ft.ElevatedButton(
        "Сохранить метрики в JSON",
        icon=ft.Icons.DOWNLOAD,
        disabled=True,
        width=220,
        height=45,
        style=ft.ButtonStyle(
            color=ft.Colors.WHITE,
            bgcolor=ft.Colors.TEAL,
            shape=ft.RoundedRectangleBorder(radius=8),
        )
    )
    model_details_container = ft.Container(expand=True, padding=10)

    tab0 = ft.Column([], scroll=ft.ScrollMode.AUTO, expand=True)
    tab1 = ft.Column([
        ft.Text("1. Загрузка данных", size=26, weight=ft.FontWeight.BOLD, color=ft.Colors.INDIGO_800),
        ft.Divider(color=ft.Colors.INDIGO_200, height=25),
        ft.Row([
            ft.ElevatedButton(
                "Выбрать .xlsx файл",
                icon=ft.Icons.FOLDER_OPEN,
                on_click=lambda _: file_picker.pick_files(allowed_extensions=["xlsx"]),
                width=200,
                height=45,
                style=ft.ButtonStyle(bgcolor=ft.Colors.INDIGO_100, color=ft.Colors.INDIGO_800)
            ),
            ft.Container(width=20),
            file_path_text
        ], alignment=ft.MainAxisAlignment.CENTER),
        ft.Container(height=25),
        ft.Row([process_button], alignment=ft.MainAxisAlignment.CENTER),
        ft.Container(height=20),
        ft.Row([progress_ring, status_text], alignment=ft.MainAxisAlignment.CENTER, spacing=15)
    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=20, width=800)

    tab2 = ft.Column([], scroll=ft.ScrollMode.AUTO, width=850)
    tab3_top = ft.Column([], alignment=ft.MainAxisAlignment.START, width=850)
    tab3_bottom = ft.Column([
        ft.Row([model_dropdown, ft.Container(width=20), save_json_button], alignment=ft.MainAxisAlignment.START),
        ft.Container(height=20),
        model_details_container
    ], spacing=20, width=850)
    tab3 = ft.Column([tab3_top, ft.Divider(color=ft.Colors.INDIGO_100), tab3_bottom], expand=True, scroll=ft.ScrollMode.AUTO)

    def on_file_pick(e):
        nonlocal selected_file_path
        if e.files:
            selected_file_path = e.files[0].path
            file_path_text.value = os.path.basename(selected_file_path)
            process_button.disabled = False
            status_text.value = "Файл выбран. Нажмите для обработки."
            status_text.color = ft.Colors.INDIGO_700
        else:
            selected_file_path = None
            file_path_text.value = "Файл не выбран"
            process_button.disabled = True
            status_text.value = "Ожидание выбора .xlsx файла"
        page.update()

    def process_data_clicked(e):
        nonlocal selected_file_path
        global df_original, X_processed, y_processed, model_results
        if not selected_file_path:
            return

        process_button.disabled = True
        progress_ring.visible = True
        status_text.value = "Обработка данных..."
        status_text.color = ft.Colors.INDIGO_700
        page.update()

        try:
            X, y, df_clean, report = preprocess_data(selected_file_path)
            if X is None:
                raise ValueError("Не удалось загрузить данные.")

            X_processed, y_processed = X, y

            status_text.value = "Обучение моделей..."
            model_results = train_and_evaluate_models(X, y)

            tab0.controls = [generate_raw_data_table(df_original)]

            tab2.controls = [
                generate_preprocessing_report(df_original, X_processed, report),
                ft.Divider(color=ft.Colors.INDIGO_100),
                generate_statistics_report(df_clean)
            ]

            r2_plot = create_overall_r2_plot(model_results)
            tab3_top.controls = []
            if r2_plot:
                tab3_top.controls.append(ft.Image(src=r2_plot, width=850, fit=ft.ImageFit.CONTAIN))

            model_dropdown.options = [ft.dropdown.Option(text=name) for name in model_results.keys()]
            model_dropdown.disabled = False
            model_dropdown.on_change = update_model_display
            save_json_button.disabled = False
            model_details_container.content = None
            status_text.value = "Готово! Перейдите во вкладки для анализа."
            status_text.color = ft.Colors.GREEN_800

        except Exception as ex:
            error_msg = str(ex)
            status_text.value = f"Ошибка: {error_msg[:100]}..."
            status_text.color = ft.Colors.RED_700
            page.snack_bar = ft.SnackBar(
                content=ft.Text(f"Ошибка: {error_msg}", size=14),
                bgcolor=ft.Colors.RED_100,
                action="Закрыть",
                action_color=ft.Colors.RED_700
            )
            page.snack_bar.open = True
        finally:
            progress_ring.visible = False
            process_button.disabled = False
            page.update()

    def update_model_display(e):
        if not model_results or not model_dropdown.value:
            return
        name = model_dropdown.value
        data = model_results[name]
        y_pred = data["y_pred"]
        rmse = np.sqrt(data["mse"])

        pred_plot = create_prediction_vs_actual_plot(y_test, y_pred, name)

        metrics_col = ft.Column([
            ft.Text(f"Модель: {name}", size=20, weight=ft.FontWeight.BOLD, color=ft.Colors.INDIGO_800),
            ft.Divider(color=ft.Colors.INDIGO_100),
            ft.Text("Метрики качества", size=16, weight=ft.FontWeight.W_600, color=ft.Colors.INDIGO_700),
            ft.Container(
                content=ft.Column([
                    ft.Row([ft.Text("R²:", weight=ft.FontWeight.W_500, width=100), ft.Text(f"{data['r2']:.4f}", width=100)]),
                    ft.Row([ft.Text("MAE:", weight=ft.FontWeight.W_500, width=100), ft.Text(f"{data['mae']:.4f}", width=100)]),
                    ft.Row([ft.Text("MSE:", weight=ft.FontWeight.W_500, width=100), ft.Text(f"{data['mse']:.4f}", width=100)]),
                    ft.Row([ft.Text("RMSE:", weight=ft.FontWeight.W_600, width=100), ft.Text(f"{rmse:.4f}", width=100, color=ft.Colors.INDIGO)]),
                ], spacing=8),
                padding=15,
                border=ft.border.all(1, ft.Colors.INDIGO_100),
                border_radius=10,
                bgcolor=ft.Colors.INDIGO_50,
            ),
            ft.Divider(color=ft.Colors.INDIGO_100),
        ], spacing=15)

        if pred_plot:
            metrics_col.controls.append(
                ft.Column([
                    ft.Text("График предсказания", size=16, weight=ft.FontWeight.W_600, color=ft.Colors.INDIGO_700),
                    ft.Image(src=pred_plot, width=550, height=450, fit=ft.ImageFit.CONTAIN)
                ], spacing=10)
            )

        model_details_container.content = metrics_col
        page.update()

    def on_save_json(e):
        if e.path and model_results:
            out = {}
            for name, data in model_results.items():
                mse = data["mse"]
                out[name] = {
                    "r2": round(data["r2"], 6),
                    "mae": round(data["mae"], 6),
                    "mse": round(data["mse"], 6),
                    "rmse": round(np.sqrt(mse), 6)
                }
            try:
                with open(e.path, 'w', encoding='utf-8') as f:
                    json.dump(out, f, indent=4, ensure_ascii=False)
                page.snack_bar = ft.SnackBar(
                    content=ft.Text(f"Метрики сохранены в {os.path.basename(e.path)}", size=14),
                    bgcolor=ft.Colors.GREEN_100,
                    action="OK",
                    action_color=ft.Colors.GREEN_800
                )
                page.snack_bar.open = True
            except Exception as ex:
                page.snack_bar = ft.SnackBar(ft.Text(f"Ошибка записи: {ex}"), bgcolor=ft.Colors.RED_100)
                page.snack_bar.open = True
            page.update()

    process_button.on_click = process_data_clicked

    tabs = ft.Tabs(
        selected_index=0,
        animation_duration=300,
        tabs=[
            ft.Tab(text="Загрузка", content=tab1),
            ft.Tab(text="Исходные данные", content=tab0),
            ft.Tab(text="Анализ данных", content=tab2),
            ft.Tab(text="Результаты моделей", content=tab3),
        ],
        expand=True,
        indicator_color=ft.Colors.INDIGO,
        label_color=ft.Colors.INDIGO_800,
        unselected_label_color=ft.Colors.GREY_600,
    )

    page.add(
        ft.Container(
            content=ft.Column([
                ft.Text("Анализ уровня счастья", size=32, weight=ft.FontWeight.BOLD, color=ft.Colors.INDIGO_900, text_align=ft.TextAlign.CENTER),
                ft.Text("Сравнение регрессионных моделей", size=16, color=ft.Colors.GREY_700, text_align=ft.TextAlign.CENTER),
                ft.Divider(height=20, color=ft.Colors.TRANSPARENT),
                tabs
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=10,
            expand=True
        )
    )


if __name__ == "__main__":
    ft.app(target=main)