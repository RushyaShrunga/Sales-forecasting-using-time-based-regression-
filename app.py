from pathlib import Path
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_file
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
STATIC_DIR = BASE_DIR / "static"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

FEATURE_COLUMNS = [
    "time_index",
    "year",
    "month",
    "quarter",
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_3",
]


def create_sample_department_data():
    rng = np.random.default_rng(7)
    departments = {
        "Electronics": {"base": 42000, "growth": 620, "season": 5200},
        "Furniture": {"base": 30000, "growth": 380, "season": 3600},
        "Clothing": {"base": 26000, "growth": 440, "season": 4700},
        "Grocery": {"base": 36000, "growth": 290, "season": 2500},
    }
    dates = pd.date_range(start="2021-01-01", periods=48, freq="MS")
    rows = []

    for department, values in departments.items():
        for index, date in enumerate(dates):
            seasonal_curve = values["season"] * np.sin(2 * np.pi * date.month / 12)
            holiday_boost = 6500 if date.month in [11, 12] else 0
            noise = rng.normal(0, 1200)
            sales = values["base"] + values["growth"] * index + seasonal_curve + holiday_boost + noise
            rows.append(
                {
                    "date": date,
                    "department": department,
                    "sales": round(max(0, sales), 2),
                }
            )

    return pd.DataFrame(rows)


def clean_sales_data(df):
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    required_columns = {"date", "department", "sales"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            "Your CSV must contain columns named date, department, and sales. "
            f"Missing: {', '.join(sorted(missing_columns))}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["department"] = df["department"].astype(str).str.strip()
    df["department"] = df["department"].replace({"": np.nan, "nan": np.nan})
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df = df.dropna(subset=["date", "department", "sales"])
    df = df.sort_values(["department", "date"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No usable rows were found after cleaning the CSV.")

    return df


def aggregate_monthly_by_department(df):
    data = df.copy()
    data["date"] = data["date"].dt.to_period("M").dt.to_timestamp()
    return (
        data.groupby(["department", "date"], as_index=False)["sales"]
        .sum()
        .sort_values(["department", "date"])
        .reset_index(drop=True)
    )


def resolve_departments(monthly_df, department_filter):
    departments = sorted(monthly_df["department"].unique())
    if not department_filter:
        return departments

    normalized = department_filter.strip().lower()
    matches = [department for department in departments if department.lower() == normalized]
    if not matches:
        available = ", ".join(departments)
        raise ValueError(f"Department '{department_filter}' was not found. Available: {available}")
    return matches


def create_time_features(data):
    data = data.copy()
    data["time_index"] = np.arange(len(data))
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["quarter"] = data["date"].dt.quarter
    data["lag_1"] = data["sales"].shift(1)
    data["lag_2"] = data["sales"].shift(2)
    data["lag_3"] = data["sales"].shift(3)
    data["rolling_mean_3"] = data["sales"].shift(1).rolling(window=3).mean()
    return data


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    non_zero = y_true != 0
    if not np.any(non_zero):
        return np.nan
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100


def build_models():
    numeric_features = [
        "time_index",
        "year",
        "quarter",
        "lag_1",
        "lag_2",
        "lag_3",
        "rolling_mean_3",
    ]
    categorical_features = ["month"]

    linear_preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    tree_preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    return {
        "Linear Regression": Pipeline(
            steps=[
                ("preprocessor", linear_preprocessor),
                ("model", LinearRegression()),
            ]
        ),
        "Random Forest Regression": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                ("model", RandomForestRegressor(n_estimators=260, random_state=42)),
            ]
        ),
    }


def forecast_future(model, history_df, periods=6):
    history = history_df[["date", "sales"]].copy().sort_values("date").reset_index(drop=True)
    forecasts = []
    month_offset = pd.tseries.frequencies.to_offset("MS")

    for _ in range(periods):
        next_date = history["date"].max() + month_offset
        placeholder = pd.DataFrame({"date": [next_date], "sales": [np.nan]})
        combined = pd.concat([history, placeholder], ignore_index=True)
        features = create_time_features(combined).iloc[[-1]][FEATURE_COLUMNS]
        predicted_sales = max(0, float(model.predict(features)[0]))

        forecasts.append({"date": next_date, "forecasted_sales": predicted_sales})
        history = pd.concat(
            [history, pd.DataFrame({"date": [next_date], "sales": [predicted_sales]})],
            ignore_index=True,
        )

    return pd.DataFrame(forecasts)


def train_department_model(department_df, periods):
    department_df = department_df.sort_values("date").reset_index(drop=True)
    if len(department_df) < 12:
        raise ValueError("needs at least 12 months of department history")

    model_df = create_time_features(department_df).dropna().reset_index(drop=True)
    split_index = int(len(model_df) * 0.8)
    train_df = model_df.iloc[:split_index]
    test_df = model_df.iloc[split_index:]

    if len(train_df) < 5 or len(test_df) < 2:
        raise ValueError("does not have enough rows after lag feature creation")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["sales"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["sales"]

    models = build_models()
    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append(
            {
                "model": name,
                "mae": mean_absolute_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "mape_percent": mean_absolute_percentage_error(y_test, y_pred),
                "r2_score": r2_score(y_test, y_pred),
            }
        )

    results_df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    best_model_name = results_df.loc[0, "model"]
    best_model = models[best_model_name]
    future_forecast = forecast_future(best_model, department_df, periods=periods)

    return results_df, future_forecast, best_model_name


def save_department_bar_chart(department_df, future_forecast, department, best_model_name, run_id):
    chart_path = STATIC_DIR / f"department_bar_{run_id}.png"
    recent_actual = department_df.tail(12).copy()
    future = future_forecast.copy()

    actual_labels = recent_actual["date"].dt.strftime("%b %Y").tolist()
    forecast_labels = future["date"].dt.strftime("%b %Y").tolist()
    labels = actual_labels + forecast_labels
    actual_values = recent_actual["sales"].tolist()
    forecast_values = future["forecasted_sales"].tolist()

    actual_x = np.arange(len(actual_values))
    forecast_x = np.arange(len(actual_values), len(actual_values) + len(forecast_values))

    plt.figure(figsize=(12, 6))
    plt.bar(actual_x, actual_values, color="#243042", label="Recent Actual Sales")
    plt.bar(forecast_x, forecast_values, color="#ef476f", label="Forecasted Sales")
    plt.axvline(len(actual_values) - 0.5, color="#94a3b8", linestyle="--", linewidth=1)
    plt.title(f"{department} Monthly Sales Forecast ({best_model_name})")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()

    return chart_path.name


def build_department_section(monthly_df, department, periods):
    department_df = monthly_df[monthly_df["department"] == department].copy()
    results_df, future_forecast, best_model_name = train_department_model(department_df, periods)
    run_id = uuid4().hex
    chart_filename = save_department_bar_chart(
        department_df, future_forecast, department, best_model_name, run_id
    )

    display_forecast = future_forecast.assign(
        date=future_forecast["date"].dt.strftime("%Y-%m-%d"),
        forecasted_sales=future_forecast["forecasted_sales"].round(2),
    )

    download_df = future_forecast.copy()
    download_df.insert(0, "department", department)

    return {
        "department": department,
        "history_months": len(department_df),
        "start_date": department_df["date"].min().date(),
        "end_date": department_df["date"].max().date(),
        "best_model_name": best_model_name,
        "results": results_df.round(2).to_dict(orient="records"),
        "forecast": display_forecast.to_dict(orient="records"),
        "chart_filename": chart_filename,
        "download_df": download_df,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    try:
        periods = int(request.form.get("periods", 6))
        periods = min(max(periods, 1), 24)
        department_filter = request.form.get("department", "").strip()

        uploaded_file = request.files.get("sales_file")
        if uploaded_file and uploaded_file.filename:
            filename = secure_filename(uploaded_file.filename)
            if not filename.lower().endswith(".csv"):
                raise ValueError("Please upload a CSV file.")
            upload_path = UPLOAD_DIR / f"{uuid4().hex}_{filename}"
            uploaded_file.save(upload_path)
            df = pd.read_csv(upload_path)
            data_source = filename
        else:
            df = create_sample_department_data()
            data_source = "sample department sales data"

        df = clean_sales_data(df)
        monthly_df = aggregate_monthly_by_department(df)
        departments = resolve_departments(monthly_df, department_filter)

        sections = []
        skipped_departments = []
        for department in departments:
            try:
                sections.append(build_department_section(monthly_df, department, periods))
            except ValueError as exc:
                skipped_departments.append(f"{department}: {exc}")

        if not sections:
            raise ValueError(
                "No selected department had enough monthly history. "
                "Use at least 12 months for each department."
            )

        forecast_filename = f"department_monthly_forecast_{uuid4().hex}.csv"
        forecast_path = OUTPUT_DIR / forecast_filename
        pd.concat([section["download_df"] for section in sections], ignore_index=True).to_csv(
            forecast_path, index=False
        )

        return render_template(
            "results.html",
            data_source=data_source,
            original_rows=len(df),
            monthly_rows=len(monthly_df),
            department_count=len(monthly_df["department"].unique()),
            start_date=monthly_df["date"].min().date(),
            end_date=monthly_df["date"].max().date(),
            sections=sections,
            skipped_departments=skipped_departments,
            forecast_filename=forecast_filename,
        )
    except Exception as exc:
        return render_template("index.html", error=str(exc)), 400


@app.route("/download/<filename>")
def download(filename):
    safe_filename = secure_filename(filename)
    forecast_path = OUTPUT_DIR / safe_filename
    if not forecast_path.exists():
        return "File not found", 404
    return send_file(forecast_path, as_attachment=True, download_name=safe_filename)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
