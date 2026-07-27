import gradio as gr
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ============================================================
# TRAIN MODELS ON STARTUP
# ============================================================

print("Loading dataset and training models...")

df_raw = pd.read_csv("west_africa_disease_dataset_v2.csv")
df = df_raw.copy()
df = df.sort_values(['country', 'year', 'month']).reset_index(drop=True)

# WHO annual scaling
who_annual_malaria = {
    'Nigeria': 29000000, 'Burkina Faso': 11000000, 'Mali': 4800000,
    'Guinea-Bissau': 600000, 'Liberia': 1200000, 'Mauritania': 350000, 'Togo': 1500000
}
who_annual_dengue = {
    'Nigeria': 120000, 'Burkina Faso': 15000, 'Mali': 8000,
    'Guinea-Bissau': 3000, 'Liberia': 5000, 'Mauritania': 2000, 'Togo': 12000
}

real_df = df[df['data_type'] == 'real'].copy()
malaria_scale = {}
dengue_scale  = {}

for country in df['country'].unique():
    country_real = real_df[real_df['country'] == country]
    years = country_real['year'].unique()
    annual_m, annual_d = [], []
    for yr in years:
        yr_data = country_real[country_real['year'] == yr]
        annual_m.append(yr_data['malaria_cases'].sum())
        annual_d.append(yr_data['dengue_cases'].sum())
    malaria_scale[country] = who_annual_malaria[country] / np.mean(annual_m)
    dengue_scale[country]  = who_annual_dengue[country]  / np.mean(annual_d)

df['malaria_cases'] = df.apply(lambda r: int(r['malaria_cases'] * malaria_scale[r['country']]), axis=1)
df['dengue_cases']  = df.apply(lambda r: int(r['dengue_cases']  * dengue_scale[r['country']]),  axis=1)

# Cleaning
df['air_quality_index'] = df['air_quality_index'].replace(0, np.nan)
df['air_quality_index'] = df.groupby(['country','month'])['air_quality_index'].transform(lambda x: x.fillna(x.median()))
df['air_quality_index'] = df.groupby('country')['air_quality_index'].transform(lambda x: x.fillna(x.median()))
df['precipitation_mm'] = df['precipitation_mm'].replace(0, np.nan)
df['precipitation_mm'] = df.groupby(['country','month'])['precipitation_mm'].transform(lambda x: x.fillna(x.median()))
df['precipitation_mm'] = df.groupby('country')['precipitation_mm'].transform(lambda x: x.fillna(x.median()))

# Feature engineering
df['season']      = df['month'].apply(lambda m: 1 if 5 <= m <= 10 else 0)
df['temp_x_rain'] = df['avg_temp_c'] * df['precipitation_mm']

lag_cols = ['malaria_lag_1','malaria_lag_2','dengue_lag_1','dengue_lag_2','rainfall_lag_1','temp_lag_1']
df = df.drop(columns=lag_cols + ['log_malaria','log_dengue'], errors='ignore')

df['malaria_lag_1']  = df.groupby('country')['malaria_cases'].shift(1)
df['malaria_lag_2']  = df.groupby('country')['malaria_cases'].shift(2)
df['dengue_lag_1']   = df.groupby('country')['dengue_cases'].shift(1)
df['dengue_lag_2']   = df.groupby('country')['dengue_cases'].shift(2)
df['rainfall_lag_1'] = df.groupby('country')['precipitation_mm'].shift(1)
df['temp_lag_1']     = df.groupby('country')['avg_temp_c'].shift(1)

df = df.dropna(subset=lag_cols).reset_index(drop=True)

df['log_malaria'] = np.log1p(df['malaria_cases'])
df['log_dengue']  = np.log1p(df['dengue_cases'])

# Train/test split
feature_cols = [
    'country', 'month', 'quarter', 'season',
    'avg_temp_c', 'precipitation_mm', 'air_quality_index',
    'uv_index', 'population_density', 'healthcare_budget',
    'temp_x_rain', 'rainfall_lag_1', 'temp_lag_1',
    'malaria_lag_1', 'malaria_lag_2', 'dengue_lag_1', 'dengue_lag_2'
]

categorical_features = ['country']
numerical_features   = [f for f in feature_cols if f != 'country']

train_df = df[df['year'] <= 2023]
X_train  = train_df[feature_cols]
y_train_m = train_df['log_malaria']
y_train_d = train_df['log_dengue']

# Train pipelines
malaria_pipeline = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ])),
    ('regressor', LinearRegression())
])
malaria_pipeline.fit(X_train, y_train_m)

dengue_pipeline = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ])),
    ('regressor', LinearRegression())
])
dengue_pipeline.fit(X_train, y_train_d)

print("Models trained successfully.")

# ============================================================
# APP CONFIG
# ============================================================

with open('country_population.json') as f:
    country_population = json.load(f)

country_meta = {
    'Nigeria':       {'population_density': 187, 'healthcare_budget': 3005,
                      'malaria_avg': 2400000, 'dengue_avg': 10000},
    'Burkina Faso':  {'population_density': 380, 'healthcare_budget': 2861,
                      'malaria_avg': 916000,  'dengue_avg': 1200},
    'Mali':          {'population_density': 15,  'healthcare_budget': 800,
                      'malaria_avg': 400000,  'dengue_avg': 650},
    'Guinea-Bissau': {'population_density': 67,  'healthcare_budget': 506,
                      'malaria_avg': 50000,   'dengue_avg': 250},
    'Liberia':       {'population_density': 49,  'healthcare_budget': 900,
                      'malaria_avg': 100000,  'dengue_avg': 400},
    'Mauritania':    {'population_density': 4,   'healthcare_budget': 1200,
                      'malaria_avg': 29000,   'dengue_avg': 160},
    'Togo':          {'population_density': 152, 'healthcare_budget': 1536,
                      'malaria_avg': 125000,  'dengue_avg': 1000},
}

month_names = {
    1:'January', 2:'February', 3:'March', 4:'April',
    5:'May', 6:'June', 7:'July', 8:'August',
    9:'September', 10:'October', 11:'November', 12:'December'
}

def update_country_defaults(country):
    meta = country_meta[country]
    return (
        meta['population_density'],
        meta['healthcare_budget'],
        meta['malaria_avg'],
        int(meta['malaria_avg'] * 0.95),
        meta['dengue_avg'],
        int(meta['dengue_avg'] * 0.95),
    )

def get_risk_level(cases, population):
    rate = (cases / population) * 100000
    if rate < 500:    return "Low"
    elif rate < 2000: return "Moderate"
    elif rate < 5000: return "High"
    else:             return "Critical"

def get_risk_style(risk):
    return {
        "Low":      "background:#dcfce7; color:#15803d;",
        "Moderate": "background:#fef9c3; color:#854d0e;",
        "High":     "background:#ffedd5; color:#9a3412;",
        "Critical": "background:#fee2e2; color:#991b1b;",
    }.get(risk, "background:#f1f5f9; color:#475569;")

def get_season_label(month):
    return "Wet Season" if 5 <= month <= 10 else "Dry Season"

def build_report(country, month, year, malaria_pred, dengue_pred,
                 malaria_risk, dengue_risk, season_label):
    season_context = (
        "This period falls within the West African wet season (May to October), "
        "when elevated rainfall and temperatures create favourable conditions for mosquito breeding."
        if season_label == "Wet Season"
        else
        "This period falls within the dry season (November to April), "
        "when reduced rainfall typically lowers mosquito breeding activity."
    )
    malaria_advice = {
        "Low":      "Routine surveillance is sufficient at this time.",
        "Moderate": "Health authorities are advised to maintain vector control measures and ensure adequate treatment supply.",
        "High":     "Urgent vector control intervention is recommended. Treatment facilities should prepare for elevated case loads.",
        "Critical": "Immediate public health response is required. Mass distribution of preventive resources and emergency treatment capacity is strongly advised.",
    }
    dengue_note = (
        f"Dengue fever risk is {dengue_risk.lower()} with {dengue_pred:,} projected cases. "
        if dengue_risk in ["High", "Critical"]
        else f"Dengue fever risk remains {dengue_risk.lower()} at {dengue_pred:,} projected cases. "
    )
    return (
        f"The model forecasts a {malaria_risk.lower()} malaria burden in {country} "
        f"for {month_names[month]} {int(year)}, with an estimated {malaria_pred:,} cases. "
        f"{season_context} "
        f"{dengue_note}"
        f"{malaria_advice[malaria_risk]}"
    )

def go_to_form():
    return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

def go_home():
    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

def predict(country, year, month,
            avg_temp_c, precipitation_mm,
            air_quality_index, uv_index,
            population_density, healthcare_budget,
            malaria_lag_1, malaria_lag_2,
            dengue_lag_1, dengue_lag_2,
            rainfall_lag_1, temp_lag_1):
    try:
        month        = int(month)
        quarter      = (month - 1) // 3 + 1
        season       = 1 if 5 <= month <= 10 else 0
        temp_x_rain  = avg_temp_c * precipitation_mm
        population   = country_population[country]
        season_label = get_season_label(month)

        sample = pd.DataFrame([{
            'country': country, 'month': month, 'quarter': quarter,
            'season': season, 'avg_temp_c': avg_temp_c,
            'precipitation_mm': precipitation_mm,
            'air_quality_index': air_quality_index,
            'uv_index': uv_index,
            'population_density': population_density,
            'healthcare_budget': healthcare_budget,
            'temp_x_rain': temp_x_rain,
            'rainfall_lag_1': rainfall_lag_1,
            'temp_lag_1': temp_lag_1,
            'malaria_lag_1': malaria_lag_1,
            'malaria_lag_2': malaria_lag_2,
            'dengue_lag_1': dengue_lag_1,
            'dengue_lag_2': dengue_lag_2,
        }])

        malaria_pred = max(0, int(np.expm1(malaria_pipeline.predict(sample)[0])))
        dengue_pred  = max(0, int(np.expm1(dengue_pipeline.predict(sample)[0])))
        malaria_risk = get_risk_level(malaria_pred, population)
        dengue_risk  = get_risk_level(dengue_pred,  population)
        report       = build_report(country, month, year, malaria_pred,
                                    dengue_pred, malaria_risk, dengue_risk,
                                    season_label)

        m_style = get_risk_style(malaria_risk)
        d_style = get_risk_style(dengue_risk)
        s_style = "background:#dcfce7; color:#15803d;" if season == 1 else "background:#e0f2fe; color:#0369a1;"

        results_html = f"""
        <div style="font-family:'Inter',sans-serif; padding:4px 0;">
          <div style="display:flex; align-items:center; gap:10px; margin-bottom:18px; flex-wrap:wrap;">
            <span style="font-size:15px; font-weight:500; color:#1a1a2e;">{country}</span>
            <span style="color:#94a3b8;">·</span>
            <span style="font-size:14px; color:#475569;">{month_names[month]} {int(year)}</span>
            <span style="font-size:11px; padding:3px 10px; border-radius:20px; font-weight:500; {s_style}">{season_label}</span>
          </div>
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:18px;">
            <div style="background:#f8fafc; border-radius:10px; border:0.5px solid #e2e8f0; padding:16px;">
              <div style="font-size:10px; font-weight:600; letter-spacing:0.08em; color:#64748b; text-transform:uppercase; margin-bottom:8px;">Malaria</div>
              <div style="font-size:26px; font-weight:500; color:#1a1a2e; margin-bottom:4px;">{malaria_pred:,}</div>
              <div style="font-size:11px; color:#94a3b8; margin-bottom:10px;">predicted cases</div>
              <span style="font-size:11px; padding:3px 10px; border-radius:20px; font-weight:500; {m_style}">{malaria_risk} risk</span>
            </div>
            <div style="background:#f8fafc; border-radius:10px; border:0.5px solid #e2e8f0; padding:16px;">
              <div style="font-size:10px; font-weight:600; letter-spacing:0.08em; color:#64748b; text-transform:uppercase; margin-bottom:8px;">Dengue fever</div>
              <div style="font-size:26px; font-weight:500; color:#1a1a2e; margin-bottom:4px;">{dengue_pred:,}</div>
              <div style="font-size:11px; color:#94a3b8; margin-bottom:10px;">predicted cases</div>
              <span style="font-size:11px; padding:3px 10px; border-radius:20px; font-weight:500; {d_style}">{dengue_risk} risk</span>
            </div>
          </div>
          <div style="background:#f0fdf9; border:0.5px solid #0d9488; border-radius:8px; padding:14px 16px; margin-bottom:16px;">
            <div style="font-size:10px; font-weight:600; letter-spacing:0.08em; color:#0d9488; text-transform:uppercase; margin-bottom:8px;">Summary report</div>
            <p style="font-size:13px; color:#1e293b; line-height:1.75; margin:0;">{report}</p>
          </div>
          <div style="background:#f8fafc; border-radius:8px; padding:10px 14px; font-size:11px; color:#64748b; line-height:1.6;">
            <span style="font-weight:500; color:#475569;">Risk scale</span> (cases per 100,000 population):
            <span style="background:#dcfce7; color:#15803d; padding:1px 7px; border-radius:10px; margin-left:4px;">Low &lt;500</span>
            <span style="background:#fef9c3; color:#854d0e; padding:1px 7px; border-radius:10px; margin-left:4px;">Moderate 500-2,000</span>
            <span style="background:#ffedd5; color:#9a3412; padding:1px 7px; border-radius:10px; margin-left:4px;">High 2,000-5,000</span>
            <span style="background:#fee2e2; color:#991b1b; padding:1px 7px; border-radius:10px; margin-left:4px;">Critical &gt;5,000</span>
          </div>
        </div>
        """
        return results_html, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

    except Exception as e:
        return f"<p style='color:red;'>Prediction error: {str(e)}</p>", gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)


FOOTER = """
<div style="background:#f0fdf9; border-top:0.5px solid #0d9488; border-radius:0 0 10px 10px; padding:12px 24px; text-align:center; font-family:Inter,sans-serif;">
  <span style="font-size:12px; color:#0d9488; font-weight:500;">
    Group 10 &nbsp;|&nbsp; Supervisor: Dr. Mrs. R.S. Babatunde &nbsp;||&nbsp; KWASU &nbsp;|&nbsp; B.Sc. Computer Science
  </span>
</div>
"""

css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.section-label {
    font-size: 10px !important; font-weight: 600 !important;
    letter-spacing: 0.08em !important; color: #0d9488 !important;
    text-transform: uppercase !important;
    border-bottom: 1.5px solid #0d9488 !important;
    padding-bottom: 4px !important; margin-bottom: 10px !important;
}
.run-btn {
    background: #0d9488 !important; color: white !important;
    border: none !important; border-radius: 6px !important;
    font-size: 14px !important; font-weight: 500 !important; height: 46px !important;
}
.run-btn:hover { background: #0f766e !important; }
.nav-btn {
    background: #1a1a2e !important; color: white !important;
    border: none !important; border-radius: 6px !important;
    font-size: 12px !important; font-weight: 500 !important;
}
.nav-btn:hover { background: #2d2d4e !important; }
label { font-size: 12px !important; color: #475569 !important; }
.home-run-btn-overlay {
    background: #0d9488 !important; color: white !important;
    border: none !important; border-radius: 6px !important;
    font-size: 13px !important; font-weight: 500 !important;
    height: 46px !important; width: 180px !important;
    display: block !important; margin: 0 auto !important;
    position: static !important;
}
.home-run-btn-overlay:hover { background: #0f766e !important; }
.gradio-container .home-run-btn-overlay { background: #0d9488 !important; }
"""

with gr.Blocks(css=css, title="Infectious Disease Forecaster") as app:

    with gr.Column(visible=True) as home_page:

        gr.HTML("""
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
        <div style="background:#1a1a2e; border-radius:10px 10px 0 0; font-family:Inter,sans-serif;">
          <div style="padding:14px 32px; display:flex; align-items:center; justify-content:space-between;">
            <div style="display:flex; align-items:center; gap:8px;">
              <div style="width:28px; height:28px; background:#0d9488; border-radius:50%; display:flex; align-items:center; justify-content:center;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M8 12h8M12 8v8"/></svg>
              </div>
              <span style="color:white; font-size:13px; font-weight:500; letter-spacing:0.03em;">IDF · West Africa</span>
            </div>
            <div style="display:flex; gap:20px;">
              <span style="color:#94a3b8; font-size:12px; cursor:pointer;">How it works</span>
              <span style="color:#94a3b8; font-size:12px; cursor:pointer;">Dataset</span>
            </div>
          </div>
          <div style="padding:40px 32px 32px; text-align:center;">
            <span style="background:rgba(13,148,136,0.15); color:#5eead4; font-size:10px; padding:3px 12px; border-radius:20px; border:0.5px solid rgba(13,148,136,0.35); display:inline-block; margin-bottom:18px;">
              Powered by Linear Regression
            </span>
            <h1 style="font-size:28px; font-weight:600; color:white; margin:0 0 12px; line-height:1.35;">
              Forecast Infectious Disease<br>
              <span style="color:#0d9488;">Outbreaks Across West Africa</span>
            </h1>
            <p style="font-size:13px; color:#94a3b8; max-width:480px; margin:0 auto 0; line-height:1.85;">
              Enter environmental conditions and prior case history to receive monthly Malaria and Dengue fever predictions with risk classifications for seven West African countries.
            </p>
          </div>
        </div>
        """)

        home_run_btn = gr.Button("Run forecast", elem_classes=["home-run-btn-overlay"])

        gr.HTML("""
        <div style="background:#1a1a2e; font-family:Inter,sans-serif;">
          <div style="padding:28px 32px 44px; text-align:center;">
            <div style="display:flex; justify-content:center; gap:56px;">
              <div><div style="font-size:24px; font-weight:600; color:#0d9488;">490</div><div style="font-size:11px; color:#64748b; margin-top:3px;">Monthly records</div></div>
              <div><div style="font-size:24px; font-weight:600; color:#0d9488;">0.99</div><div style="font-size:11px; color:#64748b; margin-top:3px;">Malaria R²</div></div>
              <div><div style="font-size:24px; font-weight:600; color:#0d9488;">7</div><div style="font-size:11px; color:#64748b; margin-top:3px;">Countries covered</div></div>
              <div><div style="font-size:24px; font-weight:600; color:#0d9488;">2</div><div style="font-size:11px; color:#64748b; margin-top:3px;">Diseases modelled</div></div>
            </div>
          </div>
          <div style="background:#0d2137; padding:24px 32px; display:flex; justify-content:space-around; gap:8px;">
            <div style="text-align:center; flex:1;">
              <div style="width:36px; height:36px; background:rgba(13,148,136,0.15); border-radius:50%; display:flex; align-items:center; justify-content:center; margin:0 auto 10px;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#0d9488" stroke-width="2"><path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2h11"/></svg>
              </div>
              <div style="font-size:11px; font-weight:500; color:#e2e8f0; margin-bottom:4px;">1. Input data</div>
              <div style="font-size:10px; color:#64748b; line-height:1.6;">Enter environmental conditions and prior case counts</div>
            </div>
            <div style="width:0.5px; background:rgba(255,255,255,0.07); margin:0 12px;"></div>
            <div style="text-align:center; flex:1;">
              <div style="width:36px; height:36px; background:rgba(13,148,136,0.15); border-radius:50%; display:flex; align-items:center; justify-content:center; margin:0 auto 10px;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#0d9488" stroke-width="2"><rect x="2" y="2" width="20" height="20" rx="2"/><path d="M7 12h10M12 7v10"/></svg>
              </div>
              <div style="font-size:11px; font-weight:500; color:#e2e8f0; margin-bottom:4px;">2. Run model</div>
              <div style="font-size:10px; color:#64748b; line-height:1.6;">Linear Regression pipeline processes your inputs</div>
            </div>
            <div style="width:0.5px; background:rgba(255,255,255,0.07); margin:0 12px;"></div>
            <div style="text-align:center; flex:1;">
              <div style="width:36px; height:36px; background:rgba(13,148,136,0.15); border-radius:50%; display:flex; align-items:center; justify-content:center; margin:0 auto 10px;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#0d9488" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
              </div>
              <div style="font-size:11px; font-weight:500; color:#e2e8f0; margin-bottom:4px;">3. Get forecast</div>
              <div style="font-size:10px; color:#64748b; line-height:1.6;">Receive case predictions, risk level, and summary report</div>
            </div>
          </div>
          <div style="padding:36px 32px; background:white;">
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:28px; align-items:start;">
              <div>
                <div style="font-size:10px; font-weight:600; letter-spacing:0.08em; color:#0d9488; text-transform:uppercase; margin-bottom:10px;">About the dataset</div>
                <h2 style="font-size:16px; font-weight:500; color:#1a1a2e; margin:0 0 10px; line-height:1.5;">Built on WHO-scaled epidemiological data</h2>
                <p style="font-size:12px; color:#475569; line-height:1.85; margin:0 0 12px;">
                  The forecasting model is trained on a curated West Africa Infectious Disease Dataset derived from the Climate-Driven Disease Spread dataset on Kaggle. Case counts are scaled to match real WHO annual burden estimates, ensuring predictions reflect real-world epidemiological magnitudes.
                </p>
                <p style="font-size:12px; color:#475569; line-height:1.85; margin:0;">
                  Real surveillance data spans 2020 to 2023, extended with statistically consistent synthetic projections through 2025 to broaden the forecasting horizon.
                </p>
              </div>
              <div style="background:#f8fafc; border-left:3px solid #0d9488; border-top:0.5px solid #e2e8f0; border-right:0.5px solid #e2e8f0; border-bottom:0.5px solid #e2e8f0; border-radius:0 10px 10px 0; padding:18px 20px;">
                <div style="font-size:10px; font-weight:600; color:#0d9488; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:14px;">Dataset facts</div>
                <div style="display:flex; flex-direction:column; gap:10px;">
                  <div style="display:flex; justify-content:space-between; border-bottom:0.5px solid #e2e8f0; padding-bottom:8px;"><span style="font-size:12px; color:#64748b;">Total records</span><span style="font-size:12px; font-weight:500; color:#1a1a2e;">490 monthly rows</span></div>
                  <div style="display:flex; justify-content:space-between; border-bottom:0.5px solid #e2e8f0; padding-bottom:8px;"><span style="font-size:12px; color:#64748b;">Coverage period</span><span style="font-size:12px; font-weight:500; color:#1a1a2e;">2020 to 2025</span></div>
                  <div style="display:flex; justify-content:space-between; border-bottom:0.5px solid #e2e8f0; padding-bottom:8px;"><span style="font-size:12px; color:#64748b;">Countries</span><span style="font-size:12px; font-weight:500; color:#1a1a2e;">7 West African nations</span></div>
                  <div style="display:flex; justify-content:space-between; border-bottom:0.5px solid #e2e8f0; padding-bottom:8px;"><span style="font-size:12px; color:#64748b;">Diseases</span><span style="font-size:12px; font-weight:500; color:#1a1a2e;">Malaria and Dengue fever</span></div>
                  <div style="display:flex; justify-content:space-between; border-bottom:0.5px solid #e2e8f0; padding-bottom:8px;"><span style="font-size:12px; color:#64748b;">Model</span><span style="font-size:12px; font-weight:500; color:#1a1a2e;">Linear Regression (OLS)</span></div>
                  <div style="display:flex; justify-content:space-between;"><span style="font-size:12px; color:#64748b;">Case scaling</span><span style="font-size:12px; font-weight:500; color:#1a1a2e;">WHO annual estimates</span></div>
                </div>
              </div>
            </div>
          </div>
        </div>
        """)

        gr.HTML(FOOTER)

    with gr.Column(visible=False) as input_page:

        gr.HTML("""
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
        <div style="background:#1a1a2e; border-radius:10px 10px 0 0; padding:20px 24px 16px; text-align:center; font-family:Inter,sans-serif;">
          <div style="font-size:18px; font-weight:600; color:white; letter-spacing:0.04em; margin-bottom:4px;">INFECTIOUS DISEASE FORECASTER</div>
          <div style="font-size:12px; color:#94a3b8; margin-bottom:14px;">West Africa &nbsp;·&nbsp; Malaria and Dengue Fever &nbsp;·&nbsp; Linear Regression</div>
          <div style="display:flex; gap:6px; flex-wrap:wrap; justify-content:center;">
            <span style="background:rgba(13,148,136,0.2); color:#5eead4; font-size:10px; padding:2px 10px; border-radius:20px; border:0.5px solid rgba(13,148,136,0.35);">Nigeria</span>
            <span style="background:rgba(13,148,136,0.2); color:#5eead4; font-size:10px; padding:2px 10px; border-radius:20px; border:0.5px solid rgba(13,148,136,0.35);">Burkina Faso</span>
            <span style="background:rgba(13,148,136,0.2); color:#5eead4; font-size:10px; padding:2px 10px; border-radius:20px; border:0.5px solid rgba(13,148,136,0.35);">Mali</span>
            <span style="background:rgba(13,148,136,0.2); color:#5eead4; font-size:10px; padding:2px 10px; border-radius:20px; border:0.5px solid rgba(13,148,136,0.35);">Guinea-Bissau</span>
            <span style="background:rgba(13,148,136,0.2); color:#5eead4; font-size:10px; padding:2px 10px; border-radius:20px; border:0.5px solid rgba(13,148,136,0.35);">Liberia</span>
            <span style="background:rgba(13,148,136,0.2); color:#5eead4; font-size:10px; padding:2px 10px; border-radius:20px; border:0.5px solid rgba(13,148,136,0.35);">Mauritania</span>
            <span style="background:rgba(13,148,136,0.2); color:#5eead4; font-size:10px; padding:2px 10px; border-radius:20px; border:0.5px solid rgba(13,148,136,0.35);">Togo</span>
          </div>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("<div class='section-label'>Location and time</div>")
                country = gr.Dropdown(choices=list(country_meta.keys()), value="Nigeria", label="Country")
                with gr.Row():
                    year  = gr.Number(value=2026, label="Year", precision=0)
                    month = gr.Dropdown(choices=[(v, k) for k, v in month_names.items()], value=8, label="Month")

                gr.Markdown("<div class='section-label'>Regional parameters</div>")
                gr.Markdown("<small style='color:#64748b;'>Pre-filled with national averages. Adjust for a specific district.</small>")
                with gr.Row():
                    population_density = gr.Number(value=country_meta['Nigeria']['population_density'], label="Population density (per km²)")
                    healthcare_budget  = gr.Number(value=country_meta['Nigeria']['healthcare_budget'],  label="Healthcare budget (USD per capita)")

                gr.Markdown(" ")
                gr.Markdown("<div class='section-label'>Previous month climate</div>")
                with gr.Row():
                    rainfall_lag_1 = gr.Number(value=180.0, label="Last month rainfall (mm)")
                    temp_lag_1     = gr.Number(value=29.0,  label="Last month temperature (°C)")

            with gr.Column(scale=1):
                gr.Markdown(" ")
                gr.Markdown("<div class='section-label'>Environmental conditions</div>")
                with gr.Row():
                    avg_temp_c       = gr.Number(value=30.5,  label="Avg temperature (°C)")
                    precipitation_mm = gr.Number(value=200.0, label="Precipitation (mm)")
                with gr.Row():
                    air_quality_index = gr.Number(value=40.0, label="Air quality index")
                    uv_index          = gr.Number(value=11.0, label="UV index")

                gr.Markdown(" ")
                gr.Markdown("<div class='section-label'>Previous month case history</div>")
                gr.Markdown("<small style='color:#64748b;'>Pre-filled with typical monthly averages. Adjust if you have more accurate figures.</small>")
                with gr.Row():
                    malaria_lag_1 = gr.Number(value=country_meta['Nigeria']['malaria_avg'],             label="Malaria cases (last month)")
                    malaria_lag_2 = gr.Number(value=int(country_meta['Nigeria']['malaria_avg'] * 0.95), label="Malaria cases (2 months ago)")
                with gr.Row():
                    dengue_lag_1 = gr.Number(value=country_meta['Nigeria']['dengue_avg'],              label="Dengue cases (last month)")
                    dengue_lag_2 = gr.Number(value=int(country_meta['Nigeria']['dengue_avg'] * 0.95),  label="Dengue cases (2 months ago)")

        gr.Markdown(" ")
        with gr.Row():
            home_btn1 = gr.Button("Home", size="sm", elem_classes=["nav-btn"])
            run_btn   = gr.Button("Run forecast", variant="primary", size="lg", elem_classes=["run-btn"])

        gr.HTML(FOOTER)

    with gr.Column(visible=False) as results_page:

        gr.HTML("""
        <div style="background:#1a1a2e; border-radius:8px; padding:13px 20px; text-align:center; font-family:Inter,sans-serif;">
          <span style="color:white; font-size:14px; font-weight:500; letter-spacing:0.03em;">FORECAST RESULTS</span>
        </div>
        """)

        with gr.Row():
            home_btn2 = gr.Button("Home", size="sm", elem_classes=["nav-btn"])
            back_btn  = gr.Button("Back to form", size="sm", elem_classes=["nav-btn"])

        results_html = gr.HTML()
        gr.HTML(FOOTER)

    home_run_btn.click(fn=go_to_form, outputs=[home_page, input_page, results_page])
    home_btn1.click(fn=go_home, outputs=[home_page, input_page, results_page])
    home_btn2.click(fn=go_home, outputs=[home_page, input_page, results_page])
    back_btn.click(
        fn=lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)),
        outputs=[home_page, input_page, results_page]
    )
    country.change(
        fn=update_country_defaults,
        inputs=[country],
        outputs=[population_density, healthcare_budget, malaria_lag_1, malaria_lag_2, dengue_lag_1, dengue_lag_2]
    )
    run_btn.click(
        fn=predict,
        inputs=[
            country, year, month,
            avg_temp_c, precipitation_mm,
            air_quality_index, uv_index,
            population_density, healthcare_budget,
            malaria_lag_1, malaria_lag_2,
            dengue_lag_1, dengue_lag_2,
            rainfall_lag_1, temp_lag_1
        ],
        outputs=[results_html, home_page, input_page, results_page]
    )

app.launch(server_name="0.0.0.0", server_port=7860)
