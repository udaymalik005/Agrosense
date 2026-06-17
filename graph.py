"""
AgroSense AI — Chapter 6 Complete Graphs Generator
====================================================
Yeh script Chapter 6 ke saare graphs generate karke
PNG files ke roop mein save karega.

Run karo: python generate_all_graphs.py
Graphs save honge: chapter6_graphs/ folder mein

Phir screenshots leke report mein add karo!
"""

import pandas as pd
import numpy as np
import json
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Output folder banao
os.makedirs("chapter6_graphs", exist_ok=True)

# ── Common layout for all graphs (dark green theme)
LAYOUT = dict(
    paper_bgcolor = "#1a3a2a",
    plot_bgcolor  = "#0d2818",
    font          = dict(color="#e8f5e9", family="Arial", size=13),
    title_font    = dict(size=18, color="#52b788", family="Arial"),
    legend        = dict(bgcolor="rgba(0,0,0,0.3)", font=dict(color="#e8f5e9")),
    margin        = dict(l=60, r=40, t=70, b=60),
)

# ── Load data
print("Loading dataset and model...")
df   = pd.read_csv("data/crop_disease_dataset.csv")
with open("models/model_metadata.json") as f:
    meta = json.load(f)

print(f"Dataset: {df.shape[0]} samples, {df.shape[1]} columns")
print(f"Model Accuracy: {meta['accuracy']*100:.2f}%")

# Color palettes
DISEASE_COLORS = {
    "Healthy":        "#52b788",
    "Leaf Blight":    "#e63946",
    "Powdery Mildew": "#f4a261",
    "Root Rot":       "#c77dff",
    "Rust Disease":   "#ff9f1c",
    "Bacterial Wilt": "#e63946",
    "Downy Mildew":   "#48cae4",
    "Mosaic Virus":   "#f72585",
    "Anthracnose":    "#7b2d8b",
    "Fusarium Wilt":  "#bc6c25",
}

print("\n" + "="*55)
print(" GENERATING ALL CHAPTER 6 GRAPHS")
print("="*55)

# ════════════════════════════════════════════════════════
# GRAPH 1 — Disease Class Distribution (Horizontal Bar)
# ════════════════════════════════════════════════════════
print("\n[1/12] Disease Class Distribution...")

counts = df['disease_label'].value_counts().reset_index()
counts.columns = ['Disease', 'Count']
counts = counts.sort_values('Count', ascending=True)

fig1 = go.Figure()
fig1.add_trace(go.Bar(
    y = counts['Disease'],
    x = counts['Count'],
    orientation = 'h',
    marker = dict(
        color = counts['Count'],
        colorscale = [[0,"#1a3a2a"],[0.4,"#2d6a4f"],[0.7,"#52b788"],[1,"#95d5b2"]],
        showscale  = False,
        line       = dict(width=0)
    ),
    text = counts['Count'],
    textposition = 'outside',
    textfont = dict(color="#e8f5e9", size=12),
))
fig1.update_layout(**LAYOUT,
    title = "Figure 6.1: Disease Class Distribution in Dataset (5,000 Samples)",
    xaxis = dict(title="Number of Samples", gridcolor="#2d4a3a", zeroline=False),
    yaxis = dict(title="", gridcolor="#2d4a3a"),
    height = 500,
)
fig1.write_image("chapter6_graphs/Fig6_1_Disease_Distribution.png", scale=2)
print("   Saved: Fig6_1_Disease_Distribution.png")

# ════════════════════════════════════════════════════════
# GRAPH 2 — Average Yield Loss by Disease (Bar Chart)
# ════════════════════════════════════════════════════════
print("[2/12] Average Yield Loss by Disease...")

avg_loss = (df[df['disease_label'] != 'Healthy']
            .groupby('disease_label')['yield_loss_percent']
            .mean()
            .sort_values(ascending=False)
            .reset_index())
avg_loss.columns = ['Disease', 'Avg_Yield_Loss']

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x = avg_loss['Disease'],
    y = avg_loss['Avg_Yield_Loss'],
    marker = dict(
        color = avg_loss['Avg_Yield_Loss'],
        colorscale = [[0,"#f4a261"],[0.5,"#e76f51"],[1,"#e63946"]],
        showscale  = False,
        line = dict(width=0)
    ),
    text = [f"{v:.1f}%" for v in avg_loss['Avg_Yield_Loss']],
    textposition = 'outside',
    textfont = dict(color="#e8f5e9", size=12),
))
fig2.update_layout(**LAYOUT,
    title  = "Figure 6.2: Average Yield Loss (%) by Disease Category",
    xaxis  = dict(title="Disease", tickangle=-25, gridcolor="#2d4a3a"),
    yaxis  = dict(title="Average Yield Loss (%)", gridcolor="#2d4a3a"),
    height = 480,
)
fig2.write_image("chapter6_graphs/Fig6_2_Yield_Loss_by_Disease.png", scale=2)
print("   Saved: Fig6_2_Yield_Loss_by_Disease.png")

# ════════════════════════════════════════════════════════
# GRAPH 3 — Model Performance Metrics Bar Chart
# ════════════════════════════════════════════════════════
print("[3/12] Model Performance Metrics...")

metrics     = ["Accuracy", "F1-Score (Weighted)", "Precision (Weighted)", "Recall (Weighted)"]
values      = [meta['accuracy'], meta['f1_score'], meta['precision'], meta['recall']]
colors_bar  = ["#52b788", "#48cae4", "#f4a261", "#c77dff"]

fig3 = go.Figure()
fig3.add_trace(go.Bar(
    x = metrics,
    y = values,
    marker_color = colors_bar,
    marker_line  = dict(width=0),
    text  = [f"{v*100:.2f}%" for v in values],
    textposition = 'outside',
    textfont = dict(color="#e8f5e9", size=13, family="Arial"),
    width = 0.5,
))
fig3.add_hline(y=0.90, line_dash="dash", line_color="#f4a261",
               annotation_text="Target: 90%",
               annotation_font_color="#f4a261")
fig3.update_layout(**LAYOUT,
    title  = "Figure 6.3: Overall Model Performance Metrics (Test Set, n=1000)",
    xaxis  = dict(title="", gridcolor="#2d4a3a"),
    yaxis  = dict(title="Score", range=[0.85, 0.98], gridcolor="#2d4a3a",
                  tickformat=".1%"),
    height = 480,
)
fig3.write_image("chapter6_graphs/Fig6_3_Model_Performance_Metrics.png", scale=2)
print("   Saved: Fig6_3_Model_Performance_Metrics.png")

# ════════════════════════════════════════════════════════
# GRAPH 4 — 5-Fold Cross-Validation Scores
# ════════════════════════════════════════════════════════
print("[4/12] Cross-Validation Scores...")

cv_scores  = meta['cv_scores']
cv_mean    = meta['cv_mean']
fold_names = [f"Fold {i+1}" for i in range(5)]

fig4 = go.Figure()
fig4.add_trace(go.Bar(
    x = fold_names,
    y = cv_scores,
    marker = dict(
        color = ["#52b788","#95d5b2","#52b788","#95d5b2","#52b788"],
        line  = dict(width=0)
    ),
    text  = [f"{v*100:.2f}%" for v in cv_scores],
    textposition = 'outside',
    textfont = dict(color="#e8f5e9", size=13),
    width = 0.5,
    name = "CV Score per Fold",
))
fig4.add_hline(y=cv_mean, line_dash="dash", line_color="#f4a261", line_width=2,
               annotation_text=f"Mean = {cv_mean*100:.2f}%",
               annotation_font=dict(color="#f4a261", size=13))
fig4.update_layout(**LAYOUT,
    title  = f"Figure 6.4: 5-Fold Cross-Validation Scores (Mean = {cv_mean*100:.2f}% ± {meta['cv_std']*100:.2f}%)",
    xaxis  = dict(title="Fold", gridcolor="#2d4a3a"),
    yaxis  = dict(title="Accuracy", range=[0.92, 0.97], gridcolor="#2d4a3a",
                  tickformat=".1%"),
    height = 450,
    showlegend = False,
)
fig4.write_image("chapter6_graphs/Fig6_4_CrossValidation_Scores.png", scale=2)
print("   Saved: Fig6_4_CrossValidation_Scores.png")

# ════════════════════════════════════════════════════════
# GRAPH 5 — Feature Importance (Horizontal Bar)
# ════════════════════════════════════════════════════════
print("[5/12] Feature Importance...")

fi     = meta['feature_importance']
labels = [k.replace('_celsius','').replace('_percent','(%)').replace('_mm',' (mm)')
           .replace('_kg_ha',' (kg/ha)').replace('_kmh',' (km/h)')
           .replace('_hours',' (hrs)').replace('_hectares',' (ha)')
           .replace('_',' ').title()
          for k in fi.keys()]
vals   = list(fi.values())

# Sort descending
sorted_pairs = sorted(zip(labels, vals), key=lambda x: x[1])
s_labels, s_vals = zip(*sorted_pairs)

fig5 = go.Figure()
fig5.add_trace(go.Bar(
    y = s_labels,
    x = s_vals,
    orientation = 'h',
    marker = dict(
        color = s_vals,
        colorscale = [[0,"#1a3a2a"],[0.3,"#2d6a4f"],[0.6,"#52b788"],[1,"#95d5b2"]],
        showscale  = False,
        line = dict(width=0)
    ),
    text = [f"{v*100:.1f}%" for v in s_vals],
    textposition = 'outside',
    textfont = dict(color="#e8f5e9", size=11),
))
fig5.update_layout(**LAYOUT,
    title  = "Figure 6.5: Feature Importance Scores (Random Forest, 200 Trees)",
    xaxis  = dict(title="Importance Score", gridcolor="#2d4a3a", zeroline=False),
    yaxis  = dict(title="", gridcolor="#2d4a3a"),
)
fig5.write_image("chapter6_graphs/Fig6_5_Feature_Importance.png", scale=2)
print("   Saved: Fig6_5_Feature_Importance.png")

# ════════════════════════════════════════════════════════
# GRAPH 6 — Per-Class F1-Score Bar Chart
# ════════════════════════════════════════════════════════
print("[6/12] Per-Class F1-Score...")

cr      = meta['classification_report']
classes = [c for c in meta['classes'] if c in cr]
f1s     = [cr[c]['f1-score'] for c in classes]
precs   = [cr[c]['precision'] for c in classes]
recs    = [cr[c]['recall'] for c in classes]

# Color by performance
def grade_color(v):
    if v >= 0.96: return "#52b788"
    if v >= 0.93: return "#95d5b2"
    if v >= 0.87: return "#f4a261"
    return "#e63946"

bar_colors = [grade_color(v) for v in f1s]
sorted_idx = sorted(range(len(f1s)), key=lambda i: f1s[i], reverse=True)
s_classes  = [classes[i] for i in sorted_idx]
s_f1s      = [f1s[i]     for i in sorted_idx]
s_colors   = [bar_colors[i] for i in sorted_idx]

fig6 = go.Figure()
fig6.add_trace(go.Bar(
    y = s_classes,
    x = s_f1s,
    orientation = 'h',
    marker = dict(color=s_colors, line=dict(width=0)),
    text  = [f"{v:.3f}" for v in s_f1s],
    textposition = 'outside',
    textfont = dict(color="#e8f5e9", size=12),
))
fig6.add_vline(x=0.90, line_dash="dash", line_color="#f4a261",
               annotation_text="90% threshold",
               annotation_font_color="#f4a261")
fig6.update_layout(**LAYOUT,
    title  = "Figure 6.6: Per-Class F1-Score (Green=Excellent, Amber=Good, Red=Moderate)",
    xaxis  = dict(title="F1-Score", range=[0.75, 1.05], gridcolor="#2d4a3a"),
    yaxis  = dict(title="", gridcolor="#2d4a3a"),
    height = 500,
)
fig6.write_image("chapter6_graphs/Fig6_6_PerClass_F1_Score.png", scale=2)
print("   Saved: Fig6_6_PerClass_F1_Score.png")

# ════════════════════════════════════════════════════════
# GRAPH 7 — Per-Class Precision vs Recall (Grouped Bar)
# ════════════════════════════════════════════════════════
print("[7/12] Precision vs Recall per Class...")

fig7 = go.Figure()
fig7.add_trace(go.Bar(
    name = "Precision",
    x    = s_classes,
    y    = [precs[classes.index(c)] for c in s_classes],
    marker_color = "#52b788",
    marker_line  = dict(width=0),
    text  = [f"{precs[classes.index(c)]:.2f}" for c in s_classes],
    textposition = 'outside',
    textfont = dict(size=9, color="#e8f5e9"),
))
fig7.add_trace(go.Bar(
    name = "Recall",
    x    = s_classes,
    y    = [recs[classes.index(c)] for c in s_classes],
    marker_color = "#48cae4",
    marker_line  = dict(width=0),
    text  = [f"{recs[classes.index(c)]:.2f}" for c in s_classes],
    textposition = 'outside',
    textfont = dict(size=9, color="#e8f5e9"),
))
fig7.update_layout(**LAYOUT,
    title    = "Figure 6.7: Per-Class Precision vs Recall Comparison",
    xaxis    = dict(title="Disease Class", tickangle=-30, gridcolor="#2d4a3a"),
    yaxis    = dict(title="Score", range=[0.7, 1.08], gridcolor="#2d4a3a",
                    tickformat=".2f"),
    barmode  = 'group',
    bargap   = 0.2,
    bargroupgap = 0.05,
    height   = 500,
)
fig7.write_image("chapter6_graphs/Fig6_7_Precision_vs_Recall.png", scale=2)
print("   Saved: Fig6_7_Precision_vs_Recall.png")

# ════════════════════════════════════════════════════════
# GRAPH 8 — Temperature vs Humidity Scatter (by Disease)
# ════════════════════════════════════════════════════════
print("[8/12] Temperature vs Humidity Scatter...")

sample_df = df.sample(min(1200, len(df)), random_state=42)

fig8 = px.scatter(
    sample_df,
    x     = 'temperature_celsius',
    y     = 'humidity_percent',
    color = 'disease_label',
    color_discrete_map = DISEASE_COLORS,
    opacity = 0.65,
    labels  = {'temperature_celsius':'Temperature (°C)',
               'humidity_percent':'Humidity (%)',
               'disease_label':'Disease Class'},
    title   = "Figure 6.8: Temperature vs Humidity Distribution by Disease Class",
)
fig8.update_traces(marker=dict(size=6))
fig8.update_layout(**LAYOUT,
    xaxis  = dict(gridcolor="#2d4a3a", zeroline=False),
    yaxis  = dict(gridcolor="#2d4a3a", zeroline=False),
)
fig8.write_image("chapter6_graphs/Fig6_8_Temp_vs_Humidity_Scatter.png", scale=2)
print("   Saved: Fig6_8_Temp_vs_Humidity_Scatter.png")

# ════════════════════════════════════════════════════════
# GRAPH 9 — Rainfall vs Soil pH Scatter
# ════════════════════════════════════════════════════════
print("[9/12] Rainfall vs Soil pH Scatter...")

fig9 = px.scatter(
    sample_df,
    x     = 'rainfall_mm',
    y     = 'soil_ph',
    color = 'disease_label',
    color_discrete_map = DISEASE_COLORS,
    opacity = 0.65,
    labels  = {'rainfall_mm':'Rainfall (mm)',
               'soil_ph':'Soil pH',
               'disease_label':'Disease Class'},
    title   = "Figure 6.9: Rainfall vs Soil pH Distribution by Disease Class",
)
fig9.update_traces(marker=dict(size=6))
fig9.update_layout(**LAYOUT,
    xaxis  = dict(gridcolor="#2d4a3a", zeroline=False),
    yaxis  = dict(gridcolor="#2d4a3a", zeroline=False),
)
fig9.write_image("chapter6_graphs/Fig6_9_Rainfall_vs_SoilPH_Scatter.png", scale=2)
print("   Saved: Fig6_9_Rainfall_vs_SoilPH_Scatter.png")

# ════════════════════════════════════════════════════════
# GRAPH 10 — Feature Correlation Heatmap
# ════════════════════════════════════════════════════════
print("[10/12] Feature Correlation Heatmap...")

num_cols = ['temperature_celsius','humidity_percent','rainfall_mm',
            'soil_ph','nitrogen_kg_ha','phosphorus_kg_ha',
            'potassium_kg_ha','leaf_wetness_hours','yield_loss_percent']
short_labels = ['Temp (°C)','Humidity (%)','Rainfall (mm)','Soil pH',
                'Nitrogen','Phosphorus','Potassium','Leaf Wetness','Yield Loss']
corr = df[num_cols].corr()

fig10 = go.Figure()
fig10.add_trace(go.Heatmap(
    z    = np.round(corr.values, 2),
    x    = short_labels,
    y    = short_labels,
    colorscale  = [[0,"#0d2818"],[0.25,"#1a3a2a"],[0.5,"#2d6a4f"],
                   [0.75,"#52b788"],[1,"#d8f3dc"]],
    text = np.round(corr.values, 2),
    texttemplate = "%{text}",
    textfont     = dict(size=11, color="white"),
    zmin = -1, zmax = 1,
    showscale = True,
    colorbar  = dict(tickfont=dict(color="#e8f5e9")),
))
fig10.update_layout(**LAYOUT,
    title  = "Figure 6.10: Pearson Feature Correlation Matrix",
    xaxis  = dict(tickangle=-35, tickfont=dict(size=11, color="#e8f5e9")),
    yaxis  = dict(tickfont=dict(size=11, color="#e8f5e9")),
)
fig10.write_image("chapter6_graphs/Fig6_10_Correlation_Heatmap.png", scale=2)
print("   Saved: Fig6_10_Correlation_Heatmap.png")

# ════════════════════════════════════════════════════════
# GRAPH 11 — Treatment Cost by Disease (Box)
# ════════════════════════════════════════════════════════
print("[11/12] Treatment Cost Distribution...")

disease_order = (df[df['disease_label']!='Healthy']
                 .groupby('disease_label')['treatment_cost_inr']
                 .median().sort_values(ascending=False).index.tolist())

fig11 = go.Figure()
for d in disease_order:
    subset = df[df['disease_label']==d]['treatment_cost_inr']
    color = DISEASE_COLORS.get(d, "#52b788")

    fig11.add_trace(go.Box(
        y    = subset,
        name = d,
        marker_color = color,
        line_color   = color,
        fillcolor    = f"rgba({int(color[1:3],16)}, {int(color[3:5],16)}, {int(color[5:7],16)}, 0.27)",
        boxmean = True,
        showlegend   = False,
    ))
fig11.update_layout(**LAYOUT,
    title  = "Figure 6.11: Treatment Cost Distribution by Disease (INR per Acre)",
    xaxis  = dict(title="Disease Class", tickangle=-25, gridcolor="#2d4a3a"),
    yaxis  = dict(title="Treatment Cost (INR/acre)", gridcolor="#2d4a3a"),
    height = 500,
)
fig11.write_image("chapter6_graphs/Fig6_11_Treatment_Cost_Distribution.png", scale=2)
print("   Saved: Fig6_11_Treatment_Cost_Distribution.png")

# ════════════════════════════════════════════════════════
# GRAPH 12 — Severity Level Distribution (Pie)
# ════════════════════════════════════════════════════════
print("[12/12] Severity Level Distribution Pie Chart...")

sev_counts = df['severity_label'].value_counts()

fig12 = go.Figure()
fig12.add_trace(go.Pie(
    labels = sev_counts.index.tolist(),
    values = sev_counts.values.tolist(),
    marker = dict(
        colors = ["#52b788","#f4a261","#e63946","#95d5b2"],
        line   = dict(color="#0d2818", width=2)
    ),
    textinfo     = "label+percent",
    textfont     = dict(color="white", size=13),
    hole         = 0.38,
    pull         = [0.05 if l=="Severe" else 0 for l in sev_counts.index],
))
fig12.update_layout(**LAYOUT,
    title  = "Figure 6.12: Disease Severity Level Distribution (5,000 Samples)",
    height = 480,
    annotations = [dict(text="5,000\nSamples", x=0.5, y=0.5,
                        font=dict(size=14, color="#e8f5e9"), showarrow=False)],
)
fig12.write_image("chapter6_graphs/Fig6_12_Severity_Distribution_Pie.png", scale=2)
print("   Saved: Fig6_12_Severity_Distribution_Pie.png")

# ════════════════════════════════════════════════════════
# BONUS — Dashboard Screenshot Placeholder Guide
# ════════════════════════════════════════════════════════
print("\n" + "="*55)
print(" ALL 12 GRAPHS GENERATED SUCCESSFULLY!")
print("="*55)
print("\nFiles saved in: chapter6_graphs/")
print("\nAll generated graphs:")
for f in sorted(os.listdir("chapter6_graphs")):
    size = os.path.getsize(f"chapter6_graphs/{f}")
    print(f"  {f}  ({size//1024} KB)")

print("\n" + "="*55)
print(" DASHBOARD SCREENSHOTS BHI LENI HAIN:")
print("="*55)
print("""
Yeh graphs automatically generate ho gaye hain.
Ab dashboard ke screenshots bhi lene hain:

STEP 1: streamlit run app.py
STEP 2: Browser mein http://localhost:8501 kholo
STEP 3: Yeh screenshots lo:

  [SS-1]  Full dashboard home screen
          → Hero banner + 5 metric cards visible
          → File: chapter6_graphs/SS1_Dashboard_Home.png

  [SS-2]  Disease Prediction — Diseased Result
          → Sidebar: Temp=32, Humidity=88, Rainfall=200,
            pH=6.1, N=45, P=32, K=40, others default
          → Click: Predict Disease
          → Screenshot: Red result card (Leaf Blight ~91%)
          → File: chapter6_graphs/SS2_Prediction_LeafBlight.png

  [SS-3]  Disease Prediction — Healthy Result  
          → Sidebar: Temp=24, Humidity=55, Rainfall=85,
            pH=6.8, N=75, P=55, K=65, others default
          → Click: Predict Disease
          → Screenshot: Green result card (Healthy ~97%)
          → File: chapter6_graphs/SS3_Prediction_Healthy.png

  [SS-4]  Data Analytics Tab
          → Click Tab 2 (Data Analytics)
          → Screenshot: Full tab with charts visible
          → File: chapter6_graphs/SS4_Analytics_Tab.png

  [SS-5]  Model Insights Tab
          → Click Tab 3 (Model Insights)
          → Screenshot: Feature importance + CV scores
          → File: chapter6_graphs/SS5_Model_Insights_Tab.png

  [SS-6]  Dataset Explorer Tab
          → Click Tab 4 (Dataset Explorer)
          → Apply some filters
          → Screenshot: Filtered table + download button
          → File: chapter6_graphs/SS6_Dataset_Explorer_Tab.png
""")

print("Report mein add karne ka order:")
print("  Fig 6.1 → Fig 6.2 → [SS-1 Dashboard] → [SS-2 Prediction]")
print("  → [SS-3 Healthy] → Fig 6.3 → Fig 6.4 → Fig 6.5")
print("  → Fig 6.6 → Fig 6.7 → Fig 6.8 → Fig 6.9")
print("  → Fig 6.10 → Fig 6.11 → Fig 6.12")
print("  → [SS-4 Analytics] → [SS-5 Model Insights] → [SS-6 Explorer]")