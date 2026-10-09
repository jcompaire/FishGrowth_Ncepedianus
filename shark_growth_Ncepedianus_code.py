#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  10 11:13:41 2026
@author: Jesus C. Compaire
@position: Assistant Researcher Scientist
@institution: Center for the Study of Marine Systems (CESIMAR)
@position: Assistant Professor of Climate System
@institution: National University of the Patagonia San Juan Bosco (UNPSJB)
@e-mail:jesus.canocompaire@uca.es
"""
# %% Load libraries
import sys
import pandas as pd
import numpy as np
from scipy import stats
# import xlsxwriter
from docx import Document

# %% Loading data
# =============================================================================
# Actual folder path where functions are stored
path = '/home/gsus/gdrive/CENPAT/LECOPEC/SevengillShark_population_growth/'
sys.path.append(path)
from shark_growth_Ncepedianus_functions import *
file = path + '/Datasets/Ncepedianus_complete_17Jul2026.xlsx'
# --- Load databse and rename length variable
df = pd.read_excel(file, sheet_name='length_dataset')
df = df.rename(columns={'TL': 'TL (cm)'})
# --- Data type conversion
df['TL (cm)'] = pd.to_numeric(df['TL (cm)'], errors='coerce')
df['diff_TL'] = pd.to_numeric(df['diff_TL'], errors='coerce')
df['dt_days'] = pd.to_numeric(df['days_at_liberty'], errors='coerce')
# --- Creating variables
# Region
conditions = [
    (df['lat'] >= -41.2),
    (df['lat'] < -41.2) & (df['lat'] >= -44),
    (df['lat'] < -44)
    ]
lat_cat = ['north', 'central', 'south']
df['region'] = np.select(conditions, lat_cat, default='unknown')
df['region_sex'] = df['region'] + '_' + df['sex'].astype(str)
dfsex = df.dropna(subset=['sex', 'TL (cm)']).copy()
# 
# %% Figure 2 - Length frequency distributions by region and sex
# =============================================================================
fig, axes = plot_regional_size_distributions(
    df = dfsex,
    value_col = 'TL (cm)',
    save_path = path + "Fig_2_length_frequency_distribution_region.tiff",
    dpi = 650
    )
# 
# %% Length Analyses
# =============================================================================
# --- 1. General
print("--- Analysis: GENERAL ---")
df['TL (cm)'].count()
df['TL (cm)'].min()
df['TL (cm)'].max()
df['TL (cm)'].mean()
df['TL (cm)'].std()
# --- 1. Sex
print("--- Analysis: SEX ---")
print(summary_stats(dfsex, 'sex', value_col='TL (cm)'))
posthoc_sex = nonparam_group_analysis(dfsex, group_col='sex')
print("---")
# --- 2. Region
print("--- Analysis: REGION---")
summary_stats(df, 'region', value_col='TL (cm)')
posthoc_region = nonparam_group_analysis(df, group_col='region')
print("---")
# --- 3. Sex inside each Region
# ------ Central
print("--- Analysis: SEX - CENTRAL---")
dfcen = df[df['region'] == "central"]
summary_stats(dfcen, 'sex', value_col='TL (cm)')
posthoc_cen = nonparam_group_analysis(dfcen, group_col='sex')
print("---")
# ------ South
print("--- Analysis: SEX - SOUTH---")
dfsou = df[df['region'] == "south"]
summary_stats(dfsou, 'sex', value_col='TL (cm)')
posthoc_sou = nonparam_group_analysis(dfsou, group_col='sex')
print("---")
# ------ North
print("--- Analysis: SEX - NORTH---")
dfnor = df[df['region'] == "north"]
summary_stats(dfnor, 'sex', value_col='TL (cm)')
posthoc_nor = nonparam_group_analysis(dfnor, group_col='sex')
print("---")
#
# %% TAG - RECAPTURED events
# =============================================================================
# --- Create variables
df['L_recap'] = df['TL (cm)']
df['L_release'] = df['L_recap'] - df['diff_TL']
df['dt_years'] = df['dt_days'] / 365.0
# --- Subset by growth
recaptured = df.dropna(subset=['serie','growth_rate'])
# -----------------------------------------------------------------------------
# Save as XLSX to share in GitHub
tag_recaptured = recaptured[[
    'region','date', 'serie', 'TL (cm)', 
    'days_at_liberty', 'diff_TL', 'growth_rate', 'sex'
    ]]
# Define writer object
with pd.ExcelWriter(path + 'tag_recaptured.xlsx', engine="xlsxwriter") as writer:
   tag_recaptured.to_excel(writer, sheet_name="tag_recaptured",
                         index=False, index_label='index', header=True)
# -----------------------------------------------------------------------------
  
# --- Excluded specimens
recaptured_unreal = recaptured[(recaptured['growth_rate'] <= 0)
                             | (recaptured['growth_rate'] > 40)] 
print(f"negative or zero: {recaptured[recaptured['growth_rate'] <= 0].shape[0]}")
print(f"implausibly high: {recaptured[recaptured['growth_rate'] > 40].shape[0]}")
# --- Included specimens
# Regular analysis - Filtered
recaptured_pos = recaptured[(recaptured['growth_rate'] > 0)
                            & (recaptured['growth_rate'] <= 40)] 
print('> 0 Growth rate < 40: (Van Dykhuizen & Mollet,  1992)')
# Sensivity analysis - Expanded (including zero-growth records)
recaptured_pos = recaptured[(recaptured['growth_rate'] >= 0) 
                            & (recaptured['growth_rate'] <= 40)] 
# Sensivity analysis - Restricted (excluding individuals with < 60 days at liberty)
recaptured_pos =  recaptured_pos[recaptured_pos['days_at_liberty'] >= 60.0] 

# --- Create Table 2  for Regular analysis - Filtered
table2 = recaptured_pos[['region','date', 'serie', 'TL (cm)', 'days_at_liberty', 'diff_TL', 'growth_rate', 'sex']]
table2 = table2.reset_index()
table2["date"] = pd.to_datetime(table2["date"]).dt.date
table2 = table2.sort_values(by="date")
table2['serie'] = list(range(1, 14)) + [1] + list(range(14, 23)) + [8] + list(range(23,31)) + [9] + list(range(31,48)) 
table2 = table2.rename(columns={'serie': 'recapture'}) 
table2 = table2.reset_index()
table2 = table2.drop(columns=["index"])
table2["date"] = pd.to_datetime(table2["date"]).dt.date
cols = ["TL (cm)","days_at_liberty","diff_TL"]
table2[cols] = table2[cols].astype(int)
table2["growth_rate"] = table2["growth_rate"].round(3)
table2 = table2.iloc[:,2:]
# Export Table 2
doc = Document()
table = doc.add_table(rows=1, cols=len(table2.columns))
table.style = "Table Grid"
# Headers
hdr_cells = table.rows[0].cells
for i, col in enumerate(table2.columns):
    hdr_cells[i].text = str(col)
# Data
for _, row in table2.iterrows():
    cells = table.add_row().cells
    for i, value in enumerate(row):
        cells[i].text = str(value)
doc.save("Table_2.docx")# Save
#
# %% MATURITY LENGTHS - Based on Irigoyen et al. (2018)
# =============================================================================
# Females
fem = recaptured_pos[recaptured_pos['sex'] == 'F']
fem_mat = recaptured_pos[
    (recaptured_pos['sex'] == 'F') & (recaptured_pos['TL (cm)'] >= 190)
    ]
fem_juv = recaptured_pos[
    (recaptured_pos['sex'] == 'F') & (recaptured_pos['TL (cm)'] < 190)
    ]
# Males
mal = recaptured_pos[recaptured_pos['sex'] == 'M']
mal_mat = recaptured_pos[
    (recaptured_pos['sex'] == 'M') & (recaptured_pos['TL (cm)'] >= 170)
    ]
mal_juv = recaptured_pos[
    (recaptured_pos['sex'] == 'M') & (recaptured_pos['TL (cm)'] < 170)
    ]
#
# %% Growth Correlation Analysis with 95% Confidence Intervals
# =============================================================================
# --- Dictionary to automatize the analyses for each group
analysis_groups = {
    "ALL": recaptured_pos,
    "FEMALE": fem,
    "FEMALE MATURE": fem_mat,
    "FEMALE JUVENILE": fem_juv,
    "MALE MATURE": mal
}
print("--- Pearson Correlation Analysis (with 95% CI) ---")
for name, group_df in analysis_groups.items():
    # Ensure there are no NaN values
    clean_group = group_df.dropna(subset=['days_at_liberty', 'diff_TL'])
    n = len(clean_group) 
    if n >= 3:
        # SciPy pearsonr
        res = stats.pearsonr(clean_group['days_at_liberty'], clean_group['diff_TL'])
        r = res.statistic
        p_val = res.pvalue        
        # Getting 95% CI
        ci = res.confidence_interval(confidence_level=0.95)
        ci_low, ci_high = ci.low, ci.high
        print(f"{name: <16} | n = {n: <2} | r = {r: .3f} (95% CI: [{ci_low: .3f}, {ci_high: .3f}]) | p = {p_val: .4f}")
    else:
        print(f"{name: <16} | n = {n: <2} | Insufficient data")
# 
# --- Subset for females and growth rate
df_growth = fem[['L_release', 'L_recap', 'dt_years', 'days_at_liberty']]
df_growth.reset_index()
# 
# %% Model fitting (MLE)
# =============================================================================
# --- 1. Set model parameters
L1 = np.array(df_growth['L_release'].values)
L2 = np.array(df_growth['L_recap'].values)
dt_years = np.array(df_growth['dt_years'].values)
Linf = 280.0 # cm (fixed according to Milessi 2008)
L0 = 40 # cm (size at birth according to Jaureguizar et al. 2022)
n = len(L2)

# --- 2. Optimizer configuration to fit the models
p0_set = [0.01]        # initial value for K
bounds_set = (0, 2.0)  # # min and max K (the second one according to fishbase) 
# https://www.fishbase.se/summary/Notorynchus_cepedianus.html

# --- 3. Running models
nonlinear_models = {
    'Fabens': model_fabens,
    'Gompertz': model_gompertz,
    'Logistic': model_logistic
}
results = {}
for name, model_func in nonlinear_models.items():
    
    # Initial values: K, log(sigma)
    init_params = [p0_set[0], np.log(15.0)]
    bounds = [
        bounds_set,      # K bounds
        (None, None)     # log(sigma) unbounded
    ]
    
    opt = minimize(
        neg_loglik_normal,
        x0=init_params,
        args=(model_func, L2, dt_years, L1, Linf),
        method='L-BFGS-B', #'SLSQP
        bounds=bounds
    )
    
    K_hat = opt.x[0]
    sigma_hat = np.exp(opt.x[1])
    
    results[name] = {
        'K': K_hat,
        'sigma': sigma_hat,
        'logLik': -opt.fun
    }
# %% MODEL COMPARISON (AIC - MLE BASED)
# =============================================================================
print("Calculating AICc from maximum likelihood...")
for name, metrics in results.items():
    results[name]['AICc'] = get_aicc_mle(
        metrics['logLik'],
        n_params=2,   # K + sigma
        n=n
    )
# %% Bootstrap confidence intervals
print("Running Bootstrap (this may take a while)...")
for name in results.keys():
    lower, upper = bootstrap_CI(name, nonlinear_models,
                                L1, L2, Linf,
                                dt_years, nboot=999)
    results[name]['CI_2.5'] = lower
    results[name]['CI_97.5'] = upper
# 
# %% RESIDUAL DIAGNOSTIC
# =============================================================================
from statsmodels.sandbox.stats.runs import runstest_1samp
print("Residual diagnostics...")
for name in results.keys():
    K_est = results[name]['K']
    func = nonlinear_models[name]
    pred = func((dt_years, L1), K_est, Linf)
    resid = L2 - pred
    # --- Normality ---
    _, shapiro_p = stats.shapiro(resid)
    results[name]['Norm_p'] = shapiro_p # p < 0.05 indicates non-normality.
    # --- Bias ---
    _, spearman_p = stats.spearmanr(L1, resid)
    results[name]['SizeDep_p'] = spearman_p # p < 0.05 indicates Bias (size-dependency).
    # --- Runs ---
    df_temp = pd.DataFrame({'resid': resid, 'L1': L1}).sort_values('L1')
    _, runs_p = runstest_1samp(df_temp['resid'].values, cutoff=0, correction=False)
    results[name]['Runs_p'] = runs_p # p < 0.05 indicates non-Randomness. 
# 
# %% Display TABLE 3 ---
# =============================================================================
df_final = pd.DataFrame(results).T
min_aic = df_final['AICc'].min()
df_final['Delta_AIC'] = df_final['AICc'] - min_aic
cols = [
    'K',
    'CI_2.5',
    'CI_97.5',
    'Delta_AIC',
    'Norm_p',
    'SizeDep_p',
    'Runs_p'
]
print(f"\nFixed Linf: {Linf} cm")
print("-" * 80)
# Rounding to 4 decimal places, except K and SE
print(df_final[cols].round(4))
print("-" * 80)
print("Note: p-values < 0.05 indicate violation of assumptions (non-normal, biased, or non-random).")
# %% Get figure CURVES
ages, curves_data = generate_growth_curves(
    results=results,
    Linf=Linf,
    L0=L0,
    max_age=50,
    n_points=100
)
#
# %% Figure 3 - Female growth curves
# =============================================================================
tl_min = min(df_growth['L_recap'])
tl_max = max(df_growth['L_recap'])
# To display and save high-resolution TIFF:
fig, ax = plot_growth_curves(
    ages = ages,
    curves_data = curves_data,
    results = results,
    Linf = Linf,
    L0 = L0,
    tl_range = (tl_min, tl_max),
    save_path = path + "Fig_3_growth_models.tiff",
    dpi=650
)
# 

