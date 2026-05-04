#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 2026
@author: Jesus C. Compaire
@position: Assistant Researcher Scientist
@institution: Center for the Study of Marine Systems (CESIMAR)
@position: Assistant Professor of Climate System
@institution: National University of the Patagonia San Juan Bosco (UNPSJB)
@e-mail:jesus.canocompaire@uca.es
"""

# %% Script description
## -- -- -- -- -- -- -- -- -- -- -- -- --
##
## With this script you will be able to repliacte the analyses and Figures 2
## and 3 of the manuscript Compaire et al. (2026a).
## To learn more about this study, you may read the manuscript Compaire et al. 
## (2026a). Whilst for a detailed the description of each variable included in
## the dataset check Compaire et al. (2026b).
##
## References:
##
## Compaire, J.C., Trobbiani, G., Mas, F., Cuevas, J.M., Pantano, C., Laporta
## M., Albornoz, L., Lagger, C., Irigoyen, A. (2026a). The broadnose sevengill
## shark, Notorynchus cepedianus, an apex predator: population structure and
## growth estimates.
## Fish and Fisheries, XX(X), XXX.
## https://doi.org/XXXXX
##
## Compaire, J.C., Trobbiani, G., Mas, F., Cuevas, J.M., Pantano, C., Laporta
## M., Albornoz, L., Lagger, C., Irigoyen, A. (2026b). Length data XXX 
## PANGAEA, https://doi.pangaea.de/10.1594/PANGAEA.932955
##
## -- -- -- -- -- -- -- -- -- -- -- -- --
#
# %% Importing packages ####
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import shapiro, levene, kruskal
import scikit_posthocs as sp
from statsmodels.sandbox.stats.runs import runstest_1samp
# %% GROWTH MODELS functions
# =============================================================================
#
# Note: Linf must be a scalar defined globally or passed as an argument.
# Here we assume Linf is a global variable fixed based on bibliography.
#
# Since the absolute age (t) is unknown in tag-recapture data, we use the 
# "interval form", which expresses recapture length (L2) as a function 
# of release length (L1) and time at liberty (dt), effectively eliminating 
# the need for age 't' and the location parameter (t0/c).
#
def model_fabens(X, K, Linf):
    """
    Von Bertalanffy Growth Model (Fabens method).
    Calculates recapture length based on release length and time at liberty.
    
    Original Age-based Equation (Beverton, 1954):
        Lt = Linf * (1 - exp(-K * (t - t0)))
    
    Parameters:
    -----------
    X : tuple or list (dt, L1)
        dt : Time at liberty (years/days)
        L1 : Length at release
    K : float
        Growth coefficient to be estimated
        
    Returns:
    --------
    L2 : Predicted length at recapture
    """
    dt, L1 = X
    return Linf - (Linf - np.asarray(L1)) * np.exp(-K * np.asarray(dt))

def model_gompertz(X, K, Linf):
    """
    Gompertz Growth Model (Interval form).
    Appropriate for species where growth is slowest at start and end.
    
    Original Age-based Equation (Gompertz, 1825):
        Lt = Linf * exp(-exp(-cg * (kg * t)))
        (Note: The constant 'cg' is eliminated in the interval form)
    
    Parameters:
    -----------
    X : tuple (dt, L1)
    K : float (Growth coefficient 'kg')
    Linf : float (Asymptotic length)
    """
    dt, L1 = X
    # Using the relationship: L2 = L1 * exp(log(Linf/L1) * exp(-K*dt))
    # Note: Ensure L1 > 0 to avoid log(0) errors
    return Linf * np.exp(np.log(np.asarray(L1) / Linf) * np.exp(-K * np.asarray(dt)))

def model_logistic(X, K, Linf):
    """
    Logistic (Robertson) Growth Model (Interval form).
    Sigmoidal curve symmetric around the inflection point.
    
    Original Age-based Equation (Robertson, 1923):
        Lt = Linf / (1 + exp(cr - kr * t))
        (Note: The constant 'cr' is eliminated in the interval form)
    
    Equation derivation for interval:
    L2 = Linf / (1 + ( (Linf/L1) - 1 ) * exp(-K * dt) )
    
    Parameters:
    -----------
    X : tuple (dt, L1)
    K : float (Growth coefficient 'kr')
    Linf : float (Asymptotic length)
    """
    dt, L1 = X
    return Linf / (1 + ((Linf / np.asarray(L1)) - 1) * np.exp(-K * np.asarray(dt)))

# %% MODEL FITTING (MLE - Maximum Likelihood Estimate)
# =============================================================================
# 
def neg_loglik_normal(params, model_func, L2_obs, dt, L1, Linf):
    """
    Negative log-likelihood assuming normal errors.
    
    params[0] = K
    params[1] = log_sigma  (log-scale to keep sigma > 0)
    Linf : Asymptotic length passed to the growth model
    """
    K = params[0]
    sigma = np.exp(params[1])
    
    L2_pred = model_func((dt, L1), K, Linf)
    resid = L2_obs - L2_pred
    
    n = len(L2_obs)
    nll = (
        0.5 * n * np.log(2 * np.pi)
        + n * np.log(sigma)
        + 0.5 * np.sum((resid / sigma) ** 2)
    )
    return nll
results = {}

# %% BOOTSTRAP Analysis cconfidence intervals
# =============================================================================
#
# Estimation of uncertainty using resampling, recommended for 
# non-normal error distributions common in tag-recapture.
def bootstrap_CI(model_name, L1, L2, dt,  nboot=999, seed=23):
    """
    Performs bootstrap resampling to calculate 95% Confidence Intervals for K.
    
    Parameters:
    -----------
    model_name : str
        'Fabens', 'Gompertz', 'Logistic'
    nboot : int
        Number of bootstrap iterations
        
    Returns:
    --------
    ci_lower, ci_upper : The 2.5th and 97.5th percentiles
    """
    rng = np.random.default_rng(seed)
    boots = []
    
    # Convert series to numpy arrays just in case
    L1_arr = np.array(L1)
    L2_arr = np.array(L2)
    dt_arr = np.array(dt)
    N = len(L1_arr)
    
    for i in range(nboot):
        # 1. Resample indices with replacement
        idx = rng.integers(0, N, N)
        
        L1b = L1_arr[idx]
        L2b = L2_arr[idx]
        dtb = dt_arr[idx]
        
        try:
            if model_name in ['Fabens', 'Gompertz', 'Logistic']:
                # Select the function from the dictionary defined previously
                # Note: We assume 'nonlinear_models' dict exists from previous step
                func = nonlinear_models[model_name]
                opt_b = minimize(
                    neg_loglik_normal,
                    x0=[0.05, np.log(5.0)],
                    args=(func, L2b, dtb, L1b, Linf),
                    method='L-BFGS-B',
                    bounds=[(0, 2.0), (None, None)]
                )
                boots.append(opt_b.x[0])
                
            # elif model_name == 'Gulland_Holt':
            #     # Recalculate transformations for the resampled data
            #     yb = (L2b - L1b) / dtb
            #     Lbarb = (L1b + L2b) / 2.0
            #     xb = Linf - Lbarb
                
            #     # IMPORTANT: Weights must follow the resampled points
            #     # sigma = 1 / sqrt(dt)
            #     sigmab = 1.0 / np.sqrt(dtb)
                
            #     popt_b, _ = curve_fit(
            #         model_gulland_holt, xb, yb, 
            #         p0=p0_set, sigma=sigmab, absolute_sigma=False
            #     )
            #     boots.append(popt_b[0])
                
        except RuntimeError:
            # Skip iterations where curve_fit fails to converge
            continue
            
    boots = np.array(boots)
    # Calculate Percentiles (95% CI)
    return np.nanpercentile(boots, [2.5, 97.5])
# Execute bootstrap
print("Running Bootstrap (this may take a moment)...")

# %% MODEL COMPARISON (AICc - MLE BASED)
# =============================================================================
print("Calculating AICc from maximum likelihood...")
# 
def get_aicc_mle(loglik, n_params, n):
    """
    Calculates AIC from maximum log-likelihood.
    
    Parameters:
    -----------
    loglik : float
        Maximum log-likelihood value
    n_params : int
        Number of estimated parameters (including sigma)
    """
    aic = -2.0 * loglik + 2.0 * n_params
    return aic + (2 * n_params * (n_params + 1)) / (n - n_params - 1)
# %% Plot MODEL GROTH CURVES
# =============================================================================
# AGE-BASED MODEL DEFINITIONS
def vbgm_age(t, K, t0):
    """Classical Von Bertalanffy Growth Model (Age-based)."""
    return Linf * (1 - np.exp(-K * (t - t0)))

def gompertz_age(t, K, b_param):
    """
    Gompertz Model (Age-based).
    b_param is the dimensionless integration coefficient.
    Equation: L(t) = Linf * exp( -b * exp(-K * t) )
    """
    return Linf * np.exp(-b_param * np.exp(-K * t))

def logistic_age(t, K, c_param):
    """
    Logistic Model (Age-based).
    c_param is the integration constant related to the inflection point.
    Equation: L(t) = Linf / (1 + c * exp(-K * t))
    """
    return Linf / (1 + c_param * np.exp(-K * t))

# %% Loading data
# =============================================================================
file = '/home/gsus/gdrive/CENPAT/LECOPEC/SevengillShark/Datasets/Ncepedianus_complete_11Feb2026.xlsx'
data = pd.read_excel(file, sheet_name='length_dataset')
df = data
df = df.rename(columns={'TL': 'TL (cm)'}) 
df["TL (cm)"] = pd.to_numeric(df["TL (cm)"], errors='coerce')
# Creating a categorical variable according to LATITUDE
conditions = [
    (df['lat'] >= -41.2),
    (df['lat'] < -41.2) & (df['lat'] >= -44),
    (df['lat'] < -44)
]
lat_cat = ['north', 'central', 'south']
df['region'] = np.select(conditions, lat_cat, default='unknown')
df['region_sex'] = df['region'] + '_' + df['sex']
df = df.dropna(subset=['TL (cm)'])
dfsex = df.dropna(subset=['sex'])

# =============================================================================
# --- Subset by growth
recaptured = data
recaptured_pos = recaptured[(recaptured['growth_rate'] > 0)
                            & (recaptured['growth_rate'] <= 40)] 
recaptured_pos = recaptured_pos.rename(columns={'TL': 'TL (cm)'}) 
print('> 0 Growth rate < 40: (Van Dykhuizen & Mollet,  1992)')
# All population
stats.shapiro(recaptured_pos['TL (cm)'])[1]
r, p_value = stats.pearsonr(
    recaptured_pos['days_at_liberty'], recaptured_pos['diff_TL']
    )
print(f"Coefficient: {r:.3f}, P-value: {p_value:.4f}, ALL (n = {recaptured_pos['days_at_liberty'].count()})")
# Females
fem = recaptured_pos[recaptured_pos['sex'] == 'F']
stats.shapiro(fem['TL (cm)'])[1]
r, p_value = stats.pearsonr(
    fem['days_at_liberty'], fem['diff_TL']
    )
print(f"Coefficient: {r:.3f}, P-value: {p_value:.4f}, FEMALE (n = {fem['days_at_liberty'].count()})")
# Males
mal = recaptured_pos[recaptured_pos['sex'] == 'M']
stats.shapiro(mal['TL (cm)'])[1]
r, p_value = stats.pearsonr(
    mal['days_at_liberty'], mal['diff_TL']
    )
print(f"Coefficient: {r:.3f}, P-value: {p_value:.4f}, MALE MATURE (n = {mal['days_at_liberty'].count()})")

# =============================================================================
fem['L_recap'] = pd.to_numeric(fem['TL (cm)'], errors='coerce')
fem['diff_TL'] = pd.to_numeric(fem['diff_TL'], errors='coerce')
fem['dt_days'] = pd.to_numeric(fem['days_at_liberty'], errors='coerce')
fem['L_release'] = fem['L_recap'] - fem['diff_TL']
fem['dt_years'] = fem['dt_days'] / 365.0
fem = fem.dropna(subset=['L_release', 'L_recap', 'dt_years'])
# %% A little bit of common statistic (Main text results + Table 1)
def summary_stats(df, group_col, value_col='TL'):
    return (
        df.groupby(group_col)[value_col]
        .agg(
            n='count',
            min='min',
            max='max',
            mean='mean',
            sd='std'
        )
        .reset_index()
    )
def nonparam_group_analysis(df, group_col, 
                            value_col='TL (cm)',
                            p_adjust='bonferroni'):
    """
    Performs normality, homoscedasticity, Kruskal–Wallis and Dunn post-hoc tests.
    """

    # Clean data
    data = df.dropna(subset=[group_col, value_col])

    print("Normality (Shapiro–Wilk by group):")
    for gname, g in data.groupby(group_col):
        if len(g[value_col]) >= 3:
            stat, p = shapiro(g[value_col])
            print(f"  {gname}: W={stat:.3f}, p={p:.4f}")
        else:
            print(f"  {gname}: n < 3, test not performed")

    # Prepare groups
    groups = [g[value_col] for _, g in data.groupby(group_col)]

    # Homoscedasticity
    stat, p = levene(*groups)
    print(f"\nLevene test: W={stat:.3f}, p={p:.4f}")

    # Kruskal–Wallis
    stat, p = kruskal(*groups)
    df_kw = len(groups) - 1
    print(f"Kruskal–Wallis: H({df_kw}) = {stat:.3f}, p={p:.4f}")

    # Post-hoc Dunn
    posthoc = sp.posthoc_dunn(
        data,
        val_col=value_col,
        group_col=group_col,
        p_adjust=p_adjust
    )
    # 🔹 Keep only lower diagonal
    posthoc = posthoc.where(
        np.tril(np.ones(posthoc.shape), k=-1).astype(bool)
    )

    return posthoc
# --- 1. General
metrics = ['count', 'min', 'max', 'mean', 'std']
summary = pd.to_numeric(df['TL (cm)'], errors='coerce').agg(metrics)
print(summary)
# --- 1. Sex
summary_stats(dfsex, 'sex', value_col='TL (cm)')
posthoc_sex = nonparam_group_analysis(dfsex, group_col='sex')
# --- 2. Region
summary_stats(df, 'region', value_col='TL (cm)')
posthoc_region = nonparam_group_analysis(df, group_col='region')
# --- 3. Sex inside each Region
# ------ Central
dfcen = df[df['region'] == "central"]
summary_stats(dfcen, 'sex', value_col='TL (cm)')
posthoc_cen = nonparam_group_analysis(dfcen, group_col='sex')
# ------ South
dfsou = df[df['region'] == "south"]
summary_stats(dfsou, 'sex', value_col='TL (cm)')
posthoc_sou = nonparam_group_analysis(dfsou, group_col='sex')
# ------ North
dfnor = df[df['region'] == "north"]
summary_stats(dfnor, 'sex', value_col='TL (cm)')
posthoc_nor = nonparam_group_analysis(dfnor, group_col='sex')

# %% Histogram for each REGION and SEX (Figure 2)
df# %% Model fitting (MLE)# %% Model fitting (MLE)# %% Model fitting (MLE)# %% Model fitting (MLE)# %% Model fitting (MLE)
dfh = df[df['sex'] == "F"]
dfm = df[df['sex'] == "M"]

# --- 1. Global Plotting Configuration ---
sns.set_style("ticks") 
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif", 
    "mathtext.fontset": "stix",
    "font.serif": ["STIXGeneral"], 
    'axes.labelweight': 'bold',
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.edgecolor': 'black',
    'figure.dpi': 300
})

# Sorting dataframes and titles
dfs = [df, dfh, dfm]
titles = ['Total', 'Females', 'Males']
labels = ['A', 'B', 'C']
region_order = ['north', 'central', 'south']
palette_custom = [sns.color_palette("colorblind")[i] for i in [4, 8, 9]]

# --- 2. Figure Initialization (3x1) ---
fig, axes = plt.subplots(3, 1, figsize=(12, 16), constrained_layout=True) #, sharey=True)

# --- 3. Loop ---
for i, ax in enumerate(axes):
    # sns.kdeplot(
    sns.histplot(
        data=dfs[i], 
        x='TL (cm)', 
        hue='region', 
        hue_order=region_order,
        fill=True,
        kde=True, 
        binwidth=10, 
        element="step", 
        palette=palette_custom,
        ax=ax,
        line_kws={'linewidth': 2.5},
        legend=(i == 0) # Legend only in the first panel
    )

    # Axes
    ax.set_xlim(40, 270) # y-limits
    ax.set_xticks(np.arange(40, 271, 35)) # x-limits
    ax.set_ylabel(r'$\mathbf{Frequency}$', fontsize=23, labelpad=8)
    
    # Only one X label sharey=True
    if i == 2:
        ax.set_xlabel(r'$\mathbf{Total\ length\ (cm)}$', fontsize=23, labelpad=8)
    else:
        ax.set_xlabel("")
    ax.tick_params(axis='both', which='major', labelsize=18)
    # Legend
    sns.move_legend(
        axes[0], 
        loc='upper right',
        bbox_to_anchor=(0.21, 0.9), 
        frameon=False,
        title=None,              # remove title of legend
        prop={'size': 23},       # fontsize in legend
        markerscale=1.5          # iconsize in legend
    )

    # Tags A, B, C
    ax.text(0.05, 0.95, labels[i], transform=ax.transAxes, 
            fontsize=32, fontweight='bold', va='top', ha='right')
    
    # Title for each panel
    ax.text(0.45, 0.80, titles[i], 
        transform=ax.transAxes, 
        fontsize=28, 
        fontweight='bold'
        # verticalalignment='center',
        # bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
        )
    
    sns.despine(ax=ax)

# --- 4. Save Figure ---
plt.savefig("Fig2_Ncepedianus_size_distribution_sex_region.tiff",
            dpi=650,
            bbox_inches='tight',
            pil_kwargs={"compression": "tiff_lzw"}
            )
# plt.savefig("Fig2_Ncepedianus_size_distribution_sex_region.png",
#            dpi=300, bbox_inches='tight')
plt.show()

# %% Model fitting (MLE)
del df
df = fem
# --- 1. Set model parameters
L1 = np.array(df['L_release'].values)
L2 = np.array(df['L_recap'].values)
dt_years = np.array(df['dt_years'].values)
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

# %% MODEL COMPARISON (AICc)
# =============================================================================
print("Calculating AICc from maximum likelihood...")
for name, metrics in results.items():
    results[name]['AICc'] = get_aicc_mle(
        metrics['logLik'],
        n_params=2,
        n=n
    )

# %% Bootstrap confidence intervals
print("Running bootstrap...")
for name in results.keys():
    lower, upper = bootstrap_CI(name, L1, L2, dt_years, nboot=999)
    results[name]['CI_2.5'] = lower
    results[name]['CI_97.5'] = upper

# %% RESIDUAL DIAGNOSTIC
# =============================================================================
print("Residual diagnostics...")
for name in results.keys():

    K_est = results[name]['K']

    func = nonlinear_models[name]
    pred = func((dt_years, L1), K_est, Linf)

    resid = L2 - pred

    # --- Normality ---
    _, shapiro_p = stats.shapiro(resid)
    results[name]['Norm_p'] = shapiro_p # p < 0.05 indicates NON-normality.

    # --- Bias ---
    _, spearman_p = stats.spearmanr(L1, resid)
    results[name]['SizeDep_p'] = spearman_p # p < 0.05 indicates Bias (size-dependency).

    # --- Runs ---
    df_temp = pd.DataFrame({'resid': resid, 'L1': L1}).sort_values('L1')
    _, runs_p = runstest_1samp(df_temp['resid'].values, cutoff=0, correction=False)
    results[name]['Runs_p'] = runs_p # p < 0.05 indicates non- Randomness. 

# %% DISPLAY FINAL TABLE ---
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
print(df_final[cols].round(4))
print("-" * 80)
print("Note: p-values < 0.05 indicate violation of assumptions (non-normal,\
      biased, or non-random).")

# %% Calculate GROWTH CURVES
# Paramters and curve calculation
ages = np.linspace(0, 50, 100) # Projection range (0 to 50 years - FishBase)
# We calculate the Y-values (mean, lower CI, upper CI) here to keep plotting clean.
curves_data = {}
# print("--- Calculation of parameters (t0, b, c) ---")
for name, metrics in results.items():
    K_mean = metrics['K']
    # Use 0.01 as default SE if not present in dictionary
    # SE_K = metrics.get('SE', 0.01) 
    # 1. Define K extremes (95% Confidence Interval)
    K_low = results[name]['CI_2.5'] #K_mean - 1.96 * SE_K
    K_upp = results[name]['CI_97.5'] #K_mean + 1.96 * SE_K
    # Calculate the Anchor Parameter based on the Model Type
    #    CRITICAL: The anchor parameter (t0, b, c) must be recalculated for 
    #    each K (mean, low, upp) to ensure all curves pass through L0 at t=0.
    if name in ['Fabens']:
        # --- VON BERTALANFFY ---
        # Solving for t0: t0 = (1/K) * ln( (Linf - L0) / Linf )
        t0_mean = (1/K_mean) * np.log((Linf - L0) / Linf)
        t0_low  = (1/K_low)  * np.log((Linf - L0) / Linf)
        t0_upp  = (1/K_upp)  * np.log((Linf - L0) / Linf)
        # Generate Y data
        y_mean = vbgm_age(ages, K_mean, t0_mean)
        y_low  = vbgm_age(ages, K_low, t0_low) # Low K -> Slower growth
        y_upp  = vbgm_age(ages, K_upp, t0_upp) # High K -> Faster growth
        anchor_val = t0_mean
        anchor_name = "t0"
    elif name == 'Gompertz':
        # --- GOMPERTZ ---
        # Solving for b: b = ln( Linf / L0 )
        # Note: In Gompertz, 'b' is independent of K.
        b_val = np.log(Linf / L0)
        y_mean = gompertz_age(ages, K_mean, b_val)
        y_low  = gompertz_age(ages, K_low, b_val)
        y_upp  = gompertz_age(ages, K_upp, b_val)
        anchor_val = b_val
        anchor_name = "b"
    elif name == 'Logistic':
        # --- LOGISTIC ---
        # Solving for c: c = (Linf - L0) / L0
        # Note: In Logistic, 'c' is independent of K.
        c_val = (Linf - L0) / L0
        y_mean = logistic_age(ages, K_mean, c_val)
        y_low  = logistic_age(ages, K_low, c_val)
        y_upp  = logistic_age(ages, K_upp, c_val)
        anchor_val = c_val
        anchor_name = "c"
        
    # Store processed data in auxiliary dict
    curves_data[name] = {
        'y_mean': y_mean,
        'y_low': y_low,
        'y_upp': y_upp,
        'anchor_str': f"{anchor_name}={anchor_val:.3f}"
    }

# %% Plot GROWTH CURVES - Figure 3
palette = sns.color_palette("colorblind")
# --- 1. Global Plotting Configuration ---
sns.set_style("ticks")
plt.rcParams.update({
    "text.usetex": False,            
    "font.family": "serif", 
    "mathtext.fontset": "stix",      
    "font.serif": ["STIXGeneral"], 
    'axes.labelweight': 'bold',
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'axes.edgecolor': 'black',
    'figure.dpi': 650                
})

# --- 2. Figure Initialization ---
fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
# Define styles
styles = {
    # 'Fabens':        {'color': 'black', 'ls': '-',  'label': 'Fabens (VBGF)'},
    'Fabens':  {'color': (sns.color_palette("colorblind")[0]),
                      'ls': ':',  'label': 'Fabens (VBGF)'},
    'Gompertz':      {'color': (sns.color_palette("colorblind")[1]),
                      'ls': '--', 'label': 'Gompertz'},
    'Logistic':      {'color': (sns.color_palette("colorblind")[2]),
                      'ls': '-.', 'label': 'Robertson'}
}

# --- 3. Plotting Curves and Bands ---
for name, st in styles.items():
    if name not in curves_data: continue
    
    data = curves_data[name]
    
    # Confidence Interval Band
    ax.fill_between(ages, data['y_low'], data['y_upp'], 
                    color=st['color'], alpha=0.13, edgecolor='none') 
    
    # Mean Curve - We format the label here for the legend
    k_val = results[name]['K']
    ax.plot(ages, data['y_mean'], 
            color=st['color'], linestyle=st['ls'], linewidth=2, 
            label=f"{st['label']} ($K$ = {k_val:.4f})")

# --- 4. Reference Elements ---
# Asymptote Linf
ax.axhline(Linf, color='gray', linestyle='--', linewidth=1.4, 
           label=fr'$L_\infty$ = {Linf} cm')

# Size at birth L0
ax.scatter([0], [L0], color='0.23', s=80, marker='o', zorder=10, 
           label=fr'$L_0$ = {L0} cm', clip_on=False)

# --- 5. Legend and Aesthetics (THE FIX) ---
# Create the legend directly with ax.legend to avoid the ValueError
leg = ax.legend(
    # loc='lower right', 
    bbox_to_anchor=(1, 0.35), # Positioned above the plot
    ncol=1,                   # 3 columns to be horizontal
    frameon=False,            # No box
    fontsize=16
)

# Bottom annotation
ax.text(
    0.02, 0.03,
    r'$t=0$ represents size at birth ($L_0$), not chronological age',
    transform=ax.transAxes,
    fontsize=14,
    style='italic'
)

# --- 6. Axes Final Touches ---
ax.set_xlabel(r'$\mathbf{Time\ since\ birth\ (years)}$')
ax.set_ylabel(r'$\mathbf{Total\ length\ (cm)}$')
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.set_xlim(0, 50)
ax.set_ylim(0, Linf + 20)

# --- Observed TL range ---
tl_min = min(df['TL (cm)'])
tl_max = max(df['TL (cm)'])
# ax.axhline(tl_min, color='black', linestyle=':', linewidth=1)
# ax.axhline(tl_max, color='black', linestyle=':', linewidth=1)
# ax.plot([0.23,0.23], [tl_min, tl_max], color='black', linewidth=4)
ax.hlines(tl_min, xmin=0, xmax=1.5, color='black', linewidth=1)
ax.hlines(tl_max, xmin=0, xmax=1.5, color='black', linewidth=1)

# Grid - Optional, but keeping it subtle if you want it
ax.grid(False)
plt.minorticks_on()
# ax.grid(True, which='major', linestyle='-', alpha=0.15)
sns.despine(top=True, right=True)
# --- 4. Export ---
plt.savefig("Fig3_Ncepedianus_female_growth_moodels.tiff",            
            dpi=650, bbox_inches='tight',
            pil_kwargs={"compression": "tiff_lzw"}
            )
# plt.savefig("Fig3_Ncepedianus_female_growth_moodels.png",
#            dpi=300, bbox_inches='tight')
plt.show()
