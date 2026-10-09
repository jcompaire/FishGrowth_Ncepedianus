#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  10 11:07:44 2026
@author: Jesus C. Compaire
@position: Assistant Researcher Scientist
@institution: Center for the Study of Marine Systems (CESIMAR)
@position: Assistant Professor of Climate System
@institution: National University of the Patagonia San Juan Bosco (UNPSJB)
@e-mail:jesus.canocompaire@uca.es
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from KDEpy import FFTKDE
from scipy import stats
from scipy.optimize import minimize # curve_fit,
from scipy.stats import shapiro, levene, kruskal
import scikit_posthocs as sp
# from statsmodels.sandbox.stats.runs import runstest_1samp
import warnings
# from docx import Document
warnings.filterwarnings("ignore", message="set_ticklabels")

# %% --- A little bit of common statistic
# =============================================================================
def summary_stats(df, group_col, value_col='TL'):
    """
    Calculates summary descriptive statistics for a specified numeric variable grouped by category.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing the grouping variable and target values.
    group_col : str
        Column name used for grouping data (e.g., 'sex', 'region').
    value_col : str, optional
        Target numeric column name to calculate statistics for (default is 'TL').

    Returns:
    --------
    summary_df : pandas.DataFrame
        DataFrame containing sample size (n), min, max, mean, standard deviation (sd),
        median, 25th percentile (q25), and 75th percentile (q75) for each group.
    """
    return (
        df.groupby(group_col)[value_col]
        .agg(
            n='count',
            min='min',
            max='max',
            mean='mean',
            sd='std',
            median='median',
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75)
        )
        .reset_index()
    )
# 
def nonparam_group_analysis(df, group_col, value_col='TL (cm)', p_adjust='bonferroni'):
    """
    Executes a complete non-parametric workflow including assumption testing (normality, 
    homoscedasticity), Kruskal–Wallis test, and Dunn's post-hoc pairwise comparisons.

    Prints intermediate test metrics (Shapiro–Wilk per group, Levene's test, Kruskal–Wallis H)
    and outputs lower-diagonal matrices for Dunn's Z-statistics and adjusted p-values.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing grouping categories and numeric values.
    group_col : str
        Column name used to categorize groups (e.g., 'sex', 'region').
    value_col : str, optional
        Target numeric column to analyze (default is 'TL (cm)').
    p_adjust : str, optional
        Method for multi-comparison p-value adjustment accepted by scikit-posthocs 
        (e.g., 'bonferroni', 'holm', 'fdr_bh') (default is 'bonferroni').

    Returns:
    --------
    p_values_filtered : pandas.DataFrame
        Lower-triangular matrix of pairwise adjusted p-values between groups.
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
    print(f"\n Levene test: W={stat:.3f}, p={p:.4f}")

    # Kruskal–Wallis
    stat, p = kruskal(*groups)
    df_kw = len(groups) - 1
    print(f"\n Kruskal–Wallis: H({df_kw}) = {stat:.3f}, p={p:.4f}")

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
    
    p_values = sp.posthoc_dunn(
        data,
        val_col=value_col,
        group_col=group_col,
        p_adjust=p_adjust
        )
    p_values_filtered = p_values.where(np.tril(np.ones(p_values.shape), k=-1).astype(bool))
    
    print("\nAdjusted p-values Matrix:")
    print(p_values_filtered.round(4))
    
    # Calculate Z-statistics from the unadjusted p-values
    p_unadj = sp.posthoc_dunn(data, val_col=value_col, group_col=group_col, p_adjust=None)
    z_matrix = np.abs(stats.norm.ppf(p_unadj / 2.0))
    z_matrix_filtered = pd.DataFrame(z_matrix, index=p_unadj.index, columns=p_unadj.columns).where(np.tril(np.ones(p_unadj.shape), k=-1).astype(bool))
    
    print("\nDunn's Z-statistics Matrix:")
    print(z_matrix_filtered.round(4))

    # return posthoc
    return p_values_filtered
#
# %% GROWTH MODELS functions
# =============================================================================
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
# 
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
# 
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
# 
# %% MODEL FITTING (MLE)
# =============================================================================
def neg_loglik_normal(params, model_func, L2_obs, dt, L1, Linf):
    """"
    Calculates negative log-likelihood assuming normally distributed errors.

    Parameters:
    -----------
    params : tuple or list
        params[0] (K) : float
            Growth coefficient to be estimated.
        params[1] (log_sigma) : float
            Natural log of residual standard deviation (log-scale forces sigma > 0).
    model_func : callable
        Growth model function in interval form (e.g., model_fabens, model_gompertz).
    L2_obs : array-like
        Observed total lengths at recapture (cm).
    dt : array-like
        Time at liberty (years).
    L1 : array-like
        Total lengths at release (cm).
    Linf : float
        Fixed asymptotic total length (cm).

    Returns:
    --------
    nll : float
        Negative log-likelihood value to be minimized.
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
#
# %% BOOTSTRAP Analysis cconfidence intervals
# =============================================================================
# Estimation of uncertainty using resampling, recommended for 
# non-normal error distributions common in tag-recapture.
def bootstrap_CI(model_name, nonlinear_models, L1, L2, Linf, dt,  nboot=999, seed=23):
    """
    Performs bootstrap resampling to calculate 95% Confidence Intervals for K.
    
    Parameters:
    -----------
    model_name : str
        'Fabens', 'Gompertz', 'Logistic'
    L1, L2 : array-like
        Release and recapture lengths.
    dt : array-like
        Time at liberty in years.
    nonlinear_models : dict
        Dictionary containing the growth model functions.
    Linf : float
        Fixed asymptotic length.
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
                    bounds=[
                        (0, 2.0),
                        (None, None)
                        ]
                )
                boots.append(opt_b.x[0])
                
        except RuntimeError:
            # Skip iterations where curve_fit fails to converge
            continue            
    boots = np.array(boots)
    # Calculate Percentiles (95% CI)
    return np.nanpercentile(boots, [2.5, 97.5])
# 
# %% MODEL COMPARISON (AICc - MLE BASED)
# =============================================================================
def get_aicc_mle(loglik, n_params, n):
    """
    Calculates small-sample corrected Akaike Information Criterion (AICc) 
    from maximum log-likelihood.

    Parameters:
    -----------
    loglik : float
        Maximum log-likelihood value obtained from optimization.
    n_params : int
        Number of estimated parameters in the model (e.g., K + sigma = 2).
    n : int
        Sample size (number of observations).

    Returns:
    --------
    aicc : float
        Small-sample corrected Akaike Information Criterion value.
    """
    aic = -2.0 * loglik + 2.0 * n_params
    return aic + (2 * n_params * (n_params + 1)) / (n - n_params - 1)
#
# %% Plotting MODEL GROTH CURVES
# =============================================================================
# AGE-BASED MODEL DEFINITIONS
def vbgm_age(t, K, t0):
    """
    Calculates total length at age using the classical age-based Von Bertalanffy 
    Growth Model (Beverton & Holt, 1957).

    Equation:
        L(t) = Linf * (1 - exp(-K * (t - t0)))

    Parameters:
    -----------
    t : array-like or float
        Relative age or time since birth (years).
    K : float
        Growth coefficient (year^-1).
    t0 : float
        Theoretical age at length zero (years).

    Returns:
    --------
    Lt : ndarray or float
        Predicted total length at age t (cm).

    Notes:
    ------
    Requires `Linf` (asymptotic total length) to be defined in the workspace.
    """
    return Linf * (1 - np.exp(-K * (t - t0)))
# 
def gompertz_age(t, K, b_param):
    """
    Calculates total length at age using the age-based Gompertz Growth Model (Gompertz, 1825).
    Appropriate for species exhibiting sigmoidal growth with early inflection.

    Equation:
        L(t) = Linf * exp(-b * exp(-K * t))

    Parameters:
    -----------
    t : array-like or float
        Relative age or time since birth (years).
    K : float
        Growth rate coefficient (year^-1).
    b_param : float
        Dimensionless integration constant related to size at birth: b = ln(Linf / L0).

    Returns:
    --------
    Lt : ndarray or float
        Predicted total length at age t (cm).

    Notes:
    ------
    Requires `Linf` (asymptotic total length) to be defined in the workspace.
    """
    return Linf * np.exp(-b_param * np.exp(-K * t))
# 
def logistic_age(t, K, c_param):
    """
    Calculates total length at age using the age-based Logistic Growth Model (Robertson, 1923).
    Sigmoidal curve symmetric around its inflection point.

    Equation:
        L(t) = Linf / (1 + c * exp(-K * t))

    Parameters:
    -----------
    t : array-like or float
        Relative age or time since birth (years).
    K : float
        Growth rate coefficient (year^-1).
    c_param : float
        Dimensionless integration constant related to size at birth: c = (Linf - L0) / L0.

    Returns:
    --------
    Lt : ndarray or float
        Predicted total length at age t (cm).

    Notes:
    ------
    Requires `Linf` (asymptotic total length) to be defined in the workspace.
    """
    return Linf / (1 + c_param * np.exp(-K * t))
#
# %% Get the CURVES values to plot
# =============================================================================
def generate_growth_curves(results, Linf, L0, max_age=50, n_points=100):
    """
    Calculates age-based growth trajectories (mean curve and 95% confidence intervals)
    anchored at size at birth (L0) at t=0.

    Parameters:
    -----------
    results : dict
        Dictionary containing estimated parameters ('K', 'CI_2.5', 'CI_97.5') per model.
    Linf : float
        Fixed asymptotic length (cm).
    L0 : float
        Size at birth (cm).
    max_age : float, optional
        Maximum projection age in years (default is 50).
    n_points : int, optional
        Number of points along the age sequence (default is 100).

    Returns:
    --------
    ages : ndarray
        Sequence of age values from 0 to max_age.
    curves_data : dict
        Processed Y-values (y_mean, y_low, y_upp) and anchor parameter strings per model.
    """
    # Ensure Linf is available in module scope for age functions
    globals()['Linf'] = Linf
    
    # Projection range (0 to 50 years - FishBase)
    ages = np.linspace(0, max_age, n_points)
    curves_data = {}

    # Map model names to their respective age-based growth functions
    age_functions = {
        'Fabens': vbgm_age,
        'Gompertz': gompertz_age,
        'Logistic': logistic_age
    }

    for name, metrics in results.items():
        
        K_mean = metrics['K']
        # 1. Define K extremes (95% Confidence Interval)
        K_low = metrics['CI_2.5']
        K_upp = metrics['CI_97.5']

        # Calculate the Anchor Parameter based on the Model Type
        #        CRITICAL: The anchor parameter (t0, b, c) must be recalculated for 
        #        each K (mean, low, upp) to ensure all curves pass through L0 at t=0.
        if name in ['Fabens']:
            # --- VON BERTALANFFY ---
            # Solving for t0: t0 = (1/K) * ln( (Linf - L0) / Linf )
            p_mean = (1 / K_mean) * np.log((Linf - L0) / Linf)
            p_low  = (1 / K_low)  * np.log((Linf - L0) / Linf)
            p_upp  = (1 / K_upp)  * np.log((Linf - L0) / Linf)
            anchor_name = "t0"

        elif name == 'Gompertz':
            # --- GOMPERTZ ---
            # Solving for b: b = ln( Linf / L0 )
            # Note: In Gompertz, 'b' is independent of K.
            p_mean = p_low = p_upp = np.log(Linf / L0)
            anchor_name = "b"

        elif name == 'Logistic':
            # --- LOGISTIC ---
            # Solving for c: c = (Linf - L0) / L0
            # Note: In Logistic, 'c' is independent of K.
            p_mean = p_low = p_upp = (Linf - L0) / L0
            anchor_name = "c"

        # Generate Y data using defined age-based functions
        func = age_functions[name]
        y_mean = func(ages, K_mean, p_mean)
        y_low  = func(ages, K_low,  p_low)   # Low K -> Slower growth
        y_upp  = func(ages, K_upp,  p_upp)   # High K -> Faster growth

        # Store processed data in auxiliary dict
        curves_data[name] = {
            'y_mean': y_mean,
            'y_low': y_low,
            'y_upp': y_upp,
            'anchor_str': f"{anchor_name}={p_mean:.3f}"
        }

    return ages, curves_data

# %% Customize FIGURES
# Fig 2.- =====================================================================
def plot_regional_size_distributions(df, value_col='TL (cm)',
                                     save_path=None, dpi=650):
    """
    Plots regional total length distributions (histograms and ISJ KDE curves) grouped by sex.
    A common ISJ bandwidth (pooled across regions) is applied per sex.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing 'sex', 'region', and total length values.
    value_col : str, optional
        Target numeric column name for size values (default is 'TL (cm)').
    save_path : str, optional
        File path to save the generated figure (e.g., 'Fig_2.tiff'). If None, figure is not saved.
    dpi : int, optional
        Resolution for saved output image (default is 650 DPI).

    Returns:
    --------
    fig, axes : matplotlib Figure and Axes objects.
    """
    # --- 1. Global Plotting Configuration ---
    sns.set_style("ticks")
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.serif": ["STIXGeneral"],
        "axes.labelweight": "bold",
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.edgecolor": "black",
        "figure.dpi": 300
    })

    # --- 2. Data filtering and plotting configuration ---
    df_clean = df[df['sex'].isin(['F', 'M'])].copy()

    region_order = ['north', 'central', 'south']
    titles = ['North', 'Central', 'South']
    labels = ['A', 'B', 'C']
    sex_order = ['F', 'M']

    palette_custom = ['#D55E00', '#737373']
    binwidth = 10

    # --- 3. Calculate a common ISJ bandwidth for each sex ---
    bandwidths = {}
    for sex in sex_order:
        data_sex_all = df_clean.loc[
            df_clean['sex'] == sex,
            value_col
        ].dropna().to_numpy()

        if len(data_sex_all) < 2:
            raise ValueError(
                f"Not enough data to estimate KDE bandwidth for sex: {sex}"
            )

        # Fit ISJ KDE to all observations of that sex
        kde = FFTKDE(
            kernel='gaussian',
            bw='ISJ'
        )
        kde.fit(data_sex_all)

        # Store the bandwidth selected by ISJ
        bandwidths[sex] = kde.bw

    print("Common bandwidths used across regions:")
    for sex in sex_order:
        print(f"{sex}: {bandwidths[sex]:.4f}")

    # --- 4. Figure Initialization ---
    fig, axes = plt.subplots(
        3, 1,
        figsize=(12, 16),
        constrained_layout=True
    )

    # --- 5. Loop over regions ---
    for i, region_name in enumerate(region_order):
        ax = axes[i]

        # Filter data for the current region
        df_region = df_clean[df_clean['region'] == region_name]

        # --- Histogram ---
        sns.histplot(
            data=df_region,
            x=value_col,
            hue='sex',
            hue_order=sex_order,
            fill=True,
            binwidth=binwidth,
            element="step",
            palette=palette_custom,
            ax=ax,
            legend=(i == 0)
        )

        # --- KDE using the common ISJ bandwidth for each sex ---
        for sex, color in zip(sex_order, palette_custom):
            data_sex = df_region.loc[
                df_region['sex'] == sex,
                value_col
            ].dropna().to_numpy()

            if len(data_sex) < 2:
                continue

            # Use the bandwidth previously calculated from pooled observations
            kde = FFTKDE(
                kernel='gaussian',
                bw=bandwidths[sex]
            )

            # Evaluation grid (extrapolada a los límites del gráfico)
            x = np.linspace(40, 270, 1000)
            # x = np.linspace(
            #     data_sex.min() - 1,
            #     data_sex.max() + 1,
            #     1000
            # )

            # Evaluate KDE
            y = kde.fit(data_sex).evaluate(x)

            # Convert density to histogram frequency scale
            y_scaled = y * len(data_sex) * binwidth

            # Plot KDE curve
            ax.plot(
                x,
                y_scaled,
                color=color,
                linewidth=2.5
            )

        # --- Axes limits and ticks ---
        ax.set_xlim(40, 270)
        ax.set_xticks(np.arange(40, 271, 35))
        ax.set_ylim(0, 70)
        ax.set_ylabel(
            r'$\mathbf{Frequency}$',
            fontsize=23,
            labelpad=8
        )

        # --- X-axis label ---
        if i == 2:
            ax.set_xlabel(
                r'$\mathbf{Total\ length\ (cm)}$',
                fontsize=23,
                labelpad=8
            )
        else:
            ax.set_xlabel("")

        # --- Tick labels ---
        ax.tick_params(
            axis='both',
            which='major',
            labelsize=18
        )

        # --- Legend ---
        if i == 0:
            sns.move_legend(
                ax,
                loc='upper left',
                bbox_to_anchor=(0.08, 0.9),
                frameon=False,
                title=None,
                prop={'size': 23},
                markerscale=1.5,
                labels=['Females', 'Males']
            )

        # --- Panel labels ---
        ax.text(
            0.05, 0.95,
            labels[i],
            transform=ax.transAxes,
            fontsize=32,
            fontweight='bold',
            va='top',
            ha='right'
        )

        # --- Region titles ---
        ax.text(
            0.45, 0.80,
            titles[i],
            transform=ax.transAxes,
            fontsize=28,
            fontweight='bold'
        )

        sns.despine(ax=ax)

    # --- 6. Save Figure Option ---
    if save_path:
        save_kwargs = {"dpi": dpi, "bbox_inches": "tight"}
        if save_path.lower().endswith(('.tiff', '.tif')):
            save_kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        plt.savefig(save_path, **save_kwargs)

    plt.show()
    return fig, axes
# 
# Fig 3.- =====================================================================
def plot_growth_curves(ages, curves_data, results, Linf, L0, tl_range,
                       save_path=None, dpi=300):
    """
    Plots customized publication-ready growth model trajectories with confidence bands.

    Parameters:
    -----------
    ages : ndarray
        Sequence of age values (0 to max_age).
    curves_data : dict
        Dictionary containing calculated y_mean, y_low, and y_upp trajectories.
    results : dict
        Dictionary containing estimated model parameters ('K', etc.).
    Linf : float
        Fixed asymptotic length (cm).
    L0 : float
        Size at birth (cm).
    tl_range : tuple or list (tl_min, tl_max)
        Minimum and maximum observed total length (cm) for reference lines.
    save_path : str, optional
        Full file path or filename to save the figure (e.g., 'growth_models_female.tiff').
        If None, the figure is displayed without saving.
    dpi : int, optional
        Resolution for output image (default is 650 DPI for JCR journal specs).

    Returns:
    --------
    fig, ax : matplotlib Figure and Axes objects.
    """
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
        'figure.dpi': dpi            
    })

    # --- 2. Figure Initialization ---
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)

    # Define styles
    styles = {
        'Fabens':   {'color': palette[0], 'ls': ':',  'label': 'Fabens (VBGF)'},
        'Gompertz':  {'color': palette[1], 'ls': '--', 'label': 'Gompertz'},
        'Logistic':  {'color': palette[2], 'ls': '-.', 'label': 'Logistic'}
    }

    # --- 3. Plotting Curves and Bands ---
    for name, st in styles.items():
        if name not in curves_data: 
            continue
        
        data = curves_data[name]
        
        # Confidence Interval Band
        ax.fill_between(
            ages, data['y_low'], data['y_upp'], 
            color=st['color'], alpha=0.13, edgecolor='none'
        ) 
        
        # Mean Curve - We format the label here for the legend
        k_val = results[name]['K']
        ax.plot(
            ages, data['y_mean'], 
            color=st['color'], linestyle=st['ls'], linewidth=2, 
            label=f"{st['label']} ($K$ = {k_val:.4f})"
        )

    # --- 4. Reference Elements ---
    # Asymptote Linf
    ax.axhline(
        Linf, color='gray', linestyle='--', linewidth=1.4, 
        label=fr'$L_\infty$ = {Linf} cm'
    )

    # Size at birth L0
    ax.scatter(
        [0], [L0], color='0.23', s=80, marker='o', zorder=10, 
        label=fr'$L_0$ = {L0} cm', clip_on=False
    )

    # --- 5. Legend and Aesthetics ---
    # Create the legend directly with ax.legend to avoid the ValueError
    ax.legend(
        bbox_to_anchor=(1, 0.35), # Positioned above/beside plot
        ncol=1,                   # 1 column vertical
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
    tl_min, tl_max = tl_range
    ax.hlines(tl_min, xmin=0, xmax=1.5, color='black', linewidth=1)
    ax.hlines(tl_max, xmin=0, xmax=1.5, color='black', linewidth=1)

    # Grid - Optional, keeping it subtle
    ax.grid(False)
    plt.minorticks_on()
    sns.despine(top=True, right=True)

    # --- 7. Optional Export ---
    if save_path:
        # Determine compression format based on file extension
        save_kwargs = {"dpi": dpi, "bbox_inches": "tight"}
        if save_path.lower().endswith(('.tiff', '.tif')):
            save_kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
            
        plt.savefig(save_path, **save_kwargs)

    plt.show()
    return fig, ax
# 
