import random
import torch
import os
import ast
import gzip
import pytz
import json
import pickle
import random
import inspect
import warnings
import numpy as np
from datetime import datetime
from collections import OrderedDict
from typing import List, Tuple
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

__all__ = [
    'datetime_now',
    'create_folder',
    'get_func_input_names',
    'auto_repr',
    'collect_class_definitions',
    'subset_loader',
    'fix_seed',
    'load_from_pickle',
    'load_json',
    'percentage_within_fold_change',
    'geometric_mean_fold_error',
    'min_max',
    'gmfe',
    'regression_test_metrics',
    'bland_altman_plot',
    'parity_plot_with_folds',
    'parity_plot_with_folds_and_percent',
]


def datetime_now(path=True):
    
    ist = pytz.timezone('Asia/Kolkata')
    dt = datetime.now(ist).strftime('%Y%m%d_%H_%M_%S' if path else '%Y/%m/%d [%H:%M:%S]')
    date, time = datetime.now(ist).strftime('Date_%Y_%m_%d'), datetime.now(ist).strftime('Time_%H_%M_%S')
    
    return dt, date, time

def create_folder(folder_name: str) -> str:
    
    if len(folder_name):
        if not os.path.isdir(folder_name):
            print(f'Creating Folder:{folder_name}')
            os.makedirs(folder_name)
        
    return folder_name

def get_func_input_names(func):
    
    sig = inspect.signature(func)
    param_list =  [param.name for param in sig.parameters.values()
            if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)]
    
    return param_list

def auto_repr(self):
    """
    Generalized __repr__ for classes.
    Constructs a multi-line string showing the class name and __init__ arguments with their current values.
    """
    cls_name = self.__class__.__name__
    try:
        sig = inspect.signature(self.__init__)
        args = [p for p in sig.parameters if p != "self"]
    except Exception:
        args = list(vars(self).keys())

    body = ",\n  ".join(f"{k}={repr(getattr(self, k, None))}" for k in args)
    
    return f"{cls_name}(\n  {body}\n)"

def collect_class_definitions(self):
    """
    Collect source code for all classes in the inheritance chain.
    """
    definitions = []
    for cls in self.__class__.__mro__[::-1]:  # base to top
        if cls in (object, nn.Module):
            continue
        definitions.append(inspect.getsource(cls))
        
    return "\n".join(definitions)


def subset_loader(dataloader: DataLoader,
                  batch_size: int,
                  subset_ratio: float = 0.2,
                  collate_fn=None
                  ) -> Tuple[DataLoader, DataLoader]:
    
    '''
    Input: 
        - A dataloader
        - Batch Size of the generated dataloader
        - Subset Ratio
        
    Output: A random subset of the dataloader
    '''
    
    complement_loader = None    
    original_dataset = dataloader.dataset
    subset_size = int(subset_ratio*len(original_dataset))
    perm = np.random.permutation(np.arange(len(original_dataset)))
    subset_indices = list(perm[:subset_size])
    subset_dataset = Subset(original_dataset, subset_indices)
    subset_loader = torch.utils.data.DataLoader(subset_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    if float(subset_ratio) != 1.0:
        complement_indices = list(perm[subset_size:])
        complement_subset_dataset = Subset(original_dataset, complement_indices)
        complement_loader = torch.utils.data.DataLoader(complement_subset_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    return subset_loader, complement_loader

def fix_seed(np_seed=30, torch_seed=30, random_seed=30):
    
    random.seed(random_seed)
    np.random.seed(np_seed)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        

def load_from_pickle(filepath: str):
    """
    Load Python object from a pickle file.
    Automatically detects gzip compression (even without .gz extension).
    """
    with open(filepath, "rb") as f:
        magic = f.read(2)
    
    if magic == b"\x1f\x8b":  # gzip magic number
        with gzip.open(filepath, "rb") as fp:
            file_ = pickle.load(fp)
    else:
        with open(filepath, "rb") as fp:
            file_ = pickle.load(fp)
    return file_

def load_json(path):

    with open(path, 'r') as fp:
        data = json.loads(fp.read())

    return data

def percentage_within_fold_change(y_true, y_pred, fold=2, transform_func=lambda x: x):
    """
    Calculate the percentage of repeated measurements within a given fold range.

    Parameters:
    - y_true (list or numpy array): The true experimental values.
    - y_pred (list or numpy array): The predicted experimental values.
    - fold (float): The fold change range (default is 2).

    Returns:
    - float: Percentage of values within the specified fold range.
    """
    y_true = transform_func(np.array(y_true, dtype=float))
    y_pred = transform_func(np.array(y_pred, dtype=float))
    
    y_true[y_true == 0] = 1e-3
    assert np.all(y_true != 0), "True values must not contain zero."
    
    fold_change = y_pred / y_true 
    max_, min_ = np.round(max(fold_change), 2), np.round(min(fold_change), 2)
    within_fold = (1/fold <= fold_change) & (fold_change <= fold)
    percentage = round(np.sum(within_fold) / len(y_true)*100, 2)
    
    return percentage, max_, min_

def geometric_mean_fold_error(y_true, y_pred, transform_func=lambda x: x):
    
    """
    Calculate Geometric Mean Fold Error (GMFE).

    Parameters:
    - y_true: array-like of true values
    - y_pred: array-like of predicted values

    Returns:
    - GMFE: float
    """
    y_true = transform_func(np.array(y_true, dtype=float))
    y_pred = transform_func(np.array(y_pred, dtype=float))
    
    # Avoid division by zero
    if np.any(y_true <= 0):
        idx = np.where(y_true <= 0)[0]
        print(f"Total: {len(idx)} with Index: {idx} contains non-positive numbers. y_true must not contain non-positive numbers.")
    if np.any(y_pred <= 0):
        idx = np.where(y_pred <= 0)[0]
        print(f"Total: {len(idx)} with Index: {idx} contains non-positive numbers. y_pred must not contain non-positive numbers.")

    valid = ((y_true > 0) & (y_pred > 0)) or ((y_true < 0) & (y_pred < 0))
    invalid = len(y_true) - len(y_pred[valid])
    
    gmfe = 10**mean_absolute_error(y_true=y_true, y_pred=y_pred)
    if invalid > 0:
        warnings.warn(f"Warning: {invalid} values were ignored due to zero or negative values.")
        print(f"{invalid} invalid (non-positive) entries found and excluded from log10 MAE calculation.")
    
    return gmfe, invalid



def min_max(y_true, y_pred, range_list, eps = 0.5, transform_func=lambda x: x):
    
    range_list = range_list.apply(lambda x: ast.literal_eval(x)).tolist()
    
    y_true = transform_func(np.array(y_true, dtype=float))
    y_pred = transform_func(np.array(y_pred, dtype=float))
    
    num = 0
    for i in range(len(y_true)):
        x = y_pred[i]
        z = range_list[i]
        if len(z) == 1:
            min_, max_ = z[0]-eps, z[0]+eps
        else:
            min_, max_ = min(z), max(z)
        if x <= max_ and x >= min_:
            num += 1
    percent = num/len(y_pred)#*100
    
    return percent

def gmfe(y_true, y_pred, eps=1e-8, base=10, transform_func=lambda x: x):
    """
    Generalized Geometric Mean Fold Error (GMFE).
    
    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.
    eps : float, optional
        Small constant added to avoid division by zero (default: 1e-8).
    base : float, optional
        Logarithm base (default: 10, use np.e for natural log).
    
    Returns
    -------
    float
        Geometric Mean Fold Error.
    """
    y_true = transform_func(np.array(y_true, dtype=float))
    y_pred = transform_func(np.array(y_pred, dtype=float))

    ratio = (np.abs(y_pred) + eps) / (np.abs(y_true) + eps)
    logs = np.log(ratio) / np.log(base)
    gmfe_value = base ** (np.mean(np.abs(logs)))

    return gmfe_value

def regression_test_metrics(y_true: np.ndarray[float] | List[float],
                 y_pred: np.ndarray[float] | List[float],
                 transform_func=lambda x: x,
                 fold_transform_fn=lambda x:x
            ) -> OrderedDict[str, float]:
    
    '''
    Input:
        - model: Random Forest Model
        - y_true: Array/List Object
        - y_pred: Array/List Object
    
    Output: A dictionary containing MSE, RMSE, MAE and r2 Values.
    '''
    if not isinstance(y_pred, np.ndarray):
        y_pred = transform_func(np.array(y_pred))
    if not isinstance(y_true, np.ndarray):
        y_true = transform_func(np.array(y_true))

    assert y_pred.shape == y_true.shape, f"Shape mismatch: y_pred -> {y_pred.shape}, y_true -> {y_true.shape}"
    
    # Calculate metrics on Test Set
    rmse_test = root_mean_squared_error(y_true, y_pred)
    mse_test = rmse_test**2
    mae_test = mean_absolute_error(y_true, y_pred)
    r2_val_test = r2_score(y_true, y_pred)
    pcc = np.corrcoef(y_true, y_pred,)[0, 1]
    fold2, _, _ = percentage_within_fold_change(y_true=y_true, y_pred=y_pred, fold=2, transform_func=fold_transform_fn)
    fold3, _, _ = percentage_within_fold_change(y_true=y_true, y_pred=y_pred, fold=3, transform_func=fold_transform_fn)
    fold5, _, _ = percentage_within_fold_change(y_true=y_true, y_pred=y_pred, fold=5, transform_func=fold_transform_fn)
    _gmfe = gmfe(y_true=y_true, y_pred=y_pred, transform_func=fold_transform_fn)
    
    
    results = OrderedDict({'mse':round(mse_test, 2),
                           'r2':round(r2_val_test, 2),
                           'rmse':round(rmse_test, 2),
                           'mae':round(mae_test, 2),
                           'PCC':round(pcc, 2),
                           'GMFE':round(_gmfe, 2),
                           'fold2':round(fold2, 2),
                           'fold3':round(fold3, 2),
                           'fold5':round(fold5, 2),
                           })
    
    return results

def bland_altman_plot(y_true, y_pred, title='', pad_frac=0.05, figsize = (10, 6),
                      x_lims=(-10, 10), y_lims=(-10, 10), return_stats=False, color='red'):
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mean_values = (y_true + y_pred) / 2
    diff_values = y_true - y_pred

    # Mean and standard deviation of differences
    mean_diff = np.mean(diff_values)
    std_diff = np.std(diff_values, ddof=1)

    # Limits of Agreement (LOA)
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    # Calculate percentage of points within LOA
    within_loa = np.sum((diff_values >= loa_lower) & (diff_values <= loa_upper))
    total_points = len(diff_values)
    percent_within_loa = (within_loa / total_points) * 100
    
    def padded_limits(values, frac):
        vmin, vmax = np.min(values), np.max(values)
        span = vmax - vmin
        if span == 0:
            span = abs(vmin) if vmin != 0 else 1.0
        pad = frac * span
        return vmin - pad, vmax + pad

    x_lims = padded_limits(mean_values, pad_frac)
    y_lims = padded_limits(np.r_[diff_values, loa_upper, loa_lower], pad_frac)

    # Plotting
    plt.figure(figsize=figsize)
    plt.scatter(mean_values, diff_values, color=color, alpha=0.6, label=title)
    plt.axhline(mean_diff, color='black', linestyle='-')
    plt.axhline(loa_upper, color='blue', linestyle='-')
    plt.axhline(loa_lower, color='blue', linestyle='-')
    plt.xlabel('Mean of True and Predicted Values', fontsize=20)
    plt.ylabel('Difference (True - Predicted)', fontsize=20)
    plt.xlim(x_lims)
    plt.ylim(y_lims)

    # Annotations for mean and LOA lines
    x_text_pos = x_lims[1]

    plt.text(x_text_pos, loa_upper + 0.1, f'+1.96 SD ({loa_upper:.2f})', fontsize=14, color='blue', ha='right', fontweight='bold')
    plt.text(x_text_pos, mean_diff + 0.1, f'Mean ({mean_diff:.2f})', fontsize=14, color='black', ha='right', fontweight='bold')
    plt.text(x_text_pos, loa_lower + 0.1, f'-1.96 SD ({loa_lower:.2f})', fontsize=14, color='blue', ha='right', fontweight='bold')

    # Vertical double-arrow between LOA lines
    x_arrow = x_lims[0] + 0.3 
    plt.annotate(
        '', 
        xy=(x_arrow, loa_upper), 
        xytext=(x_arrow, loa_lower), 
        arrowprops=dict(arrowstyle='<->', color='purple', lw=2)
    )

    # Add percentage text next to arrow
    plt.text(x_arrow - 0.1, (loa_upper + loa_lower) / 2+0.8, f'Within: {percent_within_loa:.1f}%', 
             va='top', ha='right', fontsize=13, color='purple', fontweight='bold', rotation=90)

    plt.grid(True, alpha=0.2)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.show()

    if return_stats:
        return mean_diff, std_diff, loa_upper, loa_lower, percent_within_loa
    

def parity_plot_with_folds(
    y_true,
    y_pred,
    title='',
    color='red',
    alpha=0.6,
    pad_frac=0.05,
    return_stats=False,
    figsize = (12, 6),
    title_fontsize=20,
):
    """
    Parity (x vs y) plot with y=x, 2-fold and 3-fold error bands.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted values.
    title : str
        Legend label.
    color : str
        Scatter color.
    alpha : float
        Scatter transparency.
    pad_frac : float
        Fractional padding for axis limits.
    return_stats : bool
        If True, returns fold-accuracy percentages.

    Returns
    -------
    dict (optional)
        Percent of points within 2-fold and 3-fold ranges.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    # ---------- axis limits ----------
    all_vals = np.concatenate([y_true, y_pred])
    vmin, vmax = np.min(all_vals), np.max(all_vals)
    span = vmax - vmin if vmax > vmin else max(abs(vmin), 1.0)
    pad = pad_frac * span

    xlims = (vmin - pad, vmax + pad)
    ylims = xlims

    # ---------- reference lines ----------
    x = np.linspace(*xlims, 500)

    # ---------- plot ----------
    plt.figure(figsize=figsize)

    plt.scatter(y_true, y_pred, color=color, alpha=alpha, label=title)

    # Perfect fit
    plt.plot(x, x, 'k-', lw=2, label='y = x')

    # 2-fold
    plt.plot(x, 2 * x, 'b--', lw=1.5, label='2-fold')
    plt.plot(x, 0.5 * x, 'b--', lw=1.5)

    # 3-fold
    plt.plot(x, 3 * x, 'g-.', lw=1.5, label='3-fold')
    plt.plot(x, x / 3, 'g-.', lw=1.5)

    plt.xlim(xlims)
    plt.ylim(ylims)

    plt.xlabel('True Values', fontsize=title_fontsize-3)
    plt.ylabel('Predicted Values', fontsize=title_fontsize-3)
    plt.xticks(fontsize = title_fontsize-4)
    plt.yticks(fontsize = title_fontsize-4)

    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=title_fontsize-8)
    plt.tight_layout()
    plt.show()

    # ---------- fold statistics ----------
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.maximum(y_pred / y_true, y_true / y_pred)

    within_2fold = np.mean(ratio <= 2) * 100
    within_3fold = np.mean(ratio <= 3) * 100

    if return_stats:
        return {
            "within_2fold_percent": within_2fold,
            "within_3fold_percent": within_3fold
        }
        
def parity_plot_with_folds_and_percent(
    y_true,
    y_pred,
    title='',
    color='red',
    alpha=0.6,
    pad_frac=0.05,
    percent_errors=(0.10, 0.20),
    return_stats=False,
    figsize=(7, 7)
):
    """
    Parity (x vs y) plot with:
      - y = x
      - 2-fold, 3-fold bands
      - ±percentage error bands

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted values.
    percent_errors : tuple
        Percentage errors as fractions (e.g., 0.10 = 10%).
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    # axis limits
    all_vals = np.concatenate([y_true, y_pred])
    vmin, vmax = np.min(all_vals), np.max(all_vals)
    span = vmax - vmin if vmax > vmin else max(abs(vmin), 1.0)
    pad = pad_frac * span

    xlims = (vmin - pad, vmax + pad)
    ylims = xlims
    x = np.linspace(*xlims, 600)
    plt.figure(figsize=figsize)

    plt.scatter(y_true, y_pred, color=color, alpha=alpha, label=title)
    plt.plot(x, x, 'k-', lw=2, label='y = x')
    plt.plot(x, 2 * x, 'b--', lw=1.4, label='2-fold')
    plt.plot(x, 0.5 * x, 'b--', lw=1.4)
    plt.plot(x, 3 * x, 'g--', lw=1.4, label='3-fold')
    plt.plot(x, x / 3, 'g--', lw=1.4)

    # Percentage-error bands
    for p in percent_errors:
        plt.plot(x, (1 + p)*x, color='orange', ls=':', lw=1.6,
                 label=f'±{int(p*100)}%' if p == percent_errors[0] else None)
        plt.plot(x, (1 - p)*x, color='orange', ls=':', lw=1.6)

    plt.xlim(xlims)
    plt.ylim(ylims)

    plt.xlabel('True Values', fontsize=16)
    plt.ylabel('Predicted Values', fontsize=16)

    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    # Statistics
    stats = {}

    # Fold accuracy
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.maximum(y_pred / y_true, y_true / y_pred)

    stats["within_2fold_percent"] = np.mean(ratio <= 2) * 100
    stats["within_3fold_percent"] = np.mean(ratio <= 3) * 100

    # Percentage accuracy
    for p in percent_errors:
        mask = np.abs(y_pred - y_true) <= p * np.abs(y_true)
        stats[f"within_{int(p*100)}pct_percent"] = np.mean(mask) * 100

    if return_stats:
        return stats

