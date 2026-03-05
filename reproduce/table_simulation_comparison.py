import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys

try:
    from simulation import beta as TRUE_BETA, gamma as TRUE_GAMMA, simulate_data
except ImportError:
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(root_dir)
    from simulation import beta as TRUE_BETA, gamma as TRUE_GAMMA, simulate_data

def true_g(x, g_fn):
    _, _, _, _, _, g_true = simulate_data(5, outcome="continuous", g_type=g_fn, seed=0)
    return np.vectorize(g_true)(x)

def load_results():
    data_dir = Path("output")
    files = sorted([f for f in data_dir.glob("simulation+*.json") if f.is_file()])
    
    LIST_KEYS = [
        "seed", "performance", "beta_estimate", "gamma_estimate",
        "g_pred", "time_fit", "time_hessian", "time_bootstrap",
        "hessian_summary", "bootstrap_summary", "inference_summary"
    ]

    records = []
    for f in files:
        try:
            with open(f, "r") as fh:
                raw = json.load(fh)
        except Exception: continue
        
        if not raw: continue
        
        if isinstance(raw, list):
            first = raw[0]
            rec = {
                "model": [first.get("model", "?")],
                "n": [first.get("n", "?")],
                "g_fn": [first.get("g_fn", "?")],
                "outcome": [first.get("outcome", "?")],
                "x_dist": [first.get("x_dist", "normal")]
            }
            for k in LIST_KEYS:
                rec[k] = [r.get(k) for r in raw]
        else:
            rec = raw
            
        rec["_file"] = str(f)
        rec["_model"] = rec.get("model", ["?"])[0]
        rec["_n"] = rec.get("n", ["?"])[0]
        rec["_g_fn"] = rec.get("g_fn", ["?"])[0]
        rec["_x_dist"] = rec.get("x_dist", ["normal"])[0]
        
        if "outcome" in rec and len(rec["outcome"]) > 0:
            rec["_outcome"] = rec["outcome"][0]
        else:
            rec["_outcome"] = Path(f).stem.split("+")[4] if len(Path(f).stem.split("+")) >= 5 else "?"
            
        records.append(rec)
    return records

def compute_coverage(estimates, lb_list, ub_list, true_val):
    coverage = []
    for lb, ub in zip(lb_list, ub_list):
        if lb is None or ub is None: continue
        coverage.append((np.array(lb) <= true_val) & (true_val <= np.array(ub)))
    if not coverage: return np.full_like(true_val, np.nan, dtype=float)
    return np.mean(coverage, axis=0)

def extract_inference(rec):
    inf = rec.get("hessian_summary")
    if not inf or all(not s for s in inf):
        inf = rec.get("inference_summary")
    return inf if inf else []

def main():
    records = load_results()
    print(f"Loaded {len(records)} records")
    
    x_eval = np.linspace(-3, 3, 1000)
    
    rows = []
    
    for rec in records:
        model = rec["_model"]
        outcome = rec["_outcome"]
        g_fn = rec["_g_fn"]
        x_dist = rec["_x_dist"]
        
        if x_dist != "normal": continue # Only process normal for simplicity unless requested
        
        beta_est = np.array([b for b in rec.get("beta_estimate", []) if b is not None])
        gamma_est = np.array([g for g in rec.get("gamma_estimate", []) if g is not None])
        
        if len(beta_est) == 0: continue
        
        inf_summaries = extract_inference(rec)
        beta_se_list, gamma_se_list, beta_lb_list, beta_ub_list, gamma_lb_list, gamma_ub_list = [], [], [], [], [], []
        if inf_summaries:
            for s in inf_summaries:
                if not s: continue
                beta_se_list.append(s.get("beta_se"))
                gamma_se_list.append(s.get("gamma_se"))
                beta_lb_list.append(s.get("beta_lb"))
                beta_ub_list.append(s.get("beta_ub"))
                gamma_lb_list.append(s.get("gamma_lb"))
                gamma_ub_list.append(s.get("gamma_ub"))
                
        beta_bias = beta_est.mean(axis=0) - TRUE_BETA
        gamma_bias = gamma_est.mean(axis=0) - TRUE_GAMMA
        beta_sd = beta_est.std(axis=0, ddof=1)
        gamma_sd = gamma_est.std(axis=0, ddof=1)
        
        beta_se = np.mean([s for s in beta_se_list if s is not None], axis=0) if beta_se_list else np.full(len(TRUE_BETA), np.nan)
        gamma_se = np.mean([s for s in gamma_se_list if s is not None], axis=0) if gamma_se_list else np.full(len(TRUE_GAMMA), np.nan)
        
        beta_cov = compute_coverage(beta_est, beta_lb_list, beta_ub_list, TRUE_BETA)
        gamma_cov = compute_coverage(gamma_est, gamma_lb_list, gamma_ub_list, TRUE_GAMMA)
        
        for i in range(len(TRUE_BETA)):
            rows.append({"model": model, "outcome": outcome, "g_fn": g_fn, "param": f"beta_{i}", 
                         "bias": beta_bias[i], "sd": beta_sd[i], "se": beta_se[i], "cov": beta_cov[i]})
        for i in range(len(TRUE_GAMMA)):
            rows.append({"model": model, "outcome": outcome, "g_fn": g_fn, "param": f"gamma_{i}", 
                         "bias": gamma_bias[i], "sd": gamma_sd[i], "se": gamma_se[i], "cov": gamma_cov[i]})
                         
        # g L2 error
        g_preds = [np.array(g) for g in rec.get("g_pred", []) if len(g) > 0]
        if g_preds:
            G = np.vstack(g_preds)
            if outcome == "cox":
                G = G - G[:, 500:501]
            g_true_vals = true_g(x_eval, g_fn)
            if outcome == "cox":
                g_true_vals = g_true_vals - g_true_vals[500]
                
            l2_errors = np.sqrt(np.mean((G - g_true_vals)**2, axis=1))
            rows.append({"model": model, "outcome": outcome, "g_fn": g_fn, "param": "g_L2",
                         "bias": np.nan, "sd": np.nan, "se": np.nan, "cov": np.nan,
                         "g_l2": np.mean(l2_errors)})

    df = pd.DataFrame(rows)
    # create table
    out_lines = []
    
    # pivot table
    outcomes = ["continuous", "binary", "cox"]
    g_fns = ["linear", "sfun", "sigmoid"]
    params = [f"beta_{i}" for i in range(2)] + [f"gamma_{i}" for i in range(3)] + ["g_L2"]
    
    for outcome in outcomes:
        out_lines.append(f"\\section*{{Outcome: {outcome}}}")
        
        # tabular initialization
        out_lines.append("\\begin{table}[h]")
        out_lines.append("\\centering")
        out_lines.append("\\begin{tabular}{ll|4c|4c}")
        out_lines.append("\\hline")
        out_lines.append(" & & \\multicolumn{4}{c|}{PLSI} & \\multicolumn{4}{c}{NeuralPLSI} \\\\")
        out_lines.append("$g$ fn & Param & Bias & SD & SE & Cov & Bias & SD & SE & Cov \\\\")
        out_lines.append("\\hline")
        
        for g_fn in g_fns:
            for pi, param in enumerate(params):
                row_str = f"{g_fn} & {param} & " if pi == 0 else f" & {param} & "
                
                # PLSI
                p_df = df[(df["model"]=="PLSI") & (df["outcome"]==outcome) & (df["g_fn"]==g_fn) & (df["param"]==param)]
                if len(p_df) > 0:
                    r = p_df.iloc[0]
                    if param == "g_L2":
                        plsi_str = f"{r['g_l2']:.3f} & - & - & - "
                    else:
                        plsi_str = f"{r['bias']:.3f} & {r['sd']:.3f} & {r['se']:.3f} & {r['cov']:.2f} "
                else:
                    plsi_str = "- & - & - & - "
                    
                # NeuralPLSI
                n_df = df[(df["model"]=="NeuralPLSI") & (df["outcome"]==outcome) & (df["g_fn"]==g_fn) & (df["param"]==param)]
                if len(n_df) > 0:
                    r = n_df.iloc[0]
                    if param == "g_L2":
                        npl_str = f"{r['g_l2']:.3f} & - & - & - "
                    else:
                        npl_str = f"{r['bias']:.3f} & {r['sd']:.3f} & {r['se']:.3f} & {r['cov']:.2f} "
                else:
                    npl_str = "- & - & - & - "
                    
                row_str += plsi_str + "& " + npl_str + "\\\\"
                out_lines.append(row_str)
            out_lines.append("\\hline")
        
        out_lines.append("\\end{tabular}")
        out_lines.append("\\end{table}")
        out_lines.append("\n")

    with open("logs/comparison_table.tex", "w") as f:
        f.write("\n".join(out_lines))
        
    print("Saved logs/comparison_table.tex")
    
    # Save a markdown version as well for easier viewing
    md_lines = []
    for outcome in outcomes:
        md_lines.append(f"### Outcome: {outcome}\n")
        md_lines.append("| g fn | Param | PLSI Bias | PLSI SD | PLSI SE | PLSI Cov | NeuralPLSI Bias | NeuralPLSI SD | NeuralPLSI SE | NeuralPLSI Cov |")
        md_lines.append("|---|---|---|---|---|---|---|---|---|---|")
        
        for g_fn in g_fns:
            for pi, param in enumerate(params):
                g_str = g_fn if pi == 0 else ""
                
                # PLSI
                p_df = df[(df["model"]=="PLSI") & (df["outcome"]==outcome) & (df["g_fn"]==g_fn) & (df["param"]==param)]
                if len(p_df) > 0:
                    r = p_df.iloc[0]
                    if param == "g_L2":
                        plsi_str = f"{r['g_l2']:.3f} | - | - | - "
                    else:
                        cov_val = f"{r['cov']:.2f}" if pd.notna(r['cov']) else "-"
                        se_val = f"{r['se']:.3f}" if pd.notna(r['se']) else "-"
                        plsi_str = f"{r['bias']:.3f} | {r['sd']:.3f} | {se_val} | {cov_val} "
                else:
                    plsi_str = "- | - | - | - "
                    
                # NeuralPLSI
                n_df = df[(df["model"]=="NeuralPLSI") & (df["outcome"]==outcome) & (df["g_fn"]==g_fn) & (df["param"]==param)]
                if len(n_df) > 0:
                    r = n_df.iloc[0]
                    if param == "g_L2":
                        npl_str = f"{r['g_l2']:.3f} | - | - | - "
                    else:
                        cov_val = f"{r['cov']:.2f}" if pd.notna(r['cov']) else "-"
                        se_val = f"{r['se']:.3f}" if pd.notna(r['se']) else "-"
                        npl_str = f"{r['bias']:.3f} | {r['sd']:.3f} | {se_val} | {cov_val} "
                else:
                    npl_str = "- | - | - | - "
                    
                md_lines.append(f"| {g_str} | {param} | {plsi_str} | {npl_str} |")
        md_lines.append("\n")
        
    with open("logs/comparison_table.md", "w") as f:
        f.write("\n".join(md_lines))
        
    print("Saved logs/comparison_table.md")

if __name__ == "__main__":
    main()
