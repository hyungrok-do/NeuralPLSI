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
    data_dir = Path("output/simulation")
    files = []
    for f in data_dir.glob("simulation+*.json"):
        try:
            if f.is_file():
                files.append(f)
        except PermissionError:
            files.append(f)
    files = sorted(files)
    
    LIST_KEYS = [
        "seed", "performance", "beta_est", "gamma_est",
        "beta_bias", "gamma_bias", "beta_se", "gamma_se",
        "beta_cov", "gamma_cov", "g_pred", 
        "time_fit", "time_hessian", "time_bootstrap"
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
            
        if str(rec.get("_n", "")) not in ["1000", "2000"]:
            continue

        if "ws1" in Path(f).name:
            continue

        records.append(rec)
        
    return records

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
        n_val = str(rec.get("_n", ""))
        
        if x_dist != "normal": continue
        
        beta_est = np.array([b for b in rec.get("beta_est", []) if b is not None])
        gamma_est = np.array([g for g in rec.get("gamma_est", []) if g is not None])
        
        if len(beta_est) == 0: continue
        
        beta_bias = beta_est.mean(axis=0) - TRUE_BETA
        gamma_bias = gamma_est.mean(axis=0) - TRUE_GAMMA
        beta_sd = beta_est.std(axis=0, ddof=1)
        gamma_sd = gamma_est.std(axis=0, ddof=1)
        
        if "beta_se" in rec and len(rec["beta_se"]) > 0:
            beta_se = np.array(rec["beta_se"]).mean(axis=0)
            gamma_se = np.array(rec["gamma_se"]).mean(axis=0)
            beta_cov = np.array(rec["beta_cov"]).mean(axis=0)
            gamma_cov = np.array(rec["gamma_cov"]).mean(axis=0)
        else:
            beta_se = np.full(len(TRUE_BETA), np.nan)
            gamma_se = np.full(len(TRUE_GAMMA), np.nan)
            beta_cov = np.full(len(TRUE_BETA), np.nan)
            gamma_cov = np.full(len(TRUE_GAMMA), np.nan)
        
        # Collect per-coefficient metrics
        for i in range(len(TRUE_BETA)):
            rows.append({"model": model, "n": n_val, "outcome": outcome, "g_fn": g_fn, "param": f"beta_{i}",
                         "bias": beta_bias[i], "sd": beta_sd[i], "se": beta_se[i], "cov": beta_cov[i]})

        # Squared bias for beta vector
        sq_bias_beta = np.sum(beta_bias**2)
        rows.append({"model": model, "n": n_val, "outcome": outcome, "g_fn": g_fn, "param": "beta_SqBias",
                     "bias": sq_bias_beta, "sd": np.nan, "se": np.nan, "cov": np.nan})
        for i in range(len(TRUE_GAMMA)):
            rows.append({"model": model, "n": n_val, "outcome": outcome, "g_fn": g_fn, "param": f"gamma_{i}", 
                         "bias": gamma_bias[i], "sd": gamma_sd[i], "se": gamma_se[i], "cov": gamma_cov[i]})
        
        # Squared bias for gamma vector
        sq_bias_gamma = np.sum(gamma_bias**2)
        rows.append({"model": model, "n": n_val, "outcome": outcome, "g_fn": g_fn, "param": "gamma_SqBias",
                     "bias": sq_bias_gamma, "sd": np.nan, "se": np.nan, "cov": np.nan})

        # g MSE
        g_preds = [np.array(g) for g in rec.get("g_pred", []) if len(g) > 0]
        if g_preds:
            G = np.vstack(g_preds)
            if outcome == "cox":
                G = G - G[:, 500:501]
            g_true_vals = true_g(x_eval, g_fn)
            if outcome == "cox":
                g_true_vals = g_true_vals - g_true_vals[500]
                
            mse_errors = np.mean((G - g_true_vals)**2, axis=1)
            rows.append({"model": model, "n": n_val, "outcome": outcome, "g_fn": g_fn, "param": "g_MSE",
                         "bias": np.mean(mse_errors), "sd": np.std(mse_errors, ddof=1), "se": np.nan, "cov": np.nan})

    df = pd.DataFrame(rows)
    out_lines = []
    
    outcomes = ["continuous", "binary", "cox"]
    g_fns = ["linear", "sfun", "sigmoid"]
    params = [f"beta_{i}" for i in range(len(TRUE_BETA))] + ["beta_SqBias"] + \
             [f"gamma_{i}" for i in range(len(TRUE_GAMMA))] + ["gamma_SqBias", "g_MSE"]
             
    models = ["PLSI", "NeuralPLSI"]
    n_values = ["1000", "2000"]
    
    for n_val in n_values:
        out_lines = []
        for outcome in outcomes:
            out_lines.append(f"\\section*{{Outcome: {outcome}, N: {n_val}}}")
            
            # tabular initialization
            out_lines.append("\\begin{table}[h]")
            out_lines.append("\\centering")
            out_lines.append("\\begin{tabular}{ll|cccc|cccc}")
            out_lines.append("\\hline")
            out_lines.append(" & & \\multicolumn{4}{c|}{PLSI} & \\multicolumn{4}{c}{NeuralPLSI} \\\\")
            out_lines.append("$g$ fn & Param & Bias & SD & SE & Cov & Bias & SD & SE & Cov \\\\")
            out_lines.append("\\hline")
            
            for g_fn in g_fns:
                for pi, param in enumerate(params):
                    param_str = param.replace('_', '\\_')
                    row_str = f"{g_fn} & {param_str} & " if pi == 0 else f" & {param_str} & "
                    
                    model_strs = []
                    for m in models:
                        m_df = df[(df["model"]==m) & (df["outcome"]==outcome) & (df["g_fn"]==g_fn) & (df["param"]==param) & (df["n"]==n_val)]
                        if len(m_df) > 0:
                            r = m_df.iloc[0]
                            bias_str = f"{r['bias']:.3f}"
                            if param.endswith("_SqBias") or param.endswith("_MSE"):
                                sd_str = f"{r['sd']:.3f}" if pd.notna(r['sd']) else "-"
                                m_str = f"{bias_str} & {sd_str} & - & - "
                            else:
                                sd_str = f"{r['sd']:.3f}"
                                se_val = f"{r['se']:.3f}" if pd.notna(r['se']) else "-"
                                cov_val = f"{r['cov']:.3f}" if pd.notna(r['cov']) else "-"
                                m_str = f"{bias_str} & {sd_str} & {se_val} & {cov_val} "
                        else:
                            m_str = "- & - & - & - "
                        model_strs.append(m_str)
                        
                    row_str += "& ".join(model_strs) + "\\\\"
                    out_lines.append(row_str)
                out_lines.append("\\hline")
            
            out_lines.append("\\end{tabular}")
            out_lines.append("\\end{table}")
            out_lines.append("\n")

        with open(f"output/simulation/comparison_table_n{n_val}.tex", "w") as f:
            f.write("\n".join(out_lines))
            
        print(f"Saved output/simulation/comparison_table_n{n_val}.tex")
    
    # Save a markdown version as well for easier viewing
    for n_val in n_values:
        md_lines = []
        for outcome in outcomes:
            md_lines.append(f"### Outcome: {outcome}, N: {n_val}\n")
            md_lines.append("| g fn | Param | PLSI Bias | PLSI SD | PLSI SE | PLSI Cov | NN Bias | NN SD | NN SE | NN Cov |")
            md_lines.append("|---|---|---|---|---|---|---|---|---|---|")
            
            for g_fn in g_fns:
                for pi, param in enumerate(params):
                    g_str = g_fn if pi == 0 else ""
                    
                    model_strs = []
                    for m in models:
                        m_df = df[(df["model"]==m) & (df["outcome"]==outcome) & (df["g_fn"]==g_fn) & (df["param"]==param) & (df["n"]==n_val)]
                        if len(m_df) > 0:
                            r = m_df.iloc[0]
                            bias_str = f"{r['bias']:.3f}"
                            if param.endswith("_SqBias") or param.endswith("_MSE"):
                                sd_str = f"{r['sd']:.3f}" if pd.notna(r['sd']) else "-"
                                m_str = f"{bias_str} | {sd_str} | - | -"
                            else:
                                sd_str = f"{r['sd']:.3f}"
                                cov_val = f"{r['cov']:.3f}" if pd.notna(r['cov']) else "-"
                                se_val = f"{r['se']:.3f}" if pd.notna(r['se']) else "-"
                                m_str = f"{bias_str} | {sd_str} | {se_val} | {cov_val}"
                        else:
                            m_str = "- | - | - | -"
                        model_strs.append(m_str)
                        
                    md_lines.append(f"| {g_str} | {param} | " + " | ".join(model_strs) + " |")
            md_lines.append("\n")
        
        with open(f"output/simulation/comparison_table_n{n_val}.md", "w") as f:
            f.write("\n".join(md_lines))
            
        print(f"Saved output/simulation/comparison_table_n{n_val}.md")

if __name__ == "__main__":
    main()
