
import json
import numpy as np
import pandas as pd

filename = f'output/bootstrap_PLSI_res_500_linear_cox.json'
with open(filename, 'r') as f:
    res = json.load(f)

print(np.array(pd.DataFrame(res)['beta_boot'][0]).shape)