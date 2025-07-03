import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from models.nPLSI import neuralPLSI


data = pd.read_csv('NHANES/studypop.csv')

scaler = StandardScaler()

# Telomere length outcome: log-transformed and scaled
data['lnLTL_z'] = scaler.fit_transform(np.log(data[['TELOMEAN']]))
TELO = data['lnLTL_z'].values

# Center and scale age
data['age_z'] = scaler.fit_transform(data[['age_cent']])
data['agez_sq'] = data['age_z'] ** 2

# Dummy coding
data['bmicat2'] = (data['bmi_cat3'] == 2).astype(int)
data['bmicat3'] = (data['bmi_cat3'] == 3).astype(int)
data['educat1'] = (data['edu_cat'] == 1).astype(int)
data['educat3'] = (data['edu_cat'] == 3).astype(int)
data['educat4'] = (data['edu_cat'] == 4).astype(int)
data['otherhispanic'] = (data['race_cat'] == 1).astype(int)
data['mexamerican'] = (data['race_cat'] == 2).astype(int)
data['black'] = (data['race_cat'] == 3).astype(int)

# Scale lab values
for col in ['LBXWBCSI', 'LBXLYPCT', 'LBXMOPCT', 'LBXNEPCT', 'LBXEOPCT', 'LBXBAPCT', 'ln_lbxcot']:
    data[col.lower() + '_z'] = scaler.fit_transform(data[[col]])

# Mixture matrix and log-transform
mixture_cols = ['LBX074LA', 'LBX099LA', 'LBX118LA', 'LBX138LA', 'LBX153LA', 'LBX170LA', 'LBX180LA', 'LBX187LA',
                'LBX194LA', 'LBXHXCLA', 'LBXPCBLA',
                'LBXD03LA', 'LBXD05LA', 'LBXD07LA',
                'LBXF03LA', 'LBXF04LA', 'LBXF05LA', 'LBXF08LA']

mixture = data[mixture_cols].copy()
lnmixture = np.log(mixture.replace(0, np.nan)).fillna(0)  # log(0) is undefined; handle zeros as needed: is it OK to replace them with 0?
lnmixture_z = scaler.fit_transform(lnmixture)

# Rename columns
lnmixture_z = pd.DataFrame(lnmixture_z, columns=[
    *[f'PCB{x}' for x in [74, 99, 118, 138, 153, 170, 180, 187, 194, 169, 126]],
    *[f'Dioxin{i}' for i in range(1, 4)],
    *[f'Furan{i}' for i in range(1, 5)],
], index=data.index)

# Covariate matrix
covariate_cols = ['age_z', 'agez_sq', 'male', 'bmicat2', 'bmicat3',
                  'educat1', 'educat3', 'educat4', 'otherhispanic',
                  'mexamerican', 'black', 'lbxwbcsi_z', 'lbxlypct_z',
                  'lbxmopct_z', 'lbxnepct_z', 'lbxeopct_z', 'lbxbapct_z', 'ln_lbxcot_z']

covariates = data[covariate_cols].copy()
covariates = SimpleImputer().fit_transform(covariates)
covariates = pd.DataFrame(covariates, columns=covariate_cols, index=data.index)

# Selected exposure variables (Z matrix subset)
select_z = lnmixture_z[['Furan1', 'PCB169', 'PCB126', 'PCB74', 'PCB153', 'PCB170', 'PCB180', 'PCB194']]


model = neuralPLSI(family='continuous')
#model.fit(covariates.values, select_z.values, TELO)
import matplotlib.pyplot as plt
from tqdm import tqdm
# bootstrap
g_grid = np.linspace(-3, 3, 1000)
beta_stack = []
gamma_stack = []
g_stack = []

for i in tqdm(range(500)):
    print(f'Bootstrap iteration {i+1}')
    sample_indices = np.random.choice(covariates.index, size=len(covariates), replace=True)
    covariates_sample = covariates.loc[sample_indices]
    select_z_sample = select_z.loc[sample_indices]
    TELO_sample = TELO[sample_indices]

    model.fit(select_z_sample.values, covariates_sample.values, TELO_sample)

    beta_stack.append(model.beta)
    gamma_stack.append(model.gamma)
    g_stack.append(model.g_function(g_grid))

beta_stack = np.array(beta_stack)
gamma_stack = np.array(gamma_stack)
g_stack = np.array(g_stack)
g_mean = np.mean(g_stack, axis=0)
g_ub = np.percentile(g_stack, 97.5, axis=0)
g_lb = np.percentile(g_stack, 2.5, axis=0)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.boxplot(beta_stack)
plt.title('Beta Coefficients')
plt.xticks(range(1, len(model.beta) + 1), select_z.columns, rotation=45)
plt.subplot(1, 3, 2)
plt.boxplot(gamma_stack)
plt.title('Gamma Coefficients')
plt.xticks(range(1, len(model.gamma) + 1), covariates.columns, rotation=45)
plt.subplot(1, 3, 3)

plt.plot(g_grid, g_mean, label='Mean g Function', color='black')
plt.fill_between(g_grid, g_lb, g_ub, color='lightblue', alpha=0.5, label='95% CI')
plt.legend()

plt.title('g Function Predictions')
plt.xlabel('g Grid')
plt.ylabel('g Function Value')
plt.tight_layout()
plt.savefig('output/nhanes_nplsi_results.png')
