# -*- coding: utf-8 -*-
"""
Simulation model to test for possible bias in generalized diff in diff model
(two way fixed effects) when outcome variable is binary with dropped observations.

"""
# =============================================================================
# Imports
# =============================================================================
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy
from patsy import dmatrices
import numpy as np
import linearmodels as lm
from linearmodels.panel import PanelOLS

# =============================================================================
# Define parameters
# =============================================================================
years = 2                       # Number of years in each of the two periods
nobs = 10000                    # Number of observations in each of the two groups
d_outside = 0.30                # True deforestation rate outside treated area in p2
diff = 0.40                     # Pre-treatment difference between treated and control
trend = -0.1                    # Trend in deforestation rate across periods
att = -0.16                     # Average treatment effect on the treated

# =============================================================================
# Calculate deforestation rates
# =============================================================================
m_00 = d_outside                        # Average deforestation rate outside treated area in p1
m_01 = d_outside + trend                # Average deforestation rate outside treated area in p2
m_10 = d_outside + diff                 # Average deforestation rate inside treated area in p1
m_11 = d_outside + diff + trend + att   # Average deforestation rate inside treated area in p2

# =============================================================================
# Introduce three types of randomness:
#   i: Individual-level 
#   y: year-level changes
#   e: common structure across all observations
# =============================================================================
std_i = 0.1                     # Standard error of individual-level variation
std_y = 0.01                    # Standard error of year-level variation
std_e = 0.01                    # Standard error of observation-level variation

i_err = np.random.normal(0, std_i, nobs*2)
y_err = np.random.normal(0, std_y, years*2)
e_err = np.random.normal(0, std_e, [nobs*2, years*2])

df = pd.DataFrame(data = e_err, columns = range(years*2))
df = df.add(i_err, axis = 0)
df = df.add(y_err, axis = 1)


# =============================================================================
# Add in average deforestation rates for each group
# =============================================================================
df.loc[:nobs-1, :years-1] = df.loc[:nobs-1, :years-1] + m_00
df.loc[:nobs-1, years:] = df.loc[:nobs-1, years:] + m_01
df.loc[nobs:, :years-1] = df.loc[nobs:, :years-1] + m_10
df.loc[nobs:, years:] = df.loc[nobs:, years:] + m_11

# =============================================================================
# Define treatment
# =============================================================================
df['treat'] = 0
df.loc[nobs:, 'treat'] = 1
df['idx'] = df['treat'].astype(str) + '_' + df.index.values.astype(str)  
df = df.set_index(['idx', 'treat'])

# =============================================================================
# Simulate deforestation
# =============================================================================
defor_draw = np.random.uniform(low = 0, high = 1, size = df.shape)
defor_df = (df > defor_draw).astype(int)
defor_df['defor_year'] = defor_df.idxmax(axis = 1)
defor_df.loc[defor_df.max(axis = 1)==0, 'defor_year'] = years * 2 + 1
defor_df = defor_df.rename(columns = {y: 'defor'+str(y) for y in range(years*2)})
defor_df = defor_df.reset_index()
defor_df = pd.wide_to_long(defor_df, stubnames = 'defor', i = 'idx', j = 'year')
defor_df = defor_df.reset_index()
defor_df.loc[defor_df['year']>defor_df['defor_year'], 'defor'] = np.nan
defor_df['post'] = defor_df['year'] > years-1

defor = pd.pivot_table(data = defor_df, index = 'treat', columns = 'year', 
                       values = 'defor', aggfunc = np.sum)
count = pd.pivot_table(data = defor_df, index = 'treat', columns = 'year', 
                       values = 'defor', aggfunc = "count")
defor_df = defor_df.set_index(['idx', 'year'])


# =============================================================================
# Run regression to estimate treatment effect
# =============================================================================
## Simple diff in diff
mod = PanelOLS.from_formula('defor ~ treat * post', defor_df)
res = mod.fit(cov_type='clustered', cluster_entity=True)
print(res)

## Generalized did using two-way fixed effects
# Outer is entity, inner is time
from linearmodels.panel import PanelOLS
defor_df['t'] = defor_df['treat'] * defor_df['post']
mod = PanelOLS.from_formula('defor ~ t + EntityEffects + TimeEffects', defor_df)
res = mod.fit(cov_type='clustered', cluster_entity=True)
print(res)



### KEY OBSERVATION: FE estimator yields ~ (estimate of att) = diff + att while
### simple diff in diff yields ~ (estimate of att) = att
