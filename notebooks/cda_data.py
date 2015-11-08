



import numpy as np
import pandas as pd
from statsmodels.compat.python import StringIO


def expand_grid(litu, as_numeric=False):
    n = [len(i[1]) for i in litu]   # map(len, litu)
    labels = [list(i[1]) for i in litu]
    names = [i[0] for i in litu]

    grid_slices = [slice(0, ni, 1) for ni in n]

    mesh_int = np.mgrid[grid_slices]
    if not as_numeric:
        mesh = [np.asarray(label)[i.ravel('C')]
                    for label, i in zip(labels, mesh_int)]
    else:
        mesh = mesh_int
    data = pd.DataFrame(mesh, index=names).T
    return data


# 2nd TABLE 6.14 Beetles Killed after Exposure to Carbon Disulfide

ss = '''\
LogDose Beetles killed fitted_cloglog fitted_probit fitted_logit

1.691 59  6  5.7  3.4  3.5
1.724 60 13 11.3 10.7  9.8
1.755 62 18 20.9 23.4 22.4
1.784 56 28 30.3 33.8 33.9
1.811 63 52 47.7 49.6 50.0
1.837 59 53 54.2 53.4 53.3
1.861 62 61 61.1 59.7 59.2
1.884 60 60 59.9 59.2 58.8
'''

df_beetles = pd.read_csv(StringIO(ss), delim_whitespace=True)

dt = expand_grid([("belt", ("No","Yes")),
                  ("location", ("Urban","Rural")),
                  ("gender", ("Female","Male")),
                  ("injury", ("No","Yes"))
                  ])
count = np.array([[7287,11587,3246,6134,10381,10969,6123, 6693],[996, 759, 973, 757, 812, 380,
1084, 513]]).T.ravel(order='C')
dt['count'] = count
#dt_injury = dt.pivot(dt.columns[:-1], columns=['belt'], values='count')  #doesn't work

df_injury_1d = dt
del dt


dt_injury = expand_grid([("belt", ("No","Yes")),
                          ("location", ("Urban","Rural")),
                          ("gender", ("Female","Male")),
                          #("injury", ("No","Yes"))
                          ])
count = np.array([[7287,11587,3246,6134,10381,10969,6123, 6693],[996, 759, 973, 757, 812, 380,
1084, 513]]).T
#dt[['no_inj', 'inj']] = count
df_injury_bin = dt_injury.join(pd.DataFrame(count, columns=['no_inj', 'inj']))


ss_injury = '''\
  belt location  gender injury  count
    No    Urban  Female     No   7287
    No    Urban  Female    Yes    996
    No    Urban    Male     No  11587
    No    Urban    Male    Yes    759
    No    Rural  Female     No   3246
    No    Rural  Female    Yes    973
    No    Rural    Male     No   6134
    No    Rural    Male    Yes    757
   Yes    Urban  Female     No  10381
   Yes    Urban  Female    Yes    812
   Yes    Urban    Male     No  10969
   Yes    Urban    Male    Yes    380
   Yes    Rural  Female     No   6123
   Yes    Rural  Female    Yes   1084
   Yes    Rural    Male     No   6693
   Yes    Rural    Male    Yes    513
'''
df_injury = pd.read_csv(StringIO(ss_injury), delim_whitespace=True)
