# -*- coding: utf-8 -*-

from sklearn.externals.joblib.parallel import cpu_count

NUM_CORES = cpu_count()

print(NUM_CORES)