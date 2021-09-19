import os
import pickle
from dotenv import load_dotenv
import numpy as np
load_dotenv()


def find_successors(s, a):
    input_s_a = s.copy()
    input_s_a.extend(a)
    models = []
    for i in range(1, 5):
        file = open('{}/trees_files/tree_j_{}.sav'.format(os.getenv('OUTPUTS_DIR'), i), 'rb')
        m = pickle.load(file)
        file.close()
        models.append(m)
    for i in range(1, 9):
        file = open('{}/trees_files/tree_s_{}.sav'.format(os.getenv('OUTPUTS_DIR'), i), 'rb')
        m = pickle.load(file)
        file.close()
        models.append(m)
    ans = []
    for i in range(0, 12):
        ans.append(list(models[i].predict(np.reshape(input_s_a, (1, -1))))[0])
    for i in range(len(ans)):
        ans[i] += s[i]
    print(ans)
    return ans


find_successors([-1.50335693359375,	5.96282958984375, -14.0542602539063, 5.41778564453125, 6.63388347625732,
                 0.713590919971466, 1.20638728141785, 1.28844451904297, 5.08199548721314, 4.7921838760376,
                 1.25786256790161, 0.814698100090027],
                [100, 100, -100, -100, 100, 100, 100, 100])