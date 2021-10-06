import os
import pickle
import scipy.stats as st
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import learning_model
import pandas as pd
from dotenv import load_dotenv
import numpy as np
from sklearn.preprocessing import LabelEncoder

load_dotenv()


# def find_successors(self):
class MdpModel:
    def __init__(self):
        self.data_file_name = "{}/data2021-09-13.csv".format(os.getenv('DATA_DIR'))
        self.data = pd.read_csv(self.data_file_name)
        self.trees = []
        for i in range(1, 5):
            m = load_tree(i, "j")
            self.trees.append(m)
        for i in range(1, 9):
            m = load_tree(i, "s")
            self.trees.append(m)
        self.distribution = []

    def create_successors(self):
        X_train, X_test, y_train, y_test = learning_model.prepare_data(self.data)
        self.distribution = np.ndarray((X_train.shape[0], len(self.trees)), float)
        for k, i in tqdm(enumerate(self.trees)):
            mod = 's'
            ind = k-3
            if k < 4:
                mod = 'j'
                ind = k+1
            predicted = i.predict(X_train)
            # predicted = predicted + X_train["initial_{}_{}".format(mod, ind.__str__())]
            self.distribution[:, k] = np.array(predicted)
            f = plt.figure()
            f.set_figwidth(25)
            f.set_figheight(8)
            # q25, q75 = np.percentile(predicted, [0.25, 0.75])
            bin_width = 0.1
            # bin_width = 2 * (q75 - q25) * len(predicted) ** (-1 / 3)
            bins = int(round((predicted.max() - predicted.min()) / bin_width))
            sns.displot(predicted, bins=bins, kde=True);
            # plt.hist(predicted, density=True, bins=bins, label="Data")
            # mn, mx = plt.xlim()
            # plt.xlim(mn, mx)
            # kde_xs = np.linspace(mn, mx, 300)
            # kde = st.gaussian_kde(predicted)
            # plt.plot(kde_xs, kde.pdf(kde_xs), label="PDF")
            # plt.legend(loc="upper left")
            # plt.ylabel("Probability")
            # plt.xlabel("diff")
            # plt.title("Histogram_{}".format(k));
            plt.savefig("{}/dist/dist_{}_{}".format(os.getenv('OUTPUTS_DIR'), mod, ind))
            plt.clf()
        np.savetxt("{}/dist.csv".format(os.getenv('OUTPUTS_DIR')), self.distribution, delimiter=",")

    def find_most_likely_successors(self, s, a):
        input_s_a = s.copy()
        input_s_a.extend(a)
        ans = []
        diff_ans = []
        for i in range(1, 5):
            # encoder = LabelEncoder()
            # encoder.classes_ = np.load('{}/trees_files/classes_{}_{}.npy'.format(os.getenv('OUTPUTS_DIR'), 'j', i))
            pred = self.trees[i - 1].predict(np.reshape(input_s_a, (1, -1)))
            diff_ans.append(pred[0])
        for i in range(1, 9):
            # encoder = LabelEncoder()
            # encoder.classes_ = np.load('{}/trees_files/classes_{}_{}.npy'.format(os.getenv('OUTPUTS_DIR'), 's', i))
            pred = self.trees[i - 1].predict(np.reshape(input_s_a, (1, -1)))
            diff_ans.append(pred[0])
        for i in range(len(diff_ans)):
            ans.append(diff_ans[i] + s[i])
        print(ans)
        print(diff_ans)
        return ans


def load_tree(i, mod):
    file = open('{}/trees_files/tree_{}_{}.sav'.format(os.getenv('OUTPUTS_DIR'), mod, i), 'rb')
    m = pickle.load(file)
    file.close()
    return m


mdp = MdpModel()
mdp.create_successors()
mdp.find_most_likely_successors([-1.50335693359375, 5.96282958984375, -14.0542602539063, 5.41778564453125, 6.63388347625732,
                             0.713590919971466, 1.20638728141785, 1.28844451904297, 5.08199548721314, 4.7921838760376,
                             1.25786256790161, 0.814698100090027],
                            [100, 100, -100, -100, 100, 100, 100, 100])
