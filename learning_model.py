import os
import asyncio
import pickle
from dtreeviz.trees import dtreeviz
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import pandas as pd
import shap
import graphviz
from sklearn import tree, preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import _tree
from dotenv import load_dotenv
import colors
from tqdm import tqdm

load_dotenv()


def tree_to_code(tr, feature_names):
    tree_ = tr.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    feature_names = [f.replace(" ", "_")[:-5] for f in feature_names]
    print("def predict({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, np.round(threshold, 2)))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, np.round(threshold, 2)))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)


def reg_metrics(y_t, y_pred, X_t):
    rmse = np.sqrt(mean_squared_error(y_t, y_pred))
    r2 = r2_score(y_t, y_pred)

    # Scikit-learn doesn't have adjusted r-square, hence custom code
    n = y_pred.shape[0]
    k = X_t.shape[1]
    adj_r_sq = 1 - (1 - r2) * (n - 1) / (n - 1 - k)

    return rmse, r2, adj_r_sq


def create_model(mod, ind, X_train, X_test, y_train, y_test):
    model = DecisionTreeRegressor(max_depth=4)
    # lab_enc = preprocessing.LabelEncoder()
    y_train_i = y_train["diff_{}_{}".format(mod, ind.__str__())]
    # encoded = lab_enc.fit_transform(y_train_i)
    # np.save('{}/trees_files/classes_{}_{}.npy'.format(os.getenv('OUTPUTS_DIR'), 'j', ind), lab_enc.classes_)
    model.fit(X_train, y_train_i)
    predicted = model.predict(X_test)
    y_test = y_test.reset_index(drop=True)
    # encoded_test = lab_enc.fit_transform(y_test["diff_j_" + i.__str__()])
    df = pd.DataFrame(data=[y_test["diff_{}_{}".format(mod, ind.__str__())], predicted]).transpose()
    # df = pd.DataFrame(data=[list(y_test["diff_j_" + i.__str__()]), list(predicted)]).transpose()
    df.columns = ["test", "pred"]
    # df = df[df["test"] > -10]
    # df = df[df["pred"] > -10]
    # df = df[df["pred"] < 10]
    # df = df[df["test"] < 10]
    # predicted = predicted[predicted > -10]

    scores = reg_metrics(df["test"], df["pred"], X_train)
    export_plots(mod, model, df, scores, X_train)
    # tree_to_code(model, X_train.columns)


def export_plots(mod, model, df, scores, X_train):
    filename = '{}/trees_files/tree_{}_{}.sav'.format(os.getenv('OUTPUTS_DIR'), mod, i)
    pickle.dump(model, open(filename, 'wb'))
    plt.plot(df["test"])
    plt.plot(df["pred"])
    plt.legend(['test', 'predicted'], loc="lower right")
    plt.table([["rmse", "r2", "adj_r_sq"], [scores[0], scores[1], scores[2]]], loc="upper center")
    plt.savefig("{}/evaluating_{}_{}".format(os.getenv('OUTPUTS_DIR'), mod, i))
    plt.clf()
    # score = model.score(predicted.reshape(-1, 1), y_test["diff_j_" + i.__str__()])
    # print(i, ': ', score)
    dot_data = tree.export_graphviz(model, out_file=None,
                                    feature_names=X_train.columns,

                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.save("snake_{}_{}.gv".format(mod, i.__str__()), "{}".format(os.getenv('OUTPUTS_DIR')))
    graph.render(filename="snake_{}_{}".format(mod, i.__str__()), directory="{}".format(os.getenv('OUTPUTS_DIR')))

    explainer = shap.Explainer(model)
    X = d[["initial_j_1", "initial_j_2", "initial_j_3", "initial_j_4", "initial_s_1", "initial_s_2",
              "initial_s_3", "initial_s_4", "initial_s_5", "initial_s_6", "initial_s_7", "initial_s_8", "command_1",
              "command_2", "command_3", "command_4", "command_5", "command_6", "command_7",
              "command_8"]]
    shap_values = explainer(X)
    # # visualize the first prediction's explanation
    #
    shap.plots.bar(shap_values, show=False)
    plt.savefig('{}/shap_{}_{}.png'.format(os.getenv('OUTPUTS_DIR'), mod, i.__str__()), bbox_inches='tight')
    plt.clf()
    x = []
    y = []
    for importance, name in sorted(zip(model.feature_importances_, X_train.columns), reverse=True):
        y.append(name)
        x.append(importance)
    fig, ax = plt.subplots(figsize=(25, 8))
    bars = ax.bar(y, x, width=0.5)
    plt.savefig("{}feature_importance_{}_{}".format(os.getenv('OUTPUTS_DIR'), mod, i), bbox_inches='tight')
    plt.clf()


def prepare_data(data):
    for i in range(1, 5):
        data = data[data["final_j_" + i.__str__()] > -30]
        data = data[data["final_j_" + i.__str__()] < 30]
        data = data[data["initial_j_" + i.__str__()] > -30]
        data = data[data["initial_j_" + i.__str__()] < 30]
        data["diff_j_" + i.__str__()] = data["final_j_" + i.__str__()] - data["initial_j_" + i.__str__()]
    # le.fit(data["command_{}".format(i)])
    for i in range(1, 9):
        # data["command_{}".format(i)] = data["command_{}".format(i)].astype("category")
        # data["command_{}".format(i)] = le.transform(data["command_{}".format(i)])
        data = data[data["final_s_" + i.__str__()] > -30]
        data = data[data["final_s_" + i.__str__()] < 30]
        data = data[data["initial_s_" + i.__str__()] > -30]
        data = data[data["initial_s_" + i.__str__()] < 30]
        data["diff_s_" + i.__str__()] = data["final_s_" + i.__str__()] - data["initial_s_" + i.__str__()]
    X = data[["initial_j_1", "initial_j_2", "initial_j_3", "initial_j_4", "initial_s_1", "initial_s_2",
              "initial_s_3", "initial_s_4", "initial_s_5", "initial_s_6", "initial_s_7", "initial_s_8", "command_1",
              "command_2", "command_3", "command_4", "command_5", "command_6", "command_7",
              "command_8"]]
    target_cols = ["diff_j_" + i.__str__() for i in range(1, 5)]
    target_cols.extend(["diff_s_" + i.__str__() for i in range(1, 9)])
    Y = data[target_cols]
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test


def load_data():
    file_name = "{}/data2021-09-13.csv".format(os.getenv('DATA_DIR'))
    data = pd.read_csv(file_name)
    data = data.round(1)
    data.to_csv("d.csv")
    return data


if __name__ == "__main__":
    d = load_data()
    X_t, X_t, y_t, y_t = prepare_data(d)
    pbar = tqdm(total=12)
    for i in range(1, 5):
        # model = DecisionTreeClassifier(max_depth=4)
        # lab_enc = preprocessing.LabelEncoder()
        # y_train_i = y_train["diff_j_" + i.__str__()]
        # encoded = lab_enc.fit_transform(y_train_i)
        # np.save('{}/trees_files/classes_{}_{}.npy'.format(os.getenv('OUTPUTS_DIR'), 'j', i), lab_enc.classes_)
        # model.fit(X_train, encoded)
        # predicted = model.predict(X_test)
        # encoded_test = lab_enc.fit_transform(y_test["diff_j_" + i.__str__()])
        # df = pd.DataFrame(data=[encoded_test, predicted]).transpose()
        # # df = pd.DataFrame(data=[list(y_test["diff_j_" + i.__str__()]), list(predicted)]).transpose()
        # df.columns = ["test", "pred"]
        # # df = df[df["test"] > -10]
        # # df = df[df["pred"] > -10]
        # # df = df[df["pred"] < 10]
        # # df = df[df["test"] < 10]
        # # predicted = predicted[predicted > -10]
        #
        # scores = reg_metrics(df["test"], df["pred"], X_train)
        # export_plots("j")
        # # tree_to_code(model, X_train.columns)
        create_model("j", i, X_t, X_t, y_t, y_t)
        pbar.update(1)

    for i in range(1, 9):
        # # model = DecisionTreeRegressor(max_depth=4)
        # model = DecisionTreeClassifier(max_depth=4)
        # lab_enc = preprocessing.LabelEncoder()
        # y_train_i = y_train["diff_s_" + i.__str__()]
        # encoded = lab_enc.fit_transform(y_train_i)
        # np.save('{}/trees_files/classes_{}_{}.npy'.format(os.getenv('OUTPUTS_DIR'), 's', i), lab_enc.classes_)
        # model.fit(X_train, encoded)
        # predicted = model.predict(X_test)
        # encoded_test = lab_enc.fit_transform(y_test["diff_s_" + i.__str__()])
        # df = pd.DataFrame(data=[encoded_test, predicted]).transpose()
        #
        # # y_train_i = y_train["diff_s_" + i.__str__()]
        # # model.fit(X_train, y_train_i)
        # # predicted = model.predict(X_test)
        # # df = pd.DataFrame(data=[list(y_test["diff_s_" + i.__str__()]), list(predicted)]).transpose()
        # df.columns = ["test", "pred"]
        # # df = df[df["test"] > -10]
        # # df = df[df["pred"] > -10]
        # # df = df[df["pred"] < 10]
        # # df = df[df["test"] < 10]
        # # predicted = predicted[predicted > -10]
        # scores = reg_metrics(df["test"], df["pred"], X_train)
        # export_plots("s")
        create_model("s", i, X_t, X_t, y_t, y_t)
        pbar.update(1)
    pbar.close()
