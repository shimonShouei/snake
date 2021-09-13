import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import shap
import graphviz
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing


def reg_metrics(y_t, y_pred, X_t):
    rmse = np.sqrt(mean_squared_error(y_t, y_pred))
    r2 = r2_score(y_t, y_pred)

    # Scikit-learn doesn't have adjusted r-square, hence custom code
    n = y_pred.shape[0]
    k = X_t.shape[1]
    adj_r_sq = 1 - (1 - r2) * (n - 1) / (n - 1 - k)

    return rmse, r2, adj_r_sq


def export_plots(mod):
    plt.plot(df["test"])
    plt.plot(df["pred"])
    plt.legend(['test', 'predicted'], loc="lower right")
    plt.table([["rmse", "r2", "adj_r_sq"], [scores[0], scores[1], scores[2]]], loc="upper center")
    plt.savefig("./Outputs/evaluating_{}_{}".format(mod, i))
    plt.clf()
    # score = model.score(predicted.reshape(-1, 1), y_test["diff_j_" + i.__str__()])
    # print(i, ': ', score)
    dot_data = tree.export_graphviz(model, out_file=None,
                                    feature_names=X_train.columns,
                                    class_names="diff_j_" + i.__str__(),
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("./Outputs/snake_{}_{}".format(mod, i.__str__()))
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    # visualize the first prediction's explanation

    shap.plots.bar(shap_values, show=False)
    plt.savefig('./Outputs/shap_{}_{}.png'.format(mod, i.__str__()), bbox_inches='tight')
    plt.clf()
    x = []
    y = []
    for importance, name in sorted(zip(model.feature_importances_, X_train.columns), reverse=True):
        y.append(name)
        x.append(importance)
    fig, ax = plt.subplots(figsize=(25, 8))
    bars = ax.bar(y, x, width=0.5)
    plt.savefig("./Outputs/feature_importance_{}_{}".format(mod, i), bbox_inches='tight')
    plt.clf()


le = preprocessing.LabelEncoder()
model = DecisionTreeRegressor(max_depth=4)
file_name = "./Data/data2021-09-13.csv"
data = pd.read_csv(file_name)
data.round(1)
for i in range(1, 5):
    data["diff_j_" + i.__str__()] = data["final_j_" + i.__str__()] - data["initial_j_" + i.__str__()]
# le.fit(data["command_{}".format(i)])
for i in range(1, 9):
    # data["command_{}".format(i)] = data["command_{}".format(i)].astype("category")
    # data["command_{}".format(i)] = le.transform(data["command_{}".format(i)])
    data["diff_s_" + i.__str__()] = data["final_s_" + i.__str__()] - data["initial_s_" + i.__str__()]

X = data[["initial_j_1", "initial_j_2", "initial_j_3", "initial_j_4", "initial_s_1", "initial_s_2",
          "initial_s_3", "initial_s_4", "initial_s_5", "initial_s_6", "initial_s_7", "initial_s_8", "command_1",
          "command_2", "command_3", "command_4", "command_5", "command_6", "command_7",
          "command_8"]]
target_cols = ["diff_j_" + i.__str__() for i in range(1, 5)]
target_cols.extend(["diff_s_" + i.__str__() for i in range(1, 9)])
Y = data[target_cols]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

for i in range(1, 5):

    y_train_i = y_train["diff_j_" + i.__str__()]
    model.fit(X_train, y_train_i)
    predicted = model.predict(X_test)
    df = pd.DataFrame(data=[list(y_test["diff_j_" + i.__str__()]), list(predicted)]).transpose()
    df.columns = ["test", "pred"]
    df = df[df["test"] > -10]
    df = df[df["pred"] > -10]
    df = df[df["pred"] < 10]
    df = df[df["test"] < 10]
    # predicted = predicted[predicted > -10]
    scores = reg_metrics(df["test"], df["pred"], X_train)
    export_plots("j")
for i in range(1, 9):

    y_train_i = y_train["diff_s_" + i.__str__()]
    model.fit(X_train, y_train_i)
    predicted = model.predict(X_test)
    df = pd.DataFrame(data=[list(y_test["diff_s_" + i.__str__()]), list(predicted)]).transpose()
    df.columns = ["test", "pred"]
    df = df[df["test"] > -10]
    df = df[df["pred"] > -10]
    df = df[df["pred"] < 10]
    df = df[df["test"] < 10]
    # predicted = predicted[predicted > -10]
    scores = reg_metrics(df["test"], df["pred"], X_train)
    export_plots("s")