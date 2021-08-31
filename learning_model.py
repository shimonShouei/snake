from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import shap
import graphviz
from sklearn import tree
import matplotlib.pyplot as plt

model = DecisionTreeRegressor(max_depth=5)
file_name = "./Data/data2021-08-31.csv"
data = pd.read_csv(file_name)
for i in range(1, 5):
    data["diff_j_" + i.__str__()] = data["final_j_" + i.__str__()] - data["initial_j_" + i.__str__()]
X = data[["initial_j_1", "initial_j_2", "initial_j_3", "initial_j_4", "initial_s_1", "initial_s_2",
          "initial_s_3", "initial_s_4", "initial_s_5", "initial_s_6", "initial_s_7", "initial_s_8", "command_1",
          "command_2", "command_3", "command_4", "command_5", "command_6", "command_7",
          "command_8"]]
Y = data[["diff_j_" + i.__str__() for i in range(1, 5)]]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
for i in range(1, 5):
    y = y_train["diff_j_" + i.__str__()]
    model.fit(X_train, y)
    predicted = model.predict(X_test)
    # score = model.score(predicted.reshape(-1, 1), y_test)
    # print(i, ': ', score)
    dot_data = tree.export_graphviz(model, out_file=None,
                                    feature_names=X_train.columns,
                                    class_names="diff_j_" + i.__str__(),
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("./Outputs/snake" + i.__str__())
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    # visualize the first prediction's explanation
    shap.plots.bar(shap_values, show=False)
    plt.savefig('./Outputs/shap{}.png'.format(i.__str__()))

