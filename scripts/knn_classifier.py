from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import metrics
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


class Classifier:
    def __init__(self):
        self.data = None

    @staticmethod
    def skl_knn_classification(data_train, data_test, labels, n_neighbors):
        model = KNeighborsClassifier(n_neighbors)

        model.fit(data_train, labels)
        if np.shape(data_test) == (1,):
            data_test = np.array(data_test).reshape(1, 1)
        elif len(np.shape(data_test)) == 1:
            data_test = np.array([data_test])
        else:
            data_test = np.array(data_test)

        predicted = model.predict(data_test)
        return predicted

    def test_param(self, data, parameters, group_factor):
        self.data = data
        if not isinstance(parameters, list):
            parameters = [parameters]
        data_param = data.filter(parameters, axis=1)
        k_list = [i for i in range(10,12)]
        group_labels = self.data[group_factor]
        result_data = pd.DataFrame(columns={'n_neighbors', 'predicted', 'label', 'result', 'item'})
        k_accuracy = pd.DataFrame(columns={'n_neighbors', 'accuracy'})
        for n_neighbors in k_list:
            for counter, label in enumerate(group_labels):
                # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
                data_test = data_param.iloc[counter, :]
                label = group_labels.iloc[counter]
                data_train_labels=group_labels.drop(axis=0, index=counter)
                data_train = data_param.drop(axis=0, index=counter)

                predicted = Classifier.skl_knn_classification(data_train, data_test, data_train_labels, n_neighbors)
                result = 0
                if predicted == label:
                    result = 1
                result_data = result_data.append({'n_neighbors': n_neighbors, 'predicted': int(predicted[0]),
                                                  'label': int(label), 'result': result, 'item': counter}, ignore_index=True)
            r_p = result_data['predicted'].to_numpy(dtype=int)
            r_l = result_data['label'].to_numpy(dtype=int)
            k_accuracy = k_accuracy.append({'n_neighbors': n_neighbors, 'accuracy': metrics.accuracy_score(r_p, r_l)},
                                           ignore_index=True)
        return result_data, k_accuracy

    @staticmethod
    def result_visualization(data, result_data, k_accuracy, parameters, group_label):
        plt.figure(0)
        sns.lineplot(x="n_neighbors", y="accuracy",  markers=True, dashes=False, data=k_accuracy)
        plt.show()
        idx_max, max_accuracy = k_accuracy["accuracy"].idxmax(), k_accuracy["accuracy"].max()
        best_result = result_data.loc[idx_max]
        k_max = best_result["n_neighbors"]
        best_result_data = result_data[result_data['n_neighbors'] == k_max]
        # h=data.shape[0]//2
        h = 50

        if len(parameters) == 2:
            x_min, x_max = data[parameters[0]].min() - .5, data[parameters[0]].max()
            y_min, y_max = data[parameters[1]].min() - .5, data[parameters[1]].max()
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, h), np.linspace(y_min, y_max, h))
            best_result_prediction = best_result_data["predicted"].to_numpy(dtype=int)
            prediction_grid = Classifier.skl_knn_classification(data.filter(parameters, axis=1), np.c_[xx.ravel(),
                                                                yy.ravel()], data[group_label], k_max)

            prediction_grid = prediction_grid.reshape(xx.shape)
            plt.figure(1)
            plt.set_cmap(plt.cm.Paired)
            plt.pcolormesh(xx, yy, prediction_grid)

            plt.scatter(data[parameters[0]], data[parameters[1]], c=data[group_label])
            plt.xlabel(parameters[0])
            plt.ylabel(parameters[1])

            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.xticks(())
            plt.yticks(())
            plt.show()
