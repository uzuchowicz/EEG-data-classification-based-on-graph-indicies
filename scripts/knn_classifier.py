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
    def sklearn_classification( data_train, data_test, labels, n_neighbors):
        model = KNeighborsClassifier(n_neighbors)

        model.fit(data_train, labels)
        data_test = np.array(data_test).reshape(1,1)

        predicted = model.predict(data_test)
        return predicted

    def test_param(self,data, parameters, group_factor):
        self.data = data
        if not isinstance(parameters, list):
            parameters = [parameters]
        data_param = data.filter(parameters, axis=1)
        k_list = [i for i in range(1,2)]
        group_labels = self.data[group_factor]
        result_data = pd.DataFrame(columns={'n_neighbors', 'predicted', 'label', 'result', 'item'})
        k_accuracy = pd.DataFrame(columns={'n_neighbors', 'accuracy'})
        for n_neighbors in k_list:
            for counter, label in enumerate(group_labels):
                #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
                data_test = data_param.iloc[counter,:]
                label = group_labels.iloc[counter]
                data_train_labels=group_labels.drop(axis=0, index=counter)
                data_train = data_param.drop(axis=0, index=counter)

                predicted = Classifier.sklearn_classification(data_train, data_test, data_train_labels, n_neighbors)
                result = 0
                if predicted == label:
                    result = 1
                result_data = result_data.append({'n_neighbors': n_neighbors, 'predicted': int(predicted[0]), 'label': int(label), 'result': result, 'item': counter}, ignore_index=True)
            r_p = result_data['predicted'].to_numpy(dtype=int)
            r_l = result_data['label'].to_numpy(dtype=int)
            k_accuracy = k_accuracy.append({'n_neighbors': n_neighbors, 'accuracy': metrics.accuracy_score(r_p,r_l)}, ignore_index=True)
        return result_data, k_accuracy


    @staticmethod
    def result_visualization(data, result_data, k_accuracy, parameters):
        ax = sns.lineplot(x="n_neighbors", y="accuracy", hue = "event", style = "event", markers = True, dashes = False, data = k_accuract)
        plt.show()
        sns.lineplot()
        sns.lineplot(x="n_neighbors", y="predicted", hue="event", style="event", markers=True, dashes=False,
                     data=result_data)
