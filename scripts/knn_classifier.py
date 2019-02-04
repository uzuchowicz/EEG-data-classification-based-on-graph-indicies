from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import metrics
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Classifier:
    def __init__(self):
        self.data = None

    @staticmethod
    def sklearn_classification( data_train, data_test, labels, n_neighbors):
        model = KNeighborsClassifier(n_neighbors)

        model.fit(data_train, labels)

        predicted = model.predict(data_test)
        accuracy = metrics.accuracy_score(data_train, data_test)
        return predicted, accuracy

    @staticmethod
    def test_param(data, parameter):

        k_list = [i for i in range(15)]
        group_labels = data['group']
        result_data = pd.DataFrame(columns={'preditcted', 'accuracy', 'n_neighbors', 'label','item' })

        for counter, label in enumerate(group_labels):
            #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
            data_train = data.drop(axis=0, index = counter)
            data_test = data.iloc[counter,:]
            for n_neighbors in k_list:
                predicted, accuracy = Classifier.sklearn_classification(data_train, data_test, group_labels, n_neighbors)
                result_data.append({'preditcted':predicted, 'accuracy':accuracy, 'n_neighbors':n_neighbors, 'label':label,'item':counter })
        return result_data


    @staticmethod
    def result_visualization(result_data, parameters):
        ax = sns.lineplot(x="n_neighbors", y="accuracy", hue = "event", style = "event", markers = True, dashes = False, data = result_data)
        plt.show()
        # sns.lineplot(x="n_neighbors", y="predicted", hue="event", style="event", markers=True, dashes=False,
        #              data=result_data)