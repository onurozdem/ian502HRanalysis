import pickle
import datetime

from sklearn import metrics
import matplotlib.pyplot as plt
from HRAnalysis.models import ModelDetail


class NeuralNetwork:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def train(self):
        try:
            #from keras import backend as K
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense
            from tensorflow.keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
            from tensorflow.keras.utils import to_categorical

            model_score_dict = dict()
            model_start_time = datetime.datetime.now()

            classifier = Sequential()
            # Adding the input layer and the first hidden layer
            classifier.add(Dense(units = 6, kernel_initializer='uniform', activation='relu', input_dim=len(self.x_train.columns)))

            # Adding the second hidden layer
            classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

            # Adding the output layer
            classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

            # Compiling the ANN | means applying SGD on the whole ANN
            classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

            # Fitting the ANN to the Training set
            classifier.fit(self.x_train, self.y_train, epochs=50)

            score, acc_annTrain = classifier.evaluate(self.x_train, self.y_train,batch_size=10)
            print('Train score:', score)
            print('Train accuracy:', acc_annTrain)
            # Part 3 - Making predictions and evaluating the model
            y_pred = classifier.predict(self.x_test)
            y_pred = (y_pred > 0.5)

            model_end_time = datetime.datetime.now()
            model_running_performance = model_end_time - model_start_time

            print('*'*20)
            score, acc_annTest = classifier.evaluate(self.x_test, self.y_test, batch_size=10)
            print('Test score:', score)
            print('Test accuracy:', acc_annTest)

            conf_mat = metrics.confusion_matrix(self.y_test, y_pred)

            # ROC Curve
            #pred_proba_rf = classifier.predict_proba(self.x_test)
            pred_proba_ann = []
            for i in classifier.predict_proba(self.x_test):
                pred_proba_ann.append(i)
            fpr, tpr, _ = metrics.roc_curve(self.y_test, pred_proba_ann)
            auc_ann = metrics.roc_auc_score(self.y_test, pred_proba_ann)

            plt.figure()
            lw = 3
            plt.plot(fpr, tpr, label="Neural Network, auc_ann = " + str(auc_ann))
            plt.plot([0, 1], [0, 1], color='red', lw=lw, linestyle='dashed')
            plt.legend(loc=4)
            plt.savefig('./static/images/roc_ann.png')

            # Assign all score values to dict
            model_score_dict["model_running_performance"] = (model_running_performance.seconds/60)
            model_score_dict["accuracy"] = acc_annTrain
            model_score_dict["conf_mat"] = conf_mat.tolist()
            model_score_dict["fpr"] = fpr.tolist()
            model_score_dict["tpr"] = tpr.tolist()
            model_score_dict["auc"] = auc_ann

            md = ModelDetail(**{'AlgorithmName': 'ANN', 'ModelScoreDict': str(model_score_dict)})
            md.save()

            # Export model
            """with open('./HRAnalysis/analysemodels/models/ANN.pkl', 'wb') as model_file:
                pickle.dump(classifier, model_file)"""
            classifier.save('./HRAnalysis/analysemodels/models/ANN.h5')
        except Exception as e:
            raise e
