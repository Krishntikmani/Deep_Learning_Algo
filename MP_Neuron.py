from sklearn.metrics import accuracy_score
import numpy as np


class MPNeuron:

    def __inti__(self):
        self.b =None 
    
    def model(self,x):
        return (np.sum(x)>=self.b)
    
    def predict(self,X):
        Y_pred = []
        for x in X:
            Y_pred.append(self.model(x))
        return (np.array(Y_pred))

    def fit(self,X,Y):
        accuracies = {}
        for b in range(X.shape[0]+1):
            self.b = b
            y_pred = self.predict(X)
            accuracies[b] = accuracy_score(y_pred,Y)
        best_b = max(accuracies,key=accuracies.get)
        self.b = best_b

        print("Optimal Value of parameter b is ",self.b)
        print("Accuracy over training set is ",accuracies[best_b])    
