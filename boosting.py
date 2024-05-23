import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from scipy.special import expit
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier  

class Boosting:
    def __init__(self, 
                 base_model_class=DecisionTreeClassifier,  
                 base_model_params={}, 
                 n_estimators=100, 
                 learning_rate=0.1, 
                 subsample=1.0, 
                 early_stopping_rounds=None, 
                 plot=False):
        self.base_model_class = base_model_class
        self.base_model_params = base_model_params
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.early_stopping_rounds = early_stopping_rounds
        self.plot = plot
        self.models = []
        self.gammas = []
        self.losses_train = []
        self.losses_valid = []

    def fit(self, x_train, y_train, x_valid, y_valid):

        y_pred_train = np.zeros_like(y_train, dtype=float)
        y_pred_valid = np.zeros_like(y_valid, dtype=float)

        best_score = -np.inf
        best_iteration = 0

        for i in range(self.n_estimators):
            model, gamma = self.fit_new_base_model(x_train, y_train, y_pred_train)
            self.models.append(model)
            self.gammas.append(gamma)


            y_pred_train += self.learning_rate * gamma * model.predict_proba(x_train)[:, 1]
            y_pred_valid += self.learning_rate * gamma * model.predict_proba(x_valid)[:, 1]


            loss_train = self.loss_fn(y_train, y_pred_train)
            loss_valid = self.loss_fn(y_valid, y_pred_valid)
            self.losses_train.append(loss_train)
            self.losses_valid.append(loss_valid)


            score = roc_auc_score(y_valid, expit(y_pred_valid))
            if score > best_score:
                best_score = score
                best_iteration = i
            elif self.early_stopping_rounds and i - best_iteration >= self.early_stopping_rounds:
                break

        if self.plot:
            self.plot_loss()

    def fit_new_base_model(self, x_train, y_train, y_pred_train):
        residuals = y_train - expit(y_pred_train)

        x_sample, y_sample, res_sample = resample(x_train, y_train, residuals, n_samples=int(self.subsample * len(y_train)))

        model = clone(self.base_model_class(**self.base_model_params))
        model.fit(x_sample, np.sign(res_sample))

        gamma = self.optimize_gamma(y_sample, res_sample, model, x_sample)

        return model, gamma

    def optimize_gamma(self, y, residuals, model, x):
        preds = model.predict_proba(x)[:, 1]
        return np.sum(residuals * preds) / np.sum(preds * (1 - preds))

    def loss_fn(self, y_true, y_pred):
        return np.mean(np.log(1 + np.exp(-y_true * y_pred)))

    def predict_proba(self, X):
        y_pred = np.zeros(X.shape[0], dtype=float)
        for model, gamma in zip(self.models, self.gammas):
            y_pred += self.learning_rate * gamma * model.predict_proba(X)[:, 1]
        probabilities = expit(y_pred)
        return np.vstack((1 - probabilities, probabilities)).T

    def score(self, X, y):
        return roc_auc_score(y, self.predict_proba(X)[:, 1])

    def plot_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses_train, label='Training Loss')
        plt.plot(self.losses_valid, label='Validation Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss over Iterations')
        plt.show()

    def feature_importances_(self):

        importances = np.zeros(self.models[0].feature_importances_.shape)
    
        for model in self.models:
            importances += model.feature_importances_
    
        importances /= len(self.models)
        return importances / np.sum(importances)


