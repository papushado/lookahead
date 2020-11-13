import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


class PropModel():
    def __init__(self, Cs=None, random_state=None):
        """
        Class Instance to compute the propensity weights.
        Arguments:
            Cs: Scalar or an array of paramter C in Logistic Regression Model
            If this is an array, best C is chosen through 5-fold cross validation.
        Returns:
            None
        """
        if Cs is None:
            Cs = np.arange(0.05,1.5,0.05)
        self.Cs = Cs
        self.C = None
        if np.isscalar(Cs):
            self.model = LogisticRegression(C=Cs)
            self.C = Cs
        else: #C is list
            self.model = LogisticRegressionCV(Cs=Cs, cv=5, scoring='neg_log_loss',
                                            random_state=random_state)

    def fit(self, x, xp):
        """
        Train the logistic regression model
        Arguments:
            x:  Train distribution
            xp: Shifted distribution
        Returns:
            None
        """
        n = x.shape[0]
        x_in_out = np.concatenate([x, xp])
        y_in_out = np.concatenate([-np.ones(n), np.ones(n)])
        self.model.fit(x_in_out, y_in_out)
        if self.C is None:
            self.C = self.model.C_

    def predict(self, x, clip=None):
        """
        Computes the propensity scores using the trained model
        Arguments:
            x: Instances for which score is to be computed
            clip: A tuple (min_val, max_val) to clip the scores. 
            If set to None, no clipping is done.
        """
        w = np.exp(self.model.decision_function(x))
        if clip is not None:
            w = np.maximum(w,clip[0])
            w = np.minimum(w,clip[1])
        return w

    def score(self, x, xp):
        p0 = self.model.predict_log_proba(x)
        p1 = self.model.predict_log_proba(xp)
        return -0.5*(np.mean(p1[:,1])+np.mean(p0[:,0]))
        

    def diagnostics(self, w):
        """ Computed Effective Sample Size
            Arguments:
                w: Propensity scores
            Returns
                n_eff: Number of effective samples
                w_sum: L1-norm of score
        """
        n_eff = (np.sum(w) ** 2) / (np.sum(w ** 2)) #effective sample size
        w_sum = np.sum(w) #effective sample size
        return n_eff, w_sum
