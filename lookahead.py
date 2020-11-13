import numpy as np
from utils import *

class Lookahead():
    def __init__(self, pred_model, uncert_model, prop_model,
                lam=1.0, eta=1.0, mask=None, z_score=1.65, ground_truth_model=None):        
        """
        Implements the lookahead model and the training procedure
        Arguments:
            pred_model : Model used for f(x)
            uncert_model: Model used for predicting uncertainty intervals
            prop_model: Model used for computing propensity scores
            lam: Trade-off parameter that controls accuracy and improvement
            eta: Step-size for prediction
            Mask: Boolean vector for making some features immutable
            z_score: Controls size of the confidence intervals
            ground_truth_model: f*(x) model
        """
        self.f = pred_model
        self.u = uncert_model
        self.h = prop_model
        self.lam = lam
        self.eta = eta
        self.mask =  mask
        self.z_score = z_score
        self.fstar = ground_truth_model #for evaluation only


    def train(self, x, y, num_cycles=10, init=True,
              random_state=None, verbose=False):
        """ Trains the prediction model, uncertainty model and the propensity weights scoring models
        in an interative fashion """
        self.seed = random_state
        n = x.shape[0]
        d = x.shape[1]

        # initialize:
        if init:
            vprint(verbose,'t:', 0)
            metrics0 = self.f.fit_init(x, y)
            vprint(verbose,'[f] mse: {:.4f}, la_reg: {:.4f}, norm_reg: {:.4f}, obj: {:.4f}'.format(*metrics0))
            if self.fstar is not None:
                imprv0 = self.improve(x, y)
                vprint(verbose,'[f] improve*: {:.3f}'.format(imprv0))
            vprint(verbose, '')
                
        # run cycles:
        metrics_f_t = []
        metrics_u_t = []
        for t in range(num_cycles):
            vprint(verbose,'t:', t+1)
            
            # estimate weights:
            xp = self.move_points(x, self.eta, self.mask)
            self.h.fit(x, xp)
            w = self.h.predict(x)
            vprint(verbose,'[h] n_eff: {:.2f}, w_sum: {:.2f}'.format(*self.h.diagnostics(w)))
            # print('w:', w)

            # train interval model:
            metrics_u_t = self.u.fit(x, y, w, random_state=random_state)
            vprint(verbose,'[u] loss: {:.4f}, norm_reg: {:.4f}, obj: {:.4f}'.format(*metrics_u_t))
            # vprint(verbose,'[u] loss: {:.4f}, sz_reg: {:.4f}, norm_reg: {:.4f}, obj: {:.4f}'.format(*metrics_u_t))
            cntn, intr_sz = self.contain(xp)
            vprint(verbose,'[u] size: {:.3f}, contain*: {:.3f}'.format(intr_sz, cntn))
            
            # train predictive model:
            metrics_f_t = self.f.fit(x, y, lam=self.lam, eta=self.eta,
                                     mask=self.mask, z_score=self.z_score,
                                     uncert_model=self.u)
            vprint(verbose,'[f] mse: {:.4f}, la_reg: {:.4f}, norm_reg: {:.4f}, obj: {:.4f}'.format(*metrics_f_t))

            # evaluate:
            if self.fstar is not None:
                impr = self.improve(x, y)
                vprint(verbose,'[f] improve*: {:.3f}'.format(impr))
            vprint(verbose, '')
        return metrics_f_t, metrics_u_t

    def move_points(self, x, eta=None, mask=None):
        """ Compute decisions for improvement"""
        if eta is None:
            eta = self.eta
        if mask is None:
            mask = self.mask
        return self.f.move_points(x, eta, mask)

    def mse(self, x, y):
        """ Computes the MSE Loss"""
        yhat = self.f.predict(x)
        return np.mean(np.square(y-yhat))

    def mae(self, x, y):
        """ Computes the MAE Loss"""
        yhat = self.f.predict(x)
        return np.mean(np.abs(y-yhat))

    def improve(self, x, y, eta=None, mask=None):
        """ Computes improvement"""
        assert(self.fstar is not None)
        xp = self.move_points(x, eta, mask)
        yp = self.fstar.predict(xp)
        return np.mean(yp-y)
    
    def improve_rate(self, x, y, eta=None, mask=None):
        """ Computes improvement rate"""
        xp = self.move_points(x ,eta, mask)
        yp = self.fstar.predict(xp)
        return np.mean(yp>y)

    def contain(self, x):
        """ Computes the percentage of times x lies in the uncerntainty interval
        and also computes the interval size """
        
        lb, ub = self.u.predict(x)
        intrvl_sz = np.mean(ub-lb)
        if self.fstar is not None:
            y = self.fstar.predict(x)
            contain_ = np.mean(np.logical_and(lb<=y, y<=ub))
        else:
            contain_ = np.nan
        return contain_, intrvl_sz

      

