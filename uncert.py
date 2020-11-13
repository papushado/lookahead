import copy
import torch
import numpy as np
from torch.autograd import Variable
from utils import *
from models import *

class UncertModel():
    def __init__(self, num_features, model=None):
        """ Class wrapper for the Uncertainty Model"""
        self.d = num_features
        if model is None:
            self.model = linearRegression(self.d, 1)
        else:
            self.model = model

    def eff_num_smpls(self, w):
        """ Computed effective number of samples from propensity scores """
        n = len(w)
        n_eff = (np.sum(w) ** 2) / (np.sum(w ** 2)) #effective sample size
        return np.min([np.max([np.int(np.ceil(n_eff)),2]), n])

    def weighted_mse_loss(self, preds, labels, weights):
        """ Computed MSE loss weighted by propentity weights """
        return (((preds - labels) ** 2)*weights).sum() / weights.sum()

    def predict(self, x):
        """ Predicts the upper bound uncertainty and lower bound uncertainty 
        Arguments:
            x: 2-D numpy Array
            lb: 1-D numpy lower bound scores
            ub: 1-D numpy upper bound scores
        """
        inputs = Variable(torch.from_numpy(x)).float().requires_grad_(True)
        lb, ub = self.lu(inputs)
        lb_, ub_ = lb.detach().numpy(), ub.detach().numpy()
        return lb_.squeeze(), ub_.squeeze()

    
class Bootstrap(UncertModel):
    def __init__(self, num_features, num_gs=10, z_score=1.65,
                 model=None, alpha=0., lr=0.1, num_iter=100):
        """
        Computes Uncertainty using the Bootstrap Model
        Arguments:
            num_features: Number of input features
            num_gs: Number of boostrap models
            z_score: Controls the width of the uncertainty interval
            model: Model of each individual boostrap model
            alpha: Coefficient of l2- regularizer
            lr: learning rate
            num_iter: Number of iterations to train the uncertainty model for
        """
        super().__init__(num_features, model=model)
        self.num_gs = num_gs
        self.gs = []
        for i in range(num_gs):
            self.gs.append(self.G(self.model))
        self.init_models()
        self.alpha = float(alpha)
        self.z_score = z_score
        self.lr = lr
        self.num_iter = num_iter
        self.boot_idx = None
        self.n_boot = None

    class G():
        def __init__(self, model):
            self.model = copy.deepcopy(model)
            self.optim = None
            self.loss = None
            self.objective = None
            self.l2_reg = 0.

        def set(self, optim, loss, objective, l2_reg):
            self.optim = optim
            self.loss = loss
            self.objective = objective
            self.l2_reg = l2_reg

    def fit(self, x, y, w=None, force_all=False, random_state=None):
        """ Trains the Boostrap Model """
        # initialize:
        n = x.shape[0]
        w = np.ones(n) if w is None else w
        self.n_boot = n if force_all else self.eff_num_smpls(w)
        # draw bootstrapped samples (once for all cycles):
        if self.boot_idx is None:
            rng = np.random.RandomState(random_state)
            self.boot_idx = rng.choice(n, (self.num_gs, n), replace=True)
        inputs = Variable(torch.from_numpy(x)).float().requires_grad_(True)
        labels = Variable(torch.from_numpy(y).float().unsqueeze(1))
        self.init_models()
        # train:
        for i in range(self.num_gs):
            g = self.gs[i]
            optim = torch.optim.SGD(g.model.parameters(), lr=self.lr)
            boot_importance = torch.zeros(inputs.shape[0])
            b_i = self.boot_idx[i,:self.n_boot]
            boot_importance[b_i] = torch.from_numpy(w[b_i]).float() #get importance of each point (0 if point not selected)
            for iter in range(self.num_iter):
                optim.zero_grad()
                boot_out = g.model(inputs)
                loss = self.weighted_mse_loss(boot_out, labels, boot_importance.unsqueeze(1))
                l2_reg = get_l2_reg(g.model, self.alpha, n)
                objective = loss + self.alpha * l2_reg
                objective.backward()
                optim.step()
            g.set(optim, loss, objective, l2_reg)
        metrics = [np.mean([g.loss.item() for g in self.gs]),
                   0 if self.alpha==0 else np.mean([g.l2_reg.item() for g in self.gs]),
                   np.mean([g.objective.item() for g in self.gs])]
        return metrics

    def init_models(self):
        """ Initializes the models """
        for g in self.gs:
            params = list(g.model.parameters())
            [torch.nn.init.zeros_(i) for i in params]

    def lu(self, inputs, z_score=None):
        """ Computes lower bound and upper bound uncertainty values """
        if z_score is None:
            z_score = self.z_score
        g_outputs = torch.zeros(self.num_gs, inputs.shape[0])
        for i,g in enumerate(self.gs):
            g_outputs[i] = g.model(inputs).squeeze()
        g_mean = torch.mean(g_outputs, dim=0)
        g_std = torch.std(g_outputs, dim=0)
        g_prime_l = g_mean - self.z_score * g_std
        g_prime_u = g_mean + self.z_score * g_std
        return g_prime_l.unsqueeze(1), g_prime_u.unsqueeze(1)


class BootstrapResid(Bootstrap):
    def __init__(self, num_features, f, num_gs=10, z_score=1.65,
                 model=None, alpha=None, lr=0.1, num_iter=100):
        """
        Computes Uncertainty using the Bootstrap Residual Model
        (Overloads from the Boostrap Model)
        Arguments:
            num_features: Number of input features
            f: Prediction Model f(x)
            num_gs: Number of boostrap models
            z_score: Controls the width of the uncertainty interval
            model: Model of each individual boostrap model
            alpha: Coefficient of l2- regularizer
            lr: learning rate
            num_iter: Number of iterations to train the uncertainty model for
        """
        
        super().__init__(num_features=num_features, num_gs=num_gs, z_score=z_score,
        model=model, alpha=alpha, lr=lr, num_iter=num_iter)
        self.f = f
        self.pseudo_idx = None

    def fit(self, x, y, w=None, force_all=False, random_state=None):
        """ Trains the boostrap residual model """
        # initialize:
        n = x.shape[0]
        w = np.ones(n) if w is None else w
        self.n_boot = n if force_all else self.eff_num_smpls(w)
        if self.boot_idx is None:
            np.random.seed(random_state)
            self.boot_idx = np.random.choice(n, (self.num_gs, n), replace=True)
        if self.pseudo_idx is None: #residual swap indices
            np.random.seed(random_state)
            self.pseudo_idx = np.random.choice(n, (self.num_gs, n), replace=True)
        yhat = self.f.model.predict(x)
        r = y-yhat
        inputs = Variable(torch.from_numpy(x)).float().requires_grad_(True)
        labels = Variable(torch.from_numpy(y).float().unsqueeze(1))
        resids = Variable(torch.from_numpy(r).float().unsqueeze(1))
        self.init_models()
        # train:
        for i in range(self.num_gs):
            g = self.gs[i]
            pseudo_labels = labels + resids[self.pseudo_idx[i]] # add random residuals
            optim = torch.optim.SGD(g.model.parameters(), lr=self.lr)
            w_ = torch.zeros(inputs.shape[0])
            b_i = self.boot_idx[i,:self.n_boot]
            w_[b_i] = torch.from_numpy(w[b_i]).float() #get importance of each point (0 if point not selected)
            w_ = torch.from_numpy(w).float() # use all points
            for iter in range(self.num_iter):
                optim.zero_grad()
                boot_out = g.model(inputs)
                loss = self.weighted_mse_loss(boot_out, pseudo_labels, w_.unsqueeze(1))
                l2_reg = get_l2_reg(g.model, self.alpha, n)
                objective = loss + self.alpha * l2_reg
                objective.backward()
                optim.step()
            g.set(optim, loss, objective, l2_reg)
        metrics = [np.mean([g.loss.item() for g in self.gs]),
                    0 if self.alpha==0 else np.mean([g.l2_reg.item() for g in self.gs]),
                    np.mean([g.objective.item() for g in self.gs])]
        return metrics


class QuantReg(UncertModel):
    def __init__(self, num_features, tau=[0.1, 0.9], model=None, alpha=0., lr=0.1, num_iter=100):
        super().__init__(num_features, model=model)

        """
        Computes Uncertainty using the Quantile Regression Model
        Arguments:
            num_features: Number of input features
            tau: Quantiles 
            model: Model of each individual boostrap model
            alpha: Coefficient of l2- regularizer
            lr: learning rate
            num_iter: Number of iterations to train the uncertainty model for
        """
        
        self.lu_model = self.model
        self.uu_model = copy.deepcopy(self.model)
        self.models = [self.lu_model, self.uu_model]
        self.init_models()
        self.tune_alpha = alpha<0 #TODO implement
        self.alpha = float(alpha)
        self.tau = tau
        self.lr = lr
        self.num_iter = num_iter

    def init_models(self):
        """ Initializes the models """
        for model in self.models:
            params = list(model.parameters())
            [torch.nn.init.zeros_(i) for i in params]


    def weighted_quantile_loss(self, preds, labels, weights, tau):
        """ Computes the quantile loss weighted by the propensity scores """
        error = labels - preds
        loss_ = torch.max((tau-1) * error, tau * error)
        return (loss_*weights).sum() / weights.sum()

    def fit(self, x, y, w=None, random_state = None):
        """ Trains Quantile Regression Model for Uncertainty """
        n = x.shape[0]
        w = np.ones(n) if w is None else w
        inputs = Variable(torch.from_numpy(x)).float().requires_grad_(True)
        labels = Variable(torch.from_numpy(y).float().unsqueeze(1))
        self.init_models()
        # train:
        optimizer = torch.optim.Adam([{'params':self.lu_model.parameters()},
                                         {'params':self.uu_model.parameters()}],
                                        lr=self.lr)
        weights = torch.from_numpy(w).float().unsqueeze(1)

        for iter in range(self.num_iter):

            optimizer.zero_grad()
            lu_preds = self.lu_model(inputs)
            uu_preds = self.uu_model(inputs)

            loss_lu = self.weighted_quantile_loss(lu_preds, labels, weights, self.tau[0])
            loss_uu = self.weighted_quantile_loss(uu_preds, labels, weights, self.tau[1])
            loss = loss_lu + loss_uu

            l2_reg_lu = get_l2_reg(self.lu_model, self.alpha, n)
            l2_reg_uu = get_l2_reg(self.uu_model, self.alpha, n)
            l2_reg = (l2_reg_lu + l2_reg_uu)/2

            objective = loss + self.alpha*l2_reg
            objective.backward()
            optimizer.step()

        metrics = [loss.item(),
                   0 if self.alpha<=0 else l2_reg.item(),
                   objective.item()]

        return metrics

    def lu(self, inputs):
        """ Computes lower bound and upper bound uncertainty values """
        lower = self.lu_model(inputs)
        upper = self.uu_model(inputs)
        return lower, upper