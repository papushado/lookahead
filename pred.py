import torch
import numpy as np
from torch.autograd import Variable
from models import *
from utils import *

class PredModel():
    def __init__(self, num_features, model=None, alpha=0., reg_type='none',
                 move_type='scale', lr=0.1, num_iter=50, num_iter_init=1000):
        """
        Implements the prediction model f(x)
        Arguments:
            num_features: Number of input features
            model : Model used for f(x)
            alpha: coefficient of regularization
            reg_type: Choice of regularization (None or l1 or l2)
            move_type: Choice of how decision is made (scale or clamp)
                scale: x' = x + \eta f'(x)
                clamp: x' = x + min(max( f(x), -eta), eta)
            lr: Learning rate
            num_iter: Number of iterations
            num_iter_init: Number of iterations to be trained for in the first cycle
        """
        
        self.d = num_features
        if model is None: # interface: model(inputs), model.parameters()
            self.model = linearRegression(self.d, 1)
        else:
            self.model = model
        self.init_model()
        self.set_norm_reg(alpha, reg_type)
        self.move_type = move_type #{scale, clamp}
        self.lr = lr
        self.num_iter = num_iter
        self.num_iter_init = num_iter_init
        self.mask = None

    def set_norm_reg(self, alpha=0., reg_type='none'):
        """ Sets regularization for the model """
        assert(alpha>=0)
        self.alpha = float(alpha)
        self.reg_type = reg_type #['none','l1','l2']       
        if self.alpha==0. or self.reg_type=='none':
            self.alpha = 0.
            self.reg_type = 'none'
            self.get_norm_reg = null
        elif reg_type=='l1':
            self.get_norm_reg = get_l1_reg
        elif reg_type=='l2':
            self.get_norm_reg = get_l2_reg

    def fit(self, x, y, lam=0, eta=0, mask=None, z_score=0, uncert_model=None):
        """ Train f"""
        return self.fit_(x, y, lam=lam, eta=eta, mask=mask,
                         num_iter=self.num_iter, uncert_model=uncert_model)

    def fit_init(self, x, y):
        """ Train f for the first cycle """
        return self.fit_(x, y, lam=0, eta=0, mask=None, num_iter=self.num_iter_init)

    def fit_(self, x, y, lam, eta, mask=None,
             z_score=1.65, num_iter=None, uncert_model=None):
        assert lam>=0
        lam = float(lam)
        assert eta>=0
        eta = float(eta)
        if lam>0 or eta>0:
            assert uncert_model is not None
        self.lam = lam
        self.eta = eta
        if mask is None:
            mask = np.ones(self.d)
        self.mask = mask
        mask = Variable(torch.from_numpy(mask).float().reshape(1,-1))

        # train:      
        n = x.shape[0]
        inputs = Variable(torch.from_numpy(x)).float().requires_grad_(True)
        labels = Variable(torch.from_numpy(y).float().unsqueeze(1))
        criterion = torch.nn.MSELoss()
        if self.reg_type=='l1':
            optimizer = torch.optim.LBFGS(self.model.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
            # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0)
        for iter in range(num_iter):
            def closure():
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                la_reg = self.get_la_reg(inputs, outputs, labels,
                                         uncert_model, lam, eta) #add lookahead penalty term
                norm_reg = self.get_norm_reg(self.model, self.alpha, n)
                objective = loss + self.alpha*norm_reg + lam*la_reg
                objective.backward(retain_graph=True)
                self.loss, self.la_reg, self.norm_reg, self.obj = loss, la_reg, norm_reg, objective
                return objective
            optimizer.step(closure)
        metrics = [self.loss.item(), 0 if lam<=0 else self.la_reg.item(), 
                    0 if self.alpha<=0 else self.norm_reg.item(), self.obj.item()]
        return metrics

    def get_la_reg(self, inputs, outputs, labels, uncert_model, lam=0., eta=None, mask=None):
        """ Compute the lookahead regularization loss """
        assert lam>=0
        if eta is None:
            eta = self.eta
        if mask is None:
            mask = self.mask
        mask = Variable(torch.from_numpy(mask).float().reshape(1,-1))
        la_reg = 0.
        if lam > 0:
            inputs_prime = self.move_points_torch(inputs, outputs, eta, mask)
            lu, _ = uncert_model.lu(inputs_prime)
            relu = torch.nn.ReLU()
            la_reg = torch.mean(relu(labels - lu))
        return la_reg

    def objective(self, x, y, lam, eta, uncert_model, alpha=None):
        """ 
        Computes the objective that captures the tradeoff 
        between accuracy and improvement in decisions
        """
        if alpha is None:
            alpha = self.alpha
        alpha = float(alpha)
        lam = float(lam)
        eta = float(eta)
        if uncert_model is None:
            uncert_model = self.u
        
        n = x.shape[0]
        inputs = Variable(torch.from_numpy(x)).float().requires_grad_(True)
        labels = Variable(torch.from_numpy(y).float().unsqueeze(1))
        criterion = torch.nn.MSELoss()
        outputs = self.model(inputs)
        loss = criterion(outputs, labels)
        la_reg = 0.
        if lam>0: #add lookahead penalty term
            inputs_prime = self.move_points_torch(inputs, outputs, eta, mask)
            lu, _ = uncert_model.lu(inputs_prime)
            relu = torch.nn.ReLU()
            la_reg = torch.mean(relu(labels - lu))
        norm_reg = self.get_norm_reg(self.model, alpha, n)
        objective = loss + alpha*norm_reg + lam*la_reg
        return float(objective.detach().numpy())


    def move_points(self, x, eta=None, mask=None):
        """ Computes the decision for improvement """
        if eta is None:
            eta = self.eta
        if mask is None:
            mask = self.mask
        mask = Variable(torch.from_numpy(mask).float().reshape(1,-1))
        inputs = Variable(torch.from_numpy(x).float()).requires_grad_(True)
        outputs = self.model(inputs)
        inputs_prime = self.move_points_torch(inputs, outputs, eta, mask).detach().numpy()
        return inputs_prime

    def move_points_torch(self, inputs, outputs, eta, mask):
        """ Torch function for move_points """
        gradspred, = torch.autograd.grad(outputs, inputs,
                        grad_outputs=outputs.data.new(outputs.shape).fill_(1),
                        create_graph=True)
        if self.move_type=='scale':
            inputs_prime = (inputs + eta * gradspred * mask)
        elif self.move_type=='clamp':
            inputs_prime = inputs + torch.clamp(gradspred * mask, min=-eta, max=eta)
        else:
            raise
        return inputs_prime
    
    def init_model(self):
        """" Initialize model """
        params = list(self.model.parameters())
        [torch.nn.init.zeros_(i) for i in params]

    def predict(self, x):
        """ Predicts the outputs """
        with torch.no_grad():
            inputs = Variable(torch.from_numpy(x)).float()
            outputs = self.model(inputs)
            return outputs.numpy().squeeze()