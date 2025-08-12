import numpy as np
from scipy.stats import norm, logistic
from scipy.optimize import minimize
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.optimize import approx_fprime
import pandas as pd
from tabulate import tabulate
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

class LatentVariableModel(BaseEstimator):
    
    def __init__(self, dist, l1=0.0, l2=0.0, w=0.5, reg=1e-5):
        self.param = None
        self.dist = dist
        self.l1 = l1
        self.l2 = l2
        self.w = w
        self.reg = reg
        self.best_loss = np.inf


    def __calculate_hessian(self, func, param, args, eps=1e-5):
        # Calculate Hessian of the loss using finite difference method
        n = len(param)
        hessian = np.zeros((n, n))
        
        grad = approx_fprime(param, func, eps, *args)
        
        for i in range(n):
            params_eps = np.copy(param)
            params_eps[i] += eps
            grad_eps = approx_fprime(params_eps, func, eps, *args)
            hessian[:, i] = (grad_eps - grad) / eps
        
        return hessian

    def __log_likelihood(self, param, x, y):
        p = param[:-1]
        b = param[-1]
        score = np.dot(x, p) + b
        # The original (unweighted) log-likelihood
        return np.sum( -y*self.dist.logcdf(score, loc=0, scale=1) - (1-y)*self.dist.logcdf(-score, loc=0, scale=1) )

    def __loss(self, param, x, y):
        p = param[:-1]
        b = param[-1]
        score = np.dot(x,p) + b
        diff = -2*self.w*y*self.dist.logcdf(score, loc=0, scale=1) - 2*(1-self.w)*(1-y)*self.dist.logcdf(-score, loc=0, scale=1)
        obj = np.mean(diff) + self.l1*np.linalg.norm(param, ord=1) + self.l2*np.linalg.norm(param, ord=2)
        return obj


    def fit(self, x, y, tol=1e-4, maxiter=20000, method='SLSQP', verbose=False, wald=False):
        if self.l1<0 or self.l2<0:
            raise ValueError('l1 and l2 must be non-negative!')
        if self.w<0 or self.w>1:
            raise ValueError('w should be within 0 and 1!')
        loss_values = []
        param = np.zeros(x.shape[1]+1)
        
        def callback(param):
            # callback to save the best parameters and track the loss values
            current_loss = self.__loss(param, x, y)
            loss_values.append(current_loss)
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.param = param.copy()
                if verbose: print("Loss improved, update best parameters.")
            if verbose: print("Current loss:", current_loss, "Best loss:", self.best_loss)

        
        result = minimize(
            self.__loss, 
            param, 
            args=(x, y), 
            method=method, 
            tol=tol, 
            options={'maxiter':maxiter}, 
            callback=callback
        )

        if verbose:
            print('Optimization result:', result.message)
        res_dict = {
            'result': result,
            'loss': loss_values,
        }

        if wald:
            # Wald test
            hessian_matrix = self.__calculate_hessian(self.__log_likelihood, self.param, args=(x, y))
            self.hessian_inv = np.linalg.inv(hessian_matrix+self.reg)

            se, wald_stats, p_values = self.wald_test()

            wald_df = pd.DataFrame(
                    zip(
                        self.param,
                        se.round(4),
                        wald_stats.round(4),
                        p_values.round(4)
                    ), columns=['coef','std err','z','P>|z|']
            ).sort_values(by='P>|z|', ascending=True).reset_index(inplace=False)

            res_dict['wald_result'] = wald_df

            if verbose:
                print('Wald test summary:')
                print(tabulate(wald_df, headers='keys', tablefmt='psql'))


        return res_dict

    def wald_test(self):
        se = np.sqrt(np.diag(self.hessian_inv))
        wald_stats = self.param / se
        p_values = 2*(1 - norm.cdf(np.abs(wald_stats)))
        return se, wald_stats, p_values
    
    def predict_score(self, x):
        if self.param is None:
            raise NotFittedError("Model is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        if x.shape[1] != len(self.param)-1:
            raise ValueError(f"No. of columns doesn't match the no. of features in training data!")
        
        p = self.param[:-1]
        b = self.param[-1]
        score = np.dot(x,p) + b
        return score
    
    def predict_proba(self, x):
        score = self.predict_score(x)
        proba = self.dist.cdf(score, loc=0, scale=1)
        return np.vstack([1 - proba, proba]).T

    def predict(self, x, thr=0.5):
        prob = self.predict_proba(x)[:,1]
        return (prob>=thr).astype(int)


class logisticModel(LatentVariableModel):
    def __init__(self, l1=0.0, l2=0.0, w=0.5):
        super().__init__(logistic, l1, l2, w)

class probitModel(LatentVariableModel):
    def __init__(self, l1=0.0, l2=0.0, w=0.5):
        super().__init__(norm, l1, l2, w)

class lgbModel(BaseEstimator):
    def __init__(self, eval_metric=f1_score, **kwargs):
        self.model = LGBMClassifier(**kwargs)
        self.eval_metric = eval_metric
    
    def fit(self, x, y, **kwargs):
        self.model.fit(x, y, eval_metric=self.eval_metric, **kwargs)
        return self

    def predict_proba(self, x):
        return self.model.predict_proba(x, verbosity=-1)

    def predict(self, x, thr=0.5):
        prob = self.model.predict_proba(x, verbosity=-1)[:,1]
        return (prob>=thr).astype(int)
