from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
class ClippingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, q_low=None, q_high=None):
        self.q_low = q_low
        self.q_high = q_high
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.q_low is None or self.q_high is None:
            raise ValueError("q_low and q_high must be specified manually before using transform().")
        
        clipped = np.clip(X.to_numpy().flatten(), self.q_low, self.q_high)
        return clipped.reshape(-1, 1)
    
def safe_log2(x):
    return np.log2(x + 1e-6)

def safe_sqrt(x):
    return np.sqrt(np.clip(x, a_min=0, a_max=None))

def safe_logit(x, eps=1e-6):
    """log(p / (1 - p)) for probability-like scores (e.g. FIDDLE rescore), kept finite at 0 and 1."""
    p = np.clip(x, eps, 1 - eps)
    return np.log(p / (1 - p))
