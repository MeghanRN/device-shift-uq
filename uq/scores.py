import numpy as np
from scipy.special import logsumexp

def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def uq_single_from_logits(logits):
    p = softmax(logits)
    top2 = np.partition(p, -2, axis=1)[:, -2:]
    return {
        "msp": 1.0 - p.max(axis=1),
        "entropy": -(p*np.log(p+1e-12)).sum(axis=1),
        "margin": 1.0 - (top2[:,1]-top2[:,0]),
        "energy": -logsumexp(logits, axis=1)
    }

def uq_multilabel_from_logits(logits):
    p = 1/(1+np.exp(-logits))
    conf = np.max(np.maximum(p, 1-p), axis=1)
    ent = -(p*np.log(p+1e-12) + (1-p)*np.log(1-p+1e-12))
    c = np.maximum(p, 1-p)
    top2 = np.partition(c, -2, axis=1)[:, -2:]
    return {
        "msp": 1.0-conf,
        "entropy": ent.mean(axis=1),
        "margin": 1.0-(top2[:,1]-top2[:,0]),
        "energy": -logsumexp(np.abs(logits), axis=1)
    }
