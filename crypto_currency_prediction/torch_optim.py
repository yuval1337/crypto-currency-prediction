"""
# Bundle of PyTorch optimizers

https://pytorch.org/docs/stable/optim.html

- Adadelta        Implements Adadelta algorithm.
- Adagrad         Implements Adagrad algorithm.
- Adam            Implements Adam algorithm.
- AdamW           Implements AdamW algorithm.
- SparseAdam      Implements lazy version of Adam algorithm suitable for sparse tensors.
- Adamax          Implements Adamax algorithm (a variant of Adam based on infinity norm).
- ASGD            Implements Averaged Stochastic Gradient Descent.
- LBFGS           Implements L-BFGS algorithm, heavily inspired by minFunc.
- NAdam           Implements NAdam algorithm.
- RAdam           Implements RAdam algorithm.
- RMSprop         Implements RMSprop algorithm.
- Rprop           Implements the resilient backpropagation algorithm.
- SGD
"""
from torch.optim import (
    Adadelta,
    Adagrad,
    Adam,
    AdamW,
    SparseAdam,
    Adamax,
    ASGD,
    LBFGS,
    NAdam,
    RAdam,
    RMSprop,
    Rprop,
    SGD
)
