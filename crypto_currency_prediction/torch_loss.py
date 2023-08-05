"""
# Bundle of PyTorch loss functions

https://pytorch.org/docs/stable/nn.functional.html

* binary_cross_entropy                Function that measures the Binary Cross Entropy between the target and input probabilities.
* binary_cross_entropy_with_logits    Function that measures Binary Cross Entropy between target and input logits.
* poisson_nll_loss                    Poisson negative log likelihood loss.
* cosine_embedding_loss               See CosineEmbeddingLoss for details.
* cross_entropy                       This criterion computes the cross entropy loss between input logits and target.
* ctc_loss                            The Connectionist Temporal Classification loss.
* gaussian_nll_loss                   Gaussian negative log likelihood loss.
* hinge_embedding_loss                See HingeEmbeddingLoss for details.
* kl_div                              The Kullback-Leibler divergence Loss
* l1_loss                             Function that takes the mean element-wise absolute value difference.
* mse_loss                            Measures the element-wise mean squared error.
* margin_ranking_loss                 See MarginRankingLoss for details.
* multilabel_margin_loss              See MultiLabelMarginLoss for details.
* multilabel_soft_margin_loss         See MultiLabelSoftMarginLoss for details.
* multi_margin_loss                   See MultiMarginLoss for details.
* nll_loss                            The negative log likelihood loss.
* huber_loss                          Function that uses a squared term if the absolute element-wise error falls below delta and a delta-scaled L1 term otherwise.
* smooth_l1_loss                      Function that uses a squared term if the absolute element-wise error falls below beta and an L1 term otherwise.
* soft_margin_loss                    See SoftMarginLoss for details.
* triplet_margin_loss                 See TripletMarginLoss for details
* triplet_margin_with_distance_loss   See TripletMarginWithDistanceLoss for details.
"""
from torch.nn.functional import (
    binary_cross_entropy,
    binary_cross_entropy_with_logits,
    poisson_nll_loss,
    cosine_embedding_loss,
    cross_entropy,
    ctc_loss,
    gaussian_nll_loss,
    hinge_embedding_loss,
    kl_div,
    l1_loss,
    mse_loss,
    margin_ranking_loss,
    multilabel_margin_loss,
    multilabel_soft_margin_loss,
    multi_margin_loss,
    nll_loss,
    huber_loss,
    smooth_l1_loss,
    soft_margin_loss,
    triplet_margin_loss,
    triplet_margin_with_distance_loss
)
