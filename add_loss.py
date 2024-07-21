from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import Tensor, nn
import numpy as np 
from enum import Enum
from torch.nn import functional as F
# from sentence_transformers import util
# from sentence_transformers.SentenceTransformer import SentenceTransformer

def _convert_to_tensor(a: list | np.ndarray | Tensor) -> Tensor:
    """
    Converts the input `a` to a PyTorch tensor if it is not already a tensor.

    Args:
        a (Union[list, np.ndarray, Tensor]): The input array or tensor.

    Returns:
        Tensor: The converted tensor.
    """
    if not isinstance(a, Tensor):
        a = torch.tensor(a)
    return a

def _convert_to_batch(a: Tensor) -> Tensor:
    """
    If the tensor `a` is 1-dimensional, it is unsqueezed to add a batch dimension.

    Args:
        a (Tensor): The input tensor.

    Returns:
        Tensor: The tensor with a batch dimension.
    """
    if a.dim() == 1:
        a = a.unsqueeze(0)
    return a


def _convert_to_batch_tensor(a: list | np.ndarray | Tensor) -> Tensor:
    """
    Converts the input data to a tensor with a batch dimension.

    Args:
        a (Union[list, np.ndarray, Tensor]): The input data to be converted.

    Returns:
        Tensor: The converted tensor with a batch dimension.
    """
    a = _convert_to_tensor(a)
    a = _convert_to_batch(a)
    return a

def normalize_embeddings(embeddings: Tensor) -> Tensor:
    """
    Normalizes the embeddings matrix, so that each sentence embedding has unit length.

    Args:
        embeddings (Tensor): The input embeddings matrix.

    Returns:
        Tensor: The normalized embeddings matrix.
    """
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)

def cos_sim(a: list | np.ndarray | Tensor, b: list | np.ndarray | Tensor) -> Tensor:
    """
    Computes the cosine similarity between two tensors.

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Matrix with res[i][j] = cos_sim(a[i], b[j])
    """
    a = _convert_to_batch_tensor(a)
    b = _convert_to_batch_tensor(b)

    a_norm = normalize_embeddings(a)
    b_norm = normalize_embeddings(b)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


class MultipleNegativesRankingLoss(nn.Module):
    def __init__(self, scale: float = 20.0, similarity_fct=cos_sim) -> None:
        """
        This loss expects as input a batch consisting of sentence pairs ``(a_1, p_1), (a_2, p_2)..., (a_n, p_n)``
        where we assume that ``(a_i, p_i)`` are a positive pair and ``(a_i, p_j)`` for ``i != j`` a negative pair.

        For each ``a_i``, it uses all other ``p_j`` as negative samples, i.e., for ``a_i``, we have 1 positive example
        (``p_i``) and ``n-1`` negative examples (``p_j``). It then minimizes the negative log-likehood for softmax
        normalized scores.

        This loss function works great to train embeddings for retrieval setups where you have positive pairs
        (e.g. (query, relevant_doc)) as it will sample in each batch ``n-1`` negative docs randomly.

        The performance usually increases with increasing batch sizes.

        You can also provide one or multiple hard negatives per anchor-positive pair by structuring the data like this:
        ``(a_1, p_1, n_1), (a_2, p_2, n_2)``. Then, ``n_1`` is a hard negative for ``(a_1, p_1)``. The loss will use for
        the pair ``(a_i, p_i)`` all ``p_j`` for ``j != i`` and all ``n_j`` as negatives.

        Args:
            model: SentenceTransformer model
            scale: Output of similarity function is multiplied by scale
                value
            similarity_fct: similarity function between sentence
                embeddings. By default, cos_sim. Can also be set to dot
                product (and then set scale to 1)

        References:
            - Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4: https://arxiv.org/pdf/1705.00652.pdf
            - `Training Examples > Natural Language Inference <../../examples/training/nli/README.html>`_
            - `Training Examples > Paraphrase Data <../../examples/training/paraphrases/README.html>`_
            - `Training Examples > Quora Duplicate Questions <../../examples/training/quora_duplicate_questions/README.html>`_
            - `Training Examples > MS MARCO <../../examples/training/ms_marco/README.html>`_
            - `Unsupervised Learning > SimCSE <../../examples/unsupervised_learning/SimCSE/README.html>`_
            - `Unsupervised Learning > GenQ <../../examples/unsupervised_learning/query_generation/README.html>`_

        Requirements:
            1. (anchor, positive) pairs or (anchor, positive, negative) triplets

        Relations:
            - :class:`CachedMultipleNegativesRankingLoss` is equivalent to this loss, but it uses caching that allows for
              much higher batch sizes (and thus better performance) without extra memory usage. However, it requires more
              training time.
            - :class:`MultipleNegativesSymmetricRankingLoss` is equivalent to this loss, but with an additional loss term.
            - :class:`GISTEmbedLoss` is equivalent to this loss, but uses a guide model to guide the in-batch negative
              sample selection. `GISTEmbedLoss` yields a stronger training signal at the cost of some training overhead.

        Inputs:
            +---------------------------------------+--------+
            | Texts                                 | Labels |
            +=======================================+========+
            | (anchor, positive) pairs              | none   |
            +---------------------------------------+--------+
            | (anchor, positive, negative) triplets | none   |
            +---------------------------------------+--------+

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })
                loss = losses.MultipleNegativesRankingLoss(model)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        # self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    # def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    #     """Returns token_embeddings, cls_token"""
    #     trans_features = {"input_ids": features["input_ids"], "attention_mask": features["attention_mask"]}


# khi load input vi du la 1 batch thi input dau vao phai la 1 cai dict giua input dau vao description tuong ung 

#     The embeddings for the anchor sentences are stored in embeddings_a, and the embeddings for the positive sentences are concatenated into embeddings_b.
# The similarity scores between embeddings_a and embeddings_b are computed using the provided similarity function and then scaled.
    def forward(self, embeddings_a, embeddings_b) -> Tensor:
        # reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        # embeddings_a = reps[0]
        # embeddings_b = torch.cat(reps[1:])
        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        # Example a[i] should match with b[i]
        range_labels = torch.arange(0, scores.size(0), device=scores.device)
        return self.cross_entropy_loss(scores, range_labels)

    def get_config_dict(self) -> dict[str, Any]:
        return {"scale": self.scale, "similarity_fct": self.similarity_fct.__name__}



class SiameseDistanceMetric(Enum):
    """The metric for the contrastive loss"""

    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)
    COSINE_DISTANCE = lambda x, y: 1 - F.cosine_similarity(x, y)



class OnlineContrastiveLoss(nn.Module):
    def __init__(
        self,  distance_metric=SiameseDistanceMetric.COSINE_DISTANCE, margin: float = 0.5
    ) -> None:
        """
        This Online Contrastive loss is similar to :class:`ConstrativeLoss`, but it selects hard positive (positives that
        are far apart) and hard negative pairs (negatives that are close) and computes the loss only for these pairs.
        This loss often yields better performances than ContrastiveLoss.

        Args:
            model: SentenceTransformer model
            distance_metric: Function that returns a distance between
                two embeddings. The class SiameseDistanceMetric contains
                pre-defined metrics that can be used
            margin: Negative samples (label == 0) should have a distance
                of at least the margin value.

        References:
            - `Training Examples > Quora Duplicate Questions <../../examples/training/quora_duplicate_questions/README.html>`_

        Requirements:
            1. (anchor, positive/negative) pairs
            2. Data should include hard positives and hard negatives

        Relations:
            - :class:`ContrastiveLoss` is similar, but does not use hard positive and hard negative pairs.
            :class:`OnlineContrastiveLoss` often yields better results.

        Inputs:
            +-----------------------------------------------+------------------------------+
            | Texts                                         | Labels                       |
            +===============================================+==============================+
            | (anchor, positive/negative) pairs             | 1 if positive, 0 if negative |
            +-----------------------------------------------+------------------------------+

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "sentence1": ["It's nice weather outside today.", "He drove to work."],
                    "sentence2": ["It's so sunny.", "She walked to the store."],
                    "label": [1, 0],
                })
                loss = losses.OnlineContrastiveLoss(model)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        # self.model = model
        self.margin = margin
        self.distance_metric = distance_metric

    def forward(self, embeddings_a, embeddings_b, size_average=False) -> Tensor:

        distance_matrix = self.distance_metric(embeddings_a, embeddings_b)
        range_labels = torch.arange(0, distance_matrix.size(0), device=distance_matrix.device)

        total_loss = 0
        for i in range(embeddings_a.size(0)):
            labels = range_labels == i
            negs = distance_matrix[labels == 0]
            poss = distance_matrix[labels == 1]

            # select hard positive and hard negative pairs
            negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
            positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

            positive_loss = positive_pairs.pow(2).sum()
            negative_loss = F.relu(self.margin - negative_pairs).pow(2).sum()
            loss = positive_loss + negative_loss
            total_loss += loss
        return total_loss / embeddings_a.size(0) if size_average else total_loss