from abc import abstractmethod

from pydantic import BaseModel


class SearchType(BaseModel):
    name: str = None
    k: int = 4

    @abstractmethod
    def search_kwargs(self):
        pass


class Similarity(SearchType):
    name: str = "similarity"

    def search_kwargs(self):
        return {"k": self.k}


class SimilarityScoreThreshold(SearchType):
    name: str = "similarity_score_threshold"
    score_threshold: float = 0.5

    def search_kwargs(self):
        return {
            "k": self.k,
            "score_threshold": self.score_threshold
        }


class MMR(SearchType):
    name: str = "mmr"
    fetch_k: int = 20
    lambda_mult: float = 0.5

    def search_kwargs(self):
        return {
            "k": self.k,
            "fetch_k": self.fetch_k,
            "lambda_mult": self.lambda_mult
        }
