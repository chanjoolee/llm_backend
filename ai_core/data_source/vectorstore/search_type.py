from typing import Any

from pydantic import BaseModel, Field


class SearchType(BaseModel):
    name: str = None


class Similarity(SearchType):
    name: str = "similarity"
    k: int = 4
    score_threshold: float = 0.0
    search_kwargs: dict = Field(default_factory=dict)
    _default_opensearch_search_args = {
        "approximate_search": {
           "search_type": "approximate_search",
           "boolean_filter": {"match_all": {}},
           "subquery_clause": "must",
           "efficient_filter": {}
        },
        "script_scoring": {
            "search_type": "script_scoring",
            "space_type": "cosinesimil", # "l2", "l1", "linf", "cosinesimil", "innerproduct" "hammingbit"
            "pre_filter": {"match_all": {}}
        },
        "painless_scripting": {
            "search_type": "painless_scripting",
            "space_type": "cosineSimilarity", # "l2Squared", "l1Norm", "cosineSimilarity"
            "pre_filter": {"match_all": {}}
        }
    }

    def model_post_init(self, __context: Any) -> None:
        self.search_kwargs["k"] = self.k
        self.search_kwargs["score_threshold"] = self.score_threshold
        self.search_kwargs = self._get_merged_search_type_args(self.search_kwargs)

    def _get_merged_search_type_args(self, search_kwargs):
        default_search_type = "script_scoring"
        opensearch_search_type =search_kwargs.get("search_type", default_search_type)
        return {**self._default_opensearch_search_args.get(opensearch_search_type, default_search_type), **search_kwargs}

    def __str__(self):
        return f"Similarity(k={self.k}, score_threshold={self.score_threshold}, search_kwargs={self.search_kwargs})"


class SimilarityScoreThreshold(Similarity):
    name: str = "similarity_score_threshold"


class MMR(SearchType):
    name: str = "mmr"
    k: int = 4
    fetch_k: int = 20
    lambda_mult: float = 0.5
    search_kwargs: dict = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        self.search_kwargs = {
            "k": self.k,
            "fetch_k": self.fetch_k,
            "lambda_mult": self.lambda_mult
        }
