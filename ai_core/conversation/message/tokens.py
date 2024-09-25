from dataclasses import dataclass


@dataclass
class TokensUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
