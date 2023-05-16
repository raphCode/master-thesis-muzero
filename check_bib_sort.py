from collections.abc import Iterator, Sequence

import yaml
from yaml import (
    BlockEndToken,
    BlockMappingStartToken,
    BlockSequenceStartToken,
    KeyToken,
    ScalarToken,
    Token,
)

key_order = [
    "type",
    "title",
    "issue",
    "volume",
    "edition",
    "date",
    "author",
    "publisher",
    "parent",
    "page-range",
    "url",
    "doi",
]


def check_key_order(
    tokens: Iterator[Token],
    ignore_levels: int = 0,
    key_order: Sequence[str] = key_order,
) -> None:
    last_key_idx = -1
    for t in tokens:
        if isinstance(t, BlockEndToken):
            return
        elif isinstance(t, BlockSequenceStartToken | BlockMappingStartToken):
            check_key_order(tokens, ignore_levels - 1, key_order)
        elif isinstance(t, KeyToken) and ignore_levels <= 0:
            s = next(tokens)
            assert isinstance(s, ScalarToken)
            key_idx = key_order.index(s.value)
            assert (
                key_idx > last_key_idx
            ), "Line {}: Key out of order: {} (last seen key: {})".format(
                s.start_mark.line + 1, s.value, key_order[last_key_idx]
            )
            last_key_idx = key_idx


if __name__ == "__main__":
    with open("bibliography.yml", "r") as f:
        check_key_order(yaml.scan(f), ignore_levels=2)
    print("all keys are correctly sorted!")
