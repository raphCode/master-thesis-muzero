#!/usr/bin/env python
import sys
from collections.abc import Iterator, Sequence

import yaml
from yaml import (
    BlockEndToken,
    BlockMappingStartToken,
    BlockSequenceStartToken,
    KeyToken,
    ScalarToken,
    Token,
    YAMLError,
)

key_order = [
    "type",
    "title",
    "issue",
    "volume",
    "edition",
    "value",
    "date",
    "author",
    "publisher",
    "isbn",
    "parent",
    "page-range",
    "url",
    "doi",
    "note",
]


class UnsortedKeyError(Exception):
    pass


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
            if not key_idx > last_key_idx:
                raise UnsortedKeyError(
                    "Line {}: Key out of order: {} (last seen key: {})".format(
                        s.start_mark.line + 1, s.value, key_order[last_key_idx]
                    )
                )
            last_key_idx = key_idx


if __name__ == "__main__":
    filenames = sys.argv[1:]
    if len(filenames) == 0:
        print("Needs yaml bibliography filename(s) to check!")
        sys.exit(2)
    failed = False
    for filename in filenames:  # type: str
        print(f"Checking {filename}... ", end="")
        try:
            with open(filename, "r") as f:
                check_key_order(yaml.scan(f), ignore_levels=2)
        except (YAMLError, UnsortedKeyError) as e:
            print("error:")
            print(e)
            failed = True
        else:
            print("ok")
        sys.exit(int(failed))
