#!/usr/bin/env python
import re
import sys
import textwrap
from functools import partial

import arxiv  # type: ignore [import]

"""
Entries created by this script reference an yaml anchor for the 'parent' field.
Insert the anchor like this, either on an existing or as a standalone entry:

arxiv: &arxiv
  type: repository
  title: arXiv
"""

arxiv_id_regex = re.compile(r"(\d{4}\.\d{5})(v\d+)?")


def id_from_result(result: arxiv.Result) -> str:
    return arxiv_id_regex.search(result.entry_id)[1]


def hayagriva_from_result(result: arxiv.Result) -> str:
    # pyYAML sucks for this application. I want:
    # - ordered fields
    # - an anchor reference
    # - minimal / no quoting on values
    data = dict(
        type="article",
        title=result.title,
        date=result.published.date().isoformat(),
        author=textwrap.indent(
            "".join("\n- " + author.name for author in result.authors), "  "
        ),
        parent="*arxiv",  # references an arxiv anchor
        doi=result.doi.lstrip("https://doi.org/")
        if result.doi
        else "10.48550/arXiv." + id_from_result(result),
    )
    lines = [f"arxiv{id_from_result(result)}:"]
    for k, v in data.items():
        lines.append(textwrap.indent(k + ":" + " " * (not v.startswith("n")) + v, "  "))
    return "\n".join(lines)


if __name__ == "__main__":
    eprint = partial(print, file=sys.stderr)
    args = sys.argv[1:]
    if len(args) == 0:
        eprint(
            "Needs arXiv identifiers to fetch bibliography information from!\n"
            "URLs or pdf filename with valid arxiv identifiers are also accepted."
        )
        sys.exit(1)
    ids: list[str] = []
    eprint("Extracting arXiv ids:")
    for arg in args:
        eprint(f"'{arg}'", end=": ")
        if match := arxiv_id_regex.search(arg):
            arxiv_id = match[0]
            ids.append(arxiv_id)
            eprint(arxiv_id)
        else:
            eprint("No arXiv id found")

    for result in arxiv.Search(id_list=ids).results():
        print("\n" + hayagriva_from_result(result))
