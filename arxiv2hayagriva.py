#!/usr/bin/env python
import re
import sys
import textwrap
from functools import partial
from contextlib import suppress

import arxiv  # type: ignore [import]

"""
Usage:
./arxiv2hayagriva.py [arxiv ids, urls or pdf filenames] >> bibliography.yml

Hayagriva entries are written to stdout and are appended to the bibliography.

Entries created by this script reference an yaml anchor for the 'parent' field.
Insert the anchor like this, either on an existing entry or as a standalone one:

arxiv: &arxiv
  type: repository
  title: arXiv
"""

eprint = partial(print, file=sys.stderr)

arxiv_id_regex = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


def id_from_result(result: arxiv.Result) -> str:
    return arxiv_id_regex.search(result.entry_id)[1]


def bibkey_arxiv_id(result: arxiv.Result) -> str:
    return "arxiv" + id_from_result(result)


def ask_bibkey(result: arxiv.Result) -> str:
    eprint(f"Enter bibliography key for '{result.title}': ", end="")
    bibkey = ""
    with suppress(EOFError):
        bibkey = input()
    return bibkey or bibkey_arxiv_id(result)


get_bibkey = ask_bibkey


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
    lines = [get_bibkey(result) + ":"]
    for k, v in data.items():
        lines.append(textwrap.indent(k + ":" + " " * (not v.startswith("n")) + v, "  "))
    return "\n".join(lines)


if __name__ == "__main__":
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
