#!/usr/bin/env python
import io
import re
import sys
from contextlib import closing
from typing import Any

import yaml


def get_plural(w: str) -> str:
    if re.search(r"[^aeiou]y$", w):
        return w[:-1] + "ies"
    if re.search(r"(?:ss|ch|sh|x)$", w):
        return w + "es"
    return w + "s"


subs = dict()


def add_mapping(text: str, key: str, key_suffix: str = "") -> None:
    def capitalize(text: str, only_first: bool = False) -> str:
        return re.sub(
            r"\b\w", lambda m: m.group(0).upper(), text, count=int(only_first)
        )

    # Use a custom capitalisation function:
    # - match the behavior of the typst implementation
    # - python's title() and capitalize() methods apply lowercasing
    subs[capitalize(text)] = key.upper() + key_suffix
    subs[capitalize(text, only_first=True)] = key.title() + key_suffix
    subs[text] = (
        key.lower() + key_suffix
    )  # insert original text last so it overrides any capitalization options


with open("abbrev.yaml") as f:
    abbrev = yaml.load(f, yaml.Loader)

for k, v in abbrev.items():
    if isinstance(v, dict):
        s = v["singular"]
        p = v.get("plural", None)
    else:
        s = v
        p = get_plural(s)
    add_mapping(s, k)
    if p is not None:
        add_mapping(p, k, "s")

regex_keys = sorted(subs.keys(), key=len, reverse=True)
sub_re = re.compile(r"\b(?:" + "|".join(regex_keys).replace(" ", r"\s") + r")\b")
def replace_fn(m:re.Match)->str:
    key = re.sub(r"\s", " ", m.group(0))
    return "!"+subs[key]

text = "".join(sys.stdin.readlines())
text = re.sub(sub_re, replace_fn, text)
text=text.replace(". ", ".\n")
sys.stdout.write(text)
