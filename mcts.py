from typing import List

class Node:
    children: List
    #prior: TODO

    def __init__(self, prior):
        self.prior = prior
        self.children = []

    def expanded(self) -> bool:
        return len(self.children) > 0
