

"""
Given a binary search tree, find the in order successor of a given node

Ex.

        100
         / \
        50   200
       / \    /  \
      25  55 150  220
     /      \      /
    10      60    210

"""
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


class Solution:
    def __init__(self, root):
        self.root = root

    def in_order_successor(self, node):
        if not node:
            return None

        if node.right:
            successor = node.right
            while successor:
                successor = successor.left

            return successor

        succcesor = None
        ancesstor = self.root

        while ancesstor != node:
            if ancesstor.val < node.val:
                ancesstor = ancesstor.right
            else:
                successor = ancesstor
                ancesstor = ancesstor.left

        return successor





