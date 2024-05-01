import torch

VALUES = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
ENCODED_VALUES = [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 2, 3] #0 means unknown
SCORES = [-2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 0, 1]
SUITS = ["H", "D", "C", "S"]
UNKNOWN = "?"

class Card:
    def __init__(self, value, suit):
        self.value = value
        self.suit = suit
        self.known = 0

    def show(self):
        self.known = 1
        
    def hide(self):
        self.known = 0

    def serialize(self):
        if self.known:
            return self.value
        else:
            return UNKNOWN

    def one_hot(self):
        index = VALUES.index(self.value)
        out = torch.zeros(len(VALUES))
        out[index] = 1
        return out

    def encode(self):
        if self.known:
            return torch.tensor([ENCODED_VALUES[VALUES.index(self.value)]])
        else:
            return torch.tensor([0])

    def score(self):
        return Card.static_score(self.value)
    
    @staticmethod
    def decode(code):
        return VALUES[ENCODED_VALUES.index(code)]
    
    @staticmethod
    def static_score(value):
        return SCORES[VALUES.index(value)]
        
