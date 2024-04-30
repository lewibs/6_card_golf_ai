import torch

VALUES = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUITS = ["H", "D", "C", "S"]
UNKNOWN = "??"

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
            return self.value + self.suit
        else:
            return "??"

    def one_hot(self):
        value = torch.zeros(len(VALUES))
        value[VALUES.index(self.value)] = 1
        suit = torch.zeros(len(SUITS))
        suit[SUITS.index(self.suit)] = 1
        known = torch.zeros([2])
        known = known[self.known] = 1

        return torch.cat((known, suit, value))
        
