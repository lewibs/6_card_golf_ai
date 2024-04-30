import random
from collections import namedtuple
import torch
from Card import Card, SUITS, VALUES, UNKNOWN

class Deck:
    def __init__(self):
        self.deck = []
        self.discard_pile = []

        # Initialize the deck
        for suit in SUITS:
            for value in VALUES:
                self.deck.append(Card(value, suit))

        self.shuffle()

    def shuffle(self):
        # Shuffle the deck
        random.shuffle(self.deck)

    def deal_cards(self, amount=1):
        # Deal a single card from the top of the deck
        if len(self.deck) == 0:
            return None
        else:
            cards = self.deck[-amount:]
            self.deck = self.deck[:-amount]
            return cards

    def reset_deck(self):
        # Reset the deck to its initial state
        self.deck = []
        for suit in self.suits:
            for value in self.values:
                self.deck.append(Card(value, suit))

    def discard(self, *cards):
        for card in cards:
            card.show()
            
        self.discard_pile += cards

    def serialize(self):
        return "{:<2} {:<2}".format(self.discard_pile[-1].serialize(), UNKNOWN)

    def draw_from_discard(self, amount=1):
        drawn_cards = []
        for _ in range(amount):
            drawn_cards.append(self.discard_pile.pop())
        return drawn_cards

    def encode(self):
        if not self.discard_pile:
            discarded = torch.zeros((0,))  # Return an empty tensor if discard pile is empty
        else:
            discarded = torch.cat([card.encode() for card in self.discard_pile])
        unknown = torch.zeros(len(self.deck))
        return torch.cat((discarded, unknown))

    def __len__(self):
        return len(self.deck)
