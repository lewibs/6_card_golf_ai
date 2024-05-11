import torch
import torch.nn as nn
from Card import ENCODED_VALUES, Card

FLIP_CARD_WEIGHTS = "./weights/Flip_Card.pth"

class FlipCardModel(nn.Module):
    def __init__(self):
        super(FlipCardModel, self).__init__()
        # Define additional layers for combining inputs and CardPredictor output
        # encoded values, deck, card
        self.fc_stack = nn.Sequential(
            nn.Linear(6 * len(ENCODED_VALUES), 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Linear(100, 32),
            nn.ReLU(),
            nn.Linear(32, 6),
            nn.Sigmoid()
        )

    def forward(self, game_encoded):
        own_deck = torch.stack([Card.encode_to_one_hot(c) for c in game_encoded[:6]]).flatten()

        x = self.fc_stack(own_deck)

        mask = torch.where(game_encoded[:6] != 0, torch.tensor(0), torch.tensor(1))
        x = x * mask
        x = x + mask

        return x