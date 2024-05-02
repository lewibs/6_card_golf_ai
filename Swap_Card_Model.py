import torch
import torch.nn as nn
import torch.nn.functional as F
from Card_Predictor_Model import card_predictor
from Card import ENCODED_VALUES, Card

correct = 1
wrong = 1

SWAP_CARD_WEIGHTS = "./weights/Swap_Card.pth"

class SwapCardModel(nn.Module):
    def __init__(self):
        super(SwapCardModel, self).__init__()
        self.card_predictor = card_predictor
        # Define additional layers for combining inputs and CardPredictor output
        # encoded values, deck, card
        self.fc_stack = nn.Sequential(
            nn.Linear(len(ENCODED_VALUES) + (6 * len(ENCODED_VALUES)) + (1 * len(ENCODED_VALUES)), 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Linear(100, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )

    def forward(self, game_encoded):
        card_predictions = self.card_predictor(game_encoded[:52]).unsqueeze(0)
        own_deck = torch.stack([Card.encode_to_one_hot(c) for c in game_encoded[:6]])
        current_card = Card.encode_to_one_hot(game_encoded[-1]).unsqueeze(0)
        combined_inputs = torch.cat((card_predictions, own_deck, current_card)).flatten()

        x = self.fc_stack(combined_inputs)

        mask = torch.where(game_encoded[:6] != 0, torch.tensor(0), torch.tensor(1))
        x = x * mask
        x = x + mask

        return x
