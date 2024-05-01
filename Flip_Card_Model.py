import torch
import torch.nn as nn
import torch.nn.functional as F
from Card_Predictor_Model import card_predictor
from Card import ENCODED_VALUES

FLIP_CARD_WEIGHTS = "./weights/Flip_Card.pth"

class FlipCardModel(nn.Module):
    def __init__(self):
        super(FlipCardModel, self).__init__()
        self.card_predictor = card_predictor
        # Define additional layers for combining inputs and CardPredictor output
        # encoded values, deck, card
        self.fc1 = nn.Linear(len(ENCODED_VALUES) + 6, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 6)  # Output size is 6 for optimal card picking

    def forward(self, game_encoded):
        card_predictions = self.card_predictor(game_encoded)
        own_deck = game_encoded[:6]
        combined_inputs = torch.cat((card_predictions, own_deck))

        x = self.fc1(combined_inputs)
        x = self.fc2(x)
        x = F.relu(self.fc3(x))
        mask = torch.where(own_deck != 0, torch.tensor(0), torch.tensor(1))
        x = x * mask
        x = x + mask
        return x