import torch
import torch.nn as nn
import torch.nn.functional as F
from Card_Predictor_Model import card_predictor
from Card import ENCODED_VALUES

REPLACE_OR_FLIP_WEIGHTS = "./weights/Replace_Or_Flip_Action.pth"

class ReplaceOrFlipModel(nn.Module):
    def __init__(self):
        super(ReplaceOrFlipModel, self).__init__()
        self.card_predictor = card_predictor
        # Define additional layers for combining inputs and CardPredictor output
        self.fc1 = nn.Linear(6 + len(ENCODED_VALUES) + 1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Output size is 1 for the draw action

    def forward(self, game_encoded):
        card_predictions = self.card_predictor(game_encoded[:52])
        own_deck = game_encoded[:6]
        current_card = torch.tensor([game_encoded[-1]])
        combined_inputs = torch.cat((card_predictions, own_deck, current_card))

        x = self.fc1(combined_inputs)
        x = self.fc2(x)
        x = torch.tanh(self.fc3(x))
        
        return x