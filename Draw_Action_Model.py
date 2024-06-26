import torch
import torch.nn as nn
import torch.nn.functional as F
from Card_Predictor_Model import card_predictor
from Card import ENCODED_VALUES, Card, SCORES

DRAW_ACTION_WEIGHTS = "./weights/Draw_Action.pth"

class DrawActionModel(nn.Module):
    def __init__(self):
        super(DrawActionModel, self).__init__()
        self.card_predictor = card_predictor
        # Define additional layers for combining inputs and CardPredictor output
        self.fc_stack = nn.Sequential(
            #         probable_next_val, hand, on deck
            nn.Linear((1 * len(ENCODED_VALUES)) + (6 * len(ENCODED_VALUES)) + (1 * len(ENCODED_VALUES)), 30),
            nn.ReLU(),
            nn.Linear(30, 15),
            nn.ReLU(),
            nn.Linear(15, 1),
            nn.Tanh(),
        )

    def forward(self, game_encoded):
        card_predictions = self.card_predictor(game_encoded[:52]).unsqueeze(0)
        own_deck = torch.stack([Card.encode_to_one_hot(c) for c in game_encoded[:6]])
        current_card = Card.encode_to_one_hot(game_encoded[-1]).unsqueeze(0)

        # likely_value = torch.tensor([torch.sum(torch.tensor(SCORES) * card_predictions).item()])

        combined_inputs = torch.cat((card_predictions, own_deck, current_card)).flatten()
        # combined_inputs = torch.cat((likely_value, combined_inputs))

        x = self.fc_stack(combined_inputs)

        return x