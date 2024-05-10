import torch
import torch.nn as nn
import torch.nn.functional as F
from Card_Predictor_Model import card_predictor
from Card import ENCODED_VALUES, Card, SCORES

REPLACE_OR_FLIP_WEIGHTS = "./weights/Replace_Or_Flip_Action.pth"

aditional = 1

class ReplaceOrFlipModel(nn.Module):
    def __init__(self):
        super(ReplaceOrFlipModel, self).__init__()
        self.card_predictor = card_predictor
        # Define additional layers for combining inputs and CardPredictor output
        self.fc_stack = nn.Sequential(
            # (6 * len(ENCODED_VALUES)) + (1 * len(ENCODED_VALUES))
            nn.Linear(1 + 1 + aditional, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
            nn.Tanh(),
        )

    def forward(self, game_encoded):
        card_predictions = self.card_predictor(game_encoded[:52]).unsqueeze(0)
        likely_value = torch.tensor([torch.sum(torch.tensor(SCORES) * card_predictions).item()])
        own_deck = game_encoded[:6]
        current_card = torch.tensor([Card.static_score(Card.decode(game_encoded[-1]))])
        # combined_inputs = torch.cat((own_deck, current_card)).flatten()

        pair = 0
        for i in range(6):
            other_i = i - 3 if i >= 3 else i + 3

            if own_deck[other_i] == 0 or own_deck[i] != 0:
                continue

            if own_deck[other_i] == game_encoded[-1]:
                pair = 1

        combined_inputs = torch.cat((likely_value, current_card, torch.tensor([pair])))

        x = self.fc_stack(combined_inputs)
        
        return x