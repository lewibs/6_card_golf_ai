import torch
import torch.nn as nn
import torch.nn.functional as F
from Card_Predictor_Model import card_predictor
from Card import ENCODED_VALUES, Card, SCORES

REPLACE_OR_FLIP_WEIGHTS = "./weights/Replace_Or_Flip_Action.pth"

class ReplaceOrFlipModel(nn.Module):
    def __init__(self):
        super(ReplaceOrFlipModel, self).__init__()
        self.card_predictor = card_predictor
        # Define additional layers for combining inputs and CardPredictor output
        self.fc_stack = nn.Sequential(
            # (6 * len(ENCODED_VALUES)) + (1 * len(ENCODED_VALUES))
            nn.Linear(1 + 1, 30),
            nn.ReLU(),
            nn.Linear(30, 15),
            nn.ReLU(),
            nn.Linear(15, 1),
            nn.Tanh(),
        )

    def forward(self, game_encoded):
        card_predictions = self.card_predictor(game_encoded[:52]).unsqueeze(0)
        likely_value = torch.tensor([torch.sum(torch.tensor(SCORES) * card_predictions).item()])
        own_deck = torch.stack([Card.encode_to_one_hot(c) for c in game_encoded[:6]])        
        current_card = torch.tensor([Card.static_score(Card.decode(game_encoded[-1]))])
        # combined_inputs = torch.cat((own_deck, current_card)).flatten()
        combined_inputs = torch.cat((likely_value, current_card))

        x = self.fc_stack(combined_inputs)
        
        return x