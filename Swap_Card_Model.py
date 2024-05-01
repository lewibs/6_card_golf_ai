import torch
import torch.nn as nn
import torch.nn.functional as F
from Card_Predictor_Model import card_predictor
from Card import ENCODED_VALUES, Card

SWAP_CARD_WEIGHTS = "./weights/Swap_Card.pth"

class SwapCardModel(nn.Module):
    def __init__(self):
        super(SwapCardModel, self).__init__()
        self.card_predictor = card_predictor
        # Define additional layers for combining inputs and CardPredictor output
        # encoded values, deck, card
        self.fc1 = nn.Linear(len(ENCODED_VALUES) + 6 + 1, 100)
        self.fc2 = nn.Linear(100, 32)
        self.fc3 = nn.Linear(32, 6)  # Output size is 6 for optimal card picking

    def forward(self, game_encoded):
        card_predictions = self.card_predictor(game_encoded[:52])
        own_deck = game_encoded[:6]
        current_card = torch.tensor([game_encoded[-1]])
        combined_inputs = torch.cat((card_predictions, own_deck, current_card))

        x = self.fc1(combined_inputs)
        x = self.fc2(x)
        x = F.relu(self.fc3(x))
        mask = torch.where(own_deck != 0, torch.tensor(0), torch.tensor(1))
        x = x * mask
        x = x + mask

        pairs = []


        if Card.static_score(Card.decode(int(current_card.item()))) >= 0:
            for i in range(3):
                a = int(own_deck[i].item())
                b = int(own_deck[i+3].item())
                c = int(current_card.item())

                if a != 0 and b == 0:
                    if a == c:
                        pairs.append(i+3)
                elif a == 0 and b != 0:
                    if b == c:
                        pairs.append(i)

        mask = torch.zeros(6)  # Assuming a tensor with 6 columns

        for p in pairs:
            mask[p] = 1

        x = x * mask
        x = x + mask

        return x