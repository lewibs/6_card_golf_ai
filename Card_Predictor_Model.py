import torch
from Card import ENCODED_VALUES

def card_predictor(deck_encoded):
    probs = torch.ones(len(ENCODED_VALUES)) * 4

    if (len(deck_encoded.size()) > 1):
        #if deck encoded was an array of decks I want to call this function again for each deck then return that array again
        prob_list = []
        for deck in deck_encoded:
            prob_list.append(card_predictor(deck))
        
        return torch.stack(prob_list)
    else:
        for value in deck_encoded:
            if value != 0:
                index = ENCODED_VALUES.index(value)
                probs[index] -= 1

        return probs / torch.sum(probs)