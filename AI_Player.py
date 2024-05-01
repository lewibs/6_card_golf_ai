from Player import Player
from Game import Draw_Action, Swap_Action
import random
import torch
from Draw_Action_Model import DrawActionModel, DRAW_ACTION_WEIGHTS
from Flip_Card_Model import FlipCardModel, FLIP_CARD_WEIGHTS
from Replace_Or_Flip_Model import ReplaceOrFlipModel, REPLACE_OR_FLIP_WEIGHTS
from Swap_Card_Model import SwapCardModel, SWAP_CARD_WEIGHTS

class AI_Player(Player):

    def __init__(self, id, game):
        super().__init__(id, game)

        self.draw_action = DrawActionModel()
        self.flip_action = FlipCardModel()
        self.replace_or_flip_action = ReplaceOrFlipModel()
        self.swap_card_action = SwapCardModel()

        try:
            self.draw_action.load_state_dict(torch.load(DRAW_ACTION_WEIGHTS))
        except:
            print("[WARNING] Unable to load weights for drawing action")

        try:
            self.replace_or_flip_action.load_state_dict(torch.load(REPLACE_OR_FLIP_WEIGHTS))
        except:
            print("[WARNING] Unable to load weights for replace or flip action")

        try:
            self.flip_action.load_state_dict(torch.load(FLIP_CARD_WEIGHTS))
        except:
            print("[WARNING] Unable to load weights for show card action")

        try:
            self.swap_card_action.load_state_dict(torch.load(SWAP_CARD_WEIGHTS))
        except:
            print("[WARNING] Unable to load weights for swap card action")

    def draw_card(self, prediction=torch.zeros(1)):
        prediction[0] = self.draw_action(self.game.encode()).item()
        
        if prediction < 0:
            return Draw_Action.RANDOM
        else:
            return Draw_Action.KNOWN


    def swap_card(self, card, prediction=None):
        pred = self.swap_card_action(self.game.encode(), card.encode())
        for i, item in enumerate(pred):
            prediction[i] = item
        
        index = torch.argmax(prediction).item()
        print(index)
        return index

    def show_card(self, prediction=torch.zeros(6)):
        pred = self.flip_action(self.game.encode())
        for i, item in enumerate(pred):
            prediction[i] = item
        
        return torch.argmax(prediction).item()

    def swap_or_flip(self, card, prediction=torch.zeros([1])):
        prediction[0] = self.replace_or_flip_action(self.game.encode(), card.encode()).item()

        if prediction > 0:
            return Swap_Action.SWAP
        else:
            return Swap_Action.FLIP