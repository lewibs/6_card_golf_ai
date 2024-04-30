from Player import Player
from Game import Draw_Action, Swap_Action
import random
import torch
from Draw_Action_Model import DrawActionModel, DRAW_ACTION_WEIGHTS

class AI_Player(Player):

    def __init__(self, id, game):
        super().__init__(id, game)

        self.draw_action = DrawActionModel()

        try:
            self.draw_action.load_state_dict(torch.load(DRAW_ACTION_WEIGHTS))
        except:
            print("[WARNING] Unable to load weights for drawing action")

    def draw_card(self, prediction=torch.zeros(1)):
        prediction[0] = self.draw_action(self.game.encode()).item()
        
        if prediction < 0:
            return Draw_Action.RANDOM
        else:
            return Draw_Action.KNOWN


    def swap_card(self, card, prediction=None):
        indices_of_unknown = [index for index, obj in enumerate(self.game.hands[self.id]) if not obj.known]
        rand = random.randint(0,len(indices_of_unknown)-1)
        index = indices_of_unknown[rand]
        return index

    def show_card(self, prediction=None):
        indices_of_unknown = [index for index, obj in enumerate(self.game.hands[self.id]) if not obj.known]
        rand = random.randint(0,len(indices_of_unknown)-1)
        index = indices_of_unknown[rand]
        return index

    def swap_or_flip(self, card, prediction=None):
        if random.randint(0,1):
            return Swap_Action.SWAP
        else:
            return Swap_Action.FLIP