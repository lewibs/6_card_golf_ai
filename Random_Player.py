from Player import Player
import random
from Game import Draw_Action, Swap_Action

class Random_Player(Player):
    def draw_card(self):
        if random.randint(0,1):
            return Draw_Action.RANDOM
        else:
            return Draw_Action.KNOWN

    def swap_card(self):
        indices_of_unknown = [index for index, obj in enumerate(self.game.hands[self.id]) if not obj.known]
        rand = random.randint(0,len(indices_of_unknown)-1)
        index = indices_of_unknown[rand]
        return index

    def show_card(self):
        indices_of_unknown = [index for index, obj in enumerate(self.game.hands[self.id]) if not obj.known]
        rand = random.randint(0,len(indices_of_unknown)-1)
        index = indices_of_unknown[rand]
        return index

    def swap_or_flip(self):
        if random.randint(0,1):
            return Swap_Action.SWAP
        else:
            return Swap_Action.FLIP