from Player import Player
import random
from Game import Draw_Action, Swap_Action

class User_Player(Player):
    def draw_card(self):
        card = self.game.deck.discard_pile[-1].serialize()
        if int(input(f"0 to use the {card}, or 1 to draw from the deck: ")):
            return Draw_Action.RANDOM
        else:
            return Draw_Action.KNOWN

    def swap_card(self, card):
        while True:
            index = int(input("Enter number 1-6: ")) - 1

            # Check if index is a valid integer between 1 and 6
            if index < 0 or index > 5:
                print("Invalid input. Please enter a number between 1 and 6.")
                continue  # Ask the user for input again
            
            indices_of_unknown = [i for i, obj in enumerate(self.game.hands[self.id]) if not obj.known]

            # Check if the index is in indices_of_unknown
            if index in indices_of_unknown:
                return index
            else:
                print("This index corresponds to a known card. Please choose an unknown card.")

    def show_card(self):
        while True:
            index = int(input("Enter number 1-6: ")) - 1

            # Check if index is a valid integer between 1 and 6
            if index < 0 or index > 5:
                print("Invalid input. Please enter a number between 1 and 6.")
                continue  # Ask the user for input again
            
            indices_of_unknown = [i for i, obj in enumerate(self.game.hands[self.id]) if not obj.known]

            # Check if the index is in indices_of_unknown
            if index in indices_of_unknown:
                return index
            else:
                print("This index corresponds to a known card. Please choose an unknown card.")

    def swap_or_flip(self):
        if int(input("1 to swap, or 0 to flip: ")):
            return Swap_Action.SWAP
        else:
            return Swap_Action.FLIP