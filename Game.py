from Deck import Deck
import torch
from enum import Enum

class Draw_Action(Enum):
    RANDOM = "random"
    KNOWN = "known"

class Swap_Action(Enum):
    SWAP = "swap"
    FLIP = "flip"

class Game:
    def __init__(self, players):
        self.deck = Deck()

        self.deck.shuffle()
        self.deck.discard(*self.deck.deal_cards(1))

        # [1,2,3
        # 4,5,6]
        self.hands = [[] for player in range(players)]

        for player in range(players):
            delt = self.deck.deal_cards(6)
            self.hands[player] = delt 

    def one_hot_hand(self, player):
        cards = torch.stack([card.one_hot() for card in self.hands[player]])
        return cards
    
    def serialized_hand(self, player):
        cards = [card.serialize() for card in self.hands[player]]
        return "{:<4} {:<4} {:<4}\n{:<4} {:<4} {:<4}".format(*cards)

    def serialize(self):
        string = ""
        for i in range(len(self.hands), 1, -1):
            string += f"Player {i}:\n"
            string += f"{self.serialized_hand(i-1)}\n\n"

        string += "Deck:\n"
        string += f"{self.deck.serialize()}\n\n"

        string += "You:\n"
        string += f"{self.serialized_hand(0)}\n" 

        return string

    def show_player_card(self, player, index):
        self.hands[player][index].show()

    def swap_player_card(self, player, index, card):
        self.deck.discard(self.hands[player][index])
        self.hands[player][index] = card
        card.show()

    #this flips a unknown card into discard
    def draw_card(self):
        self.deck.discard(*self.deck.deal_cards())

def run_game(players):
    rules = """
        Ready to play 6 card golf?

        THE PACK
        Standard 52 card deck

        THE DEAL
        Each player is dealt 6 cards face down from the deck. The remainder of the cards are placed face down, and the top card is turned up to start the discard pile beside it. Players arrange their 6 cards in 2 rows of 3 in front of them and turn 2 of these cards face up. The remaining cards stay face down and cannot be looked at.

        THE PLAY
        The object is for players to have the lowest value of the cards in front of them by either swapping them for lesser value cards or by pairing them up with cards of equal rank.

        Beginning with the player to the dealer's left, players take turns drawing single cards from either the stock or discard piles. The drawn card may either be swapped for one of that player's 6 cards, or discarded. If the card is swapped for one of the face down cards, the card swapped in remains face up. The round ends when all of a player's cards are face-up.

        A game is nine "holes" (deals), and the player with the lowest total score is the winner.

        SCORING
        Each ace counts 1 point.
        Each 2 counts minus 2 points.
        Each numeral card from 3 to 10 scores face value.
        Each jack or queen scores 10 points.
        Each king scores zero points.
        A pair of equal cards in the same column scores zero points for the column (even if the equal cards are 2s).
    """

    print(rules)
    
    game = Game(len(players))

    players = [Player(i, game) for i, Player in enumerate(players)]

    print("Flip two cards...")

    for i in range(len(players)):
        game.show_player_card(i, players[i].show_card())
        game.show_player_card(i, players[i].show_card())

    print(game.serialize())

    for i in range(len(players) * 4):
        print(game.serialize())
        player = i % len(players)

        print("Draw a card:")
        action = players[player].draw_card()

        if action == Draw_Action.RANDOM:
            game.draw_card()

        card = game.deck.draw_from_discard(1)[0]
        
        print(f"The card is a {card.serialize()}\n")

        print("Would you like replace a unknown card, or flip a unknown card?")
        action = players[player].swap_or_flip()
        print(f"You chose to {action.value}\n")

        if action == Swap_Action.SWAP:
            print("Which card would you like to swap it with?")
            index = players[player].swap_card()
            print(f"Swapping card {index}")
            game.swap_player_card(player, index, card)
        else:
            print("Which card would you like to flip instead?")
            game.deck.discard(card)
            index = players[player].show_card()
            print(f"Flipping card {index}")
            game.show_player_card(player, index)

    print(game.serialize())



        


