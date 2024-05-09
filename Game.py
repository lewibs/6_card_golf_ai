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
    log=True
    players=[]
    hands=[]

    def __init__(self, players):
        self.players = players

        self.reset()

        self.players = [Player(i, self) for i, Player in enumerate(players)]

    def encode(self, card=None):
        game = torch.tensor([])
        
        for i in range(len(self.hands)):
            game = torch.cat((game, self.encode_hand(i)))

        game = torch.cat((game, self.deck.encode()))

        if card:
            card = card.encode()
            #add the on deck card to the back so that it is always in a known location. THis may help with ai understanding which card they are swapping things with
            game = torch.cat((game, torch.tensor([card])))

        return game

    def encode_hand(self, player):
        cards = torch.cat([card.encode() for card in self.hands[player]])
        return cards
    
    def serialized_hand(self, player):
        cards = [card.serialize() for card in self.hands[player]]
        return "{:<2} {:<2} {:<2}\n{:<2} {:<2} {:<2}".format(*cards)

    def serialize(self):
        string = ""
        for i in range(len(self.hands), 0, -1):
            string += f"Player {i}:\n"
            string += f"{self.serialized_hand(i-1)}\n\n"

        string += "Deck:\n"
        string += f"{self.deck.serialize()}\n\n"

        # string += "Player 1:\n"
        # string += f"{self.serialized_hand(0)}\n" 

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

    def serialize_scores(self):
        scores = [(player, self.score_hand(player)) for player in range(len(self.hands))]
        scores = sorted(scores, key=lambda x: x[1])

        ret = ""

        for score in scores:
            ret += f"Player {score[0]+1}: {score[1]}\n"

        return ret



    def score_hand(self, player):
        return Game.static_score_hand(self.hands[player])

    @staticmethod
    def static_score_hand(hand):
        score = 0

        for i in range(3):
            a = hand[i]
            b = hand[i+3]

            if a.value == b.value:
                score += 0
            else:
                score += sum([a.score() * a.known, b.score()*b.known])

        return score

    def is_done(self):
        hands = []

        for hand in self.hands:
            hands.append(all(card.known for card in hand))

        return all(hands)

    def reset(self):
        player_count = len(self.players)

        self.deck = Deck()

        self.deck.shuffle()
        self.deck.discard(*self.deck.deal_cards(1))

        # [1,2,3
        # 4,5,6]
        self.hands = [[] for player in range(player_count)]

        for player in range(player_count):
            delt = self.deck.deal_cards(6)
            self.hands[player] = delt



    @staticmethod
    def initialize(players, log=True):
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

        game = Game(players)
        game.log = log

        if game.log:
            print(rules)

        return game

    @staticmethod
    #TODO i dont love this one since its explicitly returning non game info
    #TODO its awful...
    def flip_2_cards_step(game, player, prediction1=torch.zeros(6), prediction2=torch.zeros(6)):
        if game.log:
            print("Flip two cards...")

        index1 = game.players[player].show_card(prediction1)
        game.show_player_card(player, index1)
        state1 = game.encode()
        index2 = game.players[player].show_card(prediction2)
        game.show_player_card(player, index2)
        state2 = game.encode()

        return (index1, index2, state1, state2)


    @staticmethod
    def draw_card_step(game, player, prediction=torch.zeros(1), format_action=None):
        if game.log:
            print(game.serialize())

        if game.log:
            print("Draw a card:")

        action = game.players[player].draw_card(prediction)
        if format_action:
            action = format_action(action, prediction)

        if action == Draw_Action.RANDOM:
            game.draw_card()

        card = game.deck.draw_from_discard(1)[0]
        
        if game.log:
            print(f"The card is a {card.serialize()}\n")

        return card, action

    @staticmethod
    def replace_or_flip_step(game, player, card, prediction=torch.zeros(1), format_action=None):
        if game.log:
            print("Would you like replace a unknown card, or flip a unknown card?")

        action = game.players[player].swap_or_flip(card, prediction)
        if format_action:
            action = format_action(action, prediction)
        
        if game.log:
            print(f"You chose to {action.value}\n")

        return action

    @staticmethod
    def swap_card_step(game, player, card, prediction=torch.zeros(6), format_action=None):
        if game.log:
            print("Which card would you like to swap it with?")
        
        action = game.players[player].swap_card(card, prediction)
        if format_action:
            action = format_action(action, prediction)
        
        if game.log:
            print(f"Swapping card {action}")

        game.swap_player_card(player, action, card)
        return action

    @staticmethod
    def flip_card_step(game, player, card, prediction=torch.zeros(6), format_action=None):
        if game.log:
            print("Which card would you like to flip instead?")
        
        game.deck.discard(card)
        action = game.players[player].show_card(prediction)
        if format_action:
            action = format_action(action, prediction)
        
        if game.log:
            print(f"Flipping card {action}")
        
        game.show_player_card(player, action)
        return action

    @staticmethod
    def finalize_game(game):
        if game.log:
            print(game.serialize())
            print("FINAL SCORES:")
            print(game.serialize_scores())

        return [Game.static_score_hand(hand) for hand in game.hands]

def run_game(players, log = False):
    game = Game.initialize(players, log)

    for i in range(len(players)):
        Game.flip_2_cards_step(game, i)

    for i in range(len(players) * 4):
        player = i % len(players)
        card, action = Game.draw_card_step(game, player)

        action = Game.replace_or_flip_step(game, player, card)

        if action == Swap_Action.SWAP:
            Game.swap_card_step(game, player, card)
        else:
            Game.flip_card_step(game, player, card)

    return Game.finalize_game(game)




        


