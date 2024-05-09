import torch
from Game import Game
import torch.optim as optim
from Q_Learn import ReplayMemory
import numpy as np
from Draw_Action_Model import DrawActionModel, DRAW_ACTION_WEIGHTS
from Card_Predictor_Model import card_predictor
from Game import Game
from AI_Player import AI_Player
from Random_Player import Random_Player
from Game import Game, Swap_Action, Draw_Action, run_game
import matplotlib.pyplot as plt
from Card import Card, SCORES, SUITS

def draw_reward(action, top_discard, hand, probabilities):
    PAIR_REWARD = 10

    top_discard = top_discard.encode()
    hand = [c.encode() for c in hand]

    for i in range(6):
        other_i = i - 3 if i >= 3 else i + 3
        if hand[i] == 0 and hand[other_i] == top_discard:
            if action == Draw_Action.KNOWN:
                return torch.tensor([PAIR_REWARD])
            else:
                return torch.tensor([-1*PAIR_REWARD])
        elif hand[other_i] == 0 and hand[i] == top_discard:
            if action == Draw_Action.KNOWN:
                return torch.tensor([PAIR_REWARD])
            else:
                return torch.tensor([-1*PAIR_REWARD])

    known_value = Card.static_score(Card.decode(top_discard))
    likely_value = sum(torch.tensor(SCORES) * probabilities).item()

    good = True
    if known_value >= likely_value:
        if action == Draw_Action.KNOWN:
            good = False
        else:
            good = True
    else:
        if action == Draw_Action.KNOWN:
            good = True
        else:
            good = False

    diff = abs(known_value - likely_value)
    if not good:
        diff = diff * -1

    return torch.tensor([diff])

def calculate_draw_loss(prediction, reward):
    loss = prediction * reward * -1
    return loss

def start_training():
    n_games = 500
    # batch_size = 10
    # gamma = 0.99
    # eps_start = 1.0
    # eps_end = 0.1
    # eps_steps = n_games / 2
    # memory = 1000
    
    states = []
    players = [Random_Player, Random_Player]
   
    for step in range(n_games):
        game = Game.initialize(players, False)

        for i in range(len(players)):
            Game.flip_2_cards_step(game, i)

        for i in range(len(players) * 4):
            player = i % len(players)
            if player == 0:
                states.append(game.encode())


            card, action = Game.draw_card_step(game, player)

            action = Game.replace_or_flip_step(game, player, card)

            if action == Swap_Action.SWAP:
                Game.swap_card_step(game, player, card)
            else:
                Game.flip_card_step(game, player, card)
            
        Game.finalize_game(game)

    model = DrawActionModel()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    losses = []

    for state in states:
        prediction = model(state)
        draw = Card(Card.decode(state[-1]), SUITS[0])
        draw.show()

        hand = []
        for v in state[:6]:
            v = Card.decode(v)
            c = Card(v, SUITS[0])
            if v:
                c.show()
            hand.append(c)

        action = AI_Player.draw_card_prediction_to_action(prediction)
        probabilities = card_predictor(state)
        reward = draw_reward(action, draw, hand, probabilities)
        loss = calculate_draw_loss(prediction, reward)

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Plotting both arrays on the same plot
    plt.plot(losses, label='Losses')  # Plotting y1

    # Adding labels and title
    plt.xlabel('Index')  # x-axis label
    plt.ylabel('Loss')  # y-axis label
    plt.title('Plotting Losses')

    # Adding legend
    plt.legend()

    # Displaying the plot
    plt.show()

    model.eval()

    for state in states:
        draw = Card(Card.decode(state[-1]), SUITS[0])
        draw.show()
        hand = []
        for v in state[:6]:
            v = Card.decode(v)
            c = Card(v, SUITS[0])
            if v:
                c.show()
            hand.append(c)

        for i in range(6):
            other_i = i - 3 if i >= 3 else i + 3
            if (hand[i].encode() == 0 and hand[other_i].encode() == draw.encode()) or (hand[other_i].encode() == 0 and hand[i].encode() == draw.encode()):
                print(AI_Player.draw_card_prediction_to_action(model(state)))
                
if __name__ == "__main__":
    start_training()