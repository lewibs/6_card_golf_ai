from Game import Game, Swap_Action
import torch
import random
from Random_Player import Random_Player
from Swap_Card_Model import SwapCardModel, SWAP_CARD_WEIGHTS
import torch.optim as optim
import math
import numpy as np
from AI_Player import AI_Player
from Card import Card, SUITS
import torch.nn as nn

def swap_reward(game_encoded):
    card = Card(Card.decode(game_encoded[-1]), SUITS[0])
    hands = []

    hand = []
    for c in game_encoded[:6]:
        card = Card(Card.decode(c), SUITS[0])

        if c:
            card.show()
    
        hand.append(card)

    for i in range(6):
        other_i = i - 3 if i >= 3 else i + 3

        if hand[other_i].known == 0 or hand[i].known != 0:
            continue

        hold = hand[i]
        hand[i] = card
        hands.append((i, Game.static_score_hand(hand)))
        hand[i] = hold


    reward = torch.zeros(6)

    #TODO maybe look for a way to make this less greedy?
    if len(hands) > 1 and not all([item[1] == hands[0][1] for item in hands[1:]]):
        hands = sorted(hands, key=lambda x: x[1])
        for score in hands:
            min_score = hands[0][1]
            i = score[0]
            score = score[1]
            reward[i] = 1 / (score - min_score + 1)

    return reward
    
def calculate_swap_loss(prediction, target):
    loss = nn.CrossEntropyLoss()(prediction, target)

    # if loss != 0:
    #     print(loss, prediction, target)
    return loss

def start_training():
    torch.manual_seed(0)
    random.seed(0)

    print("Starting training on Swap_Card_Model:")
    n_games = 1000
    eps_start = 1.0
    eps_end = 0.1
    eps_steps = n_games / 2
    train = 0.80

    print(f"generating {n_games} worth of random data.")
    
    states = []
    players = [Random_Player, Random_Player]

    for step in range(n_games):
        game = Game.initialize(players, False)

        for i in range(len(players)):
            Game.flip_2_cards_step(game, i)

        for i in range(len(players) * 4):
            player = i % len(players)

            card, action = Game.draw_card_step(game, player)


            action = Game.replace_or_flip_step(game, player, card)
            
            if player == 0:
                states.append(game.encode(card))

            if action == Swap_Action.SWAP:
                Game.swap_card_step(game, player, card)
            else:
                Game.flip_card_step(game, player, card)
            
        Game.finalize_game(game)

    print(f"Games generated processing {len(states)} game states")
    model = SwapCardModel()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_split = math.floor(len(states) * train)

    train = states[:train_split]
    validate = states[train_split:]

    for i, state in enumerate(train):
        if i % 100 == 0:
            print(f"{i}/{len(train)} done")

        t = np.clip(i / eps_steps, 0, 1)
        eps = (1-t) * eps_start + t * eps_end

        prediction = model(state)

        #TODO randomness
        # if random.random() < eps:
        #     random_values = torch.FloatTensor(prediction.size()).uniform_(-1, 1)
        #     prediction = torch.where(torch.rand_like(prediction) < eps, random_values, prediction)

        # index = AI_Player.index_from_prediction(prediction)
        target = swap_reward(state)
        loss = calculate_swap_loss(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), SWAP_CARD_WEIGHTS)
    model.eval()

    minimized = 0
    not_minimized = 0
    not_important = 0

    print("Validating actions")
    for i, state in enumerate(validate):
        if i % 100 == 0:
            print(f"{i}/{len(train)} done")

        card = Card(Card.decode(state[-1]), SUITS[0])
        card.show()

        hand = []
        hands = []
        for c in state[:6]:
            temp_card = Card(Card.decode(c), SUITS[0])

            if c:
                temp_card.show()
        
            hand.append(temp_card)
        
        for i in range(6):
            other_i = i - 3 if i >= 3 else i + 3

            if hand[other_i].known == 0 or hand[i].known != 0:
                continue

            hold = hand[i]
            hand[i] = card
            hands.append((i, Game.static_score_hand(hand)))
            hand[i] = hold

        hands = sorted(hands, key=lambda x: x[1])
        index = AI_Player.index_from_prediction(model(state))
        if len(hands) > 1:
            min_score = hands[0][1]
            hand[index] = card
            if min_score < Game.static_score_hand(hand):
                not_minimized += 1
            else:
                minimized += 1
        else:
            not_important += 1

    print(f"Successfully minimized: {minimized}")
    print(f"Failed to minimize: {not_minimized}")
    print(f"No choice turn: {not_important}")

if __name__ == "__main__":
    start_training()