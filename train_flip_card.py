from Game import Game, Swap_Action
import torch
import random
from Random_Player import Random_Player
from Flip_Card_Model import FlipCardModel, FLIP_CARD_WEIGHTS
import torch.optim as optim
import math
import numpy as np
from AI_Player import AI_Player
from Card import Card, SUITS, SCORES
import torch.nn as nn
from Card_Predictor_Model import card_predictor
import torch.nn.functional as F
import random

def flip_reward(game_encoded):
    hand = game_encoded[:6]
    hand_mask = torch.where(hand != 0, torch.tensor(1), hand)
    inverted_hand_mask = torch.abs(hand_mask - 1)

    hand_one_hot = torch.stack([Card.encode_to_one_hot(c) for c in hand])
    probabilities = card_predictor(game_encoded)
    likely_value = torch.sum(torch.tensor(SCORES) * probabilities)
    likely_index = int(torch.round(likely_value).item())
    likely_mask = torch.cat([torch.ones(likely_index), torch.zeros(len(SCORES)-likely_index)])

    a = (hand_one_hot + inverted_hand_mask.unsqueeze(1)) * probabilities
    sums = torch.sum(a, dim=1, keepdim=True)
    sums[sums == 0] = 1
    result = torch.div(a,sums)

    rows = []
    for row in result:
        if torch.sum(row * 1) == 1:
            rows.append(row * probabilities)
        else:
            rows.append(row * likely_mask)


    result_trimmed = torch.stack(rows)

    rows = []
    for i in range(3):
        rows.append(torch.sum(result_trimmed[i] + result_trimmed[i+3]))
        rows.append(torch.sum(result_trimmed[i] + result_trimmed[i+3]))

    rows = torch.stack(rows)


    probability_of_good_card = inverted_hand_mask * rows

    return probability_of_good_card

def calculate_flip_loss(prediction, target):
    loss = nn.CrossEntropyLoss()(prediction, target)
    return loss

def start_training():
    torch.manual_seed(0)
    random.seed(0)

    print("Starting training on Flip_Card_Model:")
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
    model = FlipCardModel()
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
        target = flip_reward(state)
        loss = calculate_flip_loss(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), FLIP_CARD_WEIGHTS)
    model.eval()

    score = 0
    max_score = 0

    print("Validate")
    for i, state in enumerate(validate):
        if i % 100 == 0:
            print(f"{i}/{len(train)} done")

        prediction = model(state)
        # print(prediction)
        index = AI_Player.index_from_prediction(prediction)
        target = flip_reward(state)
        # print(target)

        max_score += torch.max(target).item()
        score += prediction[index]

    print(f'Scored probability: {score}')
    print(f"Max probability: {max_score}")

if __name__ == "__main__":
    start_training()
