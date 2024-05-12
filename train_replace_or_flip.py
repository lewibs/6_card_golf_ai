import torch
from Game import Game
import torch.optim as optim
from Q_Learn import ReplayMemory
import numpy as np
from Replace_Or_Flip_Model import ReplaceOrFlipModel, REPLACE_OR_FLIP_WEIGHTS
from Card_Predictor_Model import card_predictor
from Game import Game, Swap_Action
from AI_Player import AI_Player
from Card import Card, SUITS, SCORES
import random
import math
from Random_Player import Random_Player

def replace_or_flip_rewarder(action, game_encoded):
    card = game_encoded[-1]
    own_deck = game_encoded[:6]

    pair = 0
    for i in range(6):
        other_i = i - 3 if i >= 3 else i + 3

        if own_deck[other_i] == 0 or own_deck[i] != 0:
            continue

        if own_deck[other_i] == card:
            pair = 1

    if pair:
        if action == Swap_Action.SWAP:
            return torch.tensor([10])
        else:
            return torch.tensor([-10])
    
    probabilities = card_predictor(game_encoded)
    known_value = Card.static_score(Card.decode(card))
    likely_value = torch.sum(torch.tensor(SCORES) * probabilities).item()

    good = True
    if known_value >= likely_value:
        if action == Swap_Action.FLIP:
            good = True
        else:
            good = False
    else:
        if action == Swap_Action.FLIP:
            good = False
        else:
            good = True

    diff = abs(known_value - likely_value)

    if not good:
        diff *= -1

    return torch.tensor([diff])

def calculate_replace_or_flip_loss(prediction, reward):
    # print(prediction, reward * torch.sigmoid(reward * -1))
    reward = reward * torch.sigmoid(reward * -1)
    loss = prediction * reward
    return loss

def start_training():
    # 0
    torch.manual_seed(0)
    random.seed(3)

    print("Starting training on Replace_Or_Flip_Model:")
    n_games = 100
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

            if player == 0:
                states.append(game.encode(card))

            action = Game.replace_or_flip_step(game, player, card)

            if action == Swap_Action.SWAP:
                Game.swap_card_step(game, player, card)
            else:
                Game.flip_card_step(game, player, card)
            
        Game.finalize_game(game)

    print(f"Games generated processing {len(states)} game states")
    model = ReplaceOrFlipModel()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    losses = []

    train_split = math.floor(len(states) * train)

    train = states[:train_split]
    validate = states[train_split:]

    for i, state in enumerate(train):
        if i % 100 == 0:
            print(f"{i}/{len(train)} done")

        t = np.clip(i / eps_steps, 0, 1)
        eps = (1-t) * eps_start + t * eps_end

        prediction = model(state)

        if random.random() < eps:
            random_values = torch.FloatTensor(prediction.size()).uniform_(-1, 1)
            prediction = torch.where(torch.rand_like(prediction) < eps, random_values, prediction)

        draw = torch.tensor([state[-1]])

        hand = []
        for v in state[:6]:
            v = Card.decode(v)
            c = Card(v, SUITS[0])
            if v:
                c.show()
            hand.append(c)

        action = AI_Player.swap_or_flip_prediction_to_action(prediction)
        reward = replace_or_flip_rewarder(action, state)
        loss = calculate_replace_or_flip_loss(prediction, reward)

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), REPLACE_OR_FLIP_WEIGHTS)
    model.eval()

    correct_swap = 0
    incorrect_swap = 0
    correct_flip = 0
    incorrect_flip = 0

    print("Validating actions")
    rewards = []
    for i, state in enumerate(validate):
        if i % 100 == 0:
            print(f"{i}/{len(validate)} done")

        draw = torch.tensor([state[-1]])
        hand = state[:6]

        action = AI_Player.swap_or_flip_prediction_to_action(model(state))

        rewards.append(replace_or_flip_rewarder(action, state))

        probabilities = card_predictor(torch.cat((state, draw)))
        known_value = Card.static_score(Card.decode(draw))
        likely_value = torch.sum(torch.tensor(SCORES) * probabilities).item()

        pair = 0
        for i in range(6):
            other_i = i - 3 if i >= 3 else i + 3

            if hand[other_i] == 0 or hand[i] != 0:
                continue

            if hand[other_i] == draw:
                pair = 1

        if pair:
            if action == Swap_Action.SWAP:
                correct_swap += 1
            else:
                incorrect_flip += 1
        else:
            if known_value >= likely_value:
                if action == Swap_Action.FLIP:
                    correct_flip += 1
                else:
                    incorrect_swap += 1
            else:
                if action == Swap_Action.FLIP:
                    incorrect_flip += 1
                else:
                    correct_swap += 1
        
    print(f"Correct swap: {correct_swap}")
    print(f"Incorrect swap: {incorrect_swap}")
    print(f"Correct flip: {correct_flip}")
    print(f"Incorrect flip: {incorrect_flip}")    
    print(sum(rewards))        
    
if __name__ == "__main__":
    start_training()