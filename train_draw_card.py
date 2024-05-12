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
import random
import math

def draw_reward(action, game_encoded):
    PAIR_REWARD = 10

    probabilities = card_predictor(game_encoded)
    top_discard = game_encoded[-1]
    hand = game_encoded[:6]

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
    likely_value = torch.sum(torch.tensor(SCORES) * probabilities).item()

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
    #TODO I dont think this is calculating anything of real value??? It seems to work though not sure what is going on there?
    loss = prediction * reward * -1
    return loss

def start_training():
    # 0
    torch.manual_seed(1)
    random.seed(2)
    print("Starting training on Draw_Action_Model:")
    n_games = 1000
    eps_start = 1.0
    eps_end = 0.01
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
            if player == 0:
                states.append(game.encode())


            card, action = Game.draw_card_step(game, player)

            action = Game.replace_or_flip_step(game, player, card)

            if action == Swap_Action.SWAP:
                Game.swap_card_step(game, player, card)
            else:
                Game.flip_card_step(game, player, card)
            
        Game.finalize_game(game)


    print(f"Games generated processing {len(states)} game states")
    model = DrawActionModel()
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

        reward = draw_reward(action, state)
        loss = calculate_draw_loss(prediction, reward)

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), DRAW_ACTION_WEIGHTS)
    model.eval()

    correct_known = 0
    incorrect_known = 0
    correct_unknown = 0
    incorrect_unknown = 0

    print("Validating actions")
    for i, state in enumerate(validate):
        if i % 100 == 0:
            print(f"{i}/{len(validate)} done")

        draw = Card(Card.decode(state[-1]), SUITS[0])
        draw.show()
        hand = []
        for v in state[:6]:
            v = Card.decode(v)
            c = Card(v, SUITS[0])
            if v:
                c.show()
            hand.append(c)

        action = AI_Player.draw_card_prediction_to_action(model(state))
        for i in range(6):
            other_i = i - 3 if i >= 3 else i + 3
            if (hand[i].encode() == 0 and hand[other_i].encode() == draw.encode()) or (hand[other_i].encode() == 0 and hand[i].encode() == draw.encode()):
                if Draw_Action.KNOWN == action:
                    correct_known += 1
                    break
                else:
                    incorrect_unknown += 1
                    break
            else:
                probabilities = card_predictor(state)

                known_value = draw.score()
                likely_value = torch.sum(torch.tensor(SCORES) * probabilities).item()

                if known_value < likely_value:
                    if Draw_Action.KNOWN == action:
                        correct_known += 1
                        break
                    else:
                        incorrect_unknown += 1
                        break
                else:
                    if Draw_Action.KNOWN == action:
                        incorrect_known += 1
                        break
                    else:
                        correct_unknown += 1
                        break

    print(f"Correct known: {correct_known}")
    print(f"Incorrect known: {incorrect_known}")
    print(f"Correct random: {correct_unknown}")
    print(f"Incorrect random: {incorrect_unknown}")            
    
if __name__ == "__main__":
    start_training()