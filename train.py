import torch
from Game import Game
import torch.optim as optim
from Q_Learn import ReplayMemory
import numpy as np
from Draw_Action_Model import DrawActionModel, DRAW_ACTION_WEIGHTS
from Flip_Card_Model import FlipCardModel, FLIP_CARD_WEIGHTS
from Replace_Or_Flip_Model import ReplaceOrFlipModel, REPLACE_OR_FLIP_WEIGHTS
from Swap_Card_Model import SwapCardModel, SWAP_CARD_WEIGHTS
from Card_Predictor_Model import card_predictor
from Game import Game
from AI_Player import AI_Player
from Random_Player import Random_Player
from Game import Game, Swap_Action, Draw_Action, run_game
import random
from Q_Learn import optimize_model
import matplotlib.pyplot as plt
from Card import Card, SCORES
import statistics
from train_draw_card import start_training as train_draw_card, draw_reward
from train_replace_or_flip import start_training as train_replace_or_flip, replace_or_flip_rewarder
from train_swap_card import start_training as train_swap_card, swap_reward
from train_flip_card import start_training as train_flip_card, flip_reward
import random

def format_draw_action(action, eps, prediction=torch.zeros(1)):
    if random.random() < eps:
        if random.randint(0,1):
            prediction[0] = 1
        else:
            prediction[0] = -1

        return AI_Player.draw_card_prediction_to_action(prediction)
    else:
        return action

def format_flip_action(action, eps, prediction=torch.zeros(6)):
    if random.random() < eps:
        vals = torch.tensor([1 if p > 0 else 0 for p in prediction]) * torch.rand(6)

        for i, val in enumerate(vals):
            prediction[i] = val

        action = torch.argmax(vals)
        return action
    else:
        return action

def format_swap_action(action, eps, prediction=torch.zeros(6)):
    if random.random() < eps:
        vals = torch.tensor([1 if p > 0 else 0 for p in prediction]) * torch.rand(6)

        for i, val in enumerate(vals):
            prediction[i] = val

        action = torch.argmax(vals)
        return action
    else:
        return action

def format_replace_or_flip_action(action, eps, prediction=torch.zeros(1)):
    if random.random() < eps:
        if random.randint(0,1):
            prediction[0] = 1
        else:
            prediction[0] = -1

        return AI_Player.swap_or_flip_prediction_to_action(prediction)
    else:
        return action

def train_q_learn():
    n_games = 20
    batch_size = 10
    gamma = 0.99
    eps_start = 0.01
    eps_end = 0.01
    eps_steps = 1
    memory = 1000

    draw_action_losses = []
    flip_card_losses = []
    swap_card_losses = []
    replace_or_flip_losses = []
   
    draw_action_policy_net = DrawActionModel()
    draw_action_target_net = DrawActionModel()
    draw_action_target_net.load_state_dict(draw_action_policy_net.state_dict())
    draw_action_target_net.eval()

    draw_action_optimizer = optim.AdamW(draw_action_policy_net.parameters(), lr=1e-3)
    draw_action_memory = ReplayMemory(memory)

    flip_card_policy_net = FlipCardModel()
    flip_card_target_net = FlipCardModel()
    flip_card_target_net.load_state_dict(flip_card_policy_net.state_dict())
    flip_card_target_net.eval()

    flip_card_optimizer = optim.AdamW(flip_card_policy_net.parameters(), lr=1e-3)
    flip_card_memory = ReplayMemory(memory)

    replace_or_flip_policy_net = ReplaceOrFlipModel()
    replace_or_flip_target_net = ReplaceOrFlipModel()
    replace_or_flip_target_net.load_state_dict(replace_or_flip_policy_net.state_dict())
    replace_or_flip_target_net.eval()

    replace_or_flip_optimizer = optim.AdamW(replace_or_flip_policy_net.parameters(), lr=1e-3)
    replace_or_flip_memory = ReplayMemory(memory)

    swap_card_policy_net = SwapCardModel()
    swap_card_target_net = SwapCardModel()
    swap_card_target_net.load_state_dict(swap_card_policy_net.state_dict())
    swap_card_target_net.eval()

    swap_card_optimizer = optim.AdamW(swap_card_policy_net.parameters(), lr=1e-3)
    swap_card_memory = ReplayMemory(memory)

    #replace the ai of player with the trainable models rather then the base initialization that is used
    def init_ai(*args):
        ai = AI_Player(*args)
        ai.draw_action = draw_action_policy_net
        ai.draw_action.load_state_dict(torch.load(DRAW_ACTION_WEIGHTS))
        ai.flip_card_action = flip_card_policy_net
        ai.flip_card_action.load_state_dict(torch.load(FLIP_CARD_WEIGHTS))
        ai.swap_card_action = swap_card_policy_net
        ai.swap_card_action.load_state_dict(torch.load(SWAP_CARD_WEIGHTS))
        ai.replace_or_flip_action = replace_or_flip_policy_net
        ai.replace_or_flip_action.load_state_dict(torch.load(REPLACE_OR_FLIP_WEIGHTS))
        return ai

    #TODO maybe train with another AI will require fixing some hard coding which I dont feel like at the moment just making a player flag would work though
    game = Game.initialize([init_ai, Random_Player], log=False)

    for step in range(n_games):
        swap_card_action = None
        flip_card_action = None

        t = np.clip(step / eps_steps, 0, 1)
        eps = (1-t) * eps_start + t * eps_end

        for i in range(len(game.players)):
            prediction1 = torch.zeros(6)
            prediction2 = torch.zeros(6)
            _, _, state1, state2 = Game.flip_2_cards_step(game, i, prediction1, prediction2)

            if i == 0:
                flip1_reward = torch.tensor([flip_reward(state1)[torch.argmax(prediction1)]])
                flip_card_memory.push(game.encode().unsqueeze(0), prediction1.unsqueeze(0), state1.unsqueeze(0), flip1_reward)
                flip2_reward = torch.tensor([flip_reward(state2)[torch.argmax(prediction2)]])
                flip_card_memory.push(state1.unsqueeze(0), prediction2.unsqueeze(0), state2.unsqueeze(0), flip2_reward)

        

        for i in range(len(game.players) * 4):
            player = i % len(game.players)
            prediction = torch.zeros(1)
            card, action = Game.draw_card_step(
                game,
                player,
                prediction=prediction,
                format_action=lambda action, prediction: format_draw_action(action, eps, prediction)
            )

            if player == 0:
                draw_action = prediction
                draw_action_state = game.encode(card)
                draw_action_reward = draw_reward(action, draw_action_state)

            prediction = torch.zeros([1])
            action = Game.replace_or_flip_step(
                game,
                player,
                card,
                prediction=prediction,
                format_action=lambda action, prediction: format_replace_or_flip_action(action, eps, prediction)
            )

            if player == 0:
                replace_or_flip_action = prediction
                replace_or_flip_state = game.encode(card)
                replace_or_flip_reward = replace_or_flip_rewarder(action, replace_or_flip_state)

            if action == Swap_Action.SWAP:
                prediction = torch.zeros(6)
                index = Game.swap_card_step(
                    game, 
                    player,
                    card,
                    prediction=prediction,
                    format_action=lambda action, prediction: format_swap_action(action, eps, prediction)
                )

                if player == 0:
                    swap_card_action = prediction
                    swap_card_state = game.encode()
                    swap_card_reward = torch.tensor([swap_reward(swap_card_state)[torch.argmax(swap_card_action)]])

            else:
                prediction = torch.zeros(6)
                index = Game.flip_card_step(
                    game,
                    player,
                    card,
                    prediction=prediction,
                    format_action=lambda action, prediction: format_flip_action(action, eps, prediction)
                )

                if player == 0:
                    flip_card_action = prediction
                    flip_card_state = game.encode()
                    #get the reward from the target
                    flip_card_reward = torch.tensor([flip_reward(flip_card_state)[torch.argmax(flip_card_action)]])

            if player == 0:
                next_state = game.encode()

                #push changes to memory
                draw_action_memory.push(draw_action_state.unsqueeze(0), draw_action.unsqueeze(0), next_state.unsqueeze(0), draw_action_reward)

                if flip_card_action is not None:
                    flip_card_memory.push(flip_card_state.unsqueeze(0), flip_card_action.unsqueeze(0), next_state.unsqueeze(0), flip_card_reward)

                if swap_card_action is not None:
                    swap_card_memory.push(swap_card_state.unsqueeze(0), swap_card_action.unsqueeze(0), next_state.unsqueeze(0), swap_card_reward)

                replace_or_flip_memory.push(replace_or_flip_state.unsqueeze(0), replace_or_flip_action.unsqueeze(0), next_state.unsqueeze(0), replace_or_flip_reward)

                loss = []

                # Optimize models here:
                # Go through memory and make updates based on that
                loss_draw = optimize_model(
                    optimizer=draw_action_optimizer,
                    policy=draw_action_policy_net,
                    target=draw_action_target_net,
                    memory=draw_action_memory,
                    batch_size=batch_size,
                    gamma=gamma,
                )
                if loss_draw:
                    draw_action_losses.append(loss_draw)
                loss.append(loss_draw)

                loss_flip = optimize_model(
                    optimizer=flip_card_optimizer,
                    policy=flip_card_policy_net,
                    target=flip_card_target_net,
                    memory=flip_card_memory,
                    batch_size=batch_size,
                    gamma=gamma,
                )
                if loss_flip:
                    flip_card_losses.append(loss_flip)
                loss.append(loss_flip)

                loss_swap = optimize_model(
                    optimizer=swap_card_optimizer,
                    policy=swap_card_policy_net,
                    target=swap_card_target_net,
                    memory=swap_card_memory,
                    batch_size=batch_size,
                    gamma=gamma,
                )
                if loss_swap:
                    swap_card_losses.append(loss_swap)
                loss.append(loss_swap)

                loss_replace_or_flip = optimize_model(
                    optimizer=replace_or_flip_optimizer,
                    policy=replace_or_flip_policy_net,
                    target=replace_or_flip_target_net,
                    memory=replace_or_flip_memory,
                    batch_size=batch_size,
                    gamma=gamma,
                )
                if loss_replace_or_flip:
                    replace_or_flip_losses.append(loss_replace_or_flip)
                loss.append(loss_replace_or_flip)

        loss = [v if v is not None else 0 for v in loss]
        print("Losses: Draw_Action_Model={:<.4f} Flip_Card_Model={:<.4f} Swap_Card_Model={:<.4f} Replace_Or_Flip_Model={:<.4f}".format(*loss))
        Game.finalize_game(game)
        game.reset()

    torch.save(draw_action_policy_net.state_dict(), DRAW_ACTION_WEIGHTS)
    torch.save(flip_card_policy_net.state_dict(), FLIP_CARD_WEIGHTS)
    torch.save(replace_or_flip_policy_net.state_dict(), REPLACE_OR_FLIP_WEIGHTS)
    torch.save(swap_card_policy_net.state_dict(), SWAP_CARD_WEIGHTS)

    plt.plot(draw_action_losses, label='Draw Action Model')
    plt.plot(flip_card_losses, label='Flip Card Model')
    plt.plot(swap_card_losses, label='Swap Card Model')
    plt.plot(replace_or_flip_losses, label='Replace Or Flip Model')

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Losses for each Model')
    plt.legend()
    plt.show()

def benchmark():
    ai_score = []
    random_score = []

    print("Running games for benchmark")
    for i in range(100):
        ai, random = run_game([AI_Player, Random_Player], log=False)
        ai_score.append(ai)
        random_score.append(random)

    print("Average Scores:")
    print(f"AI: {statistics.mean(ai_score)}")
    print(f"Random: {statistics.mean(random_score)}\n")

    print("Median Scores:")
    print(f"AI: {statistics.median(ai_score)}")
    print(f"Random: {statistics.median(random_score)}")

if __name__ == "__main__":
    #single train models:
    train_draw_card()
    train_replace_or_flip()
    train_swap_card()
    train_flip_card()
    print("Done initial training")
    # benchmark()
    
    #train at same time with q-learning
    #TODO q_learn makes things worse???????
    train_q_learn()
    #check results
    print("Done q-learning")
    benchmark()
    