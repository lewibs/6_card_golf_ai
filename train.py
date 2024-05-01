import torch
from Game import Game
import torch.optim as optim
from Q_Learn import ReplayMemory
import numpy as np
from Draw_Action_Model import DrawActionModel, DRAW_ACTION_WEIGHTS
from Flip_Card_Model import FlipCardModel, FLIP_CARD_WEIGHTS
from Replace_Or_Flip_Model import ReplaceOrFlipModel, REPLACE_OR_FLIP_WEIGHTS
from Swap_Card_Model import SwapCardModel, SWAP_CARD_WEIGHTS
from Game import Game
from AI_Player import AI_Player
from Random_Player import Random_Player
from Game import Swap_Action, Draw_Action
import random
from Q_Learn import optimize_model
import matplotlib.pyplot as plt

def format_draw_action(action, eps, prediction=torch.zeros(1)):
    sample = random.random()
    
    if sample > eps:
        return action
    else:
        prediction[0] = random.uniform(-1, 1)

        # this is the same as the draw action dont change
        if prediction.item() < 0:
            return Draw_Action.RANDOM
        else:
            return Draw_Action.KNOWN

def format_flip_action(action, eps, prediction=torch.zeros(6)):
    sample = random.random()
    
    if sample > eps:
        return action
    else:
        #TODO return a random index instead of whatever they already did
        return action

def format_swap_action(action, eps, prediction=torch.zeros(6)):
    sample = random.random()
    
    if sample > eps:
        return action
    else:
        #TODO return a random index instead of whatever they already did
        return action

def format_replace_or_flip_action(action, eps, prediction=torch.zeros(1)):
    sample = random.random()
    
    if sample > eps:
        return action
    else:
        prediction[0] = random.uniform(-1, 1)

        # this is the same as the replace or flip action dont change
        if prediction > 0:
            return Swap_Action.SWAP
        else:
            return Swap_Action.FLIP

def start_training():
    n_games = 50
    batch_size = 10 #TODO maybe make this the number of turns?
    gamma = 0.99
    eps_start = 1.0
    eps_end = 0.1
    eps_steps = 2000
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

    #NOTE if an issue occurs, its very likely its from the replacement with the models in this function
    def init_ai(*args):
        ai = AI_Player(*args)
        ai.draw_action = draw_action_policy_net
        ai.flip_card_action = flip_card_policy_net
        ai.swap_card_action = swap_card_policy_net
        ai.replace_or_flip_action = replace_or_flip_policy_net
        return ai

    game = Game.initialize([init_ai, Random_Player], log=False)

    for step in range(n_games):
        swap_card_action = None
        flip_card_action = None
        
        state = game.encode()

        t = np.clip(step / eps_steps, 0, 1)
        eps = (1-t) * eps_start + t * eps_end

        for i in range(len(game.players)):
            prediction1 = torch.zeros(6)
            prediction2 = torch.zeros(6)
            #TODO format to allow randomness???
            state1, state2 = Game.flip_2_cards_step(game, i, prediction1, prediction2)

            if i == 0:
                #TODO
                flip1_reward = torch.tensor([1])
                flip_card_memory.push(state.unsqueeze(0), prediction1.unsqueeze(0), state1.unsqueeze(0), flip1_reward)
                flip2_reward = torch.tensor([1])
                flip_card_memory.push(state1.unsqueeze(0), prediction2.unsqueeze(0), state2.unsqueeze(0), flip2_reward)
                next_state = state2
                state = state2

        

        for i in range(len(game.players) * 4):

            player = i % len(game.players)
            prediction = torch.zeros(1)
            card = Game.draw_card_step(
                game,
                player,
                prediction=prediction,
                format_action=lambda action, prediction: format_draw_action(action, eps, prediction)
            )

            if player == 0:
                draw_action = prediction
                #TODO
                draw_action_reward = torch.tensor([1])

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
                #TODO
                replace_or_flip_reward = torch.tensor([1])

            if action == Swap_Action.SWAP:
                prediction = torch.zeros(6)
                Game.swap_card_step(
                    game, 
                    player,
                    card,
                    prediction=prediction,
                    format_action=lambda action, prediction: format_swap_action(action, eps, prediction)
                )

                if player == 0:
                    swap_card_action = prediction
                    swap_card_reward = torch.tensor([1])

            else:
                prediction = torch.zeros(6)
                Game.flip_card_step(
                    game,
                    player,
                    card,
                    prediction=prediction,
                    format_action=lambda action, prediction: format_flip_action(action, eps, prediction)
                )

                if player == 0:
                    flip_card_action = prediction
                    flip_card_reward = torch.tensor([1])

            #IF player is player 0 then save the stuff to add to memory in the next step


            if player == 0:
                next_state = game.encode()

                #push changes to memory
                draw_action_memory.push(state.unsqueeze(0), draw_action.unsqueeze(0), next_state.unsqueeze(0), draw_action_reward)

                if flip_card_action is not None:
                    flip_card_memory.push(state.unsqueeze(0), flip_card_action.unsqueeze(0), next_state.unsqueeze(0), flip_card_reward)

                if swap_card_action is not None:
                    swap_card_memory.push(state.unsqueeze(0), swap_card_action.unsqueeze(0), next_state.unsqueeze(0), swap_card_reward)

                replace_or_flip_memory.push(state.unsqueeze(0), replace_or_flip_action.unsqueeze(0), next_state.unsqueeze(0), replace_or_flip_reward)

                state = next_state

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
                replace_or_flip_losses.append(loss_replace_or_flip)
                loss.append(loss_replace_or_flip)

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

if __name__ == "__main__":
    start_training()