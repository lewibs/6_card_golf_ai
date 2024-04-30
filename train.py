import torch
from Card import ENCODED_VALUES
from Game import Game
import torch.optim as optim
from Q_Learn import ReplayMemory
import numpy as np
from Draw_Action_Model import DrawActionModel, DRAW_ACTION_WEIGHTS
from Game import Game
from AI_Player import AI_Player
from Random_Player import Random_Player
from Game import Swap_Action, Draw_Action
import random
from Q_Learn import optimize_model

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


def start_training():
    n_games = 1
    batch_size = 10
    gamma = 0.99
    eps_start = 1.0
    eps_end = 0.1
    eps_steps = 2000
    memory = 1000
   
    draw_action_policy_net = DrawActionModel()
    draw_action_target_net = DrawActionModel()
    draw_action_target_net.load_state_dict(draw_action_policy_net.state_dict())
    draw_action_target_net.eval()

    draw_action_optimizer = optim.AdamW(draw_action_policy_net.parameters(), lr=1e-3)
    draw_action_memory = ReplayMemory(memory)

    #NOTE if an issue occurs, its very likely its from the replacement with the models in this function
    def init_ai(*args):
        ai = AI_Player(*args)
        ai.draw_action = draw_action_policy_net
        return ai

    game = Game.initialize([init_ai, Random_Player], log=False)

    for step in range(n_games):

        state = game.encode()

        t = np.clip(step / eps_steps, 0, 1)
        eps = (1-t) * eps_start + t * eps_end

        for i in range(len(game.players)):
            Game.flip_2_cards_step(game, i)

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
                draw_action_reward = 1

            action = Game.replace_or_flip_step(game, player, card)

            if action == Swap_Action.SWAP:
                Game.swap_card_step(game, player, card)
            else:
                Game.flip_card_step(game, player, card)

            #IF player is player 0 then save the stuff to add to memory in the next step


        next_state = game.encode()

        #push changes to memory
        draw_action_memory.push(state, draw_action, next_state, draw_action_reward)
        state = next_state

        # Optimize models here:
        # Go through memory and make updates based on that
        optimize_model(
            optimizer=draw_action_optimizer,
            policy=draw_action_policy_net,
            target=draw_action_target_net,
            memory=draw_action_memory,
            batch_size=batch_size,
            gamma=gamma,
        )

        Game.finalize_game(game)
        game.reset()

    torch.save(draw_action_policy_net.state_dict(), DRAW_ACTION_WEIGHTS) 

if __name__ == "__main__":
    start_training()