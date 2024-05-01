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

#reward the flip action where hand is the current state, and index is the card recently flipped
def flip_reward(hand, index):
    other_index = index - 3 if index >= 3 else index + 3
    a = hand[index]
    b = hand[other_index]

    a = a.encode() if isinstance(a, Card) else int(a.item())
    b = b.encode() if isinstance(b, Card) else int(b.item())

    if a == 0 or b == 0:
        return torch.tensor([0])
    
    score_a = Card.static_score(Card.decode(a))
    score_b = Card.static_score(Card.decode(b))

    if a == b:
        final_score = 0
    else:
        final_score = sum([score_a, score_b])

    single_score = score_b

    score_difference = single_score - final_score

    if score_difference > 0:
        #final score is better
        return torch.tensor([1])
    elif score_difference < 0:
        #single score is better
        return torch.tensor([-1])
    else:
        if final_score <= 0:
            return torch.tensor([5])

        return torch.tensor([0])

#reward the swap action where hand is the current state, and index is the card recently added
def swap_reward(hand, index):
    hand_clone = hand[:] #clone for safety
    unknown_indexes = [index for index, value in enumerate([v.known for v in hand_clone]) if value == 0]

    score = Game.static_score_hand(hand_clone)
    scores = []

    for swap_index in unknown_indexes:
        hand_clone = hand[:]
        hold = hand_clone[index]
        hand_clone[index] = hand_clone[swap_index]
        hand_clone[swap_index] = hold
        scores.append(Game.static_score_hand(hand_clone))

    if len(scores) == 0:
        return torch.tensor([0])

    min_score = min(scores)

    if score <= min_score:
        return torch.tensor([1])
    else:
        #-1*abs(min_score-score)
        return torch.tensor([-1])

#reward decision to either replace with the card, or flip randomly, game encoded is the current state
def replace_or_flip_rewarder(action, card, game_encoded):
    hand = game_encoded[:6]
    card_encoded = card.encode()
    probalilities = card_predictor(game_encoded)

    #reward pair
    for i in range(6):
        other_i = i - 3 if i >= 3 else i + 3
        if hand[i] == 0 and hand[other_i] == card_encoded and card.score() >= 0:
            if Swap_Action.SWAP == action:
                return torch.tensor([1])
            else:
                return torch.tensor([-1])
            
    likely_score = np.dot(probalilities, SCORES)

    if likely_score < 5.5:
        if card.score() < 5.5:
            if action == Swap_Action.SWAP:
                return torch.tensor([1])
            else:
                return torch.tensor([0])
        else:
            if action == Swap_Action.SWAP:
                return torch.tensor([-1])
            else:
                return torch.tensor([1])
    else:
        if card.score() < 5.5:
            if action == Swap_Action.SWAP:
                return torch.tensor([1])
            else:
                return torch.tensor([-1])
        else:
            return torch.tensor([0])

#reward random draw vs using the discard pile, game encoded is the current state
def draw_reward(action, card, game_encoded):
    hand = game_encoded[:6]
    probalilities = card_predictor(game_encoded)
    likely_score = np.dot(probalilities, SCORES)

    if likely_score < 5.5:
        if card.score() < 5.5:
            if action == Draw_Action.KNOWN:
                return torch.tensor([1])
            else:
                return torch.tensor([0])
        else:
            if action == Draw_Action.KNOWN:
                return torch.tensor([-1])
            else:
                return torch.tensor([1])
    else:
        if card.score() < 5.5:
            if action == Draw_Action.KNOWN:
                return torch.tensor([1])
            else:
                return torch.tensor([-1])
        else:
            return torch.tensor([0])


def format_draw_action(action, eps, prediction=torch.zeros(1)):
    return action
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
    return action
    sample = random.random()
    
    if sample > eps:
        return action
    else:
        #TODO return a random index instead of whatever they already did
        return action

def format_swap_action(action, eps, prediction=torch.zeros(6)):
    return action
    sample = random.random()
    
    if sample > eps:
        return action
    else:
        #TODO return a random index instead of whatever they already did
        return action

def format_replace_or_flip_action(action, eps, prediction=torch.zeros(1)):
    return action
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
    n_games = 100
    batch_size = 10
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

    draw_action_optimizer = optim.AdamW(draw_action_policy_net.parameters(), lr=1e-4)
    draw_action_memory = ReplayMemory(memory)

    flip_card_policy_net = FlipCardModel()
    flip_card_target_net = FlipCardModel()
    flip_card_target_net.load_state_dict(flip_card_policy_net.state_dict())
    flip_card_target_net.eval()

    flip_card_optimizer = optim.AdamW(flip_card_policy_net.parameters(), lr=1e-4)
    flip_card_memory = ReplayMemory(memory)

    replace_or_flip_policy_net = ReplaceOrFlipModel()
    replace_or_flip_target_net = ReplaceOrFlipModel()
    replace_or_flip_target_net.load_state_dict(replace_or_flip_policy_net.state_dict())
    replace_or_flip_target_net.eval()

    replace_or_flip_optimizer = optim.AdamW(replace_or_flip_policy_net.parameters(), lr=1e-4)
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
            index1, index2, state1, state2 = Game.flip_2_cards_step(game, i, prediction1, prediction2)

            if i == 0:
                flip1_reward = flip_reward(state1[:6], index1)
                flip_card_memory.push(state.unsqueeze(0), prediction1.unsqueeze(0), state1.unsqueeze(0), flip1_reward)
                flip2_reward = flip_reward(state2[:6], index2)
                flip_card_memory.push(state1.unsqueeze(0), prediction2.unsqueeze(0), state2.unsqueeze(0), flip2_reward)
                next_state = state2
                state = state2

        

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
                draw_action_reward = draw_reward(action, card, state)

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
                replace_or_flip_reward = replace_or_flip_rewarder(action, card, state)

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
                    swap_card_reward = swap_reward(game.hands[0], index)

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
                    flip_card_reward = flip_reward(game.hands[0], index)

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