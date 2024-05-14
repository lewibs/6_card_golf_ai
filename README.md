
# Objective
My objective with this project was to learn some basics in ai using a simple card game.

# 6 Card Golf
6 Card Golf is a card game in which the objective is to get the lowest score in the game by flipping cards that have low values or getting pairs.

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

# Architecture
The AI Agent is utilizing 4 NNs. In the game there are 4 clear distinct actions:
1. The decision to draw a known card, or an unknown card.
2. The decision to replace a card with the drawn card, or flip a random card.
3. The decision to of where to place the card.
4. The decision of which random card to flip.

## Draw_Action_Model
The draw action ai model utilizes a simple network. This model could drastically be improved upon. It is trained on 3200 randomly generated actions which are rewarded based a greedy reward which foucuses on making sure the user will be able to create a pair, or if that is not possible determining the statistacally most likely value and deciding if the known draw card is over or under that.

```py
    #NN Structure:
    nn.Linear((1 * len(ENCODED_VALUES)) + (6 * len(ENCODED_VALUES)) + (1 * len(ENCODED_VALUES)), 30),
    nn.ReLU(),
    nn.Linear(30, 15),
    nn.ReLU(),
    nn.Linear(15, 1),
    nn.Tanh(),
```

```
Training Results:
Correct known: 374
Incorrect known: 301
Correct random: 122
Incorrect random: 3
```

As can be seen, it overly picks the known card. This is most likely due to a bab reward loss calcualtion however it gets fixed in the Q-learning.

## Replace_Or_Flip_Model
This model is the simplest one. It takes 3 inputs, prediced random card value, known card value, and if there is the possiblility of a pair. As a result it was only trained on 320 randomly generated actions and then validated on 80 actions.

```py
    #NN Structure
    nn.Linear(1 + 1 + 1, 3),
    nn.ReLU(),
    nn.Linear(3, 1),
    nn.Tanh(),
```

```
Training Results
Correct swap: 31
Incorrect swap: 0
Correct flip: 40
Incorrect flip: 9
```

## Swap_Card_Model
This model is used to determine what placement of the card will yeild the lowest score. The nn will return the probabilities of what the best action will be. When training this NN it gets the target value by determining which card placement yeilds the lowest values and then normalizes the returned prediction. Loss is calculated using CrossEntropyLoss 

```py
    nn.Linear(len(ENCODED_VALUES) + (6 * len(ENCODED_VALUES)) + (1 * len(ENCODED_VALUES)), 1000),
    nn.ReLU(),
    nn.Linear(1000, 1000),
    nn.ReLU(),
    nn.Linear(1000, 100),
    nn.ReLU(),
    nn.Linear(100, 32),
    nn.ReLU(),
    nn.Linear(32, 6)
```

```
Training Results:
Successfully minimized: 372
Failed to minimize: 30
No choice turn: 398
```

## Flip_Card_Model
The only input into this model is its own hand it looks at all of the probabilities. The way that the target/reward is generated is fairly interesting. In sort, it creates a one hot encoding of the players hand, and then a reverse of that, multiplies it by the probabilities of each card, trims all of the rows to only show the cards that are lower then the predicted value, and then also includes the probability of the desired card for a match, sums all of the rows, and then the card with the highest value, has the higest probability of being the best card.

```py
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
```


```py
    nn.Linear(6 * len(ENCODED_VALUES), 1000),
    nn.ReLU(),
    nn.Linear(1000, 1000),
    nn.ReLU(),
    nn.Linear(1000, 100),
    nn.ReLU(),
    nn.Linear(100, 32),
    nn.ReLU(),
    nn.Linear(32, 6),
    nn.Sigmoid()
```

# Q-Learning
Finally the model is trained on 50 games using Q-learning to fine tune the model. The value of each action is based off of the resulting score of the game, along with the immediate value of that action

```
Average Scores:
AI: 17.89
Random: 32.58

Median Scores:
AI: 18.0
Random: 32.0
```
