import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from Deck import Deck
# from torch.utils.data import Dataset
# from torch.utils.data import random_split
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
from Card import ENCODED_VALUES

def card_predictor(deck_encoded):
    probs = torch.ones(len(ENCODED_VALUES)) * 4

    for value in deck_encoded:
        if value != 0:
            index = ENCODED_VALUES.index(value)
            probs[index] -= 1

    return probs / torch.sum(probs)

# WEIGHTS = "./weights/Card_Predictor.pth"


# class CardDrawDataset(Dataset):
#     def __init__(self, decks):
#         data = []
#         targets = []
#         for i in range(decks):
#             deck = Deck()
#             for c in range(52):
#                 start = deck.encode()
#                 card = deck.deal_cards(1)[0]
#                 target = card.one_hot()
#                 deck.discard(card)
#                 data.append(start)
#                 targets.append(target)

#         self.data = data
#         self.targets = targets 

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         sample = {'input': self.data[idx], 'target': self.targets[idx]}
#         return sample

# class CardPredictor(nn.Module):
#     def __init__(self):
#         super(CardPredictor, self).__init__()
#         self.fc1 = nn.Linear(52, 52)  # 52 to represent the cards on the table
#         self.fc2 = nn.Linear(52, 30) 
#         self.fc3 = nn.Linear(30, 13)   # Output layer for 13 types of cards

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return F.softmax(x, dim=1)

# def train():
#     DECKS = 1000
#     EPOCHS = 10
#     TRAINING_SPLIT = 0.8
#     BATCH_SIZE = 32
#     WORKERS = 4

#     model = CardPredictor()
    
#     try:
#         model.load_state_dict(torch.load(WEIGHTS))
#     except:
#         print("unable to load weights")

#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#     dataset = CardDrawDataset(DECKS)

#     train_size = int(TRAINING_SPLIT * len(dataset))
#     val_size = len(dataset) - train_size 

#     # Split the dataset into training and validation sets
#     train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

#     train_loader = DataLoader(train_dataset, num_workers=WORKERS, batch_size=BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(val_dataset, num_workers=WORKERS, batch_size=BATCH_SIZE)
    
#     train_losses = []
#     val_losses = []
    
#     for epoch in range(EPOCHS):
#         model.train()  # Set the model to training mode
#         running_loss = 0.0
#         for i, batch in enumerate(train_loader):
#             inputs = batch['input']
#             labels = batch['target'].argmax(dim=1)  # Convert one-hot targets to class labels
#             optimizer.zero_grad()
#             outputs = model(inputs.float())  # Forward pass
#             loss = criterion(outputs, labels)  # Compute the loss
#             loss.backward()  # Backpropagation
#             optimizer.step()  # Update the weights
#             running_loss += loss.item()
#         train_loss = running_loss / len(train_loader)
#         train_losses.append(train_loss)
#         print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {train_loss}")

#         # Validation loop
#         model.eval()  # Set the model to evaluation mode
#         val_loss = 0.0
#         with torch.no_grad():
#             for batch in val_loader:
#                 inputs = batch['input']
#                 labels = batch['target'].argmax(dim=1)
#                 outputs = model(inputs.float())
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item()
#         val_loss = val_loss / len(val_loader)
#         val_losses.append(val_loss)
#         print(f"Epoch {epoch+1}/{EPOCHS}, Validation Loss: {val_loss}")

#     print("Saving changes")
#     torch.save(model.state_dict(), WEIGHTS)

#     print("Training finished.")
    
#     # Plotting the training and validation losses
#     plt.figure(figsize=(10, 5))
#     plt.plot(range(1, EPOCHS + 1), train_losses, label='Training Loss')
#     plt.plot(range(1, EPOCHS + 1), val_losses, label='Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Training and Validation Losses')
#     plt.legend()
#     plt.show()

# if __name__ == "__main__":
#     # test()
#     # train()