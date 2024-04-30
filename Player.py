from abc import ABC, abstractmethod

class Player(ABC):
    def __init__(self, id, game):
        self.id = id
        self.game = game

    @abstractmethod
    def draw_card(self):
        pass

    @abstractmethod
    def swap_card(self):
        pass

    @abstractmethod
    def show_card(self, index):
        pass

    @abstractmethod
    def swap_or_flip(self):
        pass