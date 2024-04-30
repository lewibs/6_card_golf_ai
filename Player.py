from abc import ABC, abstractmethod

class Player(ABC):
    def __init__(self, id, game):
        self.id = id
        self.game = game

    @abstractmethod
    def draw_card(self, prediction=None):
        pass

    @abstractmethod
    def swap_card(self, card, prediction=None):
        pass

    @abstractmethod
    def show_card(self, prediction=None):
        pass

    @abstractmethod
    def swap_or_flip(self, card, prediction=None):
        pass