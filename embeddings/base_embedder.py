from abc import ABC, abstractmethod

class BaseEmbedder(ABC):

    @abstractmethod
    def encode(self, text: str):
        pass