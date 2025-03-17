from abc import ABC, abstractmethod

class Memory(ABC):
    def __init__(self, type:str, information:str):
        self.type = type
        self.information = information
        
    @abstractmethod
    def get_type(self):
        pass
