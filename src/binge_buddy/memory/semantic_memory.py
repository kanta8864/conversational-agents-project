
from binge_buddy.memory.memory import Memory

class SemanticMemory(Memory):
    def __init__(self, type:str, information:str):
        super().__init__(type, information)
    
    def get_type(self):
        return "Semantic"
