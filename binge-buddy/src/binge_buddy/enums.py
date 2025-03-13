from enum import Enum

# defines a set of attributes
class Attribute(str, Enum):
    Likes = "Likes"
    Dislikes = "Dislikes"
    Favorite = "Favorite"
    Want_To_Watch = "Wants To Watch"
    Platform = "Platform"
    Genre = "Genre"
    Personality = "Personality"
    Watching_Habit = "Watching Habit"
    Frequency = "Frequency"
    Avoid = "Avoid"
    Tone = "Tone"
    Character_Preference = "Character Preference"
    Show_Length = "Show Length"
    Rewatcher = "Rewatcher"
    Popularity = "Popularity"

# defines a list of available actions
class Action(str, Enum):
    Create = "Create"
    Update = "Update"