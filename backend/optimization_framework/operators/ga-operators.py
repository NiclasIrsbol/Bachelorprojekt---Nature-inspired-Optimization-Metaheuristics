import random

def populationselection():
    return []

def crossover():
    return []

def flipbitMutation(bit):
    index = random.randint(0,(len(bit)-1))
    new_char = "0" if bit[index] == "1" else "1"
    newbit = bit[:index] + new_char + bit[index+1:]
    return newbit
