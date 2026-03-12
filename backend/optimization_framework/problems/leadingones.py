def fitnessLeadingOnes(bitstring):
    fitness = 0
    for i in range(len(bitstring)):
        if (bitstring[i] == "1"):
            fitness +=1
        else:
            break
    return fitness