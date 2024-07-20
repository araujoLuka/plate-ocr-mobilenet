import os

weightsDir: str = __file__.replace("utils/weights.py","weights/") # $PROJECT_ROOT/weights/

def weights_list():
    trainings: list = []
    weights: list = []

    # List only dirs in the weights directory
    for model in os.listdir(weightsDir):
        if os.path.isdir(f"{weightsDir}/{model}"):
            trainings.append(model)
    
    # List weights in each training directory
    for training in trainings:
        for model in os.listdir(f"{weightsDir}/{training}"):
            if model.endswith(".pth"):
                weights.append(f"{training}/{model}")
    
    # List weights in the weights directory
    for model in os.listdir(weightsDir):
        if model.endswith(".pth"):
            weights.append(model)

    # Sort the list by name
    weights.sort()
    
    print("Available weights:")
    for idx, model in enumerate(weights):
        print(f"{idx+1}: {model}")
    
    return weights 

def weights_select(weight: list) -> tuple[str, int]:
    inp = int(input("Which model do you want to load? (0 to skip): "))
    if inp == 0:
        print("Select skipped!")
        return "", 1
    elif inp > 0 and inp <= len(weight):
        return weightsDir + weight[inp-1], 0

    print("Invalid input!")
    return "", -1
 
def weights_use_select() -> bool:
    inp = input("Do you want to load a pre-trained model? (y/n): ")
    if inp.lower() == 'y':
        return True
    elif inp.lower() != 'n':
        print("Invalid input! Skipping...")
    return False
