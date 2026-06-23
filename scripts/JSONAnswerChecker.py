#Receives input:
#1) original FinQA dataset json file
#2) generated programs csv from qwen (examples subset of original dataset json)
#Evaluates the generated programs and checks to see which answers are correct.
#Outputs:
#1) numValid; Number of valid generated programs (not raw))
#2) propvalid; Proportion of valid generated programs
#3) accuracy; Accuracy of final executed answers
#Note: If generated answer is true_answer*100, it is considered correct.
#Note: If generated answer is within 1% of true_answer, it is also correct.

import json
import pandas as pd
from pathlib import Path
from FinQAProgramEvaluator import EvalProgram

#Input FinQA original data reference json file
x = open("C:/Users/japotte2/Downloads/FinQAnogit/FinQA/dataset/train.json")
original = json.load(x)

#Input generated programs csv from qwen
generated = pd.read_csv("C:/Users/japotte2/Downloads/FinQAnogit/outputs/finqa_qwen3_programs.csv")

n = len(original) #number of examples in original
m = len(generated) #number of examples in generated (subset of original)
numValid = sum(generated["is_valid_program"]) #number of valid generations
propValid = numValid/m #proportion of valid generations


# example = original[144]

# QA = example["qa"]

# ans = QA["exe_ans"]



#Create 
answers = ["NA"] * n
tPrograms = ["NA"] * n
ids = ["NA"] * n

for i in range(0,n):
    answers[i] = original[i]["qa"]["exe_ans"]
    tPrograms[i] = original[i]["qa"]["program"]
    ids[i] = original[i]["id"]

vExamples = generated.loc[generated["is_valid_program"]]
o = len(vExamples) #number of valid generations

corrects = [0] * o

for i in range(0,o):
    gAnswer = EvalProgram(vExamples["generated_program"].iloc[i])
    tAnswer = answers[ids.index(vExamples["query_example_id"].iloc[i])]
    
    print(gAnswer)
    print(tAnswer)
    print(" ")
    
    
    if type(gAnswer) != str and type(tAnswer) != str:
        if 0.99*tAnswer <= gAnswer and gAnswer <= 1.01*tAnswer:
            corrects[i] = 1
        elif 99*tAnswer <= gAnswer and gAnswer <= 101*tAnswer:
            corrects[i] = 1
    elif type(gAnswer) == bool and type(tAnswer) == str:
        if gAnswer and tAnswer == "yes":
            corrects[i] = 1
        elif (not gAnswer) and tAnswer == "no":
            corrects[i] = 1

accuracy = sum(corrects)/o
#print(accuracy)