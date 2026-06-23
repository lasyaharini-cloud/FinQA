#Joseph Pottern, 6/18/2026
#Syntax of input message must be comma-separated sequence of op(arg1,arg2)
#where op in ops, and arg1 & arg2 must be either numbers, #{n: whole number},
#or CONST_{number}, where the #n is the result of the (n-1)th operation 
#evaluation. 
#Example: msg = divide(3,10), multiply(#0,CONST_100)
#This should return 30. Space after comma is optional.

def EvalProgram(message):
    import numpy as np
    #Operations list
    ops = ["add","subtract","multiply","divide","exp", "greater"] 
    trueops = ["+","-","*","/","**",">"]
    
    #Input message
    #message = "divide(16,2), greater(9,#0)"
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    #Get rid of last ")"
    msg = message[0:len(message)-1] 
    
    #Remove spaces
    msg = msg.replace(" ","")
    
    #Place minus sign for potential M1 (minus one)
    msg = msg.replace("M","-")
    
    #Split between operations (steps)
    steps = msg.split("),")
    
    #Initialize memory to list of "NA"
    memory = ["NA"]*len(steps)
    
    ##############
    #Function to evaluate operation (step)
    def stepeval(step:str):
        
        #Split between op and args
        argop = step.split("(")
        
        #Extract operation symbol (in python)
        opstr = argop[0]
        opI = ops.index(opstr)
        trueop = trueops[opI]
        
        #Extract each arg
        argsstr = argop[1]
        [arg1str, arg2str] = argsstr.split(",")
        
        #Turn memory (#) or constants (CONST_) into the correct number
        if "#" in arg1str:
            I1 = eval(arg1str[1:len(arg1str)])
            arg1str = str(memory[I1])
        elif "CONST_" in arg1str:
            conspl1 = arg1str.split("_")
            arg1str = conspl1[1]
            
        if "#" in arg2str:
            I2 = eval(arg2str[1:len(arg2str)])
            arg2str = str(memory[I2])
        elif "CONST_" in arg2str:
            conspl2 = arg2str.split("_")
            arg2str = conspl2[1]
        
        #Return arg1 (operation) arg2 as the answer
        if trueop == "/" and eval(arg2str) == 0:
            ans = 9999999999
        else:
            ans = eval(arg1str + trueop + arg2str)
        return ans
    ##################
    
    #Call iteratively to fill up the memory
    for i in range(0,len(steps)):
        memory[i] = stepeval(steps[i])
    
    return memory[-1]

if __name__ == "__main__":
    import sys
    EvalProgram(sys.argv)
