import random
import math
import numpy as np
from numpy.linalg import matrix_transpose as tr

# Sudoku solver using the simulated annealing optimization.

# A sudoku puzzle is composed of a square 9x9 board divided into 3 rows and 3 columns of smaller 3x3 boxes.
# The goal is to fill the board with digits from 1 to 9 such that
#   each number appears only once for each row column and 3x3 box;
#   each row, column, and 3x3 box should contain all 9 digits.
# The solver should take a matrix as input
# where empty squares are represented by a standard symbol (let's say '0'),
# while known squares should be represented by the corresponding digit (1,... ,9). 

# Hints for Constraint Propagation and Backtracking:
# 1. Each cell should be a variable that can take values ​​in the domain (1,...,9).
# 2. The two types of constraints in the definition form as many types of constraints:
#   - Direct constraints impose that no two equal digits appear for each row, column, and box;
#   - Indirect constraints impose that each digit must appear in each row, column, and box.
# 3. You can think of other types of (pairwise) constraints to further improve the constraint propagation phase.
# Note: most puzzles found in the magazines can be solved with only the constraint propagation step.

# propagate through possible numbers for an empty box


BOARD_LEN = 9

# sudoku = np.array(
#         [[5,6,8,0,0,0,0,1,9],
#          [0,0,0,0,0,0,0,0,0],
#          [3,0,2,9,1,5,0,8,0],
#          [0,0,6,0,7,1,5,4,3],
#          [8,0,0,0,4,0,6,0,0],
#          [4,0,3,0,5,6,0,9,8],
#          [7,0,4,0,0,8,2,6,1],
#          [6,0,0,0,0,0,0,0,0],
#          [0,1,9,0,0,4,0,3,5]])
# sudoku = np.array(
#         [[4,0,0,0,0,0,0,0,0],
#          [0,0,0,0,4,0,0,0,0],
#          [0,0,0,0,0,0,0,5,0],
#          [2,0,0,0,0,0,0,0,0],
#          [0,0,0,2,0,0,0,0,0],
#          [0,0,0,0,0,0,0,2,0],
#          [0,0,0,9,0,0,0,0,0],
#          [0,0,5,0,0,4,0,0,0],
#          [0,0,4,0,0,0,0,0,0]])
sudoku = np.array(
    [[0 for _ in range(BOARD_LEN)] for _ in range(BOARD_LEN)]
)

def solveSudoku_simulatedAnnealing(sudoku, initialTemp = 100, coolingRate = 0.95, maxIteration = 500):
    
    def getPossibleValuePool(currentState, rowIdx, colIdx):
        possibleValuePool = [True]*BOARD_LEN
        
        # 1. check row
        for value in currentState[rowIdx]:
            if value != 0:
                possibleValuePool[value - 1] = False
                
        # 2. check column
        for value in tr(currentState)[colIdx]:
            if value != 0:
                possibleValuePool[value - 1] = False
                
        # 3. check square of 3*3
        rowOrgIdx = rowIdx
        colOrgIdx = colIdx
        # set the origin index of row for 3*3
        if rowIdx % 3 == 1:
            rowOrgIdx = rowIdx - 1
        elif rowIdx % 3 == 2:
            rowOrgIdx = rowIdx - 2
        # set the origin index of column for 3*3
        if colIdx % 3 == 1:
            colOrgIdx = colIdx - 1
        elif colIdx % 3 == 2:
            colOrgIdx = colIdx - 2
            
        for row_n in range(3):
            for col_n in range(3):
                if rowOrgIdx + row_n == rowIdx and colOrgIdx + col_n == colIdx:
                    continue
                value = currentState[rowOrgIdx + row_n][colOrgIdx + col_n]
                if value != 0:
                    possibleValuePool[value - 1] = False
        
        return possibleValuePool
    
    
    def getNeighborState(possibleValuePool, currentState, rowIdx, colIdx):
        isStateAvailable = True
        newlyAssigned = False
        value = currentState[rowIdx][colIdx]  
        
        if value == 0:
            if possibleValuePool.count(True) == 1:
                currentState[rowIdx][colIdx] = possibleValuePool.index(True) + 1
                newlyAssigned = True
            elif possibleValuePool.count(True) == 0:    # the branch is stuck! wrong path!!
                isStateAvailable = False
                        
        return currentState, newlyAssigned, isStateAvailable
    
    # Decide whether to accept the new state
    # if delta_e < 0 or random.random() < math.exp(-delta_e / temperature):
    #     current_state = neighbor_state[:]
    #     current_conflicts = neighbor_conflicts
    
    def exploreState(currentState, temperature, iteration):
        success = False
        
        # Check maximum iteration
        if iteration > maxIteration:
            return currentState, temperature, iteration, success
        
        # Check if there are certain-valued boxes and update them.
        currentState, allPossibleValuePool, isStateChanged, isStateAvailable = travelEmptyBoxes(currentState)
        
        if not isStateAvailable:
            return  currentState, temperature, iteration, success
        
        if not isStateChanged:
            # Make branches with possible states (values).
            currentState, temperature, iteration, success = travelWithPossibleStates(allPossibleValuePool, currentState, temperature, iteration)
            # Hmmmmmmmmmm.......
            if success or iteration > maxIteration:
                return currentState, temperature, iteration, success
            
        else:
            # Check availability.
            availability = np.array(allPossibleValuePool).flatten().sum()
            if availability == 0:    # all False (= no further availability)  
                # Finish a branch.
                if currentState.sum() == sum(range(1,10))*9:
                    success = True
                else:
                    success = False
                return  currentState, temperature, iteration, success
            
        # keep a branch going.
        return exploreState(currentState, temperature, iteration + 1) 
        
    
    def travelEmptyBoxes(currentState):
        isStateAvailable = True
        isStateChanged = False
        allPossibleValuePool = [[[False for _ in range(BOARD_LEN)] for _ in range(BOARD_LEN)] for _ in range(BOARD_LEN)]
        # [[[0]*BOARD_LEN]*BOARD_LEN] # unexpected behavior
        
        for rowIdx in range(BOARD_LEN):
            for colIdx in range(BOARD_LEN):
                if currentState[rowIdx][colIdx] != 0:
                    continue
                
                possibleValuePool = getPossibleValuePool(currentState, rowIdx, colIdx)
                currentState, isNewState, isStateAvailable = getNeighborState(possibleValuePool, currentState, rowIdx, colIdx)
                
                if isNewState:
                    possibleValuePool = getPossibleValuePool(currentState, rowIdx, colIdx)
                
                isStateChanged = isStateChanged | isNewState
                allPossibleValuePool[rowIdx][colIdx] = possibleValuePool
        
        return currentState, allPossibleValuePool, isStateChanged, isStateAvailable
    
    
    def travelWithPossibleStates(allPossibleValuePool, currentState, temperature, iteration):
        success = False
        costs = dict()
        
        for rowIdx in range(BOARD_LEN):
            for colIdx in range(BOARD_LEN):
                if currentState[rowIdx][colIdx] != 0:
                    continue
                
                cost = allPossibleValuePool[rowIdx][colIdx].count(True)
                costs.update({(rowIdx, colIdx): cost})
            
        costs = dict(sorted(costs.items(), key=lambda item: item[1]))
        for boxCoord in costs.keys():
            for possibilityIdx in range(BOARD_LEN):
                if allPossibleValuePool[boxCoord[0]][boxCoord[1]][possibilityIdx] == True:
                    possibleState = currentState.copy()
                    possibleState[boxCoord[0]][[boxCoord[1]]] = possibilityIdx + 1
                    # The possibility branch
                    currentState, temperature, iteration, success = exploreState(possibleState, temperature, iteration + 1)
                    
                    if success or iteration > maxIteration: break
                    else: continue
            if success: break
        
        return currentState, temperature, iteration, success
    
    # Iterating
    success = False
    iteration = 0
    state = sudoku.copy()
    #allPossibleValuePool = [[[]]]
    temperature = initialTemp
    #temperature *= coolingRate
    
    # "box means empty box."
    # First, get the possible values for a box.
    # If there is only one value, assign it and 
    # If there is none, choose the box with lowest possible values and try them (this is where probability gives in the possibility of choosing higher cost ones)
    
    # Let's roll
    state, temperature, iteration, success = exploreState(state, temperature, iteration + 1)
    print(f"Result:\n{state}\ntemperature: {temperature}\niteration: {iteration}\nsuccess: {success}")
    
    #state[0][4] = 5
    countOriginal = 0
    countState = 0
    for rowIdx in range(BOARD_LEN):
        for colIdx in range(BOARD_LEN):
            if state[rowIdx][colIdx] == 0:
                countState += 1
            if sudoku[rowIdx][colIdx] == 0:
                countOriginal += 1
    print(f"number of boxes: {countState} / {countOriginal}") # x / 43
    
    
# The initiating point
solveSudoku_simulatedAnnealing(sudoku)

# background: algorithm, screenshots, samples with different levels, optimization, explaining code




# costs = dict()
# tf = [[[0 for _ in range(BOARD_LEN)] for _ in range(BOARD_LEN)] for _ in range(BOARD_LEN)]
# #tf = [[False]*BOARD_LEN]*BOARD_LEN
# for rowIdx in range(BOARD_LEN):
#     for colIdx in range(BOARD_LEN):
#         if sudoku[rowIdx][colIdx] != 0:
#             continue
        
#         #print(tf[rowIdx][colIdx])
#         #tf[rowIdx][colIdx] = True
        
#         costs.update({(rowIdx, colIdx): colIdx % 3})
#         costs = dict(sorted(costs.items(), key=lambda item: item[1]))
        
# keys = costs.keys()
# tfCheck = np.array(tf).flatten()
# print(np.array(tf))
# print(tfCheck.prod())
# mtrx = [[1,2,3,4,5,6,7,8,9]]*9
# mtrxCheck = np.array(mtrx)
# print(mtrxCheck.sum())
# print(sum(range(1,10))*9)
# for k in keys:
#     print(k[0])