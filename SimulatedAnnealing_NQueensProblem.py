              
import random
import math

def n_queens_simulated_annealing(n, initial_temp, cooling_rate, max_iterations):
    # Initialize the board with one queen per row placed randomly
    state = [random.randint(0, n-1) for _ in range(n)]
    
    # Helper function to calculate the number of conflicts
    def count_conflicts(state):
        conflicts = 0
        for i in range(n):
            for j in range(i + 1, n):
                # Queens in the same column or on the same diagonal
                if state[i] == state[j] or abs(state[i] - state[j]) == (j - i):
                    conflicts += 1
        return conflicts
    
    # Generate a neighboring state (select random row and random column)
    def generate_neighbor(state):
        neighbor = state[:]
        row = random.randint(0, n - 1)
        new_col = random.randint(0, n - 1)
        while new_col == state[row]:
            new_col = random.randint(0, n - 1)
        neighbor[row] = new_col
        return neighbor
    
    # Initial state and temperature
    current_state = state[:]
    current_conflicts = count_conflicts(current_state)
    temperature = initial_temp
    
    for iteration in range(max_iterations):
        # If no conflicts, solution is found
        if current_conflicts == 0:
            break
        
        # Generate a neighbor and calculate conflicts
        neighbor_state = generate_neighbor(current_state)
        neighbor_conflicts = count_conflicts(neighbor_state)
        delta_e = neighbor_conflicts - current_conflicts
        
        # Decide whether to accept the new state
        if delta_e < 0 or random.random() < math.exp(-delta_e / temperature):
            current_state = neighbor_state[:]
            current_conflicts = neighbor_conflicts
            
        # Cool down the temperature
        temperature *= cooling_rate
        
    return current_state, current_conflicts, iteration

# Parameters for the 8-Queens problem
n = 8
initial_temp = 100.0
cooling_rate = 0.95
max_iterations = 10000

# Run the simulated annealing algorithm
solution_state, final_conflicts, iterations = n_queens_simulated_annealing(n, initial_temp, cooling_rate, max_iterations)

print(solution_state, final_conflicts, iterations)
            