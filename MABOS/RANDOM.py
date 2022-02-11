import random
import numpy as np
import math
from solution import solution
import time
import matplotlib.pyplot as plt

class RANDOM_BANDIT():
    def __init__(self, objf, lb, ub, dim, SearchAgents_no, Max_iter):

        super(RANDOM_BANDIT, self).__init__()
        self.objf = objf
        self.s = solution()
        self.s.optimizer = "RANDOM_BANDIT"
        self.s.objfname = self.objf.__name__
        self.s.executionTime = 0
        self.s.R_list = []

        # initialize the location and Energy of the rabbit
        self.best_pos = np.zeros(dim)
        self.best_sol = float("inf")  # change this to -inf for maximization problems

        self.dim = dim
        self.lb= lb
        self.ub= ub
        self.SearchAgents_no = SearchAgents_no
        self.Max_iter = Max_iter
        if not isinstance(self.lb, list):
            self.lb = [self.lb for _ in range(dim)]
            self.ub = [self.ub for _ in range(dim)]
        self.lb = np.asarray(self.lb)
        self.ub = np.asarray(self.ub)
        # Initialize the locations of Harris' hawks
        self.Positions = np.asarray(
            [x * (self.ub - self.lb) + self.lb for x in np.random.uniform(0, 1, (self.SearchAgents_no, self.dim))]
        )
        self.fitness = np.full(self.SearchAgents_no, np.inf)
        # Initialize convergence
        self.convergence_curve = np.zeros(self.Max_iter)
        ############################
        print('RANDOM BANDIT is now tackling  "' + self.objf.__name__ + '"')
        self.timerStart = time.time()
        self.s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
        ############################
        self.t = 0  # Loop counter
        self.f = 0 ### fitness evaluation
        self.ratio = 1.0
        self.K = 20
        ##################################################################
        self.c = 0 
        self.Q = np.zeros(3)
        self.R =[]
        self.exp_R= []
        self.count = np.zeros_like(self.Q)  # N(a)=0                         
        self.reward_list = []       # List of rewards
        self.reward_avg_list = []    # List of averaged rewards
        self.begin = True 

        # self.s = self.optimize()
        
    def optimize(self):

        if self.t == 0:
            for i in range(0, self.SearchAgents_no):
                # Check boundries
                self.Positions[i, :] = np.clip(self.Positions[i, :], self.lb, self.ub)
                # fitness of locations
                self.fitness[i] = self.objf(self.Positions[i, :])
                # Update the location of Rabbit
                if self.fitness[i] < self.best_sol:  # Change this to > for maximization problem
                    self.best_sol = self.fitness[i].copy()
                    self.best_pos = self.Positions[i, :].copy()
            self.convergence_curve[self.t] = self.best_sol
            self.t = self.t + 1

        r = np.random.random()
        if r < (1/3):
            self.HHO()
        elif r < (2/3):
            self.DE()
        else:
            self.WOA()

        return self.s
                
    def soft(self, ucb, index):
        ucb = ucb - np.max(ucb)
        return np.exp(ucb[index]) / (np.sum(np.exp(ucb)) )
        
    ########################################################################################################## HHO 
    def HHO(self):
        # Main loop
        print('HHO is optimizing  "' + self.objf.__name__ + '"')
        while self.t < self.Max_iter:
            E1 = 2 * (1 - (self.t / self.Max_iter))  # factor to show the decreaing energy of rabbit

            # Update the location of Harris' hawks
            for i in range(0, self.SearchAgents_no):
                E0 = 2 * random.random() - 1  # -1<E0<1
                Escaping_Energy = E1 * (
                    E0
                )  # escaping energy of rabbit Eq. (3) in the paper
                # -------- Exploration phase Eq. (1) in paper -------------------
                if abs(Escaping_Energy) >= 1:
                    # Harris' hawks perch randomly based on 2 strategy:
                    q = random.random()
                    rand_Hawk_index = math.floor(self.SearchAgents_no * random.random())
                    Positions_rand = self.Positions[rand_Hawk_index, :]
                    if q < 0.5:
                        # perch based on other family members
                        self.Positions[i, :] = Positions_rand - random.random() * abs(
                            Positions_rand - 2 * random.random() * self.Positions[i, :]
                        )
                    elif q >= 0.5:
                        # perch on a random tall tree (random site inside group's home range)
                        self.Positions[i, :] = (self.best_pos - self.Positions.mean(0)) - random.random() * (
                            (self.ub - self.lb) * random.random() + self.lb
                        )
                # -------- Exploitation phase -------------------
                elif abs(Escaping_Energy) < 1:
                    # Attacking the rabbit using 4 strategies regarding the behavior of the rabbit

                    # phase 1: ----- surprise pounce (seven kills) ----------
                    # surprise pounce (seven kills): multiple, short rapid dives by different hawks

                    r = random.random()  # probablity of each event

                    if (
                        r >= 0.5 and abs(Escaping_Energy) < 0.5
                    ):  # Hard besiege Eq. (6) in paper
                        self.Positions[i, :] = (self.best_pos) - Escaping_Energy * abs(
                            self.best_pos - self.Positions[i, :]
                        )

                    if (
                        r >= 0.5 and abs(Escaping_Energy) >= 0.5
                    ):  # Soft besiege Eq. (4) in paper
                        Jump_strength = 2 * (
                            1 - random.random()
                        )  # random jump strength of the rabbit
                        self.Positions[i, :] = (self.best_pos - self.Positions[i, :]) - Escaping_Energy * abs(
                            Jump_strength * self.best_pos - self.Positions[i, :]
                        )

                    # phase 2: --------performing team rapid dives (leapfrog movements)----------

                    if (
                        r < 0.5 and abs(Escaping_Energy) >= 0.5
                    ):  # Soft besiege Eq. (10) in paper
                        # rabbit try to escape by many zigzag deceptive motions
                        Jump_strength = 2 * (1 - random.random())
                        Positions1 = self.best_pos - Escaping_Energy * abs(
                            Jump_strength * self.best_pos - self.Positions[i, :]
                        )
                        Positions1 = np.clip(Positions1, self.lb, self.ub)

                        if self.objf(Positions1) < self.fitness[i]:  # improved move?
                            self.Positions[i, :] = Positions1.copy()
                        else:  # hawks perform levy-based short rapid dives around the rabbit
                            Positions2 = (
                                self.best_pos
                                - Escaping_Energy
                                * abs(Jump_strength * self.best_pos - self.Positions[i, :])
                                + np.multiply(np.random.randn(self.dim), self.Levy())
                            )
                            Positions2 = np.clip(Positions2, self.lb, self.ub)
                            if self.objf(Positions2) < self.fitness[i]:
                                self.Positions[i, :] = Positions2.copy()
                    if (
                        r < 0.5 and abs(Escaping_Energy) < 0.5
                    ):  # Hard besiege Eq. (11) in paper
                        Jump_strength = 2 * (1 - random.random())
                        Positions1 = self.best_pos - Escaping_Energy * abs(
                            Jump_strength * self.best_pos - self.Positions.mean(0)
                        )
                        Positions1 = np.clip(Positions1, self.lb, self.ub)

                        if self.objf(Positions1) < self.fitness[i]:  # improved move?
                            self.Positions[i, :] = Positions1.copy()
                        else:  # Perform levy-based short rapid dives around the rabbit
                            Positions2 = (
                                self.best_pos
                                - Escaping_Energy
                                * abs(Jump_strength * self.best_pos - self.Positions.mean(0))
                                + np.multiply(np.random.randn(self.dim), self.Levy())
                            )
                            Positions2 = np.clip(Positions2, self.lb, self.ub)
                            if self.objf(Positions2) < self.fitness[i]:
                                self.Positions[i, :] = Positions2.copy()

            for i in range(0, self.SearchAgents_no):
                # Check boundries
                self.Positions[i, :] = np.clip(self.Positions[i, :], self.lb, self.ub)
                # fitness of locations
                self.fitness[i] = self.objf(self.Positions[i, :])
                # Update the location of Rabbit
                if self.fitness[i] < self.best_sol:  # Change this to > for maximization problem
                    self.best_sol = self.fitness[i].copy()
                    self.best_pos = self.Positions[i, :].copy()

            self.f += self.SearchAgents_no
            self.convergence_curve[self.t] = self.best_sol
            if self.t % 1 == 0:
                print(
                    [
                        "At iteration "
                        + str(self.t +1)
                        + " the best fitness is "
                        + str(self.best_sol)
                    ]
                )
            self.t = self.t + 1
            self.s.R_list.append(0)

            r = np.random.random()
            if r < (1/3):
                self.HHO()
            elif r < (2/3):
                self.DE()
            else:
                self.WOA()

        timerEnd = time.time()
        self.s.stopiter = self.f
        self.s.executionTime = timerEnd - self.timerStart
        self.s.convergence = self.convergence_curve

    def Levy(self):
        beta = 1.5
        sigma = (
            math.gamma(1 + beta)
            * math.sin(math.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        u = 0.01 * np.random.randn(self.dim) * sigma
        v = np.random.randn(self.dim)
        zz = np.power(np.absolute(v), (1 / beta))
        step = np.divide(u, zz)
        return step
    ########################################################################################################## DE 
    def DE(self):
        mutation_factor = 0.5
        crossover_ratio = 0.7
        # # solution
        # # population_fitness = np.array(Pos_fitness)
        # self.fitness = np.array([float("inf") for _ in range(self.SearchAgents_no)])
        # # calculate fitness for all the population
        # for i in range(self.SearchAgents_no):
        #     temp = self.objf(self.Positions[i, :])
        #     self.fitness[self.SearchAgents_no-1] = temp
        #     # is leader ?
        #     if temp < self.best_sol:
        #         self.best_sol = temp
        #         self.best_pos = self.Positions[i, :]
        # self.f += self.SearchAgents_no

        # self.fitness = np.array([float("inf") for _ in range(self.SearchAgents_no)])
        # self.fitness[self.SearchAgents_no-1] = self.objf(self.Positions[self.SearchAgents_no-1, :])
        # start work
        print('DE is optimizing  "' + self.objf.__name__ + '"')
        while self.t < self.Max_iter:
            # loop through population
            for i in range(self.SearchAgents_no):
                # 1. Mutation
                # select 3 random solution except current solution
                ids_except_current = [_ for _ in range(self.SearchAgents_no) if _ != i]
                id_1, id_2, id_3 = random.sample(ids_except_current, 3)
                mutant_sol = []
                for d in range(self.dim):
                    d_val = self.Positions[id_1, d] + mutation_factor * (
                        self.Positions[id_2, d] - self.Positions[id_3, d]
                    )
                    # 2. Recombination
                    rn = random.uniform(0, 1)
                    if rn > crossover_ratio:
                        d_val = self.Positions[i, d]
                    # add dimension value to the mutant solution
                    mutant_sol.append(d_val)
                # 3. Replacement / Evaluation
                # clip new solution (mutant)
                mutant_sol = np.clip(mutant_sol, self.lb, self.ub)
                # calc fitness
                mutant_fitness = self.objf(mutant_sol)
                # s.func_evals += 1
                # replace if mutant_fitness is better
                # breakpoint()
                # if i == 10:
                #     breakpoint()
                if mutant_fitness < self.fitness[i]:
                    self.Positions[i, :] = mutant_sol
                    self.fitness[i] = mutant_fitness

                    # update leader
                    if mutant_fitness < self.best_sol:
                        self.best_sol = mutant_fitness
                        self.best_pos = mutant_sol

            self.f += self.SearchAgents_no
            self.convergence_curve[self.t] = self.best_sol
            if self.t % 1 == 0:
                print(
                    ["At iteration " + str(self.t + 1) + " the best fitness is " + str(self.best_sol)]
                )
            # increase iterations
            self.t = self.t + 1
            self.s.R_list.append(1)

            r = np.random.random()
            if r < (1/3):
                self.HHO()
            elif r < (2/3):
                self.DE()
            else:
                self.WOA()
                

        timerEnd = time.time()
        self.s.executionTime = timerEnd - self.timerStart
        self.s.convergence = self.convergence_curve

    ########################################################################################################## WOA
    def WOA(self):
        print('WOA is optimizing  "' + self.objf.__name__ + '"')
        # Main loop
        while self.t < self.Max_iter:
            a = 2 - self.t * ((2) / self.Max_iter)
            # a decreases linearly fron 2 to 0 in Eq. (2.3)
            
            # a2 linearly decreases from -1 to -2 to calculate t in Eq. (3.12)
            a2 = -1 + self.t * ((-1) / self.Max_iter)

            # Update the Position of search agents
            for i in range(0, self.SearchAgents_no):
                r1 = random.random()  # r1 is a random number in [0,1]
                r2 = random.random()  # r2 is a random number in [0,1]

                A = 2 * a * r1 - a  # Eq. (2.3) in the paper
                C = 2 * r2  # Eq. (2.4) in the paper

                b = 1
                #  parameters in Eq. (2.5)
                l = (a2 - 1) * random.random() + 1  #  parameters in Eq. (2.5)

                p = random.random()  # p in Eq. (2.6)

                for j in range(0, self.dim):

                    if p < 0.5:
                        if abs(A) >= 1:
                            rand_leader_index = math.floor(
                                self.SearchAgents_no * random.random()
                            )
                            X_rand = self.Positions[rand_leader_index, :]
                            D_X_rand = abs(C * X_rand[j] - self.Positions[i, j])
                            self.Positions[i, j] = X_rand[j] - A * D_X_rand

                        elif abs(A) < 1:
                            D_Leader = abs(C * self.best_pos[j] - self.Positions[i, j])
                            self.Positions[i, j] = self.best_pos[j] - A * D_Leader

                    elif p >= 0.5:

                        distance2Leader = abs(self.best_pos[j] - self.Positions[i, j])
                        # Eq. (2.5)
                        self.Positions[i, j] = (
                            distance2Leader * math.exp(b * l) * math.cos(l * 2 * math.pi)
                            + self.best_pos[j]
                        )

            for i in range(0, self.SearchAgents_no):
                # Return back the search agents that go beyond the boundaries of the search space
                # Positions[i,:]=checkBounds(Positions[i,:],lb,ub)
                self.Positions[i,:] = np.clip(self.Positions[i, :], self.lb , self.ub)
                # Calculate objective function for each search agent
                self.fitness[i] = self.objf(self.Positions[i, :])
                # Update the leader
                if self.fitness[i] < self.best_sol:  # Change this to > for maximization problem
                    self.best_sol = self.fitness[i]
                    # Update alpha
                    self.best_pos = self.Positions[i, :].copy()  # copy current whale position into the leader position

            self.convergence_curve[self.t] = self.best_sol

            if self.t % 1 == 0:
                print(
                    ["At iteration " + str(self.t +1) + " the best fitness is " + str(self.best_sol)]
                )
            self.t = self.t + 1
            self.s.R_list.append(2)

            r = np.random.random()
            if r < (1/3):
                self.HHO()
            elif r < (2/3):
                self.DE()
            else:
                self.WOA()

        timerEnd = time.time()
        self.s.stopiter = self.f
        self.s.executionTime = timerEnd - self.timerStart
        self.s.convergence = self.convergence_curve




