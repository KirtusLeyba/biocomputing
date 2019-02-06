from thread import *

import matplotlib.pyplot as plt
import numpy as np
import sys

from sklearn.decomposition import PCA
from matplotlib import animation
import matplotlib.cm as cm
import matplotlib as mpl


##random seed
if(len(sys.argv) >= 2):
	np.random.seed(int(sys.argv[1]))

##run type
if(len(sys.argv) >= 3):
    runtype = int(sys.argv[2])
else:
    runtype = 0


def makeDistMatrix(threadList):
    distMat = np.zeros((len(threadList.genes), len(threadList.genes)))
    for i in range(len(threadList.genes)):
        for j in range(len(threadList.genes)):
            distMat[i][j] = manDist(threadList.genes[i], threadList.genes[j])
    return distMat

def run_pca(population):
	pca = PCA(2)
	x = np.zeros((len(population), len(population[0].genes)))
	for i in range(len(population)):
		for j in range(len(population[i].genes)):
			x[i][j] = population[i].genes[j].g
	pca = pca.fit(x)
	return pca

def getPCATransform(pca, population):
	x = np.zeros((len(population), len(population[0].genes)))
	for i in range(len(population)):
		for j in range(len(population[i].genes)):
			x[i][j] = population[i].genes[j].g
	return pca.transform(x)


### returns the cpuCost (in cpuCycles) of an ordering of threads
### Times the energyConsumption (in joules)
def cpuCost(numCores, numThreads, threadList, adjMatrix, cpuWidth):
    T = np.zeros(numCores) ### T is the time left on each CPU before a thread switch
    C_star = np.zeros(numCores) ### C_star is the total cpu cost accumulated on each cpu core
    z = 0 ### z is the index of the next thread to add onto the cpu

    ### Step 1: Load first numCores threads onto the cores
    for i in range(numCores):
        T[i] += threadList.genes[z].cost
        z += 1
        x = int(i%cpuWidth)
        y = int(i/cpuWidth)
        threadList.genes[z].x=x
        threadList.genes[z].y=y

    ### Step 2: Iteratively add threads and accumulate cost
    while(z < numThreads):
        ### Find the shortest thread currently on the cpu
        minThreadIndex = list(T).index(min(T))
        minThreadCost = T[minThreadIndex]
        ### add this cost to all cores
        ### and subtract time from all running cores
        for i in range(numCores):
            C_star[i] += minThreadCost
            T[i] -= minThreadCost
        ### replace shortest thread with next thread in order
        T[minThreadIndex] = threadList.genes[z].cost
        x = int(minThreadIndex%cpuWidth)
        y = int(minThreadIndex/cpuWidth)
        threadList.genes[z].x=x
        threadList.genes[z].y=y
        z+=1

    return np.max(C_star)*energyConsumption(adjMatrix, threadList)

### returns the manhatten distance between two threads
def manDist(a, b):
    return abs(a.x-b.x) + abs(a.y-b.y)

### returns the energy consumption given an adjacency matrix
### and a thread list
def energyConsumption(adjMatrix, threadList):
    e = 0.0

    for i in range(len(adjMatrix)):
        for j in range(len(adjMatrix[i])):
            dist = manDist(threadList.genes[i], threadList.genes[j])
            e += adjMatrix[i][j]*dist

    return e

### mutation operator, performs a swap mutation on an individual
def mutate(threadList, mutateProb):
    l = len(threadList.genes)
    for t in threadList.genes:
        roll = np.random.random()
        if(roll < mutateProb):
            ### swap mutation
            a = np.random.randint(0,l)
            b = np.random.randint(0,l)
            threadList.genes[a], threadList.genes[b] = threadList.genes[b], threadList.genes[a]
    return threadList

### crossover operator, combines two threadlists and returns 1 offspring
def crossover(a, b):
    child = Indiv()
    l = len(a.genes)
    for i in range(l):
        roll = np.random.random()
        if(roll < 0.5):
            for j in range(l):
                if(not a.genes[j] in child.genes):
                    child.genes.append(a.genes[j])
                    break
        else:
            for j in range(l):
                if(not b.genes[j] in child.genes):
                    child.genes.append(b.genes[j])
                    break
    return child

### selection operator. Generates new population
def select(pop, adjMatrix, mutateProb):
    newPop = []

    while(len(newPop) < len(pop)):
        ia1 = np.random.randint(0, len(pop))
        ia2 = np.random.randint(0, len(pop))
        e1 = pop[ia1].fitness
        e2 = pop[ia2].fitness
        if(e1 <= e2):
            a = pop[ia1]
        else:
            a = pop[ia2]

        ib1 = np.random.randint(0, len(pop))
        ib2 = np.random.randint(0, len(pop))
        e1 = pop[ib1].fitness
        e2 = pop[ib2].fitness
        if(e1 <= e2):
            b = pop[ib1]
        else:
            b = pop[ib2]
    
        newPop.append(crossover(a, b))
        newPop[-1] = mutate(newPop[-1], mutateProb)


    return newPop

### creates a random individual
def randomIndividual(isize, gridSize, costs):
    indiv = Indiv()
    g = 0
    for i in range(gridSize):
        for j in range(gridSize):
            indiv.genes.append(Thread(i, j, g, costs[g]))
            g+=1
    indiv.genes = np.random.permutation(indiv.genes)
    return indiv

### loading adjMatrix
adjMatrixFile = "../data/adj_matrix.txt"
adjMatrix = []
with open(adjMatrixFile) as fp:
    line = fp.readline()
    while(line):
        adjMatrix.append([])
        s = line.split('\t')
        for n in s:
            adjMatrix[-1].append(float(n))
        line = fp.readline()

### loading thread costs
costFile = "../data/CPU_cost.txt"
costs = []
with open(costFile) as fp:
    line = fp.readline()
    line = fp.readline()
    while(line):
        costs.append(float(line.split('\t')[1]))
        line = fp.readline()

#### GA parameters
if(runtype == 0):
    mutateProb = 0.0001
    popSize = 100
    generations = 100
    indivSize = 64
    gridSize = 8

    interestingGenerations = [58, 59, 78, 79]

    #### Running the single objective problem
    pop = []
    for i in range(popSize):
        pop.append(randomIndividual(indivSize, gridSize, costs))

    fitnessArray = []

    pca = run_pca(pop)

    pcaList = []

    interestingIndividuals = []

    for i in range(generations):

        x = getPCATransform(pca, pop)
        pcaList.append(x)

        fitnessArray.append([])
        for n in pop:
            f = energyConsumption(adjMatrix, n)
            fitnessArray[-1].append(f)
            n.fitness = f

        if(i in interestingGenerations):
            interestingIndividuals.append(pop[fitnessArray[-1].index(min(fitnessArray[-1]))])

        newPop = select(pop, adjMatrix, mutateProb)
        pop = newPop

        print("Generation " + str(i) + ", min: " + str(min(fitnessArray[-1])))

    x = []
    ymean = []
    ymin = []
    ymax = []

    i = 0
    for f in fitnessArray:
        x.append(i)
        ymean.append(np.mean(f))
        ymin.append(min(f))
        ymax.append(max(f))
        i += 1

    plt.figure()
    plt.plot(x,ymean, label="Mean Energy Consumption")
    plt.plot(x, ymax, label="Max Energy Consumption")
    plt.plot(x, ymin, label="Min Energy Consumption")
    plt.legend()
    plt.savefig("../vis/ga_single_perf.eps")


    ### visualizing individuals:
    ### Initial Worst
    # plt.figure()
    # plt.title("Initial Worst Individual")
    # plt.imshow(makeDistMatrix(interestingIndividuals[0]), cmap=cm.Reds)
    # plt.colorbar(label="Thread Distance")
    # plt.xlabel("Thread Number")
    # plt.ylabel("Thread Number")
    # plt.show()
    for i in range(len(interestingGenerations)):
        plt.figure()
        gen = interestingGenerations[i]
        plt.title("Best Individual of Generation " + str(gen))
        plt.imshow(makeDistMatrix(interestingIndividuals[i]), cmap=cm.Reds)
        plt.colorbar(label="Thread Distance")
        plt.xlabel("Thread Number")
        plt.ylabel("Thread Number")
        plt.savefig("../vis/gen" + str(gen) + ".eps")

    ## differences
    for i in range(0,len(interestingGenerations),2):
        plt.figure()
        gen = interestingGenerations[i]
        plt.title("Difference Between generation " + str(gen) + " and generation " + str(gen+1))
        plt.imshow(makeDistMatrix(interestingIndividuals[i+1])-makeDistMatrix(interestingIndividuals[i]), cmap=cm.Reds)
        plt.colorbar(label="Thread Distance")
        plt.xlabel("Thread Number")
        plt.ylabel("Thread Number")
        plt.savefig("../vis/gen" + str(gen+1) + "minusGen" + str(gen) + ".eps")

    ### color stuff
    n = mpl.colors.Normalize(vmin = np.amin(fitnessArray), vmax=np.amax(fitnessArray))
    m = mpl.cm.ScalarMappable(norm=n, cmap=mpl.cm.viridis_r)

    x,y = pcaList[0].T
    means = [[np.mean(x)], [np.mean(y)]]
    fig, ax = plt.subplots()
    scat = ax.scatter(x,y, c = fitnessArray[0], cmap=cm.viridis_r)
    scat.set_facecolor(m.to_rgba(fitnessArray[0]))
    scat.set_clim(vmin = np.amin(fitnessArray), vmax=np.amax(fitnessArray))
    line2, = ax.plot(means, color='red')
    ax.set_title("Generation " + str(i))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(scat, label="Energy Consumption(J)",\
        ticks = [np.amin(fitnessArray), (np.amax(fitnessArray)+np.amin(fitnessArray))/2.0, np.amax(fitnessArray)])

    def init():
        x,y = pcaList[0].T
        means = [[np.mean(x)], [np.mean(y)]]
        scat.set_offsets(pcaList[0])
        scat.set_facecolor(m.to_rgba(fitnessArray[0]))
        ax.set_title("Generation " + str(i))
        means[0].append(np.mean(x))
        means[1].append(np.mean(y))
        line2.set_data(means[0], means[1])
        return scat,

    def animate(i):
        x,y = pcaList[i].T
        scat.set_offsets(pcaList[i])
        scat.set_facecolor(m.to_rgba(fitnessArray[i]))
        ax.set_title("Generation " + str(i))
        means[0].append(np.mean(x))
        means[1].append(np.mean(y))
        line2.set_data(means[0], means[1])
        return scat,

    fps = 5
    anim = animation.FuncAnimation(fig, animate, \
        init_func=init, frames = generations, interval=1000/fps, blit=False)

    # Set up formatting for the movie files
    anim.save('../vis/evolve.gif', writer='imagemagick', fps=5)


#### Running the multi objective problem
if(runtype == 1):
    mutateProb = 0.0001
    popSize = 100
    generations = 100
    indivSize = 64
    gridSize = 8

    numCores = 16
    numThreads = 64
    cpuWidth = 4

    interestingGenerations = []
    pop = []
    for i in range(popSize):
        pop.append(randomIndividual(indivSize, gridSize, costs))

    fitnessArray = []

    pca = run_pca(pop)

    pcaList = []

    interestingIndividuals = []

    for i in range(generations):

        x = getPCATransform(pca, pop)
        pcaList.append(x)

        fitnessArray.append([])
        for n in pop:
            f = cpuCost(numCores, numThreads, n, adjMatrix, cpuWidth)
            fitnessArray[-1].append(f)
            n.fitness = f

        if(i in interestingGenerations):
            interestingIndividuals.append(pop[fitnessArray[-1].index(min(fitnessArray[-1]))])

        newPop = select(pop, adjMatrix, mutateProb)
        pop = newPop

        print("Generation " + str(i) + ", min: " + str(min(fitnessArray[-1])))

    x = []
    ymean = []
    ymin = []
    ymax = []

    i = 0
    for f in fitnessArray:
        x.append(i)
        ymean.append(np.mean(f))
        ymin.append(min(f))
        ymax.append(max(f))
        i += 1

    plt.figure()
    plt.plot(x,ymean, label="Mean Energy Consumption")
    plt.plot(x, ymax, label="Max Energy Consumption")
    plt.plot(x, ymin, label="Min Energy Consumption")
    plt.legend()
    plt.savefig("../vis/multi/ga_multi_perf.eps")


    ### visualizing individuals:
    ### Initial Worst
    # plt.figure()
    # plt.title("Initial Worst Individual")
    # plt.imshow(makeDistMatrix(interestingIndividuals[0]), cmap=cm.Reds)
    # plt.colorbar(label="Thread Distance")
    # plt.xlabel("Thread Number")
    # plt.ylabel("Thread Number")
    # plt.show()
    for i in range(len(interestingGenerations)):
        plt.figure()
        gen = interestingGenerations[i]
        plt.title("Best Individual of Generation " + str(gen))
        plt.imshow(makeDistMatrix(interestingIndividuals[i]), cmap=cm.Reds)
        plt.colorbar(label="Thread Distance")
        plt.xlabel("Thread Number")
        plt.ylabel("Thread Number")
        plt.savefig("../vis/multi/gen" + str(gen) + ".eps")

    ## differences
    for i in range(0,len(interestingGenerations),2):
        plt.figure()
        gen = interestingGenerations[i]
        plt.title("Difference Between generation " + str(gen) + " and generation " + str(gen+1))
        plt.imshow(makeDistMatrix(interestingIndividuals[i+1])-makeDistMatrix(interestingIndividuals[i]), cmap=cm.Reds)
        plt.colorbar(label="Thread Distance")
        plt.xlabel("Thread Number")
        plt.ylabel("Thread Number")
        plt.savefig("../vis/multi/gen" + str(gen+1) + "minusGen" + str(gen) + ".eps")

    ### color stuff
    n = mpl.colors.Normalize(vmin = np.amin(fitnessArray), vmax=np.amax(fitnessArray))
    m = mpl.cm.ScalarMappable(norm=n, cmap=mpl.cm.viridis_r)

    x,y = pcaList[0].T
    means = [[np.mean(x)], [np.mean(y)]]
    fig, ax = plt.subplots()
    scat = ax.scatter(x,y, c = fitnessArray[0], cmap=cm.viridis_r)
    scat.set_facecolor(m.to_rgba(fitnessArray[0]))
    scat.set_clim(vmin = np.amin(fitnessArray), vmax=np.amax(fitnessArray))
    line2, = ax.plot(means, color='red')
    ax.set_title("Generation " + str(i))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(scat, label="Energy Consumption(J)",\
        ticks = [np.amin(fitnessArray), (np.amax(fitnessArray)+np.amin(fitnessArray))/2.0, np.amax(fitnessArray)])

    def init():
        x,y = pcaList[0].T
        means = [[np.mean(x)], [np.mean(y)]]
        scat.set_offsets(pcaList[0])
        scat.set_facecolor(m.to_rgba(fitnessArray[0]))
        ax.set_title("Generation " + str(i))
        means[0].append(np.mean(x))
        means[1].append(np.mean(y))
        line2.set_data(means[0], means[1])
        return scat,

    def animate(i):
        x,y = pcaList[i].T
        scat.set_offsets(pcaList[i])
        scat.set_facecolor(m.to_rgba(fitnessArray[i]))
        ax.set_title("Generation " + str(i))
        means[0].append(np.mean(x))
        means[1].append(np.mean(y))
        line2.set_data(means[0], means[1])
        return scat,

    fps = 5
    anim = animation.FuncAnimation(fig, animate, \
        init_func=init, frames = generations, interval=1000/fps, blit=False)

    # Set up formatting for the movie files
    anim.save('../vis/multi/evolveMulti.gif', writer='imagemagick', fps=5)

if(runtype == 2):
    runCount = 20
    mutateProb = 0.0001
    popSize = 50
    generations = 100
    indivSize = 64
    gridSize = 8
    maxFits = np.zeros((generations, runCount))
    meanFits = np.zeros((generations, runCount))
    minFits = np.zeros((generations, runCount))

    for r in range(runCount):
        #### Running the single objective problem
        pop = []
        for i in range(popSize):
            pop.append(randomIndividual(indivSize, gridSize, costs))

        fitnessArray = []

        pca = run_pca(pop)

        pcaList = []
        for i in range(generations):

            x = getPCATransform(pca, pop)
            pcaList.append(x)

            fitnessArray.append([])
            for n in pop:
                f = energyConsumption(adjMatrix, n)
                fitnessArray[-1].append(f)
                n.fitness = f
            maxFits[i, r] = max(fitnessArray[-1])
            meanFits[i, r] = np.mean(fitnessArray[-1])
            minFits[i, r] = min(fitnessArray[-1])
            newPop = select(pop, adjMatrix, mutateProb)
            pop = newPop

            print("Generation " + str(i) + ", min: " + str(min(fitnessArray[-1])))

x = []
meanMax = []
meanMin = []
meanMean = []
for i in range(generations):
    x.append(i)
    meanMax.append(np.mean(maxFits[i]))
    meanMin.append(np.mean(minFits[i]))
    meanMean.append(np.mean(meanFits[i]))

plt.figure()
plt.title("Multiple Runs")
plt.plot(x, meanMean, label="Mean Mean Energy Consumption")
plt.plot(x, meanMax, label="Mean Max Energy Consumption")
plt.plot(x, meanMin, label="Mean Min Energy Consumption")
plt.xlabel("Generation")
plt.ylabel("Fitness (Energy Consumption)")
plt.legend()
plt.savefig("../vis/means.eps")