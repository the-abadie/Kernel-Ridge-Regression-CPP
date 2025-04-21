import numpy as np
import numpy.linalg as lin
import sys
from copy import deepcopy

argTrain  = int(sys.argv[1])
argExtend = True if int(sys.argv[2]) == 1 else False
stratify  = True if int(sys.argv[3]) == 1 else False
k         = int(sys.argv[4])
inpath    = str(sys.argv[5])

sigmaMax :int = 10
lambdaMax:int = 10
sigmas  = np.array([2**i for i in  range(0,sigmaMax)])
lambdas = np.array([10**i for i in range(-lambdaMax, 0)])

print("Base training set size:", argTrain)
if argExtend: print("Extending Data. This will result in a fivefold increase of training data.")

def getZ(label:str) -> int:    
    elements="H   He\
        Li  Be  B   C   N   O   F   Ne\
        Na  Mg  Al  Si  P   S   Cl  Ar\
        K   Ca  Sc  Ti  V   Cr  Mn  Fe  Co  Ni  Cu  Zn  Ga  Ge  As  Se  Br  Kr\
        Rb  Sr  Y   Zr  Nb  Mo  Tc  Ru  Rh  Pd  Ag  Cd  In  Sn  Sb  Te  I   Xe\
        Cs  Ba  La  Ce  Pr  Nd  Pm  Sm  Eu  Gd  Tb  Dy  Ho  Er  Tm  Yb\
        Lu  Hf  Ta  W   Re  Os  Ir  Pt  Au  Hg  Tl  Pb  Bi  Po  At  Rn\
        Fr  Ra  Ac  Th  Pa  U".split()    
    
    return elements.index(label)+1

def importQM7(structure_file:str, energy_file:str):
    """
    Return: Z, R, E\n
    Z: list of 1D-arrays containing atomic identities\n
    R: list of 2D-arrays containing atomic positions\n
    E: 1D-array containing atomization energy\n
    """
    structures = open(structure_file,  'r').readlines()

    Z = []
    R = []
    E = []
    n_max = 0

    for line in range(len(structures)):
        x = structures[line].split()

        #Check for start of molecule structure data:
        if len(x) == 1:
            n_atoms = int(x[0])
            if n_atoms > n_max: n_max = n_atoms

            Zs   = np.zeros(n_atoms)
            xyzs = np.zeros((n_atoms, 3))

            #Go through every atom in the molecule:
            atom_index = 0
            for j in range(line+2, line+2+n_atoms):
                Zs  [atom_index] = getZ(structures[j].split()[0])
                xyzs[atom_index] = np.array([float(val) for val in structures[j].split()[1:]])

                atom_index += 1
            
            Z.append(Zs)
            R.append(xyzs)
        
    file = open(energy_file,  'r').readlines()
    for line in range(len(file)):
        E.append(float(file[line].split()[0]))
    
    return Z, R, E

def coulomb_matrix(Z, R, n_max, f):
    n_max  = n_max

    #Generate Descriptors, eigenvalues of Coulomb Matrix M

    n_atoms = len(Z)

    M = np.zeros((n_atoms, n_atoms))

    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                M[i][j] = 0.5 * (Z[i])**2.4
            else:
                M[i][j] = (Z[i]*Z[j]) / (lin.norm(R[i] - R[j])**2) / f

    return M
    
def coulomb_eigenvalues(Z, R, n_max:int, outputname:str, extend:bool = False):

    #Generate Descriptors, eigenvalues of Coulomb Matrix M
    if extend:
        fs = [1., 2/3, 0.995, 1.005, 3.]
        Z2 = deepcopy(Z)*len(fs)
        R2 = deepcopy(R)*len(fs) 

    else:
        fs = [1.]
        Z2 = deepcopy(Z)
        R2 = deepcopy(R) 

    coulomb_eVs = np.zeros((len(Z2), n_max))

    for k in range(len(Z2)):
    
        f = fs[k//len(Z)]
        M = coulomb_matrix(Z2[k], R2[k], n_max, f=f)

        eigenValues = lin.eigvals(M)
        sorted_eVal = np.array(sorted(eigenValues, key=abs, reverse=True))

        n_atoms = len(sorted_eVal)
        #Append 0s to match molecule with largest number of eigenvalues
        if n_atoms == n_max:
            coulomb_eVs[k] = sorted_eVal
        else:
            coulomb_eVs[k] = np.concatenate((sorted_eVal, [0]*(n_max-n_atoms)))

    np.savetxt(fname=outputname, X=coulomb_eVs, delimiter=" ", newline="\n")
    return coulomb_eVs

def stratSplit(nData:int, nTrain:int, k:int, y):
    assert nData  > nTrain
    assert nData == len(y)

    sortedIndex = np.argsort(y)
    strata = [[] for _ in range(k)]

    max_strata = nTrain // k
    
    for i in range(k):
        sample = i
        selected = 0 
        
        while selected < nData:
            if sample >= nData: break
            strata[i].append(sortedIndex[sample])
            sample   += k
            selected += 1

    for i in range(len(strata)):
        np.random.shuffle(strata[i])
    
    trainingIndex = np.array([])
    testingIndex  = np.array([])

    for i in range(k):
        per_strata = max_strata

        if len(trainingIndex) + max_strata > nTrain:
            per_strata = nTrain - len(trainingIndex)

        if (i == k-1) and (len(trainingIndex) + per_strata < nTrain):
            per_strata = nTrain - len(trainingIndex)
            
        trainingIndex = np.concatenate((trainingIndex, strata[i][:per_strata]), 
                                        casting="unsafe", dtype=int)
        testingIndex  = np.concatenate((testingIndex,  strata[i][per_strata:]), 
                                        casting="unsafe", dtype=int)

    assert len(trainingIndex) == nTrain        , \
        f"Unexpected array size. {len(trainingIndex)} != {nTrain}"
        
    assert len(testingIndex ) == nData - nTrain, \
        f"Unexpected array size. {len(trainingIndex)} != {nData - nTrain}"
    
    return trainingIndex, testingIndex

def stratified_k_fold(X, y, k:int = 5):
    assert len(X) == len(y), "X and y must have same length."
    assert k > 1           , "k must be greater than 1."

    sortedXy = np.array(sorted(np.column_stack((X, y)),
                  key = lambda x: x[-1]))
    
    strata = []

    n_per_fold = len(X) // k
    for i in range(k-1):
        strata.append(sortedXy[i*n_per_fold : (i+1)*n_per_fold])
    strata.append(sortedXy[(k-1)*n_per_fold:])

    for stratum in strata:
        np.random.shuffle(stratum)

    folds = [[] for i in range(k)]

    for stratum in strata:
        index = 0
        for i in range(len(stratum)):
            folds[index].append(stratum[i])

            index += 1
            if index % 5 == 0: index = 0

    folded = strata[0]

    for i in range(1, k):
        folded = np.concat((folded, strata[i]))

    np.savetxt(inpath + "/coulomb_train.txt", folded[:, :-1])
    np.savetxt(inpath + "/PBE0_train.txt"   , folded[:, -1])

model_size = argTrain

Z_small, R_small, E_small = importQM7(structure_file = inpath + "qm7_small.txt", 
                                      energy_file    = inpath + "PBE0_small.txt")
Z_rest,  R_rest,  E_rest  = importQM7(structure_file = inpath + "qm7_rest.txt", 
                                      energy_file    = inpath + "PBE0_rest.txt")

strat_train, strat_test = stratSplit(nData  = len(Z_rest),
                                     nTrain = model_size - len(Z_small),
                                     k = 5,
                                     y = E_rest)

Z_train, R_train, E_train = Z_small, R_small, E_small
Z_test , R_test , E_test  = [], [], []


for i in range(len(strat_train)):
    Z_train.append(Z_rest[strat_train[i]])
    R_train.append(R_rest[strat_train[i]])
    E_train.append(E_rest[strat_train[i]])

for i in range(len(strat_test)):
    Z_test.append(Z_rest[strat_test[i]])
    R_test.append(R_rest[strat_test[i]])
    E_test.append(E_rest[strat_test[i]])

fs = [2/3, 0.995, 1.005, 3]
if argExtend:
    E_super = deepcopy(E_train) * 5

    index = 0
    for n in range(len(fs)):
        index = len(E_train)*(n+1)

        for i in range(index, index+len(E_train)):
            if   (fs[n] == 2/3  ) or (fs[n] == 3):
                E_super[i] = 0
            elif (fs[n] == 0.995) or (fs[n] == 1.005):
                E_super[i] *= 1.005

    E_train = E_super

if (len(max(Z_train, key=len)) != len(max(Z_test, key=len))):
    print("Mismatch in maximum molecule size between training and testing sets. Passing n_max to eigenvalue generator.")

n_max = max(len(max(Z_train, key=len)),
            len(max(Z_test,  key=len)))

training_data = coulomb_eigenvalues(Z = Z_train, R = R_train, n_max = n_max, 
                                    outputname = inpath + "coulomb_train.txt", 
                                    extend=argExtend)
training_trgt = np.array(E_train)

print(np.shape(training_data), np.shape(training_trgt))

np.savetxt(fname= inpath + "PBE0_train.txt", X=training_trgt)

if stratify:
    stratified_k_fold(training_data, training_trgt, k=k)
    

testing_data  = coulomb_eigenvalues(Z = Z_test, R = R_test, n_max = n_max, 
                                    outputname = inpath + "coulomb_test.txt",
                                    extend = False)
testing_trgt  = np.array(E_test)
np.savetxt(fname = inpath + "PBE0_test.txt", X=testing_trgt)

np.savetxt(fname = inpath + "sigmas.txt" , X=sigmas )
np.savetxt(fname = inpath + "lambdas.txt", X=lambdas)
