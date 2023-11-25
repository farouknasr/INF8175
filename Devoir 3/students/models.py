import nn
from backend import PerceptronDataset, RegressionDataset, DigitClassificationDataset
import numpy as np

class PerceptronModel(object):
    def __init__(self, dimensions: int) -> None:
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self) -> nn.Parameter:
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"

        return nn.DotProduct(x,self.get_weights()) # Calcul du produit scalaire entre w et x de l'équation (1)

    def get_prediction(self, x: nn.Constant) -> int:
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"

        # Implémentation de l'équation (1):
        if nn.as_scalar(self.run(x)) >= 0: # Si le produit scalaire est non-négatif
            return 1                       # Retourner 1
        return -1                          # Sinon, retourner -1

    def train(self, dataset: PerceptronDataset) -> None:
        """
        Train the perceptron until convergence.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"

        # Implémentation de la procédure d'entraînement:
        while True: 
            pasBienClass = False                  
            for x, y in dataset.iterate_once(1) : # Tant que tous les exemples ne sont pas bien classés, itérer (1 à 1)
                # Implémentation de l'équation (2):
                if self.get_prediction(x) != nn.as_scalar(y): # Mettre à jour le poids si pas bien classé
                    self.w.update(x,nn.as_scalar(y))
                    pasBienClass = True 
            # Mettre fin à l'entraînement si précesion est de 100%
            if not pasBienClass:
                break


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self) -> None:
        # Initialize your model parameters here
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"

        self.matCouchesCach = []         # Initialiser la matrice des couches cachées
        self.dimCouchesCach = [300, 300] # Entre 100 et 400
        self.tailleMiniBatch = 1         # Entre 1 et la taille de l'ensemble des données
        self.alpha = 0.1                 # Taux d'apprentissage entre 0.001 et 1.0
        self.nbCouchesCach = 2           # Entre 1 et 3  

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"

        y = x
        for couche in self.matCouchesCach:        
            # Implémentation de l'équation (4)
            y = nn.AddBias(nn.Linear(nn.ReLU(nn.AddBias(nn.Linear(y, couche[0]), couche[1])), couche[2]), couche[3])
        return y

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"

        return nn.SquareLoss(self.run(x), y) # Erreur quadratique à minimiser

    def train(self, dataset: RegressionDataset) -> None:
        """
        Trains the model.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"

        self.tailleMiniBatch = int(0.1*dataset.x.shape[0]) # 10% des données à entraîner 
        while len(dataset.x) % self.tailleMiniBatch != 0:  # Verification de divisibilité
            self.miniBatchTaille += 1

        self.matCouchesCach = [
            ([
                nn.Parameter(dataset.x.shape[1], self.dimCouchesCach[i]),
                nn.Parameter(1,self.dimCouchesCach[i]),
                nn.Parameter(self.dimCouchesCach[i], dataset.x.shape[1]),
                nn.Parameter(1,1)
            ])
            for i in range(self.nbCouchesCach)
        ]
        
        while True:
            pertes = []
            for x, y in dataset.iterate_once(self.tailleMiniBatch):
                perte = self.get_loss(x,y)
                listeParam = []
                for couche in self.matCouchesCach:
                    for param in couche:
                        listeParam.append(param)
                gradients = nn.gradients(perte, listeParam) # Calcul des gradients
                for i in range(len(listeParam)):
                    param = listeParam[i]
                    param.update(gradients[i], -self.alpha) # Mise à jour du paramètre
                pertes.append(nn.as_scalar(perte))
            # T.q. EQM est supérieure à 0.02, itérer:
            if np.mean(pertes) <= 0.02:
                break


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self) -> None:
        # Initialize your model parameters here
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"

        self.alpha = 0.09
        self.tailleMiniBatch = 1
        self.matCouchesCach = [
            [
                nn.Parameter(784, 150),
                nn.Parameter(1, 150),
                nn.Parameter(150, 784),
                nn.Parameter(1,784)
            ],
            [
                nn.Parameter(784, 100),
                nn.Parameter(1, 100),
                nn.Parameter(100, 10),
                nn.Parameter(1,10)
            ],
        ]
        

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"

        y = x
        for couche in self.matCouchesCach:
            y = nn.AddBias(nn.Linear(nn.ReLU(nn.AddBias(nn.Linear(y, couche[0]), couche[1])), couche[2]), couche[3])
        return y

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset: DigitClassificationDataset) -> None:
        """
        Trains the model.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        
        self.tailleMiniBatch = int(0.005 * dataset.x.shape[0])
        while len(dataset.x) % self.tailleMiniBatch != 0:  
            self.tailleMiniBatch += 1

        while True:
            for x, y in dataset.iterate_once(self.tailleMiniBatch):
                perte = self.get_loss(x,y)
                listeParam = []
                for couche in self.matCouchesCach:
                    for param in couche:
                        listeParam.append(param)
                gradients = nn.gradients(perte, listeParam)
                for i in range(len(listeParam)):
                    param = listeParam[i]
                    param.update(gradients[i], -self.alpha)
            # T.q. précision est inférieure à 97%, itérer:
            if dataset.get_validation_accuracy() > 0.97:
                break

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        self.learningRate = 0.09
        self.batch_size = 1
        self.hiddenLayers = [
            [
                nn.Parameter(784, 150),
                nn.Parameter(1, 150),
                nn.Parameter(150, 784),
                nn.Parameter(1,784)
            ],
            [
                nn.Parameter(784, 100),
                nn.Parameter(1, 100),
                nn.Parameter(100, 10),
                nn.Parameter(1,10)
            ],
        ]
    
    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        y = x
        for layer in self.hiddenLayers:
            # Un réseau de neurones à deux couches y = W2*ReLU(W1*X+b1)+b2
            y = nn.AddBias(nn.Linear(nn.ReLU(nn.AddBias(nn.Linear(y, layer[0]), layer[1])), layer[2]), layer[3])
        return y
    
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        return nn.SoftmaxLoss(self.run(x), y)      
   
    def train(self, dataset):
        """
        Trains the model.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        # Sous-ensemble d'entrainement qui correspond a 0.5% des données.
        self.batch_size = int(0.005 * dataset.x.shape[0])
        # Verifie que la taille totale du jeu de données soit divisible par la taille du mini-batch.
        while len(dataset.x) % self.batch_size != 0:
            self.batch_size += 1
        # Itére sur le dataSet jusqu'a atteindre une précision d’au moins 97%.
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x,y)
                # Construit une lists des paramètres.
                parameterList = []
                for layer in self.hiddenLayers:
                    for parameter in layer:
                        parameterList.append(parameter)
                # Calcule le gradient de chaque paramètre et mes a jour le réseau.
                gradients = nn.gradients(loss, parameterList)
                for i in range(len(parameterList)):
                    parameter = parameterList[i]
                    parameter.update(gradients[i], -self.learningRate)
            # Si on attiend une précision d’au moins 97% on arrete l'entrainement.
            if dataset.get_validation_accuracy() > 0.972:
                break

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self) -> None:
        # Initialize your model parameters here
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"

        self.alpha = 0.09
        self.tailleMiniBatch = 1
        self.matCouchesCach = [
            [
                nn.Parameter(784, 150),
                nn.Parameter(1, 150),
                nn.Parameter(150, 784),
                nn.Parameter(1,784)
            ],
            [
                nn.Parameter(784, 100),
                nn.Parameter(1, 100),
                nn.Parameter(100, 10),
                nn.Parameter(1,10)
            ],
        ]

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"

        y = x
        for couche in self.matCouchesCach:
            y = nn.AddBias(nn.Linear(nn.ReLU(nn.AddBias(nn.Linear(y, couche[0]), couche[1])), couche[2]), couche[3])
        return y

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset: DigitClassificationDataset) -> None:
        """
        Trains the model.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        
        self.tailleMiniBatch = int(0.005 * dataset.x.shape[0])
        while len(dataset.x) % self.tailleMiniBatch != 0:  
            self.tailleMiniBatch += 1

        while True:
            for x, y in dataset.iterate_once(self.tailleMiniBatch):
                perte = self.get_loss(x,y)
                listeParam = []
                for couche in self.matCouchesCach:
                    for param in couche:
                        listeParam.append(param)
                gradients = nn.gradients(perte, listeParam)
                for i in range(len(listeParam)):
                    param = listeParam[i]
                    param.update(gradients[i], -self.alpha)
            # T.q. précision est inférieure à 97%, itérer:
            if dataset.get_validation_accuracy() > 0.97:
                break