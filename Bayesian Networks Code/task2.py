import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import entropy
import time

# As in task1 we are reading data from our taining file named "cardiovascular_data-discretized-train.csv", you can also change the file name by replacing it with the file below
trainData = pd.read_csv('cardiovascular_data-discretized-train.csv')

# Here is the structure for our Bayesian Network, it can be changed according to data.
structureBN = [
    ('age', 'target'), ('gender', 'target'), ('height', 'target'),
    ('weight', 'target'), ('ap_hi', 'target'), ('ap_lo', 'target'),
    ('cholesterol', 'target'), ('gluc', 'target'), ('smoke', 'target'),
    ('alco', 'target'), ('active', 'target')
]

# Here we are making an instance of our bayesian modek to copare it with other model as in this task with gaussuan model.
modelBN = BayesianModel(structureBN)
modelBN.fit(trainData)

# Similarlay from task 1 we are discretizing our data
discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
discretizedData = pd.DataFrame(discretizer.fit_transform(trainData[['age', 'height', 'weight', 'ap_hi', 'ap_lo']]),
                                 columns=['age', 'height', 'weight', 'ap_hi', 'ap_lo'])
discretizedData[['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'target']] = trainData[
    ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'target']]

# splitind the data since target table is our Expected output so we are keeping it as our y variable and the rest as input.
xGP = trainData[['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']]
yGP = trainData['target']
XTrainGP, XTestGP, yTrainGP, yTestGP = train_test_split(xGP, yGP, test_size=0.2, random_state=42)

#here we are training our gaussian model
kernel = 1.0 * RBF()
gpModel = GaussianProcessClassifier(kernel=kernel, random_state=42)
startTimeGP = time.time()
gpModel.fit(XTrainGP, yTrainGP)
endTimeGP = time.time()
inferenceTimeGP = endTimeGP - startTimeGP

#Testing and making predictions ehre  
startTimePgp = time.time()
predictionsGP = gpModel.predict(XTestGP)
endTimePgp = time.time()
inferenceTimePgp = endTimePgp - startTimePgp
accuracyGP = accuracy_score(yTestGP, predictionsGP)
f1GP = f1_score(yTestGP, predictionsGP)
balancedAccuracyGP = balanced_accuracy_score(yTestGP, predictionsGP)

# here we are using bayesion network to compare tjhis network with our GP network 
testData = pd.read_csv('cardiovascular_data-discretized-test.csv')
discretizedTestData = pd.DataFrame(discretizer.transform(testData[['age', 'height', 'weight', 'ap_hi', 'ap_lo']]),
                                      columns=['age', 'height', 'weight', 'ap_hi', 'ap_lo'])
inferenceBN = VariableElimination(modelBN)

predictionsBN = []
startTimePbn = time.time()
for _, row in discretizedTestData.iterrows():
    evidenceDict = row.to_dict()
    if 'target' in evidenceDict:
        evidenceDict.pop('target') 
    evidenceDict = {key: int(value) for key, value in evidenceDict.items()}

    predictionBN = inferenceBN.map_query(variables=['target'], evidence=evidenceDict)
    predictionsBN.append(predictionBN['target'])

endTimePbn = time.time()
inferenceTimePbn = endTimePbn - startTimePbn
accuracyBN = accuracy_score(testData['target'], predictionsBN)
f1BN = f1_score(testData['target'], predictionsBN)
balancedAccuracyBN = balanced_accuracy_score(testData['target'], predictionsBN)
klDivergence = entropy(testData['target'].value_counts(normalize=True),
                        pd.Series(predictionsBN).value_counts(normalize=True))


print("Gaussian Process Metrics:")
print("Accuracy:", accuracyGP)
print("F1 Score:", f1GP)
print("Balanced Accuracy:", balancedAccuracyGP)
print("Inference Time:", inferenceTimeGP)
print("\nBayesian Network Metrics:")
print("Accuracy:", accuracyBN)
print("F1 Score:", f1BN)
print("Balanced Accuracy:", balancedAccuracyBN)
print("KL Divergence:", klDivergence)
print("Inference Time:", inferenceTimePbn)
