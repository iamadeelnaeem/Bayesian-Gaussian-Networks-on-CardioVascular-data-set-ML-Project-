import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import entropy
import time

#It is used to read data from our training data set, if you want to change any file you have to place that file in the same 
# directory as this code and than just replace training file name with 'cardiovascular_data-discretized-train.csv' in commas. 
trainingData = pd.read_csv('cardiovascular_data-discretized-train.csv')

# Here the structure is defined for our Bayesian Network, You can change the structure depending up-on your dataset
structure = [
    ('age', 'target'), ('gender', 'target'), ('height', 'target'),
    ('weight', 'target'), ('ap_hi', 'target'), ('ap_lo', 'target'),
    ('cholesterol', 'target'), ('gluc', 'target'), ('smoke', 'target'),
    ('alco', 'target'), ('active', 'target')
]

# Here object of bayesian model is being made to use it and it is being imported from pgmpy library
model = BayesianModel(structure)
model.fit(trainingData)

# Here is the function to descritize our data from ranges to specific values.
#as our input ranges from 100 to 180 so we descritize our large values to just 1 or 2 or 3.
def discretizingData(data, features):
    discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
    return pd.DataFrame(discretizer.fit_transform(data[features]), columns=features)

fd = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
dData = discretizingData(trainingData, fd)
dData['gender'] = trainingData['gender']
dData['cholesterol'] = trainingData['cholesterol']
dData['gluc'] = trainingData['gluc']
dData['smoke'] = trainingData['smoke']
dData['alco'] = trainingData['alco']
dData['active'] = trainingData['active']
dData['target'] = trainingData['target']

testingData = pd.read_csv('cardiovascular_data-discretized-test.csv')
dTestingData = discretizingData(testingData, fd)
dTestingData['gender'] = testingData['gender']
dTestingData['cholesterol'] = testingData['cholesterol']
dTestingData['gluc'] = testingData['gluc']
dTestingData['smoke'] = testingData['smoke']
dTestingData['alco'] = testingData['alco']
dTestingData['active'] = testingData['active']
dTestingData['target'] = testingData['target']

inference = VariableElimination(model)

#From here querries are being written you can also add your own querry
queryTarget0 = inference.query(variables=['target'], evidence={'age': 2, 'height': 3, 'weight': 3, 'ap_hi': 3,
                                                                   'ap_lo': 3, 'cholesterol': 1, 'gluc': 1,
                                                                   'smoke': 0, 'alco': 0, 'active': 1})
queryTarget1 = inference.query(variables=['target'], evidence={'age': 2, 'height': 3, 'weight': 3, 'ap_hi': 3,
                                                                   'ap_lo': 3, 'cholesterol': 1, 'gluc': 1,
                                                                   'smoke': 0, 'alco': 0, 'active': 1})

predictions = []
probs = []
startTime = time.time()
print("Number of iterations are:")
i=1
for _, row in dTestingData.iterrows():
    prediction = inference.map_query(variables=[], evidence=row.to_dict())
    predictions.append(prediction['target'])
    print(i)
    i=i+1

endTime = time.time()
inferenceTime = endTime - startTime
accuracy = accuracy_score(dTestingData['target'], predictions)
f1 = f1_score(dTestingData['target'], predictions)
balancedAccuracy = balanced_accuracy_score(dTestingData['target'], predictions)
klDivergence = entropy(dTestingData['target'].value_counts(normalize=True),
                        pd.Series(predictions).value_counts(normalize=True))
brierScore = ((dTestingData['target'] - pd.Series(probs)) ** 2).mean()

print("P(target=0|evidence):", queryTarget0.values[0])
print("P(target=1|evidence):", queryTarget1.values[1])
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Balanced Accuracy:", balancedAccuracy)
print("KL Divergence:", klDivergence)
print("Brier Score:", brierScore)
print("Inference Time:", inferenceTime)
