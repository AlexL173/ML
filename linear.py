import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf


def build_model(my_learning_rate):
    """Create and compile a simple linear regression model."""
    # Most simple tf.keras models are sequential.
    # A sequential model contains one or more layers.
    model = tf.keras.models.Sequential()
    # Describe the topography of the model.
    # The topography of a simple linear regression model
    # is a single node in a single layer.
    model.add(tf.keras.layers.Dense(units=1,input_shape=(1,)))
    # Compile the model topography into code that
    # TensorFlow can efficiently execute. Configure
    # training to minimize the model's mean squared error.
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=my_learning_rate),loss="mean_squared_error", metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

def train_model(model, feature, label, epochs, batch_size):
    """Train the model by feeding it data."""
    # Feed the feature values and the label values to the
    # model. The model will train for the specified number
    # of epochs, gradually learning how the feature values
    # relate to the label values.
    history = model.fit(x=feature,
    y=label,
    batch_size=batch_size,
    epochs=epochs)
    # Gather the trained model's weight and bias.
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]
    # The list of epochs is stored separately from the
    # rest of history.
    epochs = history.epoch
    # Gather the history (a snapshot) of each epoch.
    hist = pd.DataFrame(history.history)
    # Specifically gather the model's root mean
    #squared error at each epoch.
    rmse = hist["root_mean_squared_error"]
    return trained_weight, trained_bias, epochs, rmse

print("Defined create_model and train_model")

#Define the plotting functions
def plot_the_model(trained_weight, trained_bias, feature, label):
    """Plot the trained model against the training feature and label."""
    # Label the axes.
    plt.xlabel("feature")
    plt.ylabel("label")
    # Plot the feature values vs. label values.
    plt.scatter(feature, label)
    # Create a red line representing the model. The red line starts
    # at coordinates (x0, y0) and ends at coordinates (x1, y1).
    x0 = 0
    y0 = trained_bias.item(0)
    x1 = my_feature[-1]
    y1 = trained_bias.item(0) + (trained_weight.item(0) * x1)
    print("values", x0, x1, y0, y1)
    plt.plot([x0, x1], [y0, y1], c='r')
    # Render the scatter plot and the red line.
    plt.show()


def plot_the_loss_curve(epochs, rmse):
    """Plot the loss curve, which shows loss vs. epoch."""
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")
    plt.plot(epochs, rmse, label="Loss")
    plt.legend()
    plt.ylim([rmse.min()*0.97, rmse.max()])
    plt.show()

print("Defined the plot_the_model and plot_the_loss_curve functions.")




my_feature = ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
my_label = ([5.0, 8.8, 9.6, 14.2, 18.8, 19.5, 21.4, 26.8, 28.9, 32.0, 33.8, 38.2])




df = pd.read_csv('Reef Check Data- Substrate 022723.csv', engine="python")

vals = []
valsSiteLocation = []

curr = df._get_value(0, 'site_id')
valsSiteLocation.append(curr)
runningTotal = 0

for i in range(len(df['site_id'])):
    if df._get_value(i,'site_id') == curr:
        try:
            runningTotal += int(df._get_value(i, 'total'))
        except:
            runningTotal += 0
    else:
        print(i)
        print(runningTotal)
        curr = df._get_value(i,'site_id')
        vals.append(runningTotal)
        valsSiteLocation.append(df._get_value(i,'site_id'))
        runningTotal = 0
        try:
            runningTotal += int(df._get_value(i, 'total'))
        except:
            runningTotal += 0

vals.append(runningTotal)
valsSiteLocation.append(df._get_value(i, 'site_id'))

df2 = pd.read_csv('Reef Check Data- Parrotfish-Diadema 022723.csv', engine="python")

valsParrot = []
valsSiteLocation2 = []
valsDiadema = []

fishTotal = 0
for i in range(len(df2['site_id'])):
    if df2._get_value(i,'organism_code') == "Parrotfish":
        try:
            fishTotal += (int(df2._get_value(i, 's1 (0-20m)')))
            fishTotal += (int(df2._get_value(i, 's2 (25-45m)')))
            fishTotal += (int(df2._get_value(i, 'ss3 (50-70m)')))
            fishTotal += (int(df2._get_value(i, 's4 (75-95m)')))
            valsParrot.append(fishTotal)
            fishTotal = 0
            valsSiteLocation2.append(df2._get_value(i,'site_id'))
        except:
            valsParrot.append(-1)
            fishtotal = 0
            #valsSiteLocation2.append(df2._get_value(i,'site_id'))

    if df2._get_value(i,'organism_code') == "Diadema":
        try:
            valsSiteLocation2.append(df2._get_value(i,'site_id'))
            fishTotal += (int(df2._get_value(i, 's1 (0-20m)')))
            fishTotal += (int(df2._get_value(i, 's2 (25-45m)')))
            fishTotal += (int(df2._get_value(i, 'ss3 (50-70m)')))
            fishTotal += (int(df2._get_value(i, 's4 (75-95m)')))
            valsDiadema.append(fishTotal)
            fishTotal = 0
        except:
            valsDiadema.append(-1)
            fishtotal = 0
            #valsSiteLocation2.append(df2._get_value(i,'site_id'))

#print(len(valsSiteLocation2))
#print(len(valsSiteLocation))

#for i in range(len(valsSiteLocation)):
#    if (valsSiteLocation[i] != valsSiteLocation2[i]):
#        print(valsSiteLocation[i])
#        print(valsSiteLocation2[i])
#        print(i)
#        break

#print("Success")

finalLabels = vals[:100]
finalP = valsParrot[:100]
finalD = valsDiadema[:100]
x = 5532

print(vals[0])
print(vals[1])
print(vals[2])

print(valsParrot[0])
print(valsParrot[1])
print(valsParrot[2])

my_feature = (finalP)
my_label = (finalD)

learning_rate=0.1
epochs= 100
my_batch_size= 12
my_model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(my_model, my_feature,my_label, epochs,my_batch_size)
plot_the_model(trained_weight, trained_bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)
