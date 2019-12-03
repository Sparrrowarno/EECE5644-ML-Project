import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(10,10))
x_data = ['0','1', '2', '3', '4', '5', '6', '7','8','9',]
y_data = [41, 38, 35, 37, 32, 30, 24, 31, 33, 28]
y_data2 = [44, 42, 41,35, 32, 41, 45, 50, 42, 34]
weights = np.zeros((2,10))
bar_width=0.3
plt.bar(x=range(len(x_data)), height=y_data, label='VGG16',
    color='#66ccff', alpha=0.5, width=bar_width)
plt.bar(x=np.arange(len(x_data))+bar_width, height=y_data2,
    label='InceptionV3', color='#ffcc66', alpha=0.8, width=bar_width)
plt.xticks(np.arange(len(x_data))+bar_width/2, x_data)

for x, y in enumerate(y_data):
    plt.text(x, y , '%s' % y, ha='center', va='bottom')
for x, y in enumerate(y_data2):
    plt.text(x+bar_width, y, '%s' % y, ha='center', va='bottom')
for i in range(10):
    weights[0][i]=(y_data[i]/(y_data[i] + y_data2[i]))
for i in range(10):
    weights[1][i]=(y_data2[i]/(y_data[i] + y_data2[i]))
print(weights)

plt.title("correct predictions by classes per model")
plt.xlabel("class")
plt.ylabel("number")
plt.legend()
plt.show()