import matplotlib.pyplot as plt

#VGG16 loss
plt.figure(figsize=(10,10))
names = ['1', '2', '3', '4','5', '6', '7', '8', '9','10','11','12','13','14']
x = range(len(names))
y = [9.8463, 5.5614, 3.3349, 2.2491, 1.7561, 1.3346, 1.1192, 1.0759, 1.0245, 1.0079, 0.9856, 0.9773, 0.9732, 0.9692]
y1 = [8.8849, 6.8734, 4.0007, 2.7764, 1.9461, 1.6657, 1.4551, 1.2433, 1.1013, 1.0722, 1.0710, 1.0934, 1.1046, 1.1375]
y2 = [10.3496,7.3515,3.2654,2.3916,1.6453,1.3004,1.1335,1.0681,1.0175,0.9946,0.9855,0.9797,0.9730,0.9696]
y3 = [12.1226,7.9432,3.3541,2.5225,1.7661,1.4557,1.2301,1.1180,1.0763,1.0575,1.0397,1.0287,1.0266,1.0367]
plt.xlim(-1, 14)
plt.ylim(0, 11)
plt.plot(x, y, marker='x', c='r',label=u'train_loss_0')
plt.plot(x, y1, marker='x', c='g',label=u'valid_loss_0')
plt.plot(x, y2, marker='o', mec='r', mfc='w',label=u'train_loss_9')
plt.plot(x, y3, marker='o', mec='r', mfc='w',label=u'valid_loss_9')
plt.legend()
plt.xticks(x, names, rotation=0)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"epoch")
plt.ylabel("Loss")
plt.title("VGG16-GAP Loss")
plt.show()

#VGG16 acc
plt.figure(figsize=(10,10))
names = ['1', '2', '3', '4','5', '6', '7', '8', '9','10','11','12','13','14']
x = range(len(names))
y  = [0.1145, 0.1923, 0.2795, 0.3687, 0.4306, 0.5055, 0.5631, 0.6162, 0.6574, 0.6801, 0.7023, 0.7111, 0.7104, 0.7131]
y1 = [0.1233, 0.1732, 0.2465, 0.2937, 0.3643, 0.4281, 0.4963, 0.5445, 0.5899, 0.6274, 0.6440, 0.6493, 0.6375, 0.6397]
y2 = [0.1332, 0.1843, 0.2884, 0.3608, 0.4666, 0.5310, 0.5993, 0.6391, 0.6874, 0.7291, 0.7234, 0.7356, 0.7447, 0.7459]
y3 = [0.1675, 0.1992, 0.2637, 0.3297, 0.3816, 0.4691, 0.5393, 0.5971, 0.6213, 0.6555, 0.6707, 0.6820, 0.6851, 0.6834]
yx = [9.8463, 5.5614, 3.3349, 2.2491, 1.7561, 1.3346, 1.1192, 1.0759, 1.0245, 1.0179, 0.9956, 0.9873, 0.9832, 0.9792]#defining
y1x = [9.8463, 5.5614, 3.3349, 2.2491, 1.7561, 1.3346, 1.1192, 1.0759, 1.0245, 1.0179, 0.9956, 0.9873, 0.9832, 0.9792]
y2x = [9.8463, 5.5614, 3.3349, 2.2491, 1.7561, 1.3346, 1.1192, 1.0759, 1.0245, 1.0179, 0.9956, 0.9873, 0.9832, 0.9792]
y3x = [9.8463, 5.5614, 3.3349, 2.2491, 1.7561, 1.3346, 1.1192, 1.0759, 1.0245, 1.0179, 0.9956, 0.9873, 0.9832, 0.9792]
for i in range(14):
    yx[i] = 100*y[i]
    y1x[i] = 100*y1[i]
    y2x[i] = 100*y2[i]
    y3x[i] = 100*y3[i]
#plt.plot(x, y, 'ro-')
#plt.plot(x, y1, 'bo-')
plt.xlim(-1, 14)
plt.ylim(0, 100)
plt.plot(x, yx, marker='x', c='r',label=u'train_acc_0')
plt.plot(x, y1x, marker='x', c='g',label=u'valid_acc_0')
plt.plot(x, y2x, marker='o', mec='r', mfc='w',label=u'train_acc_9')
plt.plot(x, y3x, marker='o', mec='r', mfc='w',label=u'valid_acc_9')
plt.legend()
plt.xticks(x, names, rotation=0)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"epoch")
plt.ylabel(u"Accuracy(%)")
plt.title("VGG16-GAP Accuracy")
plt.show()

#InceptionV3 loss
plt.figure(figsize=(10,10))
names = ['1', '2', '3', '4','5', '6', '7', '8', '9','10','11']
x = range(len(names))
y = [0.5230, 0.4747,0.3534,0.2973,0.1676,0.1192,0.1175,0.1169, 0.1063, 0.1032, 0.1011]
y1 = [3.4331, 2.5164,1.7432,1.2199,1.0459,0.9716,0.8544,0.8817, 0.8633, 0.8601, 0.8459]
y2 = [0.5793, 0.4919,0.3311,0.2246,0.1376,0.1002,0.0910,0.0878, 0.0881, 0.0842, 0.0833]
y3 = [2.4842, 1.6529,1.3421,1.0923,1.0692,0.8840,0.8331,0.8242, 0.8130, 0.803, 0.7892]
plt.xlim(-1, 12)
plt.ylim(0, 2.6)
plt.plot(x, y, marker='x', c='r',label=u'train_loss_0')
plt.plot(x, y1, marker='x', c='g',label=u'valid_loss_0')
plt.plot(x, y2, marker='o', mec='r', mfc='w',label=u'train_loss_9')
plt.plot(x, y3, marker='o', mec='r', mfc='w',label=u'valid_loss_9')
plt.legend()
plt.xticks(x, names, rotation=0)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"epoch")
plt.ylabel("Loss")
plt.title("VGG16-GAP Loss")
plt.show()

#VGG16 acc
plt.figure(figsize=(10,10))
names = ['1', '2', '3', '4','5', '6', '7', '8', '9','10','11',]
x = range(len(names))
y  = [0.6809, 0.7634, 0.8211,0.9037,0.9492,0.9756,0.9849,0.9854, 0.9865, 0.9871, 0.9875]
y1 = [0.5434, 0.6237, 0.6689, 0.7349, 0.7661, 0.7870, 0.7919, 0.7683, 0.7702, 0.7745, 0.7753 ]
y2 = [0.7315, 0.8134, 0.8921,0.9393,0.9799,0.9934,0.9948,0.9948, 0.9950, 0.9951, 0.9952]
y3 = [0.6255, 0.6579, 0.7389, 0.7949, 0.8161, 0.8237, 0.8279, 0.8293, 0.8331, 0.8363, 0.8369 ]
yx = [9.8463, 5.5614, 3.3349, 2.2491, 1.7561, 1.3346, 1.1192, 1.0759, 1.0245, 1.0179, 0.9956 ]#defining
y1x = [9.8463, 5.5614, 3.3349, 2.2491, 1.7561, 1.3346, 1.1192, 1.0759, 1.0245, 1.0179, 0.9956 ]
y2x = [9.8463, 5.5614, 3.3349, 2.2491, 1.7561, 1.3346, 1.1192, 1.0759, 1.0245, 1.0179, 0.9956]
y3x = [9.8463, 5.5614, 3.3349, 2.2491, 1.7561, 1.3346, 1.1192, 1.0759, 1.0245, 1.0179, 0.9956]
for i in range(11):
    yx[i] = 100*y[i]
    y1x[i] = 100*y1[i]
    y2x[i] = 100*y2[i]
    y3x[i] = 100*y3[i]
plt.xlim(-1, 11)
plt.ylim(50, 100)
plt.plot(x, yx, marker='x', c='r',label=u'train_acc_0')
plt.plot(x, y1x, marker='x', c='g',label=u'valid_acc_0')
plt.plot(x, y2x, marker='o', mec='r', mfc='w',label=u'train_acc_9')
plt.plot(x, y3x, marker='o', mec='r', mfc='w',label=u'valid_acc_9')
plt.legend()
plt.xticks(x, names, rotation=0)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"epoch")
plt.ylabel(u"Accuracy(%)")
plt.title("InceptionV3-FC Accuracy")
plt.show()


