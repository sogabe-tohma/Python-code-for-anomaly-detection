from sklearn import metrics
import matplotlib.pyplot as plt
from pylab import rcParams

# 検証用Dummyデータ
True_score = [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0 ]

#異常度の計算結果
abnormality_score = ( [2.8, 1.2, 6.2, 3.4, 5.2, 1.8, 3, 5, 7.8, 2.9, 1.8, 3.3, 4.1,
               6.7, 7.9, 1.1, 2.1, 3.1,3.9,2.8] )
false_alarm_rate, recall, thresholds = metrics.roc_curve(True_score, abnormality_score)
auc = metrics.auc(false_alarm_rate, recall)
print(auc)
print("false_alarm_rate: ", false_alarm_rate)
print("recall: ", recall)
print("thresholds: ", thresholds)

plt.plot(false_alarm_rate, recall, marker="o")
plt.xlabel("False_alarm_rate")
plt.ylabel("Recall")
plt.show()