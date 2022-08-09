from vedo import *
import numpy as np

#limit = 20600
pts = np.load('pts.npy')#[:limit]
logits = np.load('logits.npy')#[:limit]
labels = np.load('labels.npy')#[:limit]

labCols = [(1,0,0), (0,1,0)]
cols = [(201/255,201/255,201/255), (148/255,148/255,148/255), (158/255,143/255,126/255)]

co0 = [cols[i] for i in labels]
co1 = [(l[0], l[1], 0) for l in np.clip(logits,0,1)]
co2 = [labCols[np.argmax(i)] for i in np.clip(logits, 0, 1)] #shouldn't be argmax
co3 = [labCols[i] for i in labels]

#Point Cloud
s0 = Points(pts, r=10, c = co0).alpha(0.8)
s0.pointColors(co0)

#Prediction
s1 = Points(pts, r=10, c = co1).alpha(0.8)
s1.pointColors(co1)

#threshold
s2 = Points(pts, r=10, c = co2).alpha(0.8)
s2.pointColors(co2)

#ground truth
s3 = Points(pts, r=10, c = co3).alpha(0.8) #.color(co) #.color("blue")
s3.pointColors(co3)

plt1 = Plotter(N=4, axes=1, size = (400,400))
plt1.show(s0, "Point cloud", at=0)
plt1.show(s3, "Ground truth", at=1)
plt1.show(s1, "Prediction", at=2)
plt1.show(s2, "Threshold", at=3, interactive=1)

print('done')