import open3d as o3d
from vedo import show, Points
import numpy as np 

points = []
labels = []

for i in range(10) : 
    if np.random.rand() < 0.5 : 
        box = o3d.geometry.TriangleMesh.create_box(width=2*np.random.rand(), height = 2*np.random.rand(), depth = 2*np.random.rand()).translate((4*np.random.rand(), 4*np.random.rand(), 4*np.random.rand()))    
        pts = box.sample_points_poisson_disk(number_of_points=100)        
        points.append(pts.points)
        labels.append(np.ones(len(pts.points)))
    else : 
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius = np.random.rand(), resolution=10).translate((4*np.random.rand(), 4*np.random.rand(), 4*np.random.rand()))
        pts = sphere.sample_points_poisson_disk(number_of_points=100)
        points.append(pts.points)
        labels.append(np.zeros(len(pts.points)))

print('done')

points = np.vstack(np.array(points))
labels = np.concatenate(np.array(labels))


s0 = Points(points[np.where(labels == 1)], r=10).color("green").alpha(0.8)
s1 = Points(points[np.where(labels == 0)], r=10).color("red").alpha(0.8)

show(s0, s1, axes=1).close()


#pc = bodies.sample_points_poisson_disk(number_of_points=1000)



'''
bodies = []

for i in range(3) :
    bo = Box(pos=(3*np.random.rand(),3*np.random.rand(),3*np.random.rand()), length = 0.7, width = 0.7, height =1, alpha = 0.5)
    bodies.append(bo)

#for i in range(3) : 
#    sp = Sphere(pos=(3*np.random.rand(), 3*np.random.rand(), 3*np.random.rand()), r=1, c='r5', alpha=0.5, res=24, quads=False)
#    bodies.append(sp)

bo = merge(bodies)

print(len(bo.points()))



s0 = Points(bo.points(), r=10).color("green").alpha(0.8)
show(s0, axes=1).close()
'''