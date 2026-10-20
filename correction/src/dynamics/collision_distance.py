"""Signed convex-geometry distance and its robot-joint Jacobian."""
import mujoco
import numpy as np


def distance_row(scene, geom1, geom2, cutoff=.03):
    fromto=np.zeros(6)
    distance=mujoco.mj_geomDistance(scene.model,scene.data,geom1,geom2,cutoff,fromto)
    if distance>=cutoff or abs(distance)<1e-10:
        return None
    normal=(fromto[3:]-fromto[:3])/distance
    jacobians=[]
    for geom,point in ((geom1,fromto[:3]),(geom2,fromto[3:])):
        jac=np.zeros((3,scene.model.nv))
        mujoco.mj_jac(scene.model,scene.data,jac,None,point,int(scene.model.geom_bodyid[geom]))
        jacobians.append(jac[:,scene.active_v])
    jacobian=normal@(jacobians[1]-jacobians[0])
    if np.linalg.norm(jacobian)<1e-10:
        return None
    return dict(jacobian=jacobian,distance=distance,
                closest_points=fromto.reshape(2,3).copy(),normal_1_to_2=normal.copy(),
                pair=scene.model.geom(geom1).name+' / '+scene.model.geom(geom2).name)
