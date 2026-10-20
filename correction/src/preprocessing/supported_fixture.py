"""A common, explicitly recorded fixture path for the manipulation-only bench.

This does not solve locomotion. It removes high-frequency input jitter from
the prescribed external support, identically for original and method cases.
"""
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


def regularize_fixture(q,window=21):
    q=np.asarray(q,float);result=q.copy()
    if window<5 or window%2!=1 or window>len(q):raise ValueError('Odd filter window >=5 and <=episode length required')
    result[:,:3]=savgol_filter(q[:,:3],window,3,axis=0)
    quat=q[:,3:7].copy()
    for i in range(1,len(quat)):
        if quat[i]@quat[i-1]<0:quat[i]*=-1
    filtered=savgol_filter(quat,window,3,axis=0)
    filtered/=np.linalg.norm(filtered,axis=1)[:,None]
    # Preserve the actual original initial state, blending over the first window.
    u=np.clip(np.arange(len(q))/window,0,1);blend=u*u*(3-2*u)
    result[:,:3]=q[:,:3]+blend[:,None]*(result[:,:3]-q[:,:3])
    filtered=quat+blend[:,None]*(filtered-quat);filtered/=np.linalg.norm(filtered,axis=1)[:,None]
    result[:,3:7]=filtered;result[0,:7]=q[0,:7]
    delta=(Rotation.from_quat(q[:,3:7]).inv()*Rotation.from_quat(result[:,3:7])).magnitude()
    return result,dict(kind='common prescribed pelvis regularization, not walking control',window_frames=window,
        max_translation_change_m=float(np.linalg.norm(result[:,:3]-q[:,:3],axis=1).max()),
        max_rotation_change_deg=float(np.rad2deg(delta.max())),initial_state_preserved=True)
