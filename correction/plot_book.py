"""Static scientific figure for the measured real-episode correction results."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(directory):
    arrays=np.load(directory/'fk_force_audit.npz')
    report=json.loads((directory/'fk_force_audit.json').read_text())
    t=arrays['times']
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,layout='constrained')
    for label,color in [('before','#be6641'),('after','#177ca4')]:
        axes[0].plot(t,1000*np.abs(arrays[label+'_gap_m']).mean(axis=1),color=color,label=label.capitalize(),lw=2)
        axes[1].plot(t,np.linalg.norm(arrays[label+'_residual'][:,:3],axis=1),color=color,lw=2)
        axes[2].plot(t,1000*arrays[label+'_probe_max_m'],color=color,lw=2)
    axes[0].set_ylabel('Mean absolute\nsurface gap [mm]');axes[0].legend(frameon=False)
    axes[1].set_ylabel('Force-balance\nresidual [N]')
    axes[2].set_ylabel('Maximum probe\npenetration [mm]')
    axes[2].axhline(5,color='#555555',ls='--',lw=1,label='5 mm audit threshold')
    axes[2].legend(frameon=False);axes[2].set_xlabel('Original episode time [s]')
    for ax in axes:
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False);ax.set_ylim(bottom=0)
    fig.suptitle('Real Tier 1 book episode · local robot contact correction',fontsize=16)
    fig.text(.5,-.025,'Prescribed object motion · Assumed 1 kg, μ = 0.6 · Instantaneous wrench test only; slip and penetration remain',ha='center',fontsize=10)
    fig.savefig(directory/'diagnostics.png',dpi=150,bbox_inches='tight')


if __name__=='__main__': plot(Path(__file__).resolve().parent/'outputs/tier1_book')
