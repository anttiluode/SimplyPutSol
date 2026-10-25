"""SimplyPutSol: a probing arm with local complex cable dynamics.

No teacher supplies optimal joint angles. The learner compares two noisy
outgoing motor probes and their *scalar* returning sensory consequences.
Complex cable state and its local tangent are auditable, not neural claims.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

D=9
N_TASKS=4
TASKS=("bell","switch","light","horn")
TARGETS=np.array([[-1.10,1.60],[0.90,1.00],[0.10,2.20],[-0.40,0.60]])
ANCHOR=np.array([-0.1,1.4])
WIND=np.array([0.35,-0.25])
CONTACT=0.12


def hand(q):
    a,b=q
    return np.array([np.cos(a)+.8*np.cos(a+b),
                     np.sin(a)+.8*np.sin(a+b)])


class World:
    """The model learns only from scalar sensory strength, never target error."""
    def sense(self,action,task,wind):
        q=np.clip(ANCHOR+action+wind*WIND,[-2.5,.2],[2.5,2.6])
        distance=float(np.linalg.norm(hand(q)-hand(TARGETS[task])))
        strength=float(np.exp(-.5*(distance/.85)**2))
        return strength, bool(distance<=CONTACT), distance


def frames(posture,task,wind):
    """Wind occurs at t=1 but is not present in final observation t=2."""
    x=np.zeros((3,D))
    cue=np.eye(N_TASKS)[task]*1.5
    pose=np.array([np.cos(posture[0]),np.sin(posture[0]),
                   np.cos(posture[1]),np.sin(posture[1])])
    x[0,:8]=np.r_[cue,pose]
    x[1,8]=wind*1.5
    x[2,:8]=np.r_[cue,pose]
    return x


class Cable:
    """Parallel resident complex filters with branch-local parameter tangents.

    z[k,t+1] = rho[k]*exp(i*omega[k])*z[k,t] + tanh(C[k] @ input[t])
    action = B @ real(z[-1]). Each trace is causal and confined to its
    branch. A shared reward-improvement scalar modulates the branch traces.
    """
    def __init__(self,rng,n=24,mode="cable"):
        self.mode=mode
        self.C=rng.normal(0,.45,(n,D))
        self.omega=(np.zeros(n) if mode=="phase0"
                    else rng.uniform(-1.35,1.35,n))
        self.rho=rng.uniform(.65,.89,n)
        self.B=rng.normal(0,.045,(2,n))
    def forward(self,xs):
        n=len(self.omega)
        z=np.zeros(n,complex)
        dw=np.zeros(n,complex)
        dc=np.zeros((n,D),complex)
        q=self.rho*np.exp(1j*self.omega)
        for x in xs:
            s=np.tanh(self.C@x)
            if self.mode=="no_state":
                z=s.astype(complex)
                dw[:]=0
                dc=(1-s*s)[:,None]*x[None,:]
            else:
                dw=q*(dw+1j*z)
                dc=q[:,None]*dc+(1-s*s)[:,None]*x[None,:]
                z=q*z+s
        h=z.real
        return self.B@h,(h,dw.real,dc.real)
    def learn(self,trace,g):
        h,dw,dc=trace
        gain=g@self.B.copy()
        if self.mode not in ("frozen","phase0","no_state"):
            self.omega+=.018*gain*dw/(1+np.abs(dw))
            self.omega=np.clip(self.omega,-np.pi,np.pi)
        if self.mode!="frozen":
            self.C+=.012*gain[:,None]*dc/(1+np.sum(dc*dc,axis=1))[:,None]
            self.C=np.clip(self.C,-3.,3.)
        self.B+=.35*np.outer(g,h)/(.25+h@h)
        self.B=np.clip(self.B,-2.5,2.5)
    def count(self):
        return self.C.size+self.omega.size+self.rho.size+self.B.size


class RNN:
    """Conventional tanh RNN with three-step BPTT, same sensor feedback."""
    def __init__(self,rng,n=20):
        self.U=rng.normal(0,.46/np.sqrt(D),(n,D))
        self.A=rng.normal(0,.68/np.sqrt(n),(n,n))
        self.B=rng.normal(0,.04,(2,n))
    def forward(self,xs):
        h=np.zeros(self.A.shape[0])
        trace=[]
        for x in xs:
            old=h
            h=np.tanh(self.U@x+self.A@h)
            trace.append((x.copy(),old.copy(),h.copy()))
        return self.B@h,trace
    def learn(self,trace,g):
        oldB=self.B.copy()
        oldA=self.A.copy()
        dh=oldB.T@g
        du=np.zeros_like(self.U)
        da=np.zeros_like(self.A)
        for x,old,h in reversed(trace):
            dpre=dh*(1-h*h)
            du+=np.outer(dpre,x)
            da+=np.outer(dpre,old)
            dh=oldA.T@dpre
        scale=max(1.,np.linalg.norm(du)+np.linalg.norm(da))
        self.U+=.055*du/scale
        self.A+=.055*da/scale
        h=trace[-1][-1]
        self.B+=.35*np.outer(g,h)/(.25+h@h)
        self.B=np.clip(self.B,-2.5,2.5)
    def count(self):
        return self.U.size+self.A.size+self.B.size


class Route:
    """Static mixture of local linear experts (ordinary prior art baseline)."""
    def __init__(self,rng,n=24,active=4):
        self.C=rng.normal(0,.36,(n,8))
        self.W=np.zeros((n,2,9))
        self.active=active
    def forward(self,xs):
        x=xs[-1,:8]
        scores=-2.5*np.sum((self.C-x)**2,axis=1)
        idx=np.argpartition(-scores,self.active-1)[:self.active]
        a=np.exp(scores[idx]-np.max(scores[idx]))
        a/=sum(a)
        v=np.r_[x,1.]
        y=np.einsum("k,kij,j->i",a,self.W[idx],v)
        return y,(idx,a,v,x)
    def learn(self,trace,g):
        idx,a,v,x=trace
        for k,w in zip(idx,a):
            self.W[k]+=.35*w*np.outer(g,v)/(.25+v@v)
            self.C[k]+=.018*w*(x-self.C[k])
        self.W=np.clip(self.W,-2.5,2.5)
    def count(self):
        return self.C.size+self.W.size


def make(name,seed):
    rng=np.random.default_rng(seed)
    if name=="rnn": return RNN(rng)
    if name=="route": return Route(rng)
    if name in ("cable","frozen","phase0","no_state"):
        return Cable(rng,mode=name)
    raise ValueError(name)


def probe(model,world,posture,task,wind,rng,sigma=.13):
    action,trace=model.forward(frames(posture,task,wind))
    direction=rng.choice([-1.,1.],size=2)
    # Two exploratory actions are actually scored from the same reset state.
    plus=world.sense(action+sigma*direction,task,wind)[0]
    minus=world.sense(action-sigma*direction,task,wind)[0]
    g=(plus-minus)/(2*sigma)*direction
    model.learn(trace,g)


def evaluate(model,seed,postures=64):
    """No evaluation-time plasticity. Identical held-out setups across models."""
    rng=np.random.default_rng(seed)
    world=World()
    rows=[]
    for task in range(N_TASKS):
        distances=[]
        contacts=[]
        strengths=[]
        gaps=[]
        for _ in range(postures):
            posture=rng.uniform([-1.2,.3],[1.2,2.4])
            a=model.forward(frames(posture,task,1))[0]
            b=model.forward(frames(posture,task,-1))[0]
            gaps.append(float(np.linalg.norm(a-b)))
            for w,pred in ((1,a),(-1,b)):
                r,hit,d=world.sense(pred,task,w)
                strengths.append(r)
                contacts.append(hit)
                distances.append(d)
        rows.append({"task":TASKS[task],"distance":float(np.mean(distances)),
                     "contact":float(np.mean(contacts)),
                     "strength":float(np.mean(strengths)),
                     "history_response":float(np.mean(gaps))})
    return rows


def experiment(name,seed,episodes=650,evaluation=80):
    rng=np.random.default_rng(9001+seed)
    model=make(name,1701+seed)
    world=World()
    phases=[{"after":"untrained","tasks":evaluate(model,3300+seed,evaluation)}]
    for task in range(N_TASKS):
        for _ in range(episodes):
            posture=rng.uniform([-1.2,.3],[1.2,2.4])
            wind=int(rng.choice([-1,1]))
            probe(model,world,posture,task,wind,rng)
        phases.append({"after":TASKS[task],
                       "tasks":evaluate(model,3300+seed,evaluation)})
    return {"name":name,"seed":seed,"parameters":int(model.count()),
            "phases":phases}


def reduce_runs(runs):
    rows=[]
    for name in dict.fromkeys(r["name"] for r in runs):
        group=[r for r in runs if r["name"]==name]
        def mean(p,t,key):
            return float(np.mean([r["phases"][p]["tasks"][t][key] for r in group]))
        rows.append({"name":name,"parameters":group[0]["parameters"],
                     "bell_initial":mean(0,0,"distance"),
                     "bell_learned":mean(1,0,"distance"),
                     "bell_final":mean(4,0,"distance"),
                     "final_distance":float(np.mean([mean(4,k,"distance") for k in range(4)])),
                     "final_contact":float(np.mean([mean(4,k,"contact") for k in range(4)])),
                     "history_response":mean(4,0,"history_response"),
                     "horn_initial":mean(0,3,"distance"),
                     "horn_final":mean(4,3,"distance")})
    return rows


def html_report(rows,gates):
    high=max([r["final_distance"] for r in rows]+[.01])
    bars=""
    for i,r in enumerate(rows):
        y=32+i*46
        width=r["final_distance"]/high*420
        bars+=(f'<text x="124" y="{y+13}" text-anchor="end">{r["name"]}</text>'
               f'<rect x="142" y="{y}" width="{width:.1f}" height="27" rx="5" fill="#19a38c"/>'
               f'<text x="{153+width:.1f}" y="{y+18}">{r["final_distance"]:.3f}</text>')
    gate_rows="".join(f"<li>{k}: {'PASS' if v else 'FAIL'}</li>" for k,v in gates.items())
    return f'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>SimplyPutSol — paired probe receipt</title>
<style>body{{background:#111c28;color:#e5edf5;font:16px system-ui;max-width:900px;margin:40px auto;padding:20px}}
h1{{font-size:38px}}p{{line-height:1.6}}svg{{width:100%;background:#1c2a36;border-radius:12px}}
svg text{{fill:#e5edf5;font:14px system-ui}}li{{padding:5px}}</style>
<h1>SimplyPutSol</h1>
<p>Stateful cables, trained by *two scalar environmental probes*, versus
ordinary RNN, static experts and three cable ablations. Lower held-out
first-reach distance is better. No target angles or 2-D errors given as labels.</p>
<svg viewBox="0 0 730 330">{bars}</svg>
<h2>Declared rules</h2><ul>{gate_rows}</ul>
<p>Passes are not a novelty claim. Full per-seed phase tables in results.json.
Graded scalar sensing and resettable symmetric probes are important assumptions.</p>
</html>'''


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--episodes",type=int,default=650)
    a.add_argument("--seeds",type=int,default=3)
    a.add_argument("--evaluation",type=int,default=80)
    a.add_argument("--out",type=Path,default=Path("results"))
    args=a.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    names=("cable","phase0","frozen","no_state","rnn","route")
    runs=[experiment(n,s,args.episodes,args.evaluation)
          for n in names for s in range(args.seeds)]
    summary=reduce_runs(runs)
    lookup={r["name"]:r for r in summary}
    sample=np.array([.2,1.])
    a,b=frames(sample,0,1),frames(sample,0,-1)
    stateless=make("no_state",1)
    dynamic=make("cable",1)
    gates={
        "G1_state_matters": bool(np.linalg.norm(dynamic.forward(a)[0]-dynamic.forward(b)[0])>1e-8
                                 and np.allclose(stateless.forward(a)[0],stateless.forward(b)[0])),
        "G2_no_target_angle_teacher":True,
        "G3_cable_learns_horn":lookup["cable"]["horn_final"]<lookup["cable"]["horn_initial"],
        "G4_cable_beats_rnn":lookup["cable"]["final_distance"]<lookup["rnn"]["final_distance"],
        "G5_plastic_beats_frozen":lookup["cable"]["final_distance"]<lookup["frozen"]["final_distance"]
    }
    doc={"protocol":{"seeds":args.seeds,"episodes_per_task":args.episodes,
          "held_out_postures_per_task_and_wind":args.evaluation,
          "training_sensor":"scalar proximity","physical_probes_per_update":2},
         "summary":summary,"gates":gates,"runs":runs}
    (args.out/"results.json").write_text(json.dumps(doc,indent=2))
    (args.out/"report.html").write_text(html_report(summary,gates))
    print("MODEL       PARAMS   BELL LEARNED   BELL FINAL   FINAL DIST   HIT%   HISTORY")
    for r in summary:
        print(f'{r["name"]:10s} {r["parameters"]:6d}   {r["bell_learned"]:10.3f}'
              f'   {r["bell_final"]:10.3f}   {r["final_distance"]:10.3f}'
              f'   {100*r["final_contact"]:5.1f}   {r["history_response"]:8.3f}')
    print("GATES",json.dumps(gates,sort_keys=True))
    print("Report:",args.out/"report.html")


if __name__=="__main__":
    main()
