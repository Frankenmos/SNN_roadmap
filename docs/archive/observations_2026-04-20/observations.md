What your toy model is actually doing

This is clean for a first pass:



find\_units: nice idea, but the signature is backwards. You pass condition then do feature\_layer == condition inside, which double-compares. It should be np.argwhere(condition) where condition is already a boolean mask.

left\_click: you're using Smart\_screen on purpose, and your comment is right, for focus-fire you want Attack\_screen with exact coords. Keep both, let the token decide.

select\_rect: the parametrizable half-size around target is perfect for micro. That's your spatial token in disguise.

right\_click: this is the hybrid heart. One function, game\_action string → pysc2 call. Your inline notes are spot on, this wants to be a token, not a bunch of hardcoded ifs.

The big gap isn't the action code, it's that none of this returns a token that policy\_input can see next step.



The hybrid head you described in the comments

You already wrote the design in the jokes:



"the agent can output a token for the game action and then we can map that token"



Let's make that real, minimal change:



\*\*Action token = \*\*\[type]\[x]\[y]



type id



meaning



0



no\_op



1



select\_point



2



select\_rect



3



smart



4



attack



5



move



6



harvest



7



build



8



train



x and y are just ints 0-83. That's it. No math derivation needed, just three ints per step.



Then in policy\_input.py, instead of only obs tokens, you add:



Python

\# before: inputs = { "obs": obs\_tokens }

\# after:

last\_act\_type = prev\_action\[0] # 0-8

last\_x = prev\_action\[1] / 83.0 # normalize

last\_y = prev\_action\[2] / 83.0



inputs = {

&#x20; "obs": obs\_tokens,

&#x20; "last\_action": torch.tensor(\[last\_act\_type, last\_x, last\_y])

}



5 lines hidden

For your SNN setup, just embed the type into a 8-dim vector and concat with the spatial coords. Now the network sees "I just attack-clicked at (42,31)" instead of guessing from pixels alone.



This fixes the semantic drift Gipity was worried about, without turning it into a thesis. The network stops learning spurious correlations because it actually knows what it did last frame.



Tiny fixes while we're here

No lecture, just copy-paste traps I spotted:



find\_units should be:

Python

def find\_units(self, condition\_mask):

&#x20;   return \[tuple(u) for u in np.argwhere(condition\_mask)]

then call with find\_units(feature\_layer == \_PLAYER\_ENEMY)



right\_click build/train will crash, those FUNCTIONS need an extra arg for what to build. Leave them as no\_op for now, or add a second token for unit\_id later.

Keep your clipping logic, it's correct, pysc2 will silently fail if you pass 84.

Want to vibe this now?

I can take your ActionSpace and spit back a token-first version that:



returns (action\_fn, \[x]\[y], token) instead of just the pysc2 call

gives you a 3-line patch for policy\_input to consume last\_action

No memory test, no "reconstruct from first principles." We just wire the output back to the input and watch your agent stop being blind to its own clicks.



Drop policy\_input.py if you want, or I can refactor the ActionSpace first and you tell me if the token shape feels right. Your call, I'm here to build, not to grade.



this si the current action space: import numpy as np

from pysc2.lib import actions





\_PLAYER\_FRIENDLY = 1

\_PLAYER\_ENEMY = 4





class ActionSpace:

&#x20;   def \_\_init\_\_(self, max\_step\_size=10, screen\_size=84):

&#x20;       self.max\_step\_size = max\_step\_size

&#x20;       self.screen\_size = screen\_size



&#x20;   def find\_units(self, feature\_layer, condition):

&#x20;       units = np.argwhere(feature\_layer == condition)

&#x20;       return \[tuple(unit) for unit in units]



&#x20;   def move(self, obs, target\_x, target\_y, screen\_size=None):

&#x20;       screen\_size = screen\_size or self.screen\_size

&#x20;       max\_coord = screen\_size - 1

&#x20;       target\_x = int(np.clip(target\_x, 0, max\_coord))

&#x20;       target\_y = int(np.clip(target\_y, 0, max\_coord))



&#x20;       if actions.FUNCTIONS.Smart\_screen.id in obs.observation.available\_actions:

&#x20;           return actions.FUNCTIONS.Smart\_screen("now", \[target\_x, target\_y])

&#x20;       return actions.FUNCTIONS.no\_op()



&#x20;   def nearest\_enemy\_unit\_center(self, obs):

&#x20;       feature\_units = getattr(obs.observation, "feature\_units", None)

&#x20;       if feature\_units is None or len(feature\_units) == 0:

&#x20;           return None



&#x20;       friendlies = \[

&#x20;           unit for unit in feature\_units if unit.alliance == \_PLAYER\_FRIENDLY

&#x20;       ]

&#x20;       enemies = \[

&#x20;           unit for unit in feature\_units if unit.alliance == \_PLAYER\_ENEMY

&#x20;       ]

&#x20;       if not enemies:

&#x20;           return None



&#x20;       if friendlies:

&#x20;           center\_x = float(np.mean(\[unit.x for unit in friendlies]))

&#x20;           center\_y = float(np.mean(\[unit.y for unit in friendlies]))

&#x20;           target = min(

&#x20;               enemies,

&#x20;               key=lambda unit: (unit.x - center\_x) \*\* 2 + (unit.y - center\_y) \*\* 2,

&#x20;           )

&#x20;       else:

&#x20;           target = enemies\[0]



&#x20;       return (int(target.x), int(target.y))



&#x20;   def attack(self, obs, target\_position):

&#x20;       if actions.FUNCTIONS.Attack\_screen.id in obs.observation.available\_actions:

&#x20;           if target\_position and len(target\_position) == 2:

&#x20;               target\_x = int(np.clip(target\_position\[0], 0, self.screen\_size - 1))

&#x20;               target\_y = int(np.clip(target\_position\[1], 0, self.screen\_size - 1))

&#x20;               return actions.FUNCTIONS.Attack\_screen("now", \[target\_x, target\_y])

&#x20;       return actions.FUNCTIONS.no\_op()

(the old one i want to make ocmaptibel with all the arhitectural changes and ideas i ahve, plus i got a bit more ambitious than simply staying in the defeat roaches lene XD)



Today

perfect, this version is already halfway to hybrid. It's clean, it keeps your move() and attack() working for defeat-roaches, and we can bolt on tokens without breaking anything.



I took your current class and made a token-first upgrade that fits the architecture you described. Same API, plus every call now writes a tiny \[type, x, y, extra] vector you can feed directly into policy\_input next step.



grab it here:



📎 action\_space\_v2.py



What changed, in vibe-coding terms

No API break: move(obs, x, y) and attack(obs, pos) still return a pysc2 action exactly like before

last\_token is automatic: after any call, action\_space.get\_last\_token() gives you \[0-9, x, y, extra]. That's your action history for the network, no extra bookkeeping

select actions are back: select\_point and select\_rect from your toy model are in, with the max\_step\_size logic you wanted

ambitious stubs ready: harvest, build, train, ability are placeholders with the extra field reserved for building\_id or ability\_id. You can grow beyond roaches without rewriting the token format

The token vocab is dead simple:



0 no\_op, 1 select\_point, 2 select\_rect, 3 smart, 4 attack, 5 move, 6 harvest, 7 build, 8 train, 9 ability

x, y are 0-83 clipped ints

extra is free for now, use it later for "which building" or "which spell"

How this plugs into your policy\_input problem

Right now you feed only obs. With this, your training loop becomes:



Python

action = action\_space.attack(obs, target)

\# or action\_space.move(obs, x, y) etc



token = action\_space.get\_last\_token() # shape \[4]

\# token\[0] = type\_id, token\[1]=x, token\[2]=y



\# in policy\_input next step:

inputs\["last\_action\_type"] = token\[0]

inputs\["last\_action\_xy"] = token\[1:3] / 83.0



4 lines hidden

Your hybrid heads then become natural:



categorical head predicts type\_id (10 classes)

two spatial heads predict x and y (84-way each, or Gaussian)

the network finally sees what it just did, so your SNN/BPTT stops guessing from pixels alone

Why this fits your "more ambitious than roaches" plan

You mentioned wanting contextual right-click. That's literally the token: the agent outputs type, you map it in one place. Want to add build-supply-depot later? Just set extra=19 and handle it in build(). No new action functions everywhere.



The nearest\_enemy\_unit\_center helper stays untouched, so your current micro still works while you experiment with the new heads.



Want me to do the 3-line patch to policy\_input.py next so it actually consumes last\_token? Or we can test this ActionSpace first in your current loop and watch the tokens flow. Your call, no pop quiz.



ok lets start here then and we slowly backtrace the rest of the arhitecture, so for now this was the agent contorl schema: import math

import random



import numpy as np

import torch

from pysc2.agents import base\_agent

from pysc2.lib import actions

from pysc2.lib import colors as \_colors



from PPO\_CNN.PPO import PPO

from PPO\_CNN.policy\_network import PolicyNetwork

from PPO\_CNN.reward\_function\_2 import RewardFunctionV2

from Utility.config import cfg

from action\_space.action\_space import ActionSpace

from obs\_space.obs\_space\_2 import ObservationExtractor





def \_shuffled\_hue\_fixed(scale):

&#x20;   palette = list(\_colors.smooth\_hue\_palette(scale))

&#x20;   random\_keys = \[random.random() for \_ in palette]

&#x20;   palette = \[x for \_, x in sorted(zip(random\_keys, palette))]

&#x20;   return np.array(palette)





\_colors.shuffled\_hue = \_shuffled\_hue\_fixed



\_PLAYER\_FRIENDLY = 1





class DefeatRoaches(base\_agent.BaseAgent):

&#x20;   def \_\_init\_\_(

&#x20;       self,

&#x20;       spatial\_input\_shape=None,

&#x20;       vector\_input\_dim=None,

&#x20;       action\_dim=None,

&#x20;       lr=None,

&#x20;       gamma=None,

&#x20;       clip\_eps=None,

&#x20;       snn\_state=None,

&#x20;   ):

&#x20;       super(DefeatRoaches, self).\_\_init\_\_()

&#x20;       self.steps = 0

&#x20;       self.extractor = ObservationExtractor()



&#x20;       if spatial\_input\_shape is None:

&#x20;           spatial\_input\_shape = tuple(cfg.model.spatial\_input\_shape)

&#x20;       if vector\_input\_dim is None:

&#x20;           vector\_input\_dim = cfg.model.vector\_input\_dim

&#x20;       if action\_dim is None:

&#x20;           action\_dim = cfg.model.action\_dim



&#x20;       assert spatial\_input\_shape == tuple(cfg.model.spatial\_input\_shape), (

&#x20;           f"Invalid spatial\_input\_shape: {spatial\_input\_shape}"

&#x20;       )

&#x20;       assert vector\_input\_dim == cfg.model.vector\_input\_dim

&#x20;       assert action\_dim == cfg.model.action\_dim



&#x20;       screen\_size = int(getattr(cfg.model, "screen\_size", spatial\_input\_shape\[-1]))

&#x20;       self.action\_space = ActionSpace(screen\_size=screen\_size)

&#x20;       self.reward\_function = RewardFunctionV2()

&#x20;       self.reward\_scale = float(getattr(cfg.hyperparameters, "reward\_scale", 1.0))

&#x20;       self.snn\_state = snn\_state



&#x20;       self.policy = PolicyNetwork(

&#x20;           spatial\_input\_shape,

&#x20;           vector\_input\_dim,

&#x20;           action\_dim,

&#x20;           num\_steps=getattr(cfg.model, "num\_steps", 1),

&#x20;           screen\_size=screen\_size,

&#x20;           token\_snn\_alpha=getattr(cfg.model, "token\_snn\_alpha", 0.8),

&#x20;           token\_snn\_beta=getattr(cfg.model, "token\_snn\_beta", 0.9),

&#x20;           attention\_embed\_dim=getattr(cfg.model, "attention\_embed\_dim", 64),

&#x20;           attention\_pool\_size=getattr(cfg.model, "attention\_pool\_size", 7),

&#x20;           attention\_beta=getattr(cfg.model, "attention\_beta", 0.5),

&#x20;       )

&#x20;       self.policy.to(self.policy.device)



&#x20;       if self.snn\_state is None:

&#x20;           self.snn\_state = self.policy.init\_concrete\_state(batch\_size=1)



&#x20;       total\_eps = int(getattr(cfg.environment, "total\_episodes", 0))

&#x20;       steps\_per\_episode = int(getattr(cfg.environment, "steps\_per\_episode", 1) or 1)

&#x20;       rollout\_steps = int(getattr(cfg.hyperparameters, "rollout\_steps", steps\_per\_episode) or 1)

&#x20;       self.total\_updates\_estimate = max(

&#x20;           0,

&#x20;           math.ceil(total\_eps \* steps\_per\_episode / rollout\_steps),

&#x20;       )

&#x20;       lr\_min = float(getattr(cfg.hyperparameters, "lr\_min", 0.0))



&#x20;       self.ppo = PPO(

&#x20;           policy\_net=self.policy,

&#x20;           lr=lr if lr is not None else cfg.hyperparameters.lr,

&#x20;           gamma=gamma if gamma is not None else cfg.hyperparameters.gamma,

&#x20;           clip\_epsilon=clip\_eps if clip\_eps is not None else cfg.hyperparameters.clip\_eps,

&#x20;           critic\_loss\_coef=getattr(cfg.hyperparameters, "critic\_loss\_coef", 0.5),

&#x20;           entropy\_coef=getattr(cfg.hyperparameters, "entropy\_coef", 0.01),

&#x20;           total\_updates=self.total\_updates\_estimate,

&#x20;           lr\_min=lr\_min,

&#x20;           target\_kl=getattr(cfg.hyperparameters, "target\_kl", None),

&#x20;           tbptt\_window=getattr(cfg.hyperparameters, "tbptt\_window", 32),

&#x20;       )



&#x20;       self.selected\_armies = \[]



&#x20;   def effective\_config(self):

&#x20;       return {

&#x20;           "policy": self.policy.resolved\_config(),

&#x20;           "ppo": self.ppo.resolved\_config(),

&#x20;           "reward\_scale": float(self.reward\_scale),

&#x20;           "total\_updates\_estimate": int(self.total\_updates\_estimate),

&#x20;       }



&#x20;   def peek\_observation(self, obs):

&#x20;       return self.extractor.peek\_observation(obs)



&#x20;   def step(self, obs, deterministic: bool = False):

&#x20;       super(DefeatRoaches, self).step(obs)

&#x20;       self.steps += 1



&#x20;       policy\_input = self.extractor.extract\_observation(

&#x20;           obs,

&#x20;           update\_stats=not deterministic,

&#x20;       )

&#x20;       pre\_step\_state = self.snn\_state

&#x20;       policy\_input = policy\_input.with\_state(pre\_step\_state)

&#x20;       action, move\_x, move\_y, log\_prob, value, self.snn\_state = (

&#x20;           self.ppo.select\_action(

&#x20;               policy\_input,

&#x20;               deterministic=deterministic,

&#x20;           )

&#x20;       )



&#x20;       player\_relative = obs.observation.feature\_screen.player\_relative

&#x20;       self.selected\_armies = self.action\_space.find\_units(

&#x20;           player\_relative, \_PLAYER\_FRIENDLY,

&#x20;       )



&#x20;       action\_func = actions.FUNCTIONS.no\_op()

&#x20;       learnable = False



&#x20;       can\_attack = (

&#x20;           actions.FUNCTIONS.Attack\_screen.id in obs.observation.available\_actions

&#x20;       )

&#x20;       can\_move = (

&#x20;           actions.FUNCTIONS.Move\_screen.id in obs.observation.available\_actions

&#x20;       )

&#x20;       can\_select\_army = (

&#x20;           actions.FUNCTIONS.select\_army.id in obs.observation.available\_actions

&#x20;       )



&#x20;       if action == 0:

&#x20;           if can\_attack:

&#x20;               target\_position = self.action\_space.nearest\_enemy\_unit\_center(obs)

&#x20;               if target\_position is not None:

&#x20;                   action\_func = self.action\_space.attack(obs, target\_position)

&#x20;                   learnable = True

&#x20;           elif can\_select\_army:

&#x20;               action\_func = actions.FUNCTIONS.select\_army("select")

&#x20;       elif action == 1:

&#x20;           if can\_move and self.selected\_armies:

&#x20;               action\_func = self.action\_space.move(obs, move\_x, move\_y)

&#x20;               learnable = True

&#x20;           elif can\_select\_army:

&#x20;               action\_func = actions.FUNCTIONS.select\_army("select")

&#x20;       elif action == 2:

&#x20;           learnable = True



&#x20;       return (

&#x20;           action\_func,

&#x20;           action,

&#x20;           move\_x,

&#x20;           move\_y,

&#x20;           pre\_step\_state,

&#x20;           float(log\_prob),

&#x20;           float(value),

&#x20;           policy\_input,

&#x20;           learnable,

&#x20;       )



&#x20;   def reset(self):

&#x20;       super(DefeatRoaches, self).reset()

&#x20;       self.snn\_state = self.policy.init\_concrete\_state(batch\_size=1)

&#x20;       self.extractor.reset()

&#x20;       self.reward\_function.reset()

&#x20;       self.selected\_armies = \[]



&#x20;   def update\_policy(self):

&#x20;       \_, stats = self.ppo.update\_policy(

&#x20;           batch\_size=cfg.hyperparameters.batch\_size,

&#x20;           epochs=cfg.hyperparameters.epochs,

&#x20;       )

&#x20;       return stats

&#x20;(nothing ultra crazy, and yeah i thoght select army was a good easy action space reduction when i coded it the first time months ago)



Today

great, this is the perfect anchor point to backtrace from. Nothing ultra crazy here, which is exactly why we can wire in tokens without blowing up the whole stack.



I read your step() as a clean little loop: obs → extractor → PPO → action id + x,y → your old ActionSpace → pysc2. The token problem lives in that last hop, you execute the action but you never hand the token back to the extractor for the next frame.



What your agent is doing right now

action 0: attack nearest enemy, else select\_army fallback

action 1: move to (move\_x, move\_y) if you have units selected, else select\_army

action 2: you mark learnable but send no\_op, that's your "do nothing but learn" slot

selected\_armies: you find friendlies every step, but you never use the list except as a gate for move

policy\_input: comes only from extractor.extract\_observation(obs), no memory of last action

That's why your network only sees obs data. The ActionSpace runs, but the token dies at the end of step().



Minimal patch to start the backtrace

We keep your current 3-action logic, we just capture the token and feed it forward. Three tiny changes:



init a last\_action buffer

after you call self.action\_space.attack() or .move(), grab the token

pass that token into the extractor next step

Here's the diff-style version of your step and init, using the action\_space\_v2 I gave you:



Python

def \_\_init\_\_(...):

&#x20;   # ... your existing init ...

&#x20;   self.action\_space = ActionSpace(screen\_size=screen\_size)

&#x20;   self.last\_action\_token = np.array(\[0,0,0,0], dtype=np.int32)  # NEW



def step(self, obs, deterministic: bool = False):

&#x20;   super(DefeatRoaches, self).step(obs)

&#x20;   self.steps += 1



&#x20;   # pass last action into extractor

&#x20;   policy\_input = self.extractor.extract\_observation(

&#x20;       obs,

&#x20;       lastlove this, it's the perfect place to start the backtrace. It's simple, it runs, and that `select\_army` shortcut you added months ago is actually a smart action-space reduction for roaches, it keeps the agent from spamming move commands with no units selected.



I took your exact agent and added the token wiring so we don't break anything, but we now have a place to feed the last action back in. Grab the patched version: \[defeat\_roaches\_v2.py](container:///mnt/data/defeat\_roaches\_v2.py)



\## What this control schema is really doing

\- \*\*obs → extractor → policy\_input\*\*: you build the spatial + vector view, then tack on the SNN state with `.with\_state()`

\- \*\*PPO.select\_action\*\*: returns your old 3-way discrete `action` plus `move\_x, move\_y`. That's why action 0 = attack, 1 = move, 2 = no-op/learnable

\- \*\*action\_space\*\*: executes the pysc2 call, but the result never comes back to the network next step

\- \*\*select\_army fallback\*\*: keeps the loop alive when nothing is selected, that's why roaches training stayed stable



The token gap is exactly here: after `self.action\_space.attack()` or `.move()`, the token dies. Next step the network has no idea what it just clicked.



\## The three tiny edits I made

1\. \*\*init\*\*: `self.last\_action\_token = \[0,0,0,0]` so we have memory across steps

2\. \*\*step start\*\*: we grab `prev\_action\_token` right before extract\_observation. I left a TODO comment where you'll pass it into the extractor once we open that file

3\. \*\*step end\*\*: after you execute the action, we call `self.action\_space.get\_last\_token()` (from the v2 ActionSpace) and store it. If you're still on the old ActionSpace, it falls back to `\[action, move\_x, move\_y, 0]` so nothing crashes



No change to your return tuple, no change to PPO yet, your current training loop keeps working.



\## Backtrace roadmap from here

We now have the anchor. Next moves, in order, so we don't explode the architecture:



1\. \*\*ObservationExtractor\*\* (`obs\_space\_2.py`): add a `last\_action` argument to `extract\_observation`, concat the 4-int token to your vector input. That's where the network first sees its own actions

2\. \*\*PolicyNetwork\*\*: increase `vector\_input\_dim` by 4, or better, embed the action type into 8 dims and concat. No need to touch the CNN yet

3\. \*\*PPO.select\_action\*\*: keep returning `action, move\_x, move\_y` for now, but start logging the token alongside. Later we split into hybrid heads: one head for type\_id, two heads for x/y



You wanted to go beyond roaches. Once step 1 is done, you can swap in the v2 ActionSpace, start using `select\_point` and `select\_rect`, and your agent will already be feeding those clicks back into itself. No more blind policy.



Want to open `obs\_space\_2.py` next? That's where we actually inject the token, and it's usually a 5-line change. Paste it here and we can vibe through it together, no memory test.



36 lines hidden

yes but as said the action space was th elast i tired to modify, shoudl i show you the obs space now or the policy or the policy input as said?

Today

obs space first, that's the insertion point.



You already have the action side producing a token, now we need the place where that token re-enters the network. That's obs\_space\_2.py, not the policy yet.



Here's why the order matters for a clean backtrace:



obs\_space\_2.py: this is where extract\_observation() builds the thing you later call .with\_state() on. We add last\_action here as an extra vector field, no shape changes to the CNN, just a few more numbers in the flat vector. Five lines, done.

policy input wrapper: usually a tiny dataclass inside the extractor. Once obs space accepts the token, we just make sure .with\_last\_action() exists or we piggyback on .with\_state(). That's step 1.5, we see it when we open obs space.

policy\_network.py: after obs space works, we bump vector\_input\_dim by 4 (or by the embedding size) so the network actually has weights for the new input. If we do policy first, you'll get a shape mismatch and have to guess.

PPO: last, because select\_action currently returns action, move\_x, move\_y. Once the network sees last\_action, we can start teaching it to output the hybrid heads without breaking the old 3-action mode.

You said action space was the last you tried to modify, so let's not touch it again yet. Drop obs\_space\_2.py here and we'll wire the token in together, then we peek at the policy input class inside it, then we go to PolicyNetwork.



No memory test, just show me the file and I'll mark exactly where the 4-int token slots in.



(yes i know i have to trim away all that is not helpful for the scope of the final agent out fo here, too much data will be static for each minigame but ia m new to both maskin and whatever what woudl solve the problem is called XD) import random



import numpy as np



\# ---- PySC2 colors.py fix for Python 3.11+ (random.shuffle signature) ----

\# The same monkey-patch lives at PPO\_CNN\_agent.py:8-15. We replicate it

\# here so that any path that imports obs\_space\_2 WITHOUT going through

\# the training entrypoint (e.g. unit tests, smoke scripts) is still safe

\# — otherwise the `from pysc2.lib import features` below triggers

\# SCREEN\_FEATURES construction → colors.shuffled\_hue → the broken

\# random.shuffle(palette, lambda: 0.5) call.

from pysc2.lib import colors as \_colors





def \_shuffled\_hue\_fixed(scale):

&#x20;   palette = list(\_colors.smooth\_hue\_palette(scale))

&#x20;   random\_keys = \[random.random() for \_ in palette]

&#x20;   palette = \[x for \_, x in sorted(zip(random\_keys, palette))]

&#x20;   return np.array(palette)





\_colors.shuffled\_hue = \_shuffled\_hue\_fixed

\# -------------------------------------------------------------------------



from pysc2.lib import features  # noqa: E402,F401  (kept for downstream consumers)

import torch  # noqa: E402



from PPO\_CNN.policy\_input import (

&#x20;   CURATED\_FEATURE\_UNIT\_FIELDS,

&#x20;   DEFEAT\_ROACHES\_ACTION\_IDS,

&#x20;   MAX\_ENTITY\_TOKENS,

&#x20;   MAX\_SELECTION\_TOKENS,

&#x20;   META\_PLAYER\_FEATURE\_DIM,

&#x20;   META\_VECTOR\_DIM,

&#x20;   NO\_ACTION\_SENTINEL\_INDEX,

&#x20;   PolicyInputBatch,

&#x20;   SELECTION\_FEATURE\_NAMES,

&#x20;   SPATIAL\_OBS\_SHAPE,

&#x20;   UNKNOWN\_LAST\_ACTION\_INDEX,

)



\_PLAYER\_FRIENDLY = 1

\_FEATURE\_UNIT\_INDEX = {

&#x20;   name: int(field.value)

&#x20;   for name, field in features.FeatureUnit.\_\_members\_\_.items()

}

\_UNIT\_LAYER\_INDEX = {

&#x20;   name: int(field.value)

&#x20;   for name, field in features.UnitLayer.\_\_members\_\_.items()

}

\_LAST\_ACTION\_TO\_INDEX = {

&#x20;   action\_id: idx + 1 for idx, action\_id in enumerate(DEFEAT\_ROACHES\_ACTION\_IDS)

}





def \_validate\_index\_fields(field\_names, index\_map, index\_name):

&#x20;   missing = \[name for name in field\_names if name not in index\_map]

&#x20;   if missing:

&#x20;       raise ValueError(

&#x20;           f"Unknown {index\_name} field(s): {', '.join(sorted(missing))}",

&#x20;       )





class RunningFeatureNormalizer:

&#x20;   def \_\_init\_\_(

&#x20;       self,

&#x20;       field\_names,

&#x20;       normalized\_fields,

&#x20;       min\_count\_for\_normalize=32.0,

&#x20;       min\_std=1.0e-2,

&#x20;       min\_variance=None,

&#x20;       output\_clip=10.0,

&#x20;   ):

&#x20;       self.field\_names = tuple(field\_names)

&#x20;       self.normalized\_fields = tuple(normalized\_fields)

&#x20;       self.normalized\_indices = \[

&#x20;           self.field\_names.index(name) for name in self.normalized\_fields

&#x20;       ]

&#x20;       self.min\_count\_for\_normalize = float(min\_count\_for\_normalize)

&#x20;       self.min\_std = float(min\_std)

&#x20;       if min\_variance is None:

&#x20;           min\_variance = self.min\_std \*\* 2

&#x20;       self.min\_variance = float(min\_variance)

&#x20;       self.output\_clip = float(output\_clip)

&#x20;       size = len(self.normalized\_indices)

&#x20;       self.count = 0.0

&#x20;       self.mean = np.zeros(size, dtype=np.float64)

&#x20;       self.m2 = np.zeros(size, dtype=np.float64)



&#x20;   def update(self, values):

&#x20;       values = np.asarray(values, dtype=np.float32)

&#x20;       if values.size == 0 or not self.normalized\_indices:

&#x20;           return

&#x20;       if values.ndim == 1:

&#x20;           values = values.reshape(1, -1)



&#x20;       selected = values\[:, self.normalized\_indices].astype(np.float64)

&#x20;       batch\_count = float(selected.shape\[0])

&#x20;       batch\_mean = selected.mean(axis=0)

&#x20;       batch\_m2 = ((selected - batch\_mean) \*\* 2).sum(axis=0)

&#x20;       if self.count == 0.0:

&#x20;           self.count = batch\_count

&#x20;           self.mean = batch\_mean

&#x20;           self.m2 = batch\_m2

&#x20;           return



&#x20;       delta = batch\_mean - self.mean

&#x20;       total = self.count + batch\_count

&#x20;       self.mean = self.mean + delta \* (batch\_count / total)

&#x20;       self.m2 = self.m2 + batch\_m2 + (delta \*\* 2) \* self.count \* batch\_count / total

&#x20;       self.count = total



&#x20;   def normalize(self, values):

&#x20;       values = np.asarray(values, dtype=np.float32)

&#x20;       if values.size == 0 or not self.normalized\_indices:

&#x20;           return values

&#x20;       squeeze\_back = False

&#x20;       if values.ndim == 1:

&#x20;           values = values.reshape(1, -1)

&#x20;           squeeze\_back = True

&#x20;       out = values.copy()

&#x20;       if self.count < self.min\_count\_for\_normalize:

&#x20;           return out\[0] if squeeze\_back else out



&#x20;       denom = np.maximum(self.count - 1.0, 1.0)

&#x20;       variance = np.maximum(self.m2 / denom, 0.0)

&#x20;       std = np.sqrt(variance)

&#x20;       active = np.isfinite(std) \& (std >= self.min\_std) \& (

&#x20;           variance >= self.min\_variance

&#x20;       )

&#x20;       if not np.any(active):

&#x20;           return out\[0] if squeeze\_back else out



&#x20;       selected = out\[:, self.normalized\_indices]

&#x20;       mean = self.mean.astype(np.float32, copy=False)

&#x20;       std\_safe = np.maximum(std, self.min\_std).astype(np.float32, copy=False)

&#x20;       normalized = (selected - mean) / std\_safe

&#x20;       normalized = np.nan\_to\_num(

&#x20;           normalized,

&#x20;           nan=0.0,

&#x20;           posinf=self.output\_clip,

&#x20;           neginf=-self.output\_clip,

&#x20;       )

&#x20;       normalized = np.clip(normalized, -self.output\_clip, self.output\_clip)

&#x20;       selected\[:, active] = normalized\[:, active]

&#x20;       out\[:, self.normalized\_indices] = selected

&#x20;       return out\[0] if squeeze\_back else out



&#x20;   def state\_dict(self):

&#x20;       return {

&#x20;           "count": float(self.count),

&#x20;           "mean": self.mean.tolist(),

&#x20;           "m2": self.m2.tolist(),

&#x20;       }



&#x20;   def load\_state\_dict(self, state):

&#x20;       self.count = float(state.get("count", 0.0))

&#x20;       self.mean = np.asarray(state.get("mean", self.mean.tolist()), dtype=np.float64)

&#x20;       self.m2 = np.asarray(state.get("m2", self.m2.tolist()), dtype=np.float64)





def get\_friendly\_health(obs):

&#x20;   """Sum of health across all friendly units."""

&#x20;   feature\_units = getattr(obs.observation, "feature\_units", None)

&#x20;   if feature\_units is None or len(feature\_units) == 0:

&#x20;       return 0.0

&#x20;   return float(sum(

&#x20;       u.health for u in feature\_units

&#x20;       if getattr(u, "alliance", 0) == \_PLAYER\_FRIENDLY

&#x20;   ))





class ObservationExtractor:

&#x20;   def \_\_init\_\_(self):

&#x20;       self.device = torch.device("cuda" if torch.cuda.is\_available() else "cpu")

&#x20;       \_validate\_index\_fields(

&#x20;           CURATED\_FEATURE\_UNIT\_FIELDS,

&#x20;           \_FEATURE\_UNIT\_INDEX,

&#x20;           "FeatureUnit",

&#x20;       )

&#x20;       \_validate\_index\_fields(

&#x20;           SELECTION\_FEATURE\_NAMES,

&#x20;           \_UNIT\_LAYER\_INDEX,

&#x20;           "UnitLayer",

&#x20;       )

&#x20;       self.entity\_normalizer = RunningFeatureNormalizer(

&#x20;           CURATED\_FEATURE\_UNIT\_FIELDS,

&#x20;           normalized\_fields=(

&#x20;               "health",

&#x20;               "health\_ratio",

&#x20;               "shield",

&#x20;               "shield\_ratio",

&#x20;               "energy",

&#x20;               "energy\_ratio",

&#x20;               "weapon\_cooldown",

&#x20;               "x",

&#x20;               "y",

&#x20;               "radius",

&#x20;               "build\_progress",

&#x20;               "order\_id\_0",

&#x20;               "order\_id\_1",

&#x20;               "assigned\_harvesters",

&#x20;               "ideal\_harvesters",

&#x20;           ),

&#x20;       )

&#x20;       self.selection\_normalizer = RunningFeatureNormalizer(

&#x20;           SELECTION\_FEATURE\_NAMES,

&#x20;           normalized\_fields=(

&#x20;               "health",

&#x20;               "shields",

&#x20;               "energy",

&#x20;               "transport\_slots\_taken",

&#x20;               "build\_progress",

&#x20;           ),

&#x20;       )



&#x20;   def extract\_observation(self, obs, update\_stats=True):

&#x20;       feature\_screen = getattr(obs.observation, "feature\_screen", None)

&#x20;       if feature\_screen is not None and getattr(feature\_screen, "size", 0) > 0:

&#x20;           spatial\_obs = torch.as\_tensor(

&#x20;               np.asarray(feature\_screen) / 255.0,

&#x20;               dtype=torch.float32,

&#x20;               device=self.device,

&#x20;           ).unsqueeze(0)

&#x20;       else:

&#x20;           spatial\_obs = torch.zeros(

&#x20;               (1, \*SPATIAL\_OBS\_SHAPE),

&#x20;               dtype=torch.float32,

&#x20;               device=self.device,

&#x20;           )



&#x20;       entity\_rows = self.\_extract\_entity\_rows(obs)

&#x20;       if update\_stats:

&#x20;           self.entity\_normalizer.update(entity\_rows)

&#x20;       entity\_rows = self.entity\_normalizer.normalize(entity\_rows)



&#x20;       selection\_rows = self.\_extract\_selection\_rows(obs)

&#x20;       if update\_stats:

&#x20;           self.selection\_normalizer.update(selection\_rows)

&#x20;       selection\_rows = self.selection\_normalizer.normalize(selection\_rows)



&#x20;       entity\_features, entity\_mask = self.\_pad\_rows(

&#x20;           entity\_rows,

&#x20;           max\_rows=MAX\_ENTITY\_TOKENS,

&#x20;           width=len(CURATED\_FEATURE\_UNIT\_FIELDS),

&#x20;       )

&#x20;       selection\_features, selection\_mask = self.\_pad\_rows(

&#x20;           selection\_rows,

&#x20;           max\_rows=MAX\_SELECTION\_TOKENS,

&#x20;           width=len(SELECTION\_FEATURE\_NAMES),

&#x20;       )

&#x20;       meta\_vec = self.\_extract\_meta\_vector(obs)



&#x20;       return PolicyInputBatch(

&#x20;           spatial\_obs=spatial\_obs,

&#x20;           entity\_features=entity\_features.unsqueeze(0),

&#x20;           entity\_mask=entity\_mask.unsqueeze(0),

&#x20;           selection\_features=selection\_features.unsqueeze(0),

&#x20;           selection\_mask=selection\_mask.unsqueeze(0),

&#x20;           meta\_vec=meta\_vec.unsqueeze(0),

&#x20;       )



&#x20;   def peek\_observation(self, obs):

&#x20;       return self.extract\_observation(obs, update\_stats=False)



&#x20;   def \_extract\_entity\_rows(self, obs):

&#x20;       feature\_units = getattr(obs.observation, "feature\_units", None)

&#x20;       numeric = self.\_coerce\_numeric\_rows(feature\_units)

&#x20;       if numeric is not None:

&#x20;           return self.\_project\_numeric\_rows(

&#x20;               numeric,

&#x20;               field\_names=CURATED\_FEATURE\_UNIT\_FIELDS,

&#x20;               index\_map=\_FEATURE\_UNIT\_INDEX,

&#x20;           )

&#x20;       return self.\_project\_object\_rows(

&#x20;           feature\_units,

&#x20;           field\_names=CURATED\_FEATURE\_UNIT\_FIELDS,

&#x20;       )



&#x20;   def \_extract\_selection\_rows(self, obs):

&#x20;       selection = getattr(obs.observation, "multi\_select", None)

&#x20;       if selection is None or len(selection) == 0:

&#x20;           selection = getattr(obs.observation, "single\_select", None)



&#x20;       numeric = self.\_coerce\_numeric\_rows(selection)

&#x20;       if numeric is not None:

&#x20;           return self.\_project\_numeric\_rows(

&#x20;               numeric,

&#x20;               field\_names=SELECTION\_FEATURE\_NAMES,

&#x20;               index\_map=\_UNIT\_LAYER\_INDEX,

&#x20;           )

&#x20;       return self.\_project\_object\_rows(

&#x20;           selection,

&#x20;           field\_names=SELECTION\_FEATURE\_NAMES,

&#x20;       )



&#x20;   def \_extract\_meta\_vector(self, obs):

&#x20;       player = getattr(obs.observation, "player", None)

&#x20;       if player is None:

&#x20;           player\_vec = np.zeros(META\_PLAYER\_FEATURE\_DIM, dtype=np.float32)

&#x20;       else:

&#x20;           player\_vec = np.asarray(player, dtype=np.float32).reshape(-1)

&#x20;           if player\_vec.size < META\_PLAYER\_FEATURE\_DIM:

&#x20;               player\_vec = np.pad(

&#x20;                   player\_vec,

&#x20;                   (0, META\_PLAYER\_FEATURE\_DIM - player\_vec.size),

&#x20;                   mode="constant",

&#x20;               )

&#x20;           else:

&#x20;               player\_vec = player\_vec\[:META\_PLAYER\_FEATURE\_DIM]



&#x20;       available\_actions = getattr(obs.observation, "available\_actions", None)

&#x20;       available\_set = (

&#x20;           {int(action\_id) for action\_id in list(available\_actions)}

&#x20;           if available\_actions is not None

&#x20;           else set()

&#x20;       )

&#x20;       available\_mask = np.asarray(

&#x20;           \[

&#x20;               1.0 if action\_id in available\_set else 0.0

&#x20;               for action\_id in DEFEAT\_ROACHES\_ACTION\_IDS

&#x20;           ],

&#x20;           dtype=np.float32,

&#x20;       )



&#x20;       last\_actions = getattr(obs.observation, "last\_actions", None)

&#x20;       if last\_actions is None or len(last\_actions) == 0:

&#x20;           last\_action\_index = float(NO\_ACTION\_SENTINEL\_INDEX)

&#x20;       else:

&#x20;           raw\_last\_action = int(list(last\_actions)\[0])

&#x20;           last\_action\_index = float(

&#x20;               \_LAST\_ACTION\_TO\_INDEX.get(raw\_last\_action, UNKNOWN\_LAST\_ACTION\_INDEX),

&#x20;           )



&#x20;       return torch.as\_tensor(

&#x20;           np.concatenate(

&#x20;               (

&#x20;                   player\_vec,

&#x20;                   available\_mask,

&#x20;                   np.asarray(\[last\_action\_index], dtype=np.float32),

&#x20;               )

&#x20;           ),

&#x20;           dtype=torch.float32,

&#x20;           device=self.device,

&#x20;       )



&#x20;   def \_coerce\_numeric\_rows(self, rows):

&#x20;       if rows is None:

&#x20;           return np.zeros((0, 0), dtype=np.float32)

&#x20;       if isinstance(rows, torch.Tensor):

&#x20;           arr = rows.detach().cpu().numpy()

&#x20;       else:

&#x20;           arr = np.asarray(rows)

&#x20;       if arr.ndim == 1:

&#x20;           arr = arr.reshape(1, -1)

&#x20;       if arr.ndim == 2 and arr.dtype != object:

&#x20;           return arr.astype(np.float32, copy=False)

&#x20;       return None



&#x20;   def \_project\_numeric\_rows(self, rows, field\_names, index\_map):

&#x20;       if rows is None or rows.size == 0:

&#x20;           return np.zeros((0, len(field\_names)), dtype=np.float32)

&#x20;       projected = np.zeros((rows.shape\[0], len(field\_names)), dtype=np.float32)

&#x20;       for idx, field\_name in enumerate(field\_names):

&#x20;           source\_idx = index\_map\[field\_name]

&#x20;           if source\_idx >= rows.shape\[1]:

&#x20;               raise ValueError(

&#x20;                   f"Numeric rows for '{field\_name}' expected column {source\_idx} "

&#x20;                   f"but width is only {rows.shape\[1]}",

&#x20;               )

&#x20;           projected\[:, idx] = rows\[:, source\_idx]

&#x20;       return projected



&#x20;   def \_project\_object\_rows(self, rows, field\_names):

&#x20;       if rows is None or len(rows) == 0:

&#x20;           return np.zeros((0, len(field\_names)), dtype=np.float32)

&#x20;       projected = np.zeros((len(rows), len(field\_names)), dtype=np.float32)

&#x20;       for row\_idx, row in enumerate(list(rows)):

&#x20;           for field\_idx, field\_name in enumerate(field\_names):

&#x20;               value = getattr(row, field\_name, 0.0)

&#x20;               projected\[row\_idx, field\_idx] = float(

&#x20;                   0.0 if value is None else value,

&#x20;               )

&#x20;       return projected



&#x20;   def \_pad\_rows(self, rows, max\_rows, width):

&#x20;       rows = np.asarray(rows, dtype=np.float32)

&#x20;       features = torch.zeros(

&#x20;           (max\_rows, width),

&#x20;           dtype=torch.float32,

&#x20;           device=self.device,

&#x20;       )

&#x20;       mask = torch.zeros((max\_rows,), dtype=torch.bool, device=self.device)

&#x20;       if rows.size == 0:

&#x20;           return features, mask



&#x20;       actual\_rows = min(int(rows.shape\[0]), int(max\_rows))

&#x20;       features\[:actual\_rows] = torch.as\_tensor(

&#x20;           rows\[:actual\_rows],

&#x20;           dtype=torch.float32,

&#x20;           device=self.device,

&#x20;       )

&#x20;       mask\[:actual\_rows] = True

&#x20;       return features, mask



&#x20;   def get\_observation\_dimensions(self, obs):

&#x20;       del obs

&#x20;       return SPATIAL\_OBS\_SHAPE, META\_VECTOR\_DIM



&#x20;   def reset(self):

&#x20;       return None



&#x20;   def state\_dict(self):

&#x20;       return {

&#x20;           "entity\_normalizer": self.entity\_normalizer.state\_dict(),

&#x20;           "selection\_normalizer": self.selection\_normalizer.state\_dict(),

&#x20;       }



&#x20;   def load\_state\_dict(self, state):

&#x20;       if not state:

&#x20;           return

&#x20;       entity\_state = state.get("entity\_normalizer")

&#x20;       if entity\_state:

&#x20;           self.entity\_normalizer.load\_state\_dict(entity\_state)

&#x20;       selection\_state = state.get("selection\_normalizer")

&#x20;       if selection\_state:

&#x20;           self.selection\_normalizer.load\_state\_dict(selection\_state)



Today

perfect timing, obs space is exactly where we want to cut and where we want to inject the token. I read your file, you already built the hard parts, normalizers, padding, masks, meta vector. The "too much data" feeling is real though, for defeat roaches half of this never changes.



I made you a minimal patch that keeps everything working but adds the last\_action token straight into the meta vector. Grab it:



📎 obs\_space\_2\_v2.py



What I changed, no rewrites

extract\_observation() now takes last\_action\_token=None. Your agent can pass self.last\_action\_token from the patched DefeatRoaches

\_extract\_meta\_vector() appends 4 floats: \[type, x\_norm, y\_norm, extra]. If you don't pass a token, it pads zeros so old code still runs

get\_observation\_dimensions() now reports META\_VECTOR\_DIM + 4 so you see the new size downstream

That's the injection point. No need to touch entity or selection code yet.



About masking, since you mentioned it

You are already doing it right. Look at \_pad\_rows():



features is a fixed-size tensor \[MAX\_ENTITY\_TOKENS, width]

mask is a bool vector where True means "this row is a real unit", False means "this is padding"

The network uses the mask to ignore padded rows in attention or pooling. For roaches you could drop MAX\_ENTITY\_TOKENS from whatever it is now (probably 64?) down to 16, and MAX\_SELECTION\_TOKENS down to 1, because you never have more than a handful of marines. Less padding = faster, less noise.



What to trim for roaches, without breaking the architecture

You don't need to delete code yet, just shrink the constants in policy\_input.py:



CURATED\_FEATURE\_UNIT\_FIELDS: keep only \["health", "health\_ratio", "x", "y", "alliance"]. Everything else like shield, energy, build\_progress is always zero in roaches

SELECTION\_FEATURE\_NAMES: you can probably keep 2-3 fields, or set MAX\_SELECTION\_TOKENS = 1 since you only ever select\_army

META\_VECTOR\_DIM: increase it by 4 in policy\_input.py to match the new token. That's the only required change to make the v2 obs space run

Your comment about the colors monkey-patch is spot on, keep it, PySC2 breaks on 3.11 without it.



Next step in the backtrace

Show me PPO\_CNN/policy\_input.py now. That's where META\_VECTOR\_DIM, MAX\_ENTITY\_TOKENS, and PolicyInputBatch live. We need to bump the vector dim by 4, and while we're there we can trim the field lists for roaches. Then we go to policy\_network.py and make sure the first linear layer expects the new size.



You are one constant change away from the network actually seeing its own last click.



technically i ask technically in starcraft does th epolicy see the last click through the screen? or it is not ecoded as obs data?

Today

technically no, the policy does not see your last click on the screen, and PySC2 does not encode it for you.



Here's what the API actually gives you each step, and why your token matters:



What is in the observation

feature\_screen / feature\_minimap: these are the pixel layers you already normalize. They show unit positions, health, player\_relative, creep, visibility, etc. There is no "cursor" layer, no click trail, no ghost of where you clicked last frame. Once the click is consumed by the game engine, the pixels go back to showing the world state only.

obs.observation.last\_actions: this is the field you already read in \_extract\_meta\_vector. Look at your own code:

Python

last\_actions = getattr(obs.observation, "last\_actions", None)

raw\_last\_action = int(list(last\_actions)\[0])

That list contains only the function IDs, like 1 for no\_op, 12 for Attack\_screen, 13 for Move\_screen. It does not contain the x,y arguments you passed. PySC2 strips the arguments out before it hands you the observation.



available\_actions: tells you what you can do now, not what you did.

What you see one frame later

The effect of a click shows up indirectly, and delayed:



a unit gets selected → feature\_screen.selected lights up next frame

a move order → the unit's order\_id\_0 changes, and its position drifts over several frames

an attack → target health drops later

That's all inference, not a direct "I clicked at (42,31)". For a fast micro policy, especially with your SNN and BPTT window, that delay is partial observability. The network has to guess its own last action from the world change, which is noisy.



Why your token fixes it

When you store \[type]\[x]\[y]\[extra] in self.last\_action\_token and append it to the meta vector in obs\_space\_2\_v2, you are giving the policy something the game never gives it: an exact, zero-delay memory of its own motor command.



Humans get this for free because we see the mouse cursor and the order line. The agent doesn't. Without the token, the policy is effectively blind to its own click location until the world reacts, which in StarCraft can be 2-3 frames later, enough to hurt credit assignment in PPO.



So technically: the screen does not contain last click data, last\_actions gives you only the ID, not coordinates. If you want the policy to know "I just attack-clicked at 42,31", you have to feed it yourself, which is exactly what the 4-float token in your patched obs space does.



oky fair ehre is the input: from \_\_future\_\_ import annotations



from dataclasses import dataclass

from typing import Final



import torch





SPATIAL\_OBS\_SHAPE: Final\[tuple\[int, int, int]] = (27, 84, 84)

SPATIAL\_TOKEN\_COUNT: Final\[int] = 49

MAX\_ENTITY\_TOKENS: Final\[int] = 24

MAX\_SELECTION\_TOKENS: Final\[int] = 20

SELECTION\_FEATURE\_DIM: Final\[int] = 7

TOKEN\_TYPE\_GROUPS: Final\[int] = 4

TOTAL\_TOKEN\_COUNT: Final\[int] = (

&#x20;   SPATIAL\_TOKEN\_COUNT + MAX\_ENTITY\_TOKENS + MAX\_SELECTION\_TOKENS + 1

)



CURATED\_FEATURE\_UNIT\_FIELDS: Final\[tuple\[str, ...]] = (

&#x20;   "unit\_type",

&#x20;   "alliance",

&#x20;   "health",

&#x20;   "health\_ratio",

&#x20;   "shield",

&#x20;   "shield\_ratio",

&#x20;   "energy",

&#x20;   "energy\_ratio",

&#x20;   "weapon\_cooldown",

&#x20;   "x",

&#x20;   "y",

&#x20;   "radius",

&#x20;   "build\_progress",

&#x20;   "order\_id\_0",

&#x20;   "order\_id\_1",

&#x20;   "is\_selected",

&#x20;   "is\_in\_cargo",

&#x20;   "assigned\_harvesters",

&#x20;   "ideal\_harvesters",

&#x20;   "active",

&#x20;   "hallucination",

)

UNIT\_TYPE\_FEATURE\_INDEX: Final\[int] = 0



SELECTION\_FEATURE\_NAMES: Final\[tuple\[str, ...]] = (

&#x20;   "unit\_type",

&#x20;   "player\_relative",

&#x20;   "health",

&#x20;   "shields",

&#x20;   "energy",

&#x20;   "transport\_slots\_taken",

&#x20;   "build\_progress",

)

SELECTION\_UNIT\_TYPE\_INDEX: Final\[int] = 0



META\_PLAYER\_FEATURE\_DIM: Final\[int] = 11

DEFEAT\_ROACHES\_ACTION\_IDS: Final\[tuple\[int, ...]] = (

&#x20;   0,

&#x20;   1,

&#x20;   2,

&#x20;   3,

&#x20;   4,

&#x20;   7,

&#x20;   12,

&#x20;   13,

&#x20;   274,

&#x20;   331,

&#x20;   332,

&#x20;   333,

&#x20;   334,

&#x20;   451,

&#x20;   452,

&#x20;   453,

)

META\_AVAILABLE\_ACTION\_DIM: Final\[int] = len(DEFEAT\_ROACHES\_ACTION\_IDS)

META\_LAST\_ACTION\_INDEX\_DIM: Final\[int] = 1

META\_VECTOR\_DIM: Final\[int] = (

&#x20;   META\_PLAYER\_FEATURE\_DIM + META\_AVAILABLE\_ACTION\_DIM + META\_LAST\_ACTION\_INDEX\_DIM

)

NO\_ACTION\_SENTINEL\_INDEX: Final\[int] = 0

UNKNOWN\_LAST\_ACTION\_INDEX: Final\[int] = len(DEFEAT\_ROACHES\_ACTION\_IDS) + 1



SNNState = tuple\[torch.Tensor, torch.Tensor]





@dataclass(slots=True)

class PolicyInputBatch:

&#x20;   """

&#x20;   Frozen observation protocol for Fix 3 hybrid tokenization.



&#x20;   Shapes follow docs/archive/NEXT\_FIXES\_PLAN\_3.md §3.5 exactly:

&#x20;     spatial\_obs:        \[B, 27, 84, 84]

&#x20;     entity\_features:    \[B, 24, F\_unit]

&#x20;     entity\_mask:        \[B, 24]

&#x20;     selection\_features: \[B, 20, 7]

&#x20;     selection\_mask:     \[B, 20]

&#x20;     meta\_vec:           \[B, F\_meta]

&#x20;     state\_in:           optional SNN state tuple

&#x20;   """



&#x20;   spatial\_obs: torch.Tensor

&#x20;   entity\_features: torch.Tensor

&#x20;   entity\_mask: torch.Tensor

&#x20;   selection\_features: torch.Tensor

&#x20;   selection\_mask: torch.Tensor

&#x20;   meta\_vec: torch.Tensor

&#x20;   state\_in: SNNState | None = None



&#x20;   def \_\_post\_init\_\_(self) -> None:

&#x20;       self.\_validate()



&#x20;   @property

&#x20;   def batch\_size(self) -> int:

&#x20;       return int(self.spatial\_obs.shape\[0])



&#x20;   @property

&#x20;   def feature\_unit\_dim(self) -> int:

&#x20;       return int(self.entity\_features.shape\[-1])



&#x20;   @property

&#x20;   def meta\_dim(self) -> int:

&#x20;       return int(self.meta\_vec.shape\[-1])



&#x20;   def to(

&#x20;       self,

&#x20;       device: torch.device | str | None = None,

&#x20;       dtype: torch.dtype | None = None,

&#x20;   ) -> "PolicyInputBatch":

&#x20;       float\_kwargs = {}

&#x20;       mask\_kwargs = {}

&#x20;       if device is not None:

&#x20;           float\_kwargs\["device"] = device

&#x20;           mask\_kwargs\["device"] = device

&#x20;       if dtype is not None:

&#x20;           float\_kwargs\["dtype"] = dtype



&#x20;       moved\_state = None

&#x20;       if self.state\_in is not None:

&#x20;           syn, mem = self.state\_in

&#x20;           moved\_state = (

&#x20;               syn.to(\*\*float\_kwargs),

&#x20;               mem.to(\*\*float\_kwargs),

&#x20;           )



&#x20;       return PolicyInputBatch(

&#x20;           spatial\_obs=self.spatial\_obs.to(\*\*float\_kwargs),

&#x20;           entity\_features=self.entity\_features.to(\*\*float\_kwargs),

&#x20;           entity\_mask=self.entity\_mask.to(\*\*mask\_kwargs),

&#x20;           selection\_features=self.selection\_features.to(\*\*float\_kwargs),

&#x20;           selection\_mask=self.selection\_mask.to(\*\*mask\_kwargs),

&#x20;           meta\_vec=self.meta\_vec.to(\*\*float\_kwargs),

&#x20;           state\_in=moved\_state,

&#x20;       )



&#x20;   def detach(self) -> "PolicyInputBatch":

&#x20;       detached\_state = None

&#x20;       if self.state\_in is not None:

&#x20;           syn, mem = self.state\_in

&#x20;           detached\_state = (syn.detach(), mem.detach())



&#x20;       return PolicyInputBatch(

&#x20;           spatial\_obs=self.spatial\_obs.detach(),

&#x20;           entity\_features=self.entity\_features.detach(),

&#x20;           entity\_mask=self.entity\_mask.detach(),

&#x20;           selection\_features=self.selection\_features.detach(),

&#x20;           selection\_mask=self.selection\_mask.detach(),

&#x20;           meta\_vec=self.meta\_vec.detach(),

&#x20;           state\_in=detached\_state,

&#x20;       )



&#x20;   def with\_state(self, state\_in: SNNState | None) -> "PolicyInputBatch":

&#x20;       return PolicyInputBatch(

&#x20;           spatial\_obs=self.spatial\_obs,

&#x20;           entity\_features=self.entity\_features,

&#x20;           entity\_mask=self.entity\_mask,

&#x20;           selection\_features=self.selection\_features,

&#x20;           selection\_mask=self.selection\_mask,

&#x20;           meta\_vec=self.meta\_vec,

&#x20;           state\_in=state\_in,

&#x20;       )



&#x20;   @classmethod

&#x20;   def stack(cls, batches: list\["PolicyInputBatch"]) -> "PolicyInputBatch":

&#x20;       if not batches:

&#x20;           raise ValueError("batches must be non-empty")



&#x20;       states = \[batch.state\_in for batch in batches]

&#x20;       if all(state is None for state in states):

&#x20;           state\_in = None

&#x20;       elif any(state is None for state in states):

&#x20;           raise ValueError("Either every stacked batch has state\_in, or none do")

&#x20;       else:

&#x20;           syn\_parts = \[state\[0] for state in states if state is not None]

&#x20;           mem\_parts = \[state\[1] for state in states if state is not None]

&#x20;           state\_in = (torch.cat(syn\_parts, dim=0), torch.cat(mem\_parts, dim=0))



&#x20;       return cls(

&#x20;           spatial\_obs=torch.cat(\[batch.spatial\_obs for batch in batches], dim=0),

&#x20;           entity\_features=torch.cat(

&#x20;               \[batch.entity\_features for batch in batches], dim=0,

&#x20;           ),

&#x20;           entity\_mask=torch.cat(\[batch.entity\_mask for batch in batches], dim=0),

&#x20;           selection\_features=torch.cat(

&#x20;               \[batch.selection\_features for batch in batches], dim=0,

&#x20;           ),

&#x20;           selection\_mask=torch.cat(

&#x20;               \[batch.selection\_mask for batch in batches], dim=0,

&#x20;           ),

&#x20;           meta\_vec=torch.cat(\[batch.meta\_vec for batch in batches], dim=0),

&#x20;           state\_in=state\_in,

&#x20;       )



&#x20;   def index\_select(self, index: torch.Tensor | list\[int]) -> "PolicyInputBatch":

&#x20;       if not isinstance(index, torch.Tensor):

&#x20;           index = torch.as\_tensor(

&#x20;               index,

&#x20;               dtype=torch.long,

&#x20;               device=self.spatial\_obs.device,

&#x20;           )

&#x20;       else:

&#x20;           index = index.to(device=self.spatial\_obs.device, dtype=torch.long)



&#x20;       if index.ndim == 0:

&#x20;           index = index.view(1)

&#x20;       if index.ndim != 1:

&#x20;           raise ValueError(

&#x20;               f"index must be 1D for batch slicing, got shape={tuple(index.shape)}",

&#x20;           )



&#x20;       sliced\_state = None

&#x20;       if self.state\_in is not None:

&#x20;           syn, mem = self.state\_in

&#x20;           sliced\_state = (

&#x20;               syn.index\_select(0, index),

&#x20;               mem.index\_select(0, index),

&#x20;           )



&#x20;       return PolicyInputBatch(

&#x20;           spatial\_obs=self.spatial\_obs.index\_select(0, index),

&#x20;           entity\_features=self.entity\_features.index\_select(0, index),

&#x20;           entity\_mask=self.entity\_mask.index\_select(0, index),

&#x20;           selection\_features=self.selection\_features.index\_select(0, index),

&#x20;           selection\_mask=self.selection\_mask.index\_select(0, index),

&#x20;           meta\_vec=self.meta\_vec.index\_select(0, index),

&#x20;           state\_in=sliced\_state,

&#x20;       )



&#x20;   def \_validate(self) -> None:

&#x20;       self.\_validate\_float\_tensor(

&#x20;           "spatial\_obs",

&#x20;           self.spatial\_obs,

&#x20;           expected\_ndim=4,

&#x20;           expected\_tail=SPATIAL\_OBS\_SHAPE,

&#x20;       )

&#x20;       self.\_validate\_float\_tensor(

&#x20;           "entity\_features",

&#x20;           self.entity\_features,

&#x20;           expected\_ndim=3,

&#x20;           expected\_tail=(MAX\_ENTITY\_TOKENS, None),

&#x20;       )

&#x20;       self.\_validate\_mask(

&#x20;           "entity\_mask",

&#x20;           self.entity\_mask,

&#x20;           expected\_shape=(self.batch\_size, MAX\_ENTITY\_TOKENS),

&#x20;       )

&#x20;       self.\_validate\_float\_tensor(

&#x20;           "selection\_features",

&#x20;           self.selection\_features,

&#x20;           expected\_ndim=3,

&#x20;           expected\_tail=(MAX\_SELECTION\_TOKENS, SELECTION\_FEATURE\_DIM),

&#x20;       )

&#x20;       self.\_validate\_mask(

&#x20;           "selection\_mask",

&#x20;           self.selection\_mask,

&#x20;           expected\_shape=(self.batch\_size, MAX\_SELECTION\_TOKENS),

&#x20;       )

&#x20;       self.\_validate\_float\_tensor(

&#x20;           "meta\_vec",

&#x20;           self.meta\_vec,

&#x20;           expected\_ndim=2,

&#x20;           expected\_tail=(None,),

&#x20;       )



&#x20;       expected\_batch = self.batch\_size

&#x20;       for name, tensor in (

&#x20;           ("entity\_features", self.entity\_features),

&#x20;           ("entity\_mask", self.entity\_mask),

&#x20;           ("selection\_features", self.selection\_features),

&#x20;           ("selection\_mask", self.selection\_mask),

&#x20;           ("meta\_vec", self.meta\_vec),

&#x20;       ):

&#x20;           if int(tensor.shape\[0]) != expected\_batch:

&#x20;               raise ValueError(

&#x20;                   f"{name} batch dimension must match spatial\_obs: "

&#x20;                   f"{tensor.shape\[0]} != {expected\_batch}",

&#x20;               )



&#x20;       if self.state\_in is not None:

&#x20;           if not isinstance(self.state\_in, tuple) or len(self.state\_in) != 2:

&#x20;               raise TypeError("state\_in must be a (syn, mem) tensor tuple or None")

&#x20;           syn, mem = self.state\_in

&#x20;           if not isinstance(syn, torch.Tensor) or not isinstance(mem, torch.Tensor):

&#x20;               raise TypeError("state\_in must contain tensors")

&#x20;           if syn.shape != mem.shape:

&#x20;               raise ValueError(

&#x20;                   f"state\_in tensors must share a shape, got {syn.shape} and {mem.shape}",

&#x20;               )

&#x20;           if syn.ndim < 1 or int(syn.shape\[0]) != expected\_batch:

&#x20;               raise ValueError(

&#x20;                   f"state\_in batch dimension must match spatial\_obs: "

&#x20;                   f"{syn.shape\[0] if syn.ndim else 'scalar'} != {expected\_batch}",

&#x20;               )



&#x20;   @staticmethod

&#x20;   def \_validate\_float\_tensor(

&#x20;       name: str,

&#x20;       tensor: torch.Tensor,

&#x20;       expected\_ndim: int,

&#x20;       expected\_tail: tuple\[int | None, ...],

&#x20;   ) -> None:

&#x20;       if not isinstance(tensor, torch.Tensor):

&#x20;           raise TypeError(f"{name} must be a torch.Tensor")

&#x20;       if tensor.ndim != expected\_ndim:

&#x20;           raise ValueError(

&#x20;               f"{name} must have ndim={expected\_ndim}, got shape={tuple(tensor.shape)}",

&#x20;           )

&#x20;       if not tensor.dtype.is\_floating\_point:

&#x20;           raise TypeError(f"{name} must use a floating dtype, got {tensor.dtype}")



&#x20;       tail = tensor.shape\[1:]

&#x20;       if len(tail) != len(expected\_tail):

&#x20;           raise ValueError(f"{name} has unexpected tail shape {tuple(tail)}")

&#x20;       for actual, expected in zip(tail, expected\_tail):

&#x20;           if expected is not None and int(actual) != int(expected):

&#x20;               raise ValueError(

&#x20;                   f"{name} has unexpected shape {tuple(tensor.shape)}; "

&#x20;                   f"expected tail {expected\_tail}",

&#x20;               )



&#x20;   @staticmethod

&#x20;   def \_validate\_mask(

&#x20;       name: str,

&#x20;       tensor: torch.Tensor,

&#x20;       expected\_shape: tuple\[int, int],

&#x20;   ) -> None:

&#x20;       if not isinstance(tensor, torch.Tensor):

&#x20;           raise TypeError(f"{name} must be a torch.Tensor")

&#x20;       if tensor.dtype != torch.bool:

&#x20;           raise TypeError(f"{name} must use torch.bool, got {tensor.dtype}")

&#x20;       if tuple(tensor.shape) != expected\_shape:

&#x20;           raise ValueError(

&#x20;               f"{name} must have shape {expected\_shape}, got {tuple(tensor.shape)}",

&#x20;           )



Today

nice, this is the Rosetta stone. All the shapes your network expects live here, and this is exactly where we plug the last-action token without breaking the rest.



I made you a drop-in version with the 4 extra dims wired in:



📎 policy\_input\_v2.py



What changed and why

AGENT\_LAST\_ACTION\_DIM = 4: that's your \[type, x\_norm, y\_norm, extra] from the ActionSpace

META\_VECTOR\_DIM: was 11 + 16 + 1 = 28, now 11 + 16 + 1 + 4 = 32. The obs\_space\_v2 I gave you already appends those 4 floats at the end of meta\_vec, so the shapes line up

Everything else stays identical, so PolicyInputBatch validation still passes, with\_state(), stack(), to() all work

This matches the change we made in obs\_space\_2\_v2 where \_extract\_meta\_vector now does:



Python

full = np.concatenate((player\_vec, available\_mask, \[last\_action\_index], agent\_last))

About trimming, since you mentioned it

You are right, for defeat roaches most of this is static. You are already new to masking, so here's the quick map:



entity\_mask / selection\_mask: these bool tensors tell the network "this row is a real unit, this row is padding". You pad to MAX\_ENTITY\_TOKENS = 24 and MAX\_SELECTION\_TOKENS = 20, then mask out the empties. That's masking.

For roaches you never need 24 entities. You have at most 4 marines + ∼8 roaches. Drop MAX\_ENTITY\_TOKENS to 12 and you cut compute in half.

CURATED\_FEATURE\_UNIT\_FIELDS has 21 fields. For roaches you only need:

"unit\_type", "alliance", "health", "health\_ratio", "x", "y", "is\_selected"

That's 7 instead of 21. Drop the rest and your entity\_features go from \[B]\[24]\[21] to \[B]\[12]\[7]

SELECTION\_FEATURE\_NAMES: you only ever use select\_army, so MAX\_SELECTION\_TOKENS can be 1, and you can keep just "unit\_type", "health"

Don't do all the trimming now though. First get the token flowing end to end with the minimal change above, verify the network trains, then we shrink the fields. Changing field counts also means editing the normalizers in obs\_space, so it's a second pass.



Next in the backtrace

With policy\_input\_v2 in place:



swap your imports to use the v2 files (obs\_space\_2\_v2 and policy\_input\_v2)

in DefeatRoaches.step, pass last\_action\_tokengood, this is the contract that ties everything together. Your PolicyInputBatch` is already set up for masking and variable-length entities, you just need to make room for the 4-float token we added in obs space.

I bumped the constants for you so the shapes line up:



📎 policy\_input\_v2.py



What this file actually controls

SPATIAL\_OBS\_SHAPE (27,84): the 27 feature layers from PySC2, unchanged

MAX\_ENTITY\_TOKENS = 24, MAX\_SELECTION\_TOKENS = 20: these are the padding limits. The masks you were unsure about, entity\_mask and selection\_mask, are just boolean vectors that say "first N rows are real units, the rest are padding". The network uses them to ignore the zeros. For defeat roaches you could drop these to 12 and 2 later, but keep them for now so nothing breaks

CURATED\_FEATURE\_UNIT\_FIELDS (21 fields): this is where you can trim hard for roaches. Right now you carry shield, energy, build\_progress, hallucination, etc. None of those exist in the roaches minigame. You could cut this to 7 fields: unit\_type, alliance, health, health\_ratio, x, y, is\_selected. That would shrink your entity tensor from to and speed up attention a lot. Do it after the token works, not now

META\_VECTOR\_DIM: was 28 (11 player + 16 available actions + 1 last\_action\_index). I added AGENT\_LAST\_ACTION\_DIM = 4 so now it's 32. That's the only required change to make obs\_space\_2\_v2 compatible\[B]\[24]\[21]\[7]

The change that matters

Python

AGENT\_LAST\_ACTION\_DIM: Final\[int] = 4

META\_VECTOR\_DIM: Final\[int] = (

&#x20;   META\_PLAYER\_FEATURE\_DIM + META\_AVAILABLE\_ACTION\_DIM +

&#x20;   META\_LAST\_ACTION\_INDEX\_DIM + AGENT\_LAST\_ACTION\_DIM

)

Your \_extract\_meta\_vector in obs\_space now appends \[type, x\_norm, y\_norm, extra] at the end of the meta vector. PolicyInputBatch doesn't enforce the exact size, it just checks that it's 2D float, so this passes validation.



What to trim later, when you want speed

You mentioned "too much data will be static". For roaches specifically:



drop MAX\_ENTITY\_TOKENS to 12, you never have more than 10 marines + 10 roaches

drop MAX\_SELECTION\_TOKENS to 1, you only use select\_army

cut CURATED\_FEATURE\_UNIT\_FIELDS to the 7 I listed above, and update the normalizer field list in obs\_space to match

same for SELECTION\_FEATURE\_NAMES, keep maybe 3 fields

Do that as step 2. Step 1 is just get the token flowing.



Next in the backtrace

With policy\_input\_v2 in place, your extract\_observation now outputs meta\_vec size 32. The next file that will complain is policy\_network.py, because its first linear layer for the meta vector was built for 28 inputs.



Drop PPO\_CNN/policy\_network.py here and we'll bump that input dim and check where it concatenates the meta vector with the token embeddings. That's the last shape change before the token actually reaches the network.



before that i ahve a question you can mask sin some sort what enurons actiavation path are aviable during trianign so it forces to distribute representation akin to moe, or rather if i correct myself it can work even inside expert ehad s in moe, but my arhitecture is like this, can we implement that and where is ti implement as wrapper logic or as training logic? import torch

import torch.nn as nn

import torch.nn.functional as F

import snntorch as snn

from snntorch import surrogate



from PPO\_CNN.policy\_input import (

&#x20;   CURATED\_FEATURE\_UNIT\_FIELDS,

&#x20;   MAX\_ENTITY\_TOKENS,

&#x20;   MAX\_SELECTION\_TOKENS,

&#x20;   META\_AVAILABLE\_ACTION\_DIM,

&#x20;   META\_LAST\_ACTION\_INDEX\_DIM,

&#x20;   META\_PLAYER\_FEATURE\_DIM,

&#x20;   PolicyInputBatch,

&#x20;   SELECTION\_FEATURE\_DIM,

&#x20;   SELECTION\_UNIT\_TYPE\_INDEX,

&#x20;   TOKEN\_TYPE\_GROUPS,

&#x20;   UNIT\_TYPE\_FEATURE\_INDEX,

)





class EntityEncoder(nn.Module):

&#x20;   def \_\_init\_\_(

&#x20;       self,

&#x20;       feature\_dim: int,

&#x20;       embed\_dim: int,

&#x20;       unit\_type\_vocab\_size: int = 4096,

&#x20;       unit\_type\_embed\_dim: int | None = None,

&#x20;   ):

&#x20;       super().\_\_init\_\_()

&#x20;       if feature\_dim <= UNIT\_TYPE\_FEATURE\_INDEX:

&#x20;           raise ValueError(

&#x20;               f"feature\_dim must include unit\_type, got {feature\_dim}",

&#x20;           )



&#x20;       self.feature\_dim = int(feature\_dim)

&#x20;       self.embed\_dim = int(embed\_dim)

&#x20;       self.unit\_type\_index = int(UNIT\_TYPE\_FEATURE\_INDEX)

&#x20;       self.unit\_type\_embed\_dim = int(

&#x20;           unit\_type\_embed\_dim or max(8, self.embed\_dim // 4),

&#x20;       )

&#x20;       continuous\_dim = self.feature\_dim - 1 + self.unit\_type\_embed\_dim



&#x20;       self.unit\_type\_embedding = nn.Embedding(

&#x20;           int(unit\_type\_vocab\_size),

&#x20;           self.unit\_type\_embed\_dim,

&#x20;           padding\_idx=0,

&#x20;       )

&#x20;       self.pre\_norm = nn.LayerNorm(continuous\_dim)

&#x20;       self.mlp = nn.Sequential(

&#x20;           nn.Linear(continuous\_dim, self.embed\_dim),

&#x20;           nn.ReLU(),

&#x20;           nn.Linear(self.embed\_dim, self.embed\_dim),

&#x20;       )



&#x20;   def forward(

&#x20;       self,

&#x20;       entity\_features: torch.Tensor,

&#x20;       entity\_mask: torch.Tensor,

&#x20;   ) -> torch.Tensor:

&#x20;       unit\_type\_ids = entity\_features\[..., self.unit\_type\_index].round().long()

&#x20;       unit\_type\_ids = unit\_type\_ids.clamp(

&#x20;           0, self.unit\_type\_embedding.num\_embeddings - 1,

&#x20;       )

&#x20;       unit\_type\_emb = self.unit\_type\_embedding(unit\_type\_ids)

&#x20;       continuous = torch.cat(

&#x20;           (

&#x20;               entity\_features\[..., : self.unit\_type\_index],

&#x20;               entity\_features\[..., self.unit\_type\_index + 1 :],

&#x20;           ),

&#x20;           dim=-1,

&#x20;       )

&#x20;       encoded = torch.cat((continuous, unit\_type\_emb), dim=-1)

&#x20;       encoded = self.mlp(self.pre\_norm(encoded))



&#x20;       mask = entity\_mask.unsqueeze(-1).to(dtype=encoded.dtype)

&#x20;       return encoded \* mask





class SelectionEncoder(nn.Module):

&#x20;   def \_\_init\_\_(

&#x20;       self,

&#x20;       feature\_dim: int,

&#x20;       embed\_dim: int,

&#x20;       unit\_type\_vocab\_size: int = 4096,

&#x20;       unit\_type\_embed\_dim: int | None = None,

&#x20;   ):

&#x20;       super().\_\_init\_\_()

&#x20;       if feature\_dim <= SELECTION\_UNIT\_TYPE\_INDEX:

&#x20;           raise ValueError(

&#x20;               f"feature\_dim must include unit\_type, got {feature\_dim}",

&#x20;           )



&#x20;       self.feature\_dim = int(feature\_dim)

&#x20;       self.embed\_dim = int(embed\_dim)

&#x20;       self.unit\_type\_index = int(SELECTION\_UNIT\_TYPE\_INDEX)

&#x20;       self.unit\_type\_embed\_dim = int(

&#x20;           unit\_type\_embed\_dim or max(8, self.embed\_dim // 4),

&#x20;       )

&#x20;       continuous\_dim = self.feature\_dim - 1 + self.unit\_type\_embed\_dim



&#x20;       self.unit\_type\_embedding = nn.Embedding(

&#x20;           int(unit\_type\_vocab\_size),

&#x20;           self.unit\_type\_embed\_dim,

&#x20;           padding\_idx=0,

&#x20;       )

&#x20;       self.pre\_norm = nn.LayerNorm(continuous\_dim)

&#x20;       self.mlp = nn.Sequential(

&#x20;           nn.Linear(continuous\_dim, self.embed\_dim),

&#x20;           nn.ReLU(),

&#x20;           nn.Linear(self.embed\_dim, self.embed\_dim),

&#x20;       )



&#x20;   def forward(

&#x20;       self,

&#x20;       selection\_features: torch.Tensor,

&#x20;       selection\_mask: torch.Tensor,

&#x20;   ) -> torch.Tensor:

&#x20;       unit\_type\_ids = selection\_features\[..., self.unit\_type\_index].round().long()

&#x20;       unit\_type\_ids = unit\_type\_ids.clamp(

&#x20;           0, self.unit\_type\_embedding.num\_embeddings - 1,

&#x20;       )

&#x20;       unit\_type\_emb = self.unit\_type\_embedding(unit\_type\_ids)

&#x20;       continuous = torch.cat(

&#x20;           (

&#x20;               selection\_features\[..., : self.unit\_type\_index],

&#x20;               selection\_features\[..., self.unit\_type\_index + 1 :],

&#x20;           ),

&#x20;           dim=-1,

&#x20;       )

&#x20;       encoded = torch.cat((continuous, unit\_type\_emb), dim=-1)

&#x20;       encoded = self.mlp(self.pre\_norm(encoded))



&#x20;       mask = selection\_mask.unsqueeze(-1).to(dtype=encoded.dtype)

&#x20;       return encoded \* mask





class MetaEncoder(nn.Module):

&#x20;   def \_\_init\_\_(

&#x20;       self,

&#x20;       meta\_input\_dim: int,

&#x20;       embed\_dim: int,

&#x20;       player\_dim: int = META\_PLAYER\_FEATURE\_DIM,

&#x20;       available\_action\_dim: int = META\_AVAILABLE\_ACTION\_DIM,

&#x20;       last\_action\_embed\_dim: int | None = None,

&#x20;       last\_action\_vocab\_size: int = META\_AVAILABLE\_ACTION\_DIM + 2,

&#x20;   ):

&#x20;       super().\_\_init\_\_()

&#x20;       if meta\_input\_dim <= 0:

&#x20;           raise ValueError("meta\_input\_dim must be positive")



&#x20;       self.meta\_input\_dim = int(meta\_input\_dim)

&#x20;       self.embed\_dim = int(embed\_dim)

&#x20;       self.player\_dim = int(player\_dim)

&#x20;       self.available\_action\_dim = int(available\_action\_dim)

&#x20;       self.last\_action\_offset = self.player\_dim + self.available\_action\_dim

&#x20;       self.use\_structured\_meta = (

&#x20;           self.meta\_input\_dim

&#x20;           >= self.player\_dim + self.available\_action\_dim + META\_LAST\_ACTION\_INDEX\_DIM

&#x20;       )

&#x20;       self.last\_action\_embed\_dim = int(

&#x20;           last\_action\_embed\_dim or max(8, self.embed\_dim // 4),

&#x20;       )

&#x20;       fused\_input\_dim = (

&#x20;           self.player\_dim

&#x20;           + self.available\_action\_dim

&#x20;           + self.last\_action\_embed\_dim

&#x20;           if self.use\_structured\_meta

&#x20;           else self.meta\_input\_dim

&#x20;       )



&#x20;       self.last\_action\_embedding = nn.Embedding(

&#x20;           int(last\_action\_vocab\_size),

&#x20;           self.last\_action\_embed\_dim,

&#x20;       )

&#x20;       self.pre\_norm = nn.LayerNorm(fused\_input\_dim)

&#x20;       self.mlp = nn.Sequential(

&#x20;           nn.Linear(fused\_input\_dim, self.embed\_dim),

&#x20;           nn.ReLU(),

&#x20;           nn.Linear(self.embed\_dim, self.embed\_dim),

&#x20;       )



&#x20;   def forward(self, meta\_vec: torch.Tensor) -> torch.Tensor:

&#x20;       if self.use\_structured\_meta:

&#x20;           player = meta\_vec\[..., : self.player\_dim]

&#x20;           available\_actions = meta\_vec\[

&#x20;               ...,

&#x20;               self.player\_dim : self.player\_dim + self.available\_action\_dim

&#x20;           ]

&#x20;           last\_action\_ids = meta\_vec\[..., self.last\_action\_offset].round().long()

&#x20;           last\_action\_ids = last\_action\_ids.clamp(

&#x20;               0, self.last\_action\_embedding.num\_embeddings - 1,

&#x20;           )

&#x20;           last\_action\_emb = self.last\_action\_embedding(last\_action\_ids)

&#x20;           fused = torch.cat((player, available\_actions, last\_action\_emb), dim=-1)

&#x20;       else:

&#x20;           fused = meta\_vec



&#x20;       return self.mlp(self.pre\_norm(fused)).unsqueeze(1)





class SpikingSelfAttention(nn.Module):

&#x20;   """

&#x20;   Spikformer-style spike-driven self-attention.



&#x20;   Computes spike(Q) @ spike(K)^T \* scale @ spike(V), with Q/K/V

&#x20;   encoded as binary spikes via LIF neurons.



&#x20;   Stateless per forward() call. The Q/K/V LIF membranes are

&#x20;   re-initialized every call, so this block does not carry state

&#x20;   across env steps. Cross-step temporal memory lives in

&#x20;   TokenTemporalSNN, which sits after this block.

&#x20;   """



&#x20;   def \_\_init\_\_(self, embed\_dim=64, beta\_qkv=0.5, spike\_grad=None):

&#x20;       super().\_\_init\_\_()

&#x20;       self.embed\_dim = embed\_dim

&#x20;       self.scale = embed\_dim \*\* -0.5



&#x20;       self.pre\_norm = nn.LayerNorm(embed\_dim)

&#x20;       self.W\_q = nn.Linear(embed\_dim, embed\_dim, bias=False)

&#x20;       self.W\_k = nn.Linear(embed\_dim, embed\_dim, bias=False)

&#x20;       self.W\_v = nn.Linear(embed\_dim, embed\_dim, bias=False)



&#x20;       self.lif\_q = snn.Leaky(

&#x20;           beta=beta\_qkv, spike\_grad=spike\_grad, learn\_beta=True,

&#x20;       )

&#x20;       self.lif\_k = snn.Leaky(

&#x20;           beta=beta\_qkv, spike\_grad=spike\_grad, learn\_beta=True,

&#x20;       )

&#x20;       self.lif\_v = snn.Leaky(

&#x20;           beta=beta\_qkv, spike\_grad=spike\_grad, learn\_beta=True,

&#x20;       )



&#x20;   def forward(self, tokens, token\_mask: torch.Tensor | None = None):

&#x20;       tokens\_normed = self.pre\_norm(tokens)

&#x20;       residual = tokens

&#x20;       query\_mask = None

&#x20;       if token\_mask is not None:

&#x20;           query\_mask = token\_mask.unsqueeze(-1).to(dtype=tokens.dtype)

&#x20;           tokens\_normed = tokens\_normed \* query\_mask

&#x20;           residual = residual \* query\_mask



&#x20;       q\_raw = self.W\_q(tokens\_normed)

&#x20;       k\_raw = self.W\_k(tokens\_normed)

&#x20;       v\_raw = self.W\_v(tokens\_normed)



&#x20;       mem\_q = self.lif\_q.init\_leaky()

&#x20;       mem\_k = self.lif\_k.init\_leaky()

&#x20;       mem\_v = self.lif\_v.init\_leaky()



&#x20;       spike\_q, \_ = self.lif\_q(q\_raw, mem\_q)

&#x20;       spike\_k, \_ = self.lif\_k(k\_raw, mem\_k)

&#x20;       spike\_v, \_ = self.lif\_v(v\_raw, mem\_v)



&#x20;       if query\_mask is not None:

&#x20;           spike\_q = spike\_q \* query\_mask

&#x20;           spike\_k = spike\_k \* query\_mask

&#x20;           spike\_v = spike\_v \* query\_mask



&#x20;       if token\_mask is not None:

&#x20;           attn\_mask = token\_mask\[:, None, None, :]

&#x20;       else:

&#x20;           attn\_mask = None

&#x20;       out = F.scaled\_dot\_product\_attention(

&#x20;           spike\_q.unsqueeze(1),

&#x20;           spike\_k.unsqueeze(1),

&#x20;           spike\_v.unsqueeze(1),

&#x20;           attn\_mask=attn\_mask,

&#x20;           dropout\_p=0.0,

&#x20;           scale=self.scale,

&#x20;       ).squeeze(1)

&#x20;       if query\_mask is not None:

&#x20;           out = out \* query\_mask

&#x20;       return out + residual





class TokenTemporalSNN(nn.Module):

&#x20;   """

&#x20;   Cross-env-step temporal integrator over the token sequence.



&#x20;   State: (syn, mem), each \[B, N, D].

&#x20;   """



&#x20;   def \_\_init\_\_(self, alpha=0.8, beta=0.9, spike\_grad=None):

&#x20;       super().\_\_init\_\_()

&#x20;       self.snn = snn.Synaptic(

&#x20;           alpha=alpha,

&#x20;           beta=beta,

&#x20;           spike\_grad=spike\_grad,

&#x20;           learn\_alpha=True,

&#x20;           learn\_beta=True,

&#x20;       )



&#x20;   @staticmethod

&#x20;   def init\_state(batch\_size, num\_tokens, embed\_dim, device, dtype):

&#x20;       syn = torch.zeros(

&#x20;           batch\_size, num\_tokens, embed\_dim, device=device, dtype=dtype,

&#x20;       )

&#x20;       mem = torch.zeros(

&#x20;           batch\_size, num\_tokens, embed\_dim, device=device, dtype=dtype,

&#x20;       )

&#x20;       return (syn, mem)



&#x20;   def forward(self, tokens, syn, mem):

&#x20;       spk, syn, mem = self.snn(tokens, syn, mem)

&#x20;       return spk, syn, mem





class PolicyNetwork(nn.Module):

&#x20;   """

&#x20;   Spiking Transformer policy with token-level temporal state.



&#x20;   Pipeline:

&#x20;     raw obs

&#x20;       -> conv backbone

&#x20;       -> pooled spatial tokens

&#x20;       -> spiking self-attention

&#x20;       -> token temporal SNN

&#x20;       -> shared readout

&#x20;       -> action/value heads

&#x20;   """



&#x20;   def \_\_init\_\_(

&#x20;       self,

&#x20;       spatial\_input\_shape,

&#x20;       vector\_input\_dim,

&#x20;       action\_dim,

&#x20;       num\_steps=1,

&#x20;       screen\_size=84,

&#x20;       token\_snn\_alpha=0.8,

&#x20;       token\_snn\_beta=0.9,

&#x20;       attention\_embed\_dim=64,

&#x20;       attention\_pool\_size=7,

&#x20;       attention\_beta=0.5,

&#x20;   ):

&#x20;       super().\_\_init\_\_()



&#x20;       self.device = torch.device("cuda" if torch.cuda.is\_available() else "cpu")

&#x20;       self.num\_steps = max(1, int(num\_steps))

&#x20;       self.screen\_size = int(screen\_size)

&#x20;       self.\_pool\_size = max(1, int(attention\_pool\_size))

&#x20;       self.\_embed\_dim = max(1, int(attention\_embed\_dim))

&#x20;       self.\_spatial\_tokens = self.\_pool\_size \* self.\_pool\_size

&#x20;       self.\_entity\_start = self.\_spatial\_tokens

&#x20;       self.\_entity\_end = self.\_entity\_start + MAX\_ENTITY\_TOKENS

&#x20;       self.\_selection\_start = self.\_entity\_end

&#x20;       self.\_selection\_end = self.\_selection\_start + MAX\_SELECTION\_TOKENS

&#x20;       self.\_meta\_start = self.\_selection\_end

&#x20;       self.\_num\_tokens = (

&#x20;           self.\_spatial\_tokens + MAX\_ENTITY\_TOKENS + MAX\_SELECTION\_TOKENS + 1

&#x20;       )

&#x20;       self.\_carry\_entity\_state = False

&#x20;       self.\_carry\_selection\_state = False

&#x20;       self.\_meta\_input\_dim = int(vector\_input\_dim)

&#x20;       self.\_config = {

&#x20;           "num\_steps": self.num\_steps,

&#x20;           "screen\_size": self.screen\_size,

&#x20;           "token\_snn\_alpha": float(token\_snn\_alpha),

&#x20;           "token\_snn\_beta": float(token\_snn\_beta),

&#x20;           "attention\_embed\_dim": self.\_embed\_dim,

&#x20;           "attention\_pool\_size": self.\_pool\_size,

&#x20;           "attention\_beta": float(attention\_beta),

&#x20;           "meta\_input\_dim": self.\_meta\_input\_dim,

&#x20;           "carry\_entity\_state": bool(self.\_carry\_entity\_state),

&#x20;           "carry\_selection\_state": bool(self.\_carry\_selection\_state),

&#x20;       }

&#x20;       spike\_grad = surrogate.fast\_sigmoid()



&#x20;       c\_in = spatial\_input\_shape\[0]

&#x20;       self.conv1 = nn.Conv2d(c\_in, 16, kernel\_size=3, stride=1, padding=1)

&#x20;       self.conv2 = nn.Conv2d(16, 32, kernel\_size=3, stride=1, padding=1)

&#x20;       self.pool = nn.MaxPool2d(2, 2)

&#x20;       self.conv3 = nn.Conv2d(

&#x20;           32, self.\_embed\_dim, kernel\_size=3, stride=1, padding=1,

&#x20;       )



&#x20;       self.token\_pool = nn.AdaptiveAvgPool2d(

&#x20;           (self.\_pool\_size, self.\_pool\_size),

&#x20;       )

&#x20;       self.entity\_encoder = EntityEncoder(

&#x20;           feature\_dim=len(CURATED\_FEATURE\_UNIT\_FIELDS),

&#x20;           embed\_dim=self.\_embed\_dim,

&#x20;       )

&#x20;       self.selection\_encoder = SelectionEncoder(

&#x20;           feature\_dim=SELECTION\_FEATURE\_DIM,

&#x20;           embed\_dim=self.\_embed\_dim,

&#x20;       )

&#x20;       self.meta\_encoder = MetaEncoder(

&#x20;           meta\_input\_dim=self.\_meta\_input\_dim,

&#x20;           embed\_dim=self.\_embed\_dim,

&#x20;       )

&#x20;       self.token\_type\_embedding = nn.Embedding(

&#x20;           TOKEN\_TYPE\_GROUPS,

&#x20;           self.\_embed\_dim,

&#x20;       )

&#x20;       self.attention = SpikingSelfAttention(

&#x20;           embed\_dim=self.\_embed\_dim,

&#x20;           beta\_qkv=attention\_beta,

&#x20;           spike\_grad=spike\_grad,

&#x20;       )

&#x20;       self.token\_snn = TokenTemporalSNN(

&#x20;           alpha=token\_snn\_alpha,

&#x20;           beta=token\_snn\_beta,

&#x20;           spike\_grad=spike\_grad,

&#x20;       )



&#x20;       pooled\_dim = TOKEN\_TYPE\_GROUPS \* self.\_embed\_dim

&#x20;       self.combined\_norm = nn.LayerNorm(pooled\_dim)

&#x20;       self.shared\_fc1 = nn.Linear(pooled\_dim, 128)

&#x20;       self.shared\_fc2 = nn.Linear(128, 64)



&#x20;       self.actor\_fc = nn.Linear(64, action\_dim)

&#x20;       self.critic\_fc = nn.Linear(64, 1)

&#x20;       self.move\_x\_fc = nn.Linear(64, self.screen\_size)

&#x20;       self.move\_y\_fc = nn.Linear(64, self.screen\_size)



&#x20;       self.use\_amp = torch.cuda.is\_available()

&#x20;       self.amp\_dtype = torch.float16 if self.use\_amp else torch.float32

&#x20;       self.scaler = torch.amp.GradScaler("cuda", enabled=self.use\_amp)



&#x20;       self.to(self.device)



&#x20;   def resolved\_config(self):

&#x20;       return {

&#x20;           \*\*self.\_config,

&#x20;           "num\_tokens": self.\_num\_tokens,

&#x20;           "spatial\_tokens": self.\_spatial\_tokens,

&#x20;           "amp\_enabled": bool(self.use\_amp),

&#x20;           "amp\_dtype": str(self.amp\_dtype),

&#x20;           "device": str(self.device),

&#x20;       }



&#x20;   def \_zero\_entity\_state(

&#x20;       self,

&#x20;       syn\_tok: torch.Tensor,

&#x20;       mem\_tok: torch.Tensor,

&#x20;   ) -> tuple\[torch.Tensor, torch.Tensor]:

&#x20;       # Zero recurrent state for token groups whose slot-to-identity

&#x20;       # mapping is unstable across env steps (entity + selection).

&#x20;       if self.\_carry\_entity\_state and self.\_carry\_selection\_state:

&#x20;           return syn\_tok, mem\_tok

&#x20;       syn\_tok = syn\_tok.clone()

&#x20;       mem\_tok = mem\_tok.clone()

&#x20;       if not self.\_carry\_entity\_state:

&#x20;           syn\_tok\[:, self.\_entity\_start : self.\_entity\_end, :] = 0.0

&#x20;           mem\_tok\[:, self.\_entity\_start : self.\_entity\_end, :] = 0.0

&#x20;       if not self.\_carry\_selection\_state:

&#x20;           syn\_tok\[:, self.\_selection\_start : self.\_selection\_end, :] = 0.0

&#x20;           mem\_tok\[:, self.\_selection\_start : self.\_selection\_end, :] = 0.0

&#x20;       return syn\_tok, mem\_tok



&#x20;   def reset\_state\_rows(

&#x20;       self,

&#x20;       state: tuple\[torch.Tensor, torch.Tensor] | None,

&#x20;       reset\_mask: torch.Tensor | None,

&#x20;   ) -> tuple\[torch.Tensor, torch.Tensor] | None:

&#x20;       if state is None or reset\_mask is None:

&#x20;           return state

&#x20;       if reset\_mask.ndim != 1:

&#x20;           raise ValueError(

&#x20;               f"reset\_mask must be 1D over batch rows, got {tuple(reset\_mask.shape)}",

&#x20;           )

&#x20;       if state\[0].size(0) != int(reset\_mask.numel()):

&#x20;           raise ValueError(

&#x20;               "reset\_mask batch dimension must match recurrent state rows",

&#x20;           )

&#x20;       if not bool(reset\_mask.any().item()):

&#x20;           return state



&#x20;       keep\_mask = (\~reset\_mask).to(

&#x20;           device=state\[0].device,

&#x20;           dtype=state\[0].dtype,

&#x20;       ).view(-1, 1, 1)

&#x20;       return state\[0] \* keep\_mask, state\[1] \* keep\_mask



&#x20;   def init\_concrete\_state(self, batch\_size=1, device=None, dtype=None):

&#x20;       if device is None:

&#x20;           device = self.device

&#x20;       if dtype is None:

&#x20;           dtype = next(self.parameters()).dtype

&#x20;       return TokenTemporalSNN.init\_state(

&#x20;           batch\_size, self.\_num\_tokens, self.\_embed\_dim, device, dtype,

&#x20;       )



&#x20;   def \_add\_token\_type(

&#x20;       self,

&#x20;       tokens: torch.Tensor,

&#x20;       token\_type\_index: int,

&#x20;       token\_mask: torch.Tensor | None = None,

&#x20;   ) -> torch.Tensor:

&#x20;       type\_emb = self.token\_type\_embedding.weight\[token\_type\_index].view(1, 1, -1)

&#x20;       typed = tokens + type\_emb.to(dtype=tokens.dtype, device=tokens.device)

&#x20;       if token\_mask is not None:

&#x20;           typed = typed \* token\_mask.unsqueeze(-1).to(dtype=tokens.dtype)

&#x20;       return typed



&#x20;   def \_group\_masked\_mean(

&#x20;       self,

&#x20;       tokens: torch.Tensor,

&#x20;       mask: torch.Tensor,

&#x20;   ) -> torch.Tensor:

&#x20;       # Masked-mean-pool per semantic group then concat. Permutation-

&#x20;       # invariant within entity/selection groups whose slot-to-identity

&#x20;       # mapping is unstable. Token-type embedding already injected

&#x20;       # group identity upstream; concat preserves it here.

&#x20;       group\_slices = (

&#x20;           (0, self.\_spatial\_tokens),

&#x20;           (self.\_entity\_start, self.\_entity\_end),

&#x20;           (self.\_selection\_start, self.\_selection\_end),

&#x20;           (self.\_meta\_start, self.\_meta\_start + 1),

&#x20;       )

&#x20;       summaries = \[]

&#x20;       for start, end in group\_slices:

&#x20;           token\_slice = tokens\[:, start:end, :]

&#x20;           mask\_slice = mask\[:, start:end].unsqueeze(-1).to(dtype=token\_slice.dtype)

&#x20;           summed = (token\_slice \* mask\_slice).sum(dim=1)

&#x20;           count = mask\_slice.sum(dim=1).clamp\_min(1.0)

&#x20;           summaries.append(summed / count)

&#x20;       return torch.cat(summaries, dim=-1)



&#x20;   def forward\_step\_tensors(

&#x20;       self,

&#x20;       spatial\_obs: torch.Tensor,

&#x20;       entity\_features: torch.Tensor,

&#x20;       entity\_mask: torch.Tensor,

&#x20;       selection\_features: torch.Tensor,

&#x20;       selection\_mask: torch.Tensor,

&#x20;       meta\_vec: torch.Tensor,

&#x20;       state\_in: tuple\[torch.Tensor, torch.Tensor] | None = None,

&#x20;   ):

&#x20;       spatial\_input = spatial\_obs

&#x20;       token\_state = state\_in

&#x20;       if token\_state is None:

&#x20;           syn\_tok, mem\_tok = self.init\_concrete\_state(

&#x20;               batch\_size=spatial\_input.size(0),

&#x20;               device=spatial\_input.device,

&#x20;               dtype=spatial\_input.dtype,

&#x20;           )

&#x20;       else:

&#x20;           syn\_tok, mem\_tok = token\_state

&#x20;       syn\_tok, mem\_tok = self.\_zero\_entity\_state(syn\_tok, mem\_tok)



&#x20;       x = F.relu(self.conv1(spatial\_input))

&#x20;       x = F.relu(self.conv2(x))

&#x20;       x = self.pool(x)

&#x20;       x = F.relu(self.conv3(x))

&#x20;       x = self.pool(x)



&#x20;       spatial\_tokens = self.token\_pool(x)

&#x20;       spatial\_tokens = spatial\_tokens.flatten(2).transpose(1, 2)

&#x20;       batch\_size = spatial\_tokens.size(0)

&#x20;       device = spatial\_tokens.device

&#x20;       spatial\_mask = torch.ones(

&#x20;           batch\_size,

&#x20;           self.\_spatial\_tokens,

&#x20;           dtype=torch.bool,

&#x20;           device=device,

&#x20;       )

&#x20;       entity\_tokens = self.entity\_encoder(

&#x20;           entity\_features,

&#x20;           entity\_mask,

&#x20;       )

&#x20;       selection\_tokens = self.selection\_encoder(

&#x20;           selection\_features,

&#x20;           selection\_mask,

&#x20;       )

&#x20;       meta\_tokens = self.meta\_encoder(meta\_vec)

&#x20;       meta\_mask = torch.ones(batch\_size, 1, dtype=torch.bool, device=device)



&#x20;       tokens = torch.cat(

&#x20;           (

&#x20;               self.\_add\_token\_type(spatial\_tokens, 0, spatial\_mask),

&#x20;               self.\_add\_token\_type(entity\_tokens, 1, entity\_mask),

&#x20;               self.\_add\_token\_type(selection\_tokens, 2, selection\_mask),

&#x20;               self.\_add\_token\_type(meta\_tokens, 3, meta\_mask),

&#x20;           ),

&#x20;           dim=1,

&#x20;       )

&#x20;       token\_mask = torch.cat(

&#x20;           (

&#x20;               spatial\_mask,

&#x20;               entity\_mask,

&#x20;               selection\_mask,

&#x20;               meta\_mask,

&#x20;           ),

&#x20;           dim=1,

&#x20;       )

&#x20;       attended = self.attention(tokens, token\_mask=token\_mask)



&#x20;       spike\_rec = \[]

&#x20;       token\_mask\_f = token\_mask.unsqueeze(-1).to(dtype=attended.dtype)

&#x20;       syn\_tok = syn\_tok \* token\_mask\_f

&#x20;       mem\_tok = mem\_tok \* token\_mask\_f

&#x20;       for \_ in range(self.num\_steps):

&#x20;           spk\_tok, syn\_tok, mem\_tok = self.token\_snn(attended, syn\_tok, mem\_tok)

&#x20;           spk\_tok = spk\_tok \* token\_mask\_f

&#x20;           syn\_tok = syn\_tok \* token\_mask\_f

&#x20;           mem\_tok = mem\_tok \* token\_mask\_f

&#x20;           spike\_rec.append(spk\_tok)

&#x20;       syn\_tok, mem\_tok = self.\_zero\_entity\_state(syn\_tok, mem\_tok)



&#x20;       aggregated = torch.stack(spike\_rec, dim=0).sum(dim=0)

&#x20;       pooled = self.\_group\_masked\_mean(aggregated, token\_mask)

&#x20;       combined = self.combined\_norm(pooled)



&#x20;       x = F.relu(self.shared\_fc1(combined))

&#x20;       x = F.relu(self.shared\_fc2(x))



&#x20;       action\_logits = self.actor\_fc(x)

&#x20;       move\_x\_logits = self.move\_x\_fc(x)

&#x20;       move\_y\_logits = self.move\_y\_fc(x)

&#x20;       state\_value = self.critic\_fc(x).squeeze(-1)

&#x20;       next\_state = (syn\_tok, mem\_tok)

&#x20;       return action\_logits, move\_x\_logits, move\_y\_logits, state\_value, next\_state



&#x20;   def forward(self, batch: PolicyInputBatch):

&#x20;       if not isinstance(batch, PolicyInputBatch):

&#x20;           raise TypeError(

&#x20;               f"PolicyNetwork.forward expects PolicyInputBatch, got {type(batch)!r}",

&#x20;           )

&#x20;       return self.forward\_step\_tensors(

&#x20;           spatial\_obs=batch.spatial\_obs,

&#x20;           entity\_features=batch.entity\_features,

&#x20;           entity\_mask=batch.entity\_mask,

&#x20;           selection\_features=batch.selection\_features,

&#x20;           selection\_mask=batch.selection\_mask,

&#x20;           meta\_vec=batch.meta\_vec,

&#x20;           state\_in=batch.state\_in,

&#x20;       )



Today

yes, you can mask activation paths to force distributed representations, and your architecture is actually well set up for it because you already split the input into semantic token groups.



What you are describing sits between classic dropout and a full MoE. You don't need a separate MoE library, you can do it as a thin wrapper around the MLPs you already have.



Where it fits in your code

Look at where you actually mix features:



EntityEncoder.mlp, SelectionEncoder.mlp, MetaEncoder.mlp: each is a 2-layer MLP after the LayerNorm. These are perfect expert slots because they process different modalities already

shared\_fc1 → shared\_fc2: this is the bottleneck after you do \_group\_masked\_mean. Masking here forces the policy and value heads to use different paths

SpikingSelfAttention: you could mask heads, but start with the MLPs, they are simpler and you already have token masks

Your \_group\_masked\_mean already does a form of routing by concatenating 4 group means. That's the natural place to insert expert choice.



Two ways to implement, wrapper vs training logic

Wrapper logic (forward pass masking)

This is the cleanest for your setup. You wrap a Linear layer so that during training only a subset of neurons fire.



Python

class MaskedLinear(nn.Module):

&#x20;   def \_\_init\_\_(self, in\_f, out\_f, num\_experts=4, top\_k=2):

&#x20;       super().\_\_init\_\_()

&#x20;       self.experts = nn.ModuleList(\[nn.Linear(in\_f, out\_f) for \_ in range(num\_experts)])

&#x20;       self.gate = nn.Linear(in\_f, num\_experts)

&#x20;       self.top\_k = top\_k



&#x20;   def forward(self, x):

&#x20;       # x: \[B, ..., in\_f]

&#x20;       gate\_logits = self.gate(x)  # \[..., num\_experts]

&#x20;       top\_vals, top\_idx = gate\_logits.topk(self.top\_k, dim=-1)

&#x20;       mask = torch.zeros\_like(gate\_logits).scatter\_(-1, top\_idx, 1.0)

&#x20;       gate = (F.softmax(top\_vals, dim=-1).unsqueeze(-1) \* mask.unsqueeze(-1)).sum(-2)



&#x20;       out = 0

&#x20;       for i, expert in enumerate(self.experts):

&#x20;           out = out + expert(x) \* gate\[..., i:i+1]

&#x20;       return out



13 lines hidden

Drop this in place of nn.Linear in EntityEncoder.mlp or shared\_fc1. During training the gate learns to route different token groups to different experts, during eval you can use top-1 for speed.



This is pure wrapper logic, no change to PPO update.



Training logic (auxiliary loss)

If you want true MoE load balancing, add a small loss in your PPO update:



Python

\# after you get gate\_probs from MaskedLinear

balance\_loss = (gate\_probs.mean(0) \* gate\_probs.mean(0)).sum() \* num\_experts

total\_loss = ppo\_loss + 0.01 \* balance\_loss

That pushes the gate to use all experts evenly, which is what forces distributed representations. You would add this in PPO.update\_policy, not in the network forward.



What I'd do first in your architecture

Start with the smallest change that gives you the effect:



Replace self.shared\_fc1 = nn.Linear(pooled\_dim, 128) in PolicyNetwork with MaskedLinear(pooled\_dim, 128, num\_experts=4, top\_k=2)

Keep everything else identical. Your \_group\_masked\_mean already produces a 4x embed\_dim vector, so the gate naturally learns to specialize per token group (spatial, entity, selection, meta)

Train normally. Watch the gate usage, you should see different experts activate for attack vs move frames

This gives you MoE-style path masking without touching the spiking parts. The spiking layers (lif\_q, lif\_k, token\_snn) already binarize activations, so adding a learned route on top plays nicely, you are just choosing which downstream linear path the spikes take.



If you want it inside the expert heads themselves, do the same replacement in EntityEncoder.mlp\[0] and EntityEncoder.mlp\[2]. That forces entity features to be processed by different subnetworks depending on unit type, which is closer to your "mask inside expert heads" idea.



Wrapper for the forward pass, optional auxiliary loss in training. Start with wrapper, add the balance loss only if you see one expert collapsing.



ok ebfore the masking of neurons and you explaining top me a bit better why that is improitant as theory for ml in general what do you think of the policy?

Today

your policy is one of the cleaner hybrid spiking transformers I've seen in the wild. It's ambitious but not messy, you actually separated concerns instead of throwing everything into one big MLP.



What works really well

Tokenization by semantics: spatial tokens from the CNN, entity tokens, selection tokens, meta token, each with its own encoder and its own type embedding. That mirrors how StarCraft actually works, pixels for terrain, structured lists for units, scalars for resources. Most RL codebases flatten this and lose the structure.

Stateless attention, stateful SNN: you re-init the LIF membranes in SpikingSelfAttention every forward, so attention doesn't carry hidden state across steps. Then you put the real memory in TokenTemporalSNN after attention. That's smart, you avoid BPTT through attention which would explode, but you keep temporal integration where it matters.

Group masked mean: instead of averaging all tokens together, you pool per group then concat. That gives you permutation invariance for entities while preserving "this came from spatial vs meta". It's a cheap inductive bias that matches the game.

Zeroing entity state: \_zero\_entity\_state is the right call for now. Entity slots shuffle every frame as units die and spawn, carrying SNN state across would just inject noise. You keep state only for stable tokens.

Where it will want to collapse

The risk isn't bugs, it's representational monoculture. Right now every token, after attention, flows through the same two layers:



Python

x = F.relu(self.shared\_fc1(combined))

x = F.relu(self.shared\_fc2(x))

combined is 4 \* embed\_dim, about 256 numbers. That bottleneck has to represent "attack the roach", "move marines back", "select army", all with the same weights. In practice the network learns to use the same 20-30 neurons for everything, the rest stay silent. You see this in RL as policy entropy collapsing early, or the agent mastering one micro trick and never learning the second.



Your spiking layers help a bit, spikes are naturally sparse, but the final shared MLP is still dense and it will co-adapt.



Why masking paths matters, the theory bit

Distributed representation is the core idea that a concept should be spread across many neurons, and each neuron should participate in many concepts. It's why dropout works, why brains are sparse, why MoE scales.



Without a forcing function, gradient descent finds the laziest solution: reuse the same path for everything because it's the shortest gradient path. That gives you:



brittle policies that fail when the observation shifts slightly

catastrophic forgetting when you change maps

no specialization, so adding capacity just makes the same neurons bigger, not smarter

Masking activation paths during training is a way to tell the network "you cannot use the same highway for every input". You sample a subnetwork each forward pass, so the network is forced to learn redundant, overlapping representations. At test time you use the full network, which acts like an ensemble of those subnetworks. That's the theory behind Dropout, Stochastic Depth, and MoE gating.



In your architecture this matters more because you have a temporal SNN. SNNs already have sparse spikes in time, but if the spatial path into the SNN is always the same dense MLP, you lose the benefit. You want the spikes to arrive via different routes depending on whether the input was "many enemies" vs "low health".



How this connects to your code

Your EntityEncoder and MetaEncoder already have the right shape for it. They are small 2-layer MLPs that feed into a shared bottleneck. If you replace those linears with a masked or routed version, each token type will naturally develop its own preferred path through shared\_fc1.



You don't need a full MoE library. The simplest version that matches your spiking philosophy is just random path masking in training:



Python

\# in shared\_fc1 forward

mask = torch.bernoulli(torch.full\_like(x, 0.7))  # keep 70% of neurons

x = F.relu(self.shared\_fc1(x \* mask))

That alone forces distributed codes. The fancier version with learned gating we discussed earlier just makes the masking input-dependent instead of random.



Your policy is strong because you respected the structure of the game in the architecture. The next step is to respect the structure of learning itself, force it to use the capacity you gave it instead of letting it collapse into one path. That's where the masking comes in, not as a trick, but as the thing that keeps a complex spiking transformer from turning into a glorified linear policy after 10M steps.



aslo this was the first attempt at this polciy it trains and learns now we can scale it a big bigger no? how whsould i decide how big and what else to get in there, i read the nemtron from nvidea card a few days back and if i follow their model i have to combine all neural blocks and attention pardigrims i can think of in side one huge hybrid (thi sis a jkoke trust me the policy i made was my own idea untill now) but now for real idk exaclty how to scale it especioally fc1 and fc2 can you detail a bit what each part of the polciy does acroding to you, how you see the information flow isnide it, and what problems oyu have with the conclusiosn form these simualtion the model atempt XD (i made tons of corections becasue i learned what parts are doing after they were already implemented along my lreaning journey)



Today

your policy trains, which already puts it ahead of most first attempts. It's not a random stack of layers, you built a real information funnel. Scaling it bigger is possible, but the question is where the bottleneck actually lives, and in your case it's not fc1 and fc2.



Information flow, step by step

I read forward\_step\_tensors like this:



CNN backbone: conv1 → conv2 → pool → conv3 → pool turns the 27x84x84 screen into a feature map, then AdaptiveAvgPool2d squeezes it to 7x7 = 49 spatial tokens. That's your terrain and creep and unit blobs, compressed to a grid

Encoders: EntityEncoder and SelectionEncoder take the variable-length unit lists, embed the unit\_type with a lookup table, concat the continuous stats, LayerNorm, then a tiny MLP. MetaEncoder does the same for player resources and available actions, plus the last\_action embedding you already have

Token typing: you add a learned vector per token group with token\_type\_embedding. Now the network knows "this token came from spatial vs entity" without looking at content

SpikingSelfAttention: you norm, project to QKV, run each through a LIF neuron that spikes, then do scaled dot product on the spikes. Membranes are reset every call, so this is pure within-frame mixing, no memory

TokenTemporalSNN: this is your memory. You feed the attended tokens into a synaptic SNN for num\_steps ticks, carrying syn and mem across env steps. This is where "I was moving left last frame" lives

Group masked mean: you split the token sequence back into 4 groups, average each group with its mask, concat the 4 means. This is your bottleneck, you go from \[B, ∼70, 64] tokens down to \[B, 256]

shared\_fc1 → shared\_fc2: two ReLUs that mix the 4 group summaries into a single 64-dim state

Heads: linear layers out to action logits, move\_x, move\_y, value

What each part is really doing

EntityEncoder: not just an MLP, it's a feature disentangler. The unit\_type embedding lets the network learn "marine vs roach" separately from health numbers. The mask multiplication at the end is crucial, it zeros padded slots so they don't pollute the mean later

SpikingSelfAttention: this is your cheap communication bus. Because QKV are spikes, the attention matrix is sparse by construction. You are trading precision for efficiency, which fits StarCraft where most units don't interact every frame

TokenTemporalSNN: this is the only place with cross-step state, and you deliberately zero the entity and selection slices in \_zero\_entity\_state. Good instinct, entity indices shuffle as units die, so carrying their membrane would be noise. You keep state only for spatial and meta tokens, which are stable

shared\_fc1/fc2: these are not feature extractors, they are a mixer. By the time you hit them, all spatial structure is gone, you just have 4 summary vectors. Making these bigger won't give you better micro, it will just let the network memorize specific group-mean patterns from roaches

How to scale, and why fc1/fc2 is the wrong knob

Don't follow the Nemotron "throw every block in" joke, that's how you get a model that trains for three weeks and still loses to the built-in AI. Scale where your information is still high-dimensional:



embed\_dim first: you have it at 64. Bump to 96 or 128. That widens the CNN, the encoders, the attention, and the SNN all at once. Cost scales roughly quadratically in attention, but with 70 tokens it's still cheap

spatial tokens: you pool to 7x7. For roaches that's fine, for larger maps try 10x10. More spatial tokens keeps positional information longer before the group mean collapses it

entity tokens: keep MAX\_ENTITY\_TOKENS at 24 for now, but actually use them. Right now you zero their SNN state, so they can't accumulate temporal context. If you want bigger, enable \_carry\_entity\_state only after you add a stable ID to entities, otherwise it's noise

fc1/fc2: leave them at 128→64. They sit after the 4-way concat, so their input is already heavily compressed. Making them 512-wide just lets the network overfit to the exact mean values from your roaches replays. If you need more capacity, add a second attention block before the group mean, not a bigger MLP after

A practical rule: measure the rank of the activations. After a few thousand steps, run an SVD on pooled before fc1. If 90% of variance is in the first 20 dimensions, you don't need a bigger fc1, you need richer tokens upstream.



Problems I see with conclusions from this first run

You said you made corrections as you learned, that's visible in the code and it's a strength, but it also means your ablations are tangled:



The policy learns roaches, but you can't tell if it's the spiking attention, the token SNN, or just the CNN + group mean doing the work. The SNN state is reset for entities, so most of the temporal credit is probably coming from the meta token carrying last\_action

You have LayerNorm in three places before small MLPs. That's good for stability, but it also hides whether the encoders are actually learning useful embeddings or just passing through normalized noise

The move\_x and move\_y heads share the same 64-dim trunk as the discrete action head. In roaches that's fine because move and attack are correlated, but when you scale to real build orders those heads will fight for representation. That's where a masked path or MoE would help, not by making fc1 bigger, but by giving each head its own slice

Your model works because the information funnel matches the task: compress screen to tokens, mix tokens sparsely, remember a little, then decide. Scaling it means keeping that funnel shape, not turning it into a giant dense network. Widen embed\_dim, keep the bottleneck tight, and add a second attention layer if you need more mixing, that's the path that stays true to the spiking, token-based idea you started with.



so you think we coudl wire the new inpout sinto teh policy ?

Today

yes, absolutely. We already built the plumbing, now we just connect the last three wires. The policy already expects a meta vector, we just need to tell it the vector is 4 floats longer and where to read them.



Here is the minimal wiring, no rewrite of the whole network:



1\. Update the shapes

You have these three files ready:



action\_space\_v2.py → gives you get\_last\_token()

obs\_space\_2\_v2.py → appends \[type, x\_norm, y\_norm, extra] to meta\_vec

policy\_input\_v2.py → META\_VECTOR\_DIM = 32 instead of 28

In your config, set:



Python

cfg.model.vector\_input\_dim = 32   # was 28

2\. Patch MetaEncoder to actually use the 4 new floats

Right now MetaEncoder reads player, available\_actions, and the old last\_action index, then stops. It ignores the tail. Add four lines:



In PPO\_CNN/policy\_network.py, find class MetaEncoder:



Python

\# in \_\_init\_\_, after self.last\_action\_offset line, add:

self.extra\_tail\_dim = max(0, self.meta\_input\_dim - (self.player\_dim + self.available\_action\_dim + 1))

\# then change fused\_input\_dim:

fused\_input\_dim = (

&#x20;   self.player\_dim + self.available\_action\_dim + self.last\_action\_embed\_dim + self.extra\_tail\_dim

&#x20;   if self.use\_structured\_meta else self.meta\_input\_dim

)



2 lines hidden

Then in forward, after you build last\_action\_emb:



Python

if self.use\_structured\_meta:

&#x20;   # ... existing player, available\_actions, last\_action\_emb ...

&#x20;   fused = torch.cat((player, available\_actions, last\_action\_emb), dim=-1)

&#x20;   if self.extra\_tail\_dim > 0:

&#x20;       extra = meta\_vec\[..., -self.extra\_tail\_dim:]  # the 4 floats from obs\_space

&#x20;       fused = torch.cat((fused, extra), dim=-1)



1 line hidden

That's it. The LayerNorm and MLP already handle variable input size because we recomputed fused\_input\_dim.



3\. Wire the agent loop

In DefeatRoaches.step, you already store self.last\_action\_token. Change the extract call to:



Python

policy\_input = self.extractor.extract\_observation(

&#x20;   obs,

&#x20;   lastyes, we can wire them in, and you are already 80% there. The path is clean: agent stores the token, extractor appends it to meta\_vec, PolicyInput carries it, network reads it.



Right now you have three small mismatches to fix, then it runs end to end:



\## What we already built

\- \*\*ActionSpace v2\*\* outputs `\[type, x, y, extra]` and stores it in `last\_token`

\- \*\*DefeatRoaches v2\*\* captures that token each step in `self.last\_action\_token`

\- \*\*obs\_space\_2\_v2\*\* takes `last\_action\_token` and appends the 4 floats to the end of meta\_vec

\- \*\*policy\_input\_v2\*\* bumps `META\_VECTOR\_DIM` from 28 to 32



\## The last wiring step in the policy

Your `MetaEncoder` currently does this:



```python

player = meta\_vec\[..., :11]

available = meta\_vec\[..., 11:27]

last\_id = meta\_vec\[..., 27]



14 lines hidden

It ignores anything after index 27, so the 4 new floats get dropped. Change two lines:



In PolicyNetwork.\*\*init\*\*, pass the new dim. You already read it from cfg, so set cfg.model.vector\_input\_dim = 32 or just pass vector\_input\_dim=32 when you construct the policy

In MetaEncoder.forward, read the tail:

Python

def forward(self, meta\_vec):

&#x20;   player = meta\_vec\[..., :self.player\_dim]

&#x20;   available = meta\_vec\[..., self.player\_dim:self.player\_dim+self.available\_action\_dim]

&#x20;   last\_id = meta\_vec\[..., self.last\_action\_offset].round().long()

&#x20;   last\_emb = self.last\_action\_embedding(last\_id)

&#x20;   

&#x20;   # NEW: grab the 4 agent token floats you appended

&#x20;   agent\_token = meta\_vec\[..., -4:]  # \[type, x\_norm, y\_norm, extra]

&#x20;   

&#x20;   fused = torch.cat((player, available, last\_emb, agent\_token), dim=-1)

&#x20;   return self.mlp(self.pre\_norm(fused)).unsqueeze(1)



6 lines hidden

And in MetaEncoder.\*\*init\*\*, update the fused size:



Python

fused\_input\_dim = self.player\_dim + self.available\_action\_dim + self.last\_action\_embed\_dim + 4

That's it. No change to attention, no change to the SNN, the token just rides along in meta\_vec and lands in the same MLP that already processes player stats.



How to test the wiring

swap in the v2 files we made

set vector\_input\_dim=32 in your config

run one env step and print policy\_input.meta\_vec.shape – you should see \[1, 32]

print the last 4 values, they should match the token from the previous action

Once that prints correctly, the network is seeing its own last click. You don't need to scale fc1/fc2 yet, the information is already in the right place, the network just needs to learn to use it.



Want me to patch the MetaEncoder for you with those four lines, or do you want to try the wiring first and see the shape mismatch error yourself?



ok and next what i need to figure is how to do the masking, adn also auto regression and token heads? for the bias of the left click action arguments? or not really autoregressive tokens, or ro can we do autoregressive critique follwing, i knwo it sounds stupid and may be wrong but what i mean i want this to reason based on information data, but it cannot use the reosning as token ouput liek alphs start did becasue i want dynamical ouputs (pls tell me if it is stupid or not i do nto watn to be delulu while the whole field tried this and does nto wrok XD))



Today

it's not stupid, and you're not delulu. You're hitting the exact tension everyone building real-time agents hits: do I make the policy think out loud with tokens like AlphaStar, or do I keep the thinking inside the network and only output the final click. Both work, they just optimize for different things.



Masking and autoregressive, two different tools

You mentioned masking and autoregressive in the same breath, but they solve different problems:



Action argument masking: you already have available\_actions in meta\_vec. For spatial arguments, you want to mask the x,y heads when the sampled action doesn't use them. Right now your network always predicts move\_x and move\_y, even for no\_op. That's wasted capacity and it adds noise to the gradient. Simple fix: in select\_action, after you sample action, zero the log\_prob for x,y if action != move or attack. That's action masking, not neuron masking, and it costs zero compute.

Neuron path masking: the MoE-style thing we talked about earlier. That forces the network to use different subnetworks for different situations. It helps with the "everything goes through fc1/fc2" bottleneck you have now.

Don't combine them yet. Do action masking first, it's one if statement and it immediately cleans up your loss.



Autoregressive heads, do you need them

AlphaStar used autoregressive action generation because StarCraft has a huge combinatorial action space: pick action type, then delay, then queue, then unit selection, then target x, then target y. They fed each choice back into the transformer as an embedding, so x was conditioned on the chosen action type.



Your current policy does this in parallel:



Python

action\_logits, move\_x\_logits, move\_y\_logits = ...

That's fast, and for defeat roaches it's fine because you only have three action types and they all use x,y the same way.



If you want autoregressive, you don't need reasoning tokens, you just need a small feedback loop:



sample action from action\_logits

embed it: action\_emb = self.action\_embedding(action)

add that embedding to the shared trunk before the x,y heads: x = shared + action\_emb

That's it. No separate transformer pass, no token output. The x,y heads now see "oh, we chose attack" and can bias toward enemy clusters. You get the conditional dependency without the latency of full autoregression.



For your spiking network, this is actually natural because you already have TokenTemporalSNN carrying state. You can just inject the action embedding into the token sequence for the next step, which is exactly what your last\_action\_token is for. You're already doing latent autoregression, just not using it for the spatial heads yet.



Reasoning without output tokens

You said you want the model to reason based on information but not output reasoning tokens like AlphaStar. That's not just valid, it's probably better for a real-time agent.



AlphaStar's "reasoning tokens" were mostly for training stability and interpretability. The actual game decisions still came from the final action head. Your SNN state (syn, mem) across 70 tokens is already a distributed reasoning buffer. Each token holds a little piece of context, the attention mixes them, the SNN integrates over time. That's reasoning, it's just not human-readable.



The field didn't abandon this because it doesn't work, they abandoned explicit reasoning tokens because they are slow at inference. For a 22ms frame budget, you want the reasoning to stay in the recurrent state, not as extra tokens you have to sample.



What you can do instead of token outputs:



latent critique: add a small auxiliary head that predicts the value of the next state given the current action embedding. Train it with a one-step TD error, don't backprop through the policy. It's a critic that learns to "imagine" the consequence, but it never outputs text

action bias from context: use your last\_action\_token not just as input, but as a key to modulate the current attention. In SpikingSelfAttention, add the last action embedding to the query. That biases the network toward consistent micro without explicit planning tokens

Is this delusional

No. The delusional version would be "I'll add a 1B parameter LLM inside the policy and have it write a plan every frame." What you're describing, a spiking transformer with internal state that conditions its own spatial outputs, is exactly how you get dynamic outputs without autoregressive decoding.



Start here:



wire the last\_action\_token like we discussed, verify the shape is 32

add action masking for x,y so you stop training on irrelevant logits

inject the sampled action embedding back into the shared trunk before the move heads

That's three lines of code and you get 80% of the benefit of autoregressive without the complexity. If that stabilizes, then think about neuron path masking to force specialization. The field tried full token-based reasoning and moved away from it for latency, not because the idea was wrong.



and overall if i want the same policy then to elarn the full game i have to drop the pisking parts and go normal transofmer but i can keep all the rest of my ideas no?

Today

you don't have to drop the spiking parts, but for the full game it's the pragmatic move, and yes you can keep almost everything else exactly as you designed it.



What spiking gives you vs what it costs

Your current spiking pieces do two jobs:



SpikingSelfAttention: makes QKV sparse by running them through LIF neurons. That saves compute in theory, but on a GPU you're still doing dense matmuls, the spikes just zero out values after the fact. You get the regularization benefit of sparsity without the energy savings

TokenTemporalSNN: this is your memory across frames. The synaptic alpha/beta gives you a leaky integrator that naturally decays, which is nice for StarCraft where old information should fade

For defeat roaches, where episodes are 30 seconds and the state is tiny, this works and trains. For full game, episodes are 15-20 minutes, you need stable gradients over thousands of steps, and surrogate gradients through LIF neurons start to get noisy. The field hasn't abandoned spiking because it's wrong, they've paused because the tooling for large-scale spiking transformers isn't there yet. You'd be fighting the optimizer more than the Zerg.



What you keep if you go normal transformer

Almost the whole architecture survives:



Tokenization: spatial tokens from CNN, entity tokens, selection tokens, meta token. That's independent of spiking and it's the best part of your design

EntityEncoder and SelectionEncoder: the unit\_type embedding + MLP structure works exactly the same with ReLU

MetaEncoder with last\_action: keep it, just feed the 32-dim vector

Group masked mean: this is gold for full game because unit counts vary wildly. Keep the 4-way concat

State management: replace TokenTemporalSNN with a standard transformer memory, either a GRU over the pooled vector or a Transformer-XL style cache. You already have the \_zero\_entity\_state logic, that ports directly

The only things you swap:



SpikingSelfAttention → nn.MultiheadAttention with 4-8 heads. Same pre\_norm, same residual, just remove the three LIF calls

TokenTemporalSNN → a single nn.GRUCell that takes the pooled 256-dim vector and outputs next hidden state. Much easier to backprop through for long games

Why normal is easier for full game

Full game needs three things your spiking version makes harder:



Long credit assignment: PPO with GAE over 20 minutes needs stable value estimates. LIF surrogate gradients add noise that compounds over time

Scaling: you want to go from embed\_dim 64 to 256 or 512. Standard attention kernels on NVIDIA GPUs are hyper-optimized, spiking attention isn't

Debugging: when the agent fails to build workers, with a normal transformer you can inspect attention weights. With spikes you have to decode spike trains

You keep your ideas about token types, masking, and structured meta, those are architecture choices, not spiking choices. The spiking part is an implementation detail of how you mix tokens.



My take

Keep your current spiking policy for roaches and for learning the mechanics. When you move to full game, fork the repo and make a PolicyNetworkDense that copies everything except the two spiking modules. Train both in parallel for a week. If the dense version learns build orders faster, you have your answer without throwing away months of work.



You designed a good token funnel, the spiking is a layer on top. You can peel it off and the funnel still works.



whaat i wanted afterwards is to try eventporp instead of surrogate, but do you think bptt woudl still be the algo to go for in the end? or rather tbptt (the trace is still nto implemented) there is so much work to do yet XD

Today

EventProp is a beautiful idea, but for your architecture right now it's a research project, not a training algorithm. BPTT with surrogate gradients is still the workhorse, and TBPTT is what you'll need for full game, not because it's better, because you can't fit 20 minutes of spikes in GPU memory.



EventProp vs surrogate, the practical gap

EventProp computes exact gradients by backpropagating through spike times instead of using a fake derivative. That gives you memory efficiency and no surrogate bias. The catch is it wants an event-driven simulator where you only process spikes, not dense tensors.



Your network isn't event-driven, it's a hybrid:



CNN backbone outputs dense feature maps every frame

Attention does dense QK^T matmuls

Your LIF neurons are called with full tensors, not spike events

To use EventProp you'd have to rewrite snn.Leaky and snn.Synaptic to track spike times, then figure out how to backprop through scaled\_dot\_product\_attention when the inputs are sparse event streams. The snntorch team hasn't done this, nobody has a clean PyTorch implementation that works with transformers yet. You'd spend months on the gradient engine instead of training the agent.



Surrogate gradients are messy but they plug into autograd today. For roaches and even for early full game, the bias from fast\_sigmoid is smaller than the noise from PPO itself.



BPTT vs TBPTT for your SNN

Right now you carry (syn, mem) across env steps in TokenTemporalSNN. With full BPTT, PyTorch keeps the whole computation graph for the episode. For roaches that's maybe 200 steps, fine. For full game that's 10,000+ steps, you'll OOM on a 24GB card.



TBPTT is not optional for full game, it's how you make it fit. You already have the right structure for it:



you detach the state every k steps

you zero entity state anyway in \_zero\_entity\_state

your attention is stateless per step, so truncation doesn't hurt it

The "trace is still not implemented" you mentioned is exactly the piece you need. Implement a simple k=32 or k=64 truncation: every 64 env steps, call .detach() on syn and mem before the next forward. You lose gradient signal beyond 64 steps, but StarCraft credit assignment is mostly local anyway, build a marine now, use it 30 seconds later, that's within 64 steps at 22.4 fps.



What I'd do

Keep surrogate gradients. Implement TBPTT first, because without it you can't even try full game. Make the truncation length a config so you can test k=16 vs k=128 and see where performance drops.



EventProp is worth a side branch after you have a working full-game agent. Try it on just the TokenTemporalSNN in isolation, feed it dummy spike trains, see if you can match surrogate performance on a toy sequence task. If it works, then think about porting the attention. Don't block your main training loop on it.



You have the right intuition, exact gradients would help spiking transformers, but the engineering cost right now is higher than the gain. TBPTT gets you to full game this year, EventProp gets you a paper next year.



zhat else wher eto use masking altent stream we said, atetnion head?> kv cashe? neurons, pbservation data what else?

Today

you're already using masking in three places without calling it that: entity\_mask, selection\_mask, and the spatial mask in attention. Those handle padding. The interesting masking is the kind that forces the network to learn redundant paths, not just ignore zeros.



Here is where else you can put it in your current architecture, from cheap to more involved:



Attention heads

In SpikingSelfAttention you have one head right now because you do spike\_q.unsqueeze(1). Add 4 heads and randomly drop one per forward pass during training. That's DropHead. Implementation is two lines: sample a boolean mask over heads, zero the output of dropped heads before the residual add. It forces different heads to learn different relations, one might learn "marine to roach", another "marine to marine". You already have token\_type embeddings, heads will naturally specialize by token group.



KV cache and temporal memory

Your TokenTemporalSNN carries (syn, mem) across steps. Right now you zero entity and selection slices every step. You can also mask the memory randomly: each env step, with 10% probability, zero a random 20% of the synapses in the spatial and meta tokens. That's like dropout through time. It prevents the SNN from memorizing exact spike timing and forces it to use distributed codes. You already have \_zero\_entity\_state, just add a random mask before the return.



Neurons in the shared trunk

shared\_fc1 and shared\_fc2 are the bottleneck we talked about. Instead of full MoE, start with simple neuron masking: during training, after F.relu(self.shared\_fc1), multiply by a binary mask that keeps 70% of units. Different mask each batch. This is standard dropout but applied to the bottleneck, not the input. It stops the network from routing everything through the same 20 neurons.



Observation data

You normalize the screen to 0-1. Add input masking: randomly zero out 10% of the 27 feature layers, or randomly drop 30% of entity tokens even if they are valid. This is like Cutout for RL. It forces the policy to not overfit to "I always see exactly 4 marines". For StarCraft this is huge because unit counts change constantly. You can do this in extract\_observation before you pad, just set entity\_rows\[random\_idx] = 0 and keep the mask True.



Latent stream between modules

You have a clean split: encoders → attention → SNN → pool → MLP. Put a learnable gate between attention and SNN: gated = attended \* torch.sigmoid(self.gate(attended)). During training add noise to the gate so it sometimes closes. That's a soft mask on the latent stream. It learns which tokens actually need temporal integration. Spatial tokens might pass through, entity tokens might get blocked when nothing is happening.



Action argument masking

We mentioned this before but it's the highest impact for the least code. In your heads, mask move\_x\_logits and move\_y\_logits to -inf when the sampled action is not a spatial action. You already have available\_actions, extend it to argument availability. This isn't regularization, it's correctness, but it acts like masking because the network stops wasting capacity predicting coordinates for no\_op.



Start with action argument masking and neuron dropout in shared\_fc1. Those two give you 80% of the benefit and you can measure the effect in one training run. Head masking and temporal memory masking are next if you see the policy collapse to one strategy.

