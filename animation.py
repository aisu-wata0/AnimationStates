
from argparse import ArgumentTypeError
from typing import Any, Dict, List, Set, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import time
import random
import math
import copy

import numpy as np
from bisect import bisect_left

from python_utils_aisu.utils import Buildable, Cooldown, CooldownVarU, get_random_string
from python_utils_aisu import utils

logger = utils.loggingGetLogger(__name__)
logger.setLevel('INFO')


@dataclass
class BezierCurveCubic:
    """
    cubic-bezier(x1,y1,x2,y2) in CSS has the first point 0,0 and last 1,1
    https://cubic-bezier.com/
```python
px = [0, x1, x2, 1]`
py = [0, y1, y2, 1]
bcc = BezierCurveCubic(px, py, resolution)

bcc.y(x) # gets value of y for x
bcc.y_y(x) # gets value of y for x, but interpolates
```
    """
    px: List[int]
    py: List[int]
    resolution: float = 0.001
    curve: Dict[str, Dict[float, float]] = field(default_factory=lambda: {'x': {}, 'y': {}})
    values: Dict[str, List[float]] = field(default_factory=lambda: {'x': [], 'y': []})
    rounded: bool = True

    def __post_init__(self):
        if not self.curve:
            self.curve = {'x': {}, 'y': {}}
        if len(self.curve['x']) == 0 or len(self.curve['y']) == 0:
            self.recalculate()
    
    def getDegree(self):
        return len(self.px)
    
    def getRoundDigits(self):
        return 1 + int(-math.log10(self.resolution))

    def recalculate(self):
        self.curve = {'x': {}, 'y': {}}
        x = self.px
        y = self.py
        round_digits = self.getRoundDigits()
        for t in np.arange(0.0, 1.0, self.resolution):
            mt = 1-t
            xu = mt**3 * x[0] + 3 * t * mt**2 * x[1] + 3 * t**2 * mt * x[2] + t**3 * x[3]
            yu = mt**3 * y[0] + 3 * t * mt**2 * y[1] + 3 * t**2 * mt * y[2] + t**3 * y[3]
            if self.rounded:
                self.curve['x'][round(xu, round_digits)] = yu
                self.curve['y'][round(yu, round_digits)] = xu
            else:
                self.curve['x'][xu] = yu
                self.curve['y'][yu] = xu
        
        self.values['x'] = sorted(self.curve['x'].keys())
        self.values['y'] = sorted(self.curve['y'].keys())

    def x_index(self, x):
        return bisect_left(self.values['x'], x)

    def x_nearest(self, x):
        index = self.x_index(x)
        
        if index == 0:
            # x is smaller than the smallest precalculated x
            return self.values['x'][0]
        if index == len(self.values['x']):
            # x is larger than the largest precalculated x
            return self.values['x'][-1]
        return self.values['x'][index]

    def y(self, x):
        return self.curve['x'][self.x_nearest(x)]

    def y_i(self, x):
        index = self.x_index(x)

        if index == 0:
            # x is smaller than the smallest precalculated x
            return self.curve['x'][self.values['x'][0]]
        if index == len(self.values['x']):
            # x is larger than the largest precalculated x
            return self.curve['x'][self.values['x'][-1]]

        x1 = self.values['x'][index - 1]
        x2 = self.values['x'][index]
        y1 = self.curve['x'][x1]
        y2 = self.curve['x'][x2]

        # Interpolate the y value between the nearest precalculated x values
        t = (x - x1) / (x2 - x1)
        interpolated_y = y1 + t * (y2 - y1)
        return interpolated_y

    @classmethod
    def css(cls, x1,y1,x2,y2, resolution=0.001):
        px = [0, x1, x2, 1]
        py = [0, y1, y2, 1]
        return BezierCurveCubic(px, py, resolution)



@dataclass
class CurveT(Buildable):
    """
    Curve that represents the transition in time `x` from from 0 to 1
    by definition:
    CurveT().get(0) == 0
    CurveT().get(1) == 1

    but values in between are defined in the behavior of subclasses
    """
    kwargs: Dict[str, Any] = field(default_factory=dict)
    name: str = ''

    def a(self, arg_name):
        return self.kwargs[arg_name]

    def get(self, x: float) -> float:
        """x: float represents the percentage of completion/time elapsed of the curve"""
        raise NotImplementedError()


@dataclass
class CurveT_linear(CurveT):
    """
    y == x
    """
    name: str = 'linear'
    def get(self, x: float) -> float:
        return x

@dataclass
class CurveT_linear_delays(CurveT):
    """
    `y == 0` until `(x, y) == (delay_start, 0)`

    then linearly rises until `(x, y) == (delay_end, 1)`
    
    then `y == 1` until the end, i.e. x == 1
    """
    name: str = 'linear_delays'
    def __post_init__(self):
        self.kwargs = {
            'delay_start': 0.0,
            'delay_end': 0.0,
            **self.kwargs,
        }
    def get(self, x: float) -> float:
        delay_start = self.a('delay_start')
        delay_end = self.a('delay_end')
        return min(1.0, max(0.0, (x - delay_start) / (delay_end - delay_start)))

@dataclass
class CurveT_keyframes(CurveT):    
    """
    `keyframes: List of (x, y) keyframe tuples`

    The curve passes through all keyframes, with linear interpolation between them.
    """
    name: str = 'keyframes'
    def get(self, x: float) -> float:
        keyframes = self.a('keyframes')
        if x <= keyframes[0][0]:
            return 0.0
        if x >= keyframes[-1][0]:
            return 1.0
        
        for i in range(len(keyframes)-1):
            if keyframes[i][0] <= x <= keyframes[i+1][0]:
                x1, y1 = keyframes[i]
                x2, y2 = keyframes[i+1]
                return y1 + (x - x1) * (y2 - y1) / (x2 - x1)
        return 1.0

@dataclass
class CurveT_sigmoid(CurveT):    
    """
    Sigmoid curve defined by `steepness`. 
    The steeper the curve, the more abruptly it transitions from 0 to 1.
    """
    name: str = 'sigmoid'
    def get(self, x: float) -> float:
        steepness = self.a('steepness')
        return 1 / (1 + np.exp(-steepness * (x - 0.5)))


@dataclass
class CurveT_bezier(CurveT):    
    """
    Cubic Bezier curve defined by two control points.

    The curve starts at (0, 0) and ends at (1, 1).
    The two control points determine the shape of the curve.
    """
    name: str = 'bezier'
    def get(self, x: float) -> float:
        p1 = self.a('p1')
        p2 = self.a('p2')
        return (1-x)**3 * p1[1] + 3*(1-x)**2 * x * p2[1] + 3*(1-x) * x**2 * (1 - p2[1]) + x**3 * (1 - p1[1])


CurveT.register_classes({
    'CurveT_linear': CurveT_linear,
    'CurveT_linear_delays': CurveT_linear_delays,
    'CurveT_keyframes': CurveT_keyframes,
    'CurveT_sigmoid': CurveT_sigmoid,
    'CurveT_bezier': CurveT_bezier,
})

@dataclass
class Transition:
    # curve: Dict[str, Any]
    curve: CurveT
    cd: CooldownVarU

    def getCurve(self) -> CurveT:
        # return CurveT.build(**self.curve)
        return self.curve

    def get(self, time_counter: float) -> float:
        return self.getCurve().get(self.cd.elapsed_percent(time_counter))

    def doStart(self, time_counter: float):
        return self.cd.doStart(time_counter)
    
    @classmethod
    def build(cls, curve, cd):
        return Transition(
            curve=CurveT.build(**curve),
            cd=CooldownVarU.build(cd),
        )


class AParameters:
    """
    An abstract base class.
    Represents Animation Parameters that can be:
        - added together
        - multiplied by a scalar
    """
    @abstractmethod
    def __add__(self, other: 'AParameters') -> 'AParameters':
        """Addition method, between objects of this class."""
        pass

    @abstractmethod 
    def __radd__(self, other: 'AParameters') -> 'AParameters':
        """Addition method with reflected operands, between objects of this class."""
        pass

    @abstractmethod
    def __mul__(self, scalar: float) -> 'AParameters':
        """Multiplication method."""
        pass

    @abstractmethod
    def __rmul__(self, scalar: float) -> 'AParameters':
        """Multiplication method with reflected operands."""
        pass



class Animation(Buildable):
    def __init__(
        self,
        typ: str = "",
        duration: Dict[str, float] | float | None = 0.0,
        interval: Dict[str, float] | float | None = 0.0,
        transition: Transition | None = None,
        sentiments: Dict[str, float] | None = None,
        tags: Set[str] | None = None,
        weight = 1.0,
        hist_max = 4,
        **kwargs,
    ):
        self.typ = typ
        self.transition = transition
        self.sentiments = sentiments or {}
        self.tags = tags or set()

        self.duration = duration
        self.interval = interval

        self.state = AParameters()
        self.state_hist: List[Dict[str, Any]] = []
        self.hist_max = hist_max
        self.kwargs = kwargs
        
        self.weight = weight
        self.time_multipliers = {
            'duration': 1.0,
            'interval': 1.0,
        }
        self.__post_init__()

        # Cooldowns have instances because they might be multiplied by something temporarily
        # such as sentiments

        self.cooldowns = {
            'duration': CooldownVarU.build(self.duration),
            'interval': CooldownVarU.build(self.interval),
        }
        self.cooldowns['duration_i'] = copy.copy(self.cooldowns['duration'])
        self.cooldowns['interval_i'] = copy.copy(self.cooldowns['interval'])
        self.init()

    
    def __post_init__(self):
        pass

    def init(self):
        """
        variables to be set before a new animation cycle starts to be used until the end
        for example randomizing some parameters
        """
        pass

    def fill_default_dict(self, obj, default):
        if not obj:
            obj = default

        if isinstance(obj, dict):
            obj = {
                **default,
                **obj,
            }
        return obj

    def isActive(self):
        return self.weight > 0 and (self.cooldowns['duration_i'].getRemaining() > 0) or self.isContinuous()
    
    def isContinuous(self):
        return not self.interval or self.cooldowns['interval_i'].seconds == 0
    
    def get_weight(self) -> float:
        return self.weight
    
    def get_parameters_w(self) -> AParameters:
        return self.get_parameters() * self.get_weight()
    
    def get(self) -> AParameters:
        return self.get_parameters_w()
    
    
    def elapsed_percent(self, time_counter):
        return min(1.0, self.cooldowns['duration_i'].elapsed_percent(time_counter)) 
    
    def time_pi_duration(self, time_counter):
        return time_counter * math.pi / self.cooldowns['duration_i'].getDuration()

    def state_add(self, state: AParameters, elapsed):
        self.state_hist.append({
            'elapsed': elapsed,
            'state': state,
        })
        self.state_hist = self.state_hist[-self.hist_max:]
        return self.state_hist
    
    def update(self, elapsed, time_counter, **kwargs) -> AParameters:
        self.update_state(elapsed, time_counter, **kwargs)
        state = self.get_parameters()
        self.state_add(state, elapsed)
        return self.get_parameters_w()

    def get_parameters(self) -> AParameters:
        """
        return (last) animation parameters
        """
        return self.state

    def initCooldown(self, typ, time_counter):
        cooldown = copy.copy(self.cooldowns[typ])
        cooldown *= self.time_multipliers[typ]
        cooldown.trigger(time_counter, check=False)
        self.cooldowns[f'{typ}_i'] = cooldown
        return cooldown
    
    def set_time_multiplier(self, typ, value):
        delta = value / self.time_multipliers[typ]
        self.cooldowns[typ + "_i"] *= delta
        self.time_multipliers[typ] = value


    def update_state(self, elapsed, time_counter, **kwargs):
        duration_cd = self.cooldowns['duration_i']
        interval_cd = self.cooldowns['interval_i']
        if duration_cd.isStarted() or self.isContinuous() or interval_cd.isFinished(time_counter):
            self.animate(elapsed, time_counter, **kwargs)
            if duration_cd.getDuration() > 0:
                triggers = duration_cd.trigger(time_counter)
                if triggers > 1:
                    # End animation
                    duration_cd.clear()
                    interval_cd = self.initCooldown('interval', time_counter)
                    return
        else:
            triggers = interval_cd.trigger(time_counter)
            if triggers > 1:
                # Start animation
                interval_cd.clear()
                duration_cd = self.initCooldown('duration', time_counter)


    def animate(self, elapsed, time_counter, **kwargs):
        """
        calculate actual animation based on last state and elapsed
        set self.state
        """
        pass



@dataclass
class Changer:
    cd: Cooldown
    state_names: List[str] = field(default_factory=list)
    state_idx: int = 0
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def get(self):
        if not self.state_names:
            return None
        return self.state_names[self.state_idx]

    def update(self, time_counter, **kwargs):
        triggers = self.cd.trigger(time_counter)
        if triggers > 1:
            self.change_state(time_counter, triggers, **kwargs)
            return self.get()
        return None
    
    def a(self, arg_name):
        return self.kwargs[arg_name]
    
    @abstractmethod
    def change_state(self, time_counter, triggers=1.0, **kwargs):
        """should change self.state_idx"""
        pass

@dataclass
class ChangerRandom(Changer):
    def change_state(self, time_counter, triggers=1.0, **kwargs):
        if not self.state_names:
            return
        self.state_idx = random.randint(0, len(self.state_names) - 1)

@dataclass
class ChangerCycle(Changer):
    def change_state(self, time_counter, triggers=1.0, **kwargs):
        if not self.state_names:
            return
        self.state_idx = (self.state_idx + int(triggers)) % len(self.state_names)



class AnimationStates:
    def __init__(
        self,
        animations: Dict[str, Animation | Dict] | None = None,
        transitions: Dict[str, Dict[str, Transition | Tuple[Dict, Dict]]] | None = None,
        state_changes: Dict[str, Changer] | None = None,
        sentiments_args = None,
        sentiments: Dict[str, float] | None = None,
        **kwargs
    ):
        if not animations:
            animations = {}

        self.sentiment_animations: Dict[str, Dict[str, Animation]] = {}
        self.animations: Dict[str, Animation] = {}
        for animation_name, a in animations.items():
            if isinstance(a, Animation):
                logger.info(f"Registered {animation_name}")
                self.animations[animation_name] = a
            else:
                logger.info(f"Building   {animation_name}")
                self.animations[animation_name] = Animation.build(animation_name, **a)
            if animation_name.startswith('sentiment_'):
                sentiment = animation_name.split('_')[1]
                if sentiment not in self.sentiment_animations:
                    self.sentiment_animations[sentiment] = {}
                
                self.sentiment_animations[sentiment][animation_name] = self.animations[animation_name]
                self.animations[animation_name].weight = 0.0
        
        logger.info(f"self.sentiment_animations {self.sentiment_animations}")

        transitions = transitions or {}
        tr: Dict[str, Dict[str, Transition]] = {k: {} for k in transitions.keys()}
        for k in transitions.keys():
            for kk in transitions[k].keys():
                args = transitions[k][kk]
                if isinstance(args, tuple):
                    tr[k][kk] = Transition.build(*args)
                elif isinstance(args, Transition):
                    tr[k][kk] = args
                else:
                    ArgumentTypeError()

        self.transitions: Dict[str, Dict[str, Transition]] = {
            'by_type': {
                'default': Transition.build({'name': 'linear'}, {'seconds': 0}),
                'idle': Transition.build({'name': 'linear'}, {'seconds': 2}),
            },
            'None': {
                'default': Transition.build({'name': 'linear'}, {'seconds': 0}),
            },
            **tr,
        }

        self.state_changes: Dict[str, Changer] = {
            **(state_changes or {}),
        }

        self.animations_by_type = {}
        for a_name, a in self.animations.items():
            if a.typ not in self.animations_by_type:
                self.animations_by_type[a.typ] = []
            self.animations_by_type[a.typ].append(a_name)

        
        self.playing_types: Dict[str, set] = {}
        self.animation_name_to_changer_typ = {}

        for typ in self.animations_by_type.keys():
            typ_split = typ.split('_')
            for ti in range(len(typ_split)-1, -1, -1):
                t = '_'.join(typ_split[:ti+1])
                if t in self.state_changes.keys():
                    changer = self.state_changes[t]
                    a_list = self.animations_by_type[typ]
                    logger.info(f"Registered {a_list} into {t} changer")
                    changer.state_names = [*{*changer.state_names, *a_list}]
                    for a_name in a_list:
                        self.animation_name_to_changer_typ[a_name] = t
                        self.animations[a_name].weight = 0
                    animation_name = changer.get()
                    if animation_name:
                        self.animations[animation_name].weight = 1.0
                        self.playing_types[typ] = {animation_name}
                    break

        self.sentiments_args = sentiments_args or {}
        self.sentiments: Dict[str, float] = sentiments or {}
    
        self.transitioning: Dict[str, Dict[str, Dict[str, str]]] = {}
        self.transitions_active: Dict[str, Transition] = {}

        self.time_counter_last = time.perf_counter()
        self.kwargs = kwargs


    def add_transition(self, typ, animations_in: Dict[str, str], animations_out: Dict[str, str]| None = None):
        animations_out = (animations_out or {})

        multipliers = {
			'duration_multiplier': 1.0,
			'transition_multiplier': 1.0,
        }
        for sentiment, weight in self.sentiments.items():
            if sentiment in self.sentiments_args:
                for var_name in ['duration_multiplier', 'transition_multiplier']:
                    if var_name in self.sentiments_args[sentiment]:
                        multipliers[var_name] *= self.sentiments_args[sentiment][var_name]
        
        done = set()
        for animations in [animations_in, animations_out]:
            for a, transition_id in animations.items():
                if transition_id not in done:
                    transition = self.transitions_active[transition_id]
                    transition.cd *= multipliers['transition_multiplier']
                    done.add(transition_id)

        self.transitioning[typ] = {
            'out': animations_out,
            'in': animations_in,
        }


    def get_transition_to(self, animations_out, animation_name):
        animation = self.getAnimation(animation_name)
        typ = animation.typ
        transition = self.transitions['by_type']['default']

        typ_split = typ.split('_')
        for ti in range(len(typ_split)-1, -1, -1):
            t = '_'.join(typ_split[:ti+1])
            if t in self.transitions['by_type']:
                transition = self.transitions['by_type'][t]
                break
        
        for out_name in animations_out:
            if self.transitions and out_name in self.transitions:
                if animation_name in self.transitions[out_name]:
                    transition = self.transitions[out_name][animation_name]
        return transition

    def register_transition(self, transition: Transition):
        id_size = 10
        transition_id = get_random_string(id_size)
        while transition_id in self.transitions_active:
            transition_id = get_random_string(id_size)
        self.transitions_active[transition_id] = transition
        return transition_id

    def change_animation(self, animation_name, typ, time_counter, transition: Transition|None=None):
        logger.info(f"Animation change {animation_name}")
        if (typ in self.playing_types):
            if (animation_name in self.playing_types[typ]):
                logger.info(f"Animation already playing {animation_name}")
                # TODO refresh time_to_expire
                return
            else: 
                if typ in self.transitioning:
                    if animation_name in self.transitioning[typ]['in']:
                        logger.info(f"Animation already transitioning {animation_name}")
                        return
                animations_out = self.playing_types[typ]
                if not transition:
                    transition = self.get_transition_to(animations_out, animation_name)
            
                transition = copy.deepcopy(transition)
                transition.doStart(time_counter)
                transition_id = self.register_transition(transition)
                logger.info(f"transition_id {transition_id} duration {transition.cd.getDuration()} Animation {animation_name}  isStarted {transition.cd.isStarted()}   {self.transitions_active}")
                self.add_transition(typ, {animation_name: transition_id}, {a_name: transition_id for a_name in animations_out})
                self.playing_types[typ].add(animation_name)
                logger.info(f"self.playing_types[{typ}] = {self.playing_types[typ]}")
    

    def change_sentiments(self, sentiments):
        # get which sentiments are new, which are not in the new, and which remain
        new = set(sentiments.keys()) - set(self.sentiments.keys()) 
        removed = set(self.sentiments.keys()) - set(sentiments.keys())
        common = set(self.sentiments.keys()).intersection(set(sentiments.keys()))

        for s_name in removed:
            if s_name in self.sentiment_animations.keys():
                for a_name, a in self.sentiment_animations[s_name].items():
                    logger.info(f"Sentiment removed animation {a_name}")
                    a.weight = 0.0

        for s_name in [*new, *common]:
            if s_name in self.sentiment_animations.keys():
                for a_name, a in self.sentiment_animations[s_name].items():
                    logger.info(f"Sentiment changed animation {a_name}.weight = {sentiments[s_name]}")
                    a.weight = sentiments[s_name]

        self.sentiments = sentiments

    def update(self, time_counter = None):
        elapsed, time_counter = self.updateElapsed(time_counter)

        for typ, sc in self.state_changes.items():
            animation_name = sc.update(time_counter)
            if animation_name:
                self.change_animation(animation_name, typ, time_counter)
        
        to_del = {}
        for typ, t in self.transitioning.items():
            to_del[typ] = {'out': set(), 'in': set()}
            weights_cache = {}
            def getW(transition_id):
                if transition_id not in weights_cache:
                    transition = self.transitions_active[transition_id]
                    weight = transition.get(time_counter)
                    weights_cache[transition_id] = weight
                else:
                    weight = weights_cache[transition_id]
                return weight

            for animation_name, transition_id in t['out'].items():
                weight = getW(transition_id)
                weight_new = 1.0 - weight
                self.getAnimation(animation_name).weight = weight_new
                if weight_new <= 0.0:
                    # Finished transition
                    to_del[typ]['out'].add(animation_name)
                    changer_typ = self.animation_name_to_changer_typ[animation_name]
                    self.playing_types[changer_typ].remove(animation_name)
                    logger.info(f"Finished transition {animation_name}  self.playing_types[{changer_typ}] = {self.playing_types[changer_typ]}")
            
            for animation_name, transition_id in t['in'].items():
                weight = getW(transition_id)
                self.getAnimation(animation_name).weight = weight
                if weight >= 1.0:
                    # Finished transition
                    to_del[typ]['in'].add(animation_name)

        for typ, t in to_del.items():
            for tr_typ, names in t.items():
                for name in names:
                    logger.info(f"del self.transitioning[{typ}][{tr_typ}][{name}]")
                    del self.transitioning[typ][tr_typ][name]
            
            # Finished all transitions
            if not self.transitioning[typ]['out'].keys() and not self.transitioning[typ]['in'].keys():
                logger.info(f"Finished all transitions {typ}")
                del self.transitioning[typ]
                to_del_t = []
                for t_id, t in self.transitions_active.items():
                    if t.cd.getRemaining(time_counter) <= 0:
                        to_del_t.append(t_id)
                for t_id in to_del_t:
                    del self.transitions_active[t_id]



        for name, a in self.animations.items():
            a.update(elapsed, time_counter)

        return self.get()

    def get(self):
        parameters = self.get_parameters_neutral()
        for name, a in self.getAnimationsActive():
            parameters += a.get()
        return parameters

    def getAnimationsActive(self):
        # TODO change entire code to keep track of these in `self.playing_types` instead
        for name, a in self.animations.items():
            if a.isActive():
                yield name, a

    def updateElapsed(self, time_counter = None):
        if not time_counter:
            time_counter = time.perf_counter()
        elapsed = time_counter - self.time_counter_last
        self.time_counter_last = time_counter
        return elapsed, time_counter
    
    def getAnimation(self, animation_name):
        return self.animations[animation_name]

    @abstractmethod
    def get_parameters_neutral(self) -> AParameters:
        """
        This should return a neutral, 0 equivalent, state that animations will add to
        """
        pass
