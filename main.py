from ai import *
from entities import Match, Map, Agent, Trainer

import pygad.torchga, torch

import argparse, importlib, os, math

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='AI Project',
                    description='A project of unsupervised learning, for the AI class of UESPI-Floriano'
                    )
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-gens', '--generations', type=int, default=1000)
    parser.add_argument('-pars', '--parents', type=int, default=5)

    parser.add_argument('-p', '--presentation', action='store_true')
    parser.add_argument('-s', '--sleep', type=float, default=0.005)
    
    parser.add_argument('-t0m', '--team_0_module', type=str, default='random_ai')
    parser.add_argument('-t0c', '--team_0_class', type=str, default='RandomAI')
    parser.add_argument('-l0', '--load_0', type=str, default=None)
    
    parser.add_argument('-t1m', '--team_1_module', type=str, default='random_ai')
    parser.add_argument('-t1c', '--team_1_class', type=str, default='RandomAI')
    parser.add_argument('-l1', '--load_1', type=str, default=None)

    parser.add_argument('-mw', '--max_width', type=int, default=80)
    parser.add_argument('-mh', '--max_height', type=int, default=40)

    args = parser.parse_args()

    Map.MAX_WIDTH = args.max_width
    Map.MAX_HEIGHT = args.max_height

    class0 = getattr(importlib.import_module(f"ai.{args.team_0_module}", "ai"), args.team_0_class)
    model0: torch.nn.Module = class0(0)
    if not args.load_0 is None and os.path.isfile(args.load_0):
        state_dict = torch.load(args.load_0)
        model0.load_state_dict(state_dict)
    
    class1 = getattr(importlib.import_module(f"ai.{args.team_1_module}", "ai"), args.team_1_class)
    model1: torch.nn.Module = class1(1)
    if not args.load_1 is None and os.path.isfile(args.load_1):
        state_dict = torch.load(args.load_1)
        model1.load_state_dict(state_dict)
    
    if args.train: 
        t = Trainer(model0, model1)
        t.train(args.generations, args.parents)
    else:
        m = Match(3, model0, model1, presentation=args.presentation, sleep_time=args.sleep) 
        m.play()