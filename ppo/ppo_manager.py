from ppo.ppo import ppo


mpi_fork(args.cpu)  # run parallel code with mpi

from spinup.utils.run_utils import setup_logger_kwargs

logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

ppo(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
    ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma,
    seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
    logger_kwargs=logger_kwargs)