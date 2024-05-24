import logging
import os
import statistics
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dreamerv2.models.dense import DenseModel

from dreamer.modules.actor import Actor
from dreamer.modules.critic import Critic
from dreamer.modules.decoder import Decoder
from dreamer.modules.encoder import Encoder
from dreamer.modules.model import RSSM, ContinueModel, RewardModel
from dreamer.utils.buffer import ReplayBuffer
from dreamer.utils.utils import DynamicInfos, compute_lambda_values, create_normal_dist

logging.basicConfig(level=logging.INFO)


class Dreamer:
    def __init__(
        self,
        observation_shape,
        discrete_action_bool,
        action_size,
        writer,
        device,
        config,
    ):
        self.device = device
        self.action_size = action_size
        self.discrete_action_bool = discrete_action_bool

        if config.environment.benchmark != "isaacgym":
            self.encoder = Encoder(observation_shape, config).to(self.device)
            self.decoder = Decoder(observation_shape, config).to(self.device)
        else:
            self.modelstate_size = int(
                config.parameters.dreamer.deterministic_size
            ) + int(config.parameters.dreamer.stochastic_size)
            obs_encoder: dict = {
                "layers": 2,
                "node_size": 256,
                "dist": None,
                "activation": nn.ELU,
            }
            obs_decoder: dict = {
                "layers": 2,
                "node_size": 256,
                "dist": "normal",
                "activation": nn.ELU,
            }
            # creates latent of an observation
            self.encoder = DenseModel(
                (config.parameters.dreamer.embedded_state_size,),
                int(np.prod(observation_shape)),
                obs_encoder,
            ).to(self.device)
            # reconstructs observation from latent
            self.decoder = DenseModel(
                observation_shape,
                self.modelstate_size,
                obs_decoder,
            ).to(self.device)
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size
        self.rssm = RSSM(
            action_size,
            self.stochastic_size,
            self.deterministic_size,
            self.device,
            recurrent_model_config={
                "hidden_size": config.parameters.dreamer.rssm.recurrent_model.hidden_size,
                "activation": config.parameters.dreamer.rssm.recurrent_model.activation,
            },
            transition_model_config={
                "hidden_size": config.parameters.dreamer.rssm.transition_model.hidden_size,
                "num_layers": config.parameters.dreamer.rssm.transition_model.num_layers,
                "activation": config.parameters.dreamer.rssm.transition_model.activation,
                "min_std": config.parameters.dreamer.rssm.transition_model.min_std,
            },
            representation_model_config={
                "embedded_state_size": config.parameters.dreamer.embedded_state_size,
                "hidden_size": config.parameters.dreamer.rssm.representation_model.hidden_size,
                "num_layers": config.parameters.dreamer.rssm.representation_model.num_layers,
                "activation": config.parameters.dreamer.rssm.representation_model.activation,
                "min_std": config.parameters.dreamer.rssm.representation_model.min_std,
            },
        ).to(self.device)
        self.reward_predictor = RewardModel(
            stochastic_size=self.stochastic_size,
            deterministic_size=self.deterministic_size,
            hidden_size=config.parameters.dreamer.reward.hidden_size,
            num_layers=config.parameters.dreamer.reward.num_layers,
            activation=config.parameters.dreamer.reward.activation,
        ).to(self.device)
        # if config.parameters.dreamer.use_continue_flag:
        #     self.continue_predictor = ContinueModel(config).to(self.device)
        self.actor = Actor(
            discrete_action_bool=discrete_action_bool,
            action_size=action_size,
            stochastic_size=self.stochastic_size,
            deterministic_size=self.deterministic_size,
            hidden_size=config.parameters.dreamer.agent.actor.hidden_size,
            num_layers=config.parameters.dreamer.agent.actor.num_layers,
            activation=config.parameters.dreamer.agent.actor.activation,
            mean_scale=config.parameters.dreamer.agent.actor.mean_scale,
            init_std=config.parameters.dreamer.agent.actor.min_std,
            min_std=config.parameters.dreamer.agent.actor.init_std,
        ).to(self.device)
        self.critic = Critic(
            stochastic_size=self.stochastic_size,
            deterministic_size=self.deterministic_size,
            hidden_size=config.parameters.dreamer.agent.critic.hidden_size,
            num_layers=config.parameters.dreamer.agent.critic.num_layers,
            activation=config.parameters.dreamer.agent.critic.activation,
        ).to(self.device)

        self.buffer = ReplayBuffer(observation_shape, action_size, self.device, config)

        self.config = config.parameters.dreamer

        # optimizer
        self.model_params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.rssm.parameters())
            + list(self.reward_predictor.parameters())
        )
        if self.config.use_continue_flag:
            self.model_params += list(self.continue_predictor.parameters())

        self.model_optimizer = torch.optim.Adam(
            self.model_params, lr=self.config.model_learning_rate
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.critic_learning_rate
        )

        self.continue_criterion = nn.BCELoss()

        self.dynamic_learning_infos = DynamicInfos(self.device)
        self.behavior_learning_infos = DynamicInfos(self.device)

        self.writer = writer
        self.num_total_episode = 0

    def train(self, env):
        self.tot_timesteps = 0
        self.tot_time = 0
        self.num_steps_per_env = (
            # not fully correct. there is a while not done loop in environment_interaction
            self.config.num_interaction_episodes
            * self.config.batch_length
        )
        self.num_envs = 1
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)

        if len(self.buffer) < 1:
            interaction_info = self.environment_interaction(
                env, self.config.seed_episodes
            )

        for iteration in range(self.config.train_iterations):
            # at every epoch
            behavior_loss = defaultdict(list)
            dynamic_loss = defaultdict(list)
            # we have a fixed number of times we sample from the buffer
            # and update world model
            # and update actor-critic
            start = time.time()
            for collect_interval in range(self.config.collect_interval):
                # sampled batch_size trajs at random with length batch_length
                data = self.buffer.sample(
                    self.config.batch_size, self.config.batch_length
                )
                dynamic_learning_res = self.dynamic_learning(data)
                posteriors = dynamic_learning_res["posteriors"]
                deterministics = dynamic_learning_res["deterministics"]
                dynamic_losses = dynamic_learning_res["losses"]
                behavior_losses = self.behavior_learning(posteriors, deterministics)
                # dynamic_loss.update([dynamic_losses])
                # behavior_loss.update([behavior_losses])
                for k, v in dynamic_losses.items():
                    dynamic_loss[k].append(v)
                for k, v in behavior_losses.items():
                    behavior_loss[k].append(v)
            learn_time = time.time() - start

            # for k, v in {**dynamic_loss, **behavior_loss}.items():
            #     logging.debug(f"{k}\n{pd.Series(v).describe()}\n")

            start = time.time()
            interaction_info = self.environment_interaction(
                env, self.config.num_interaction_episodes
            )
            rewbuffer.extend(interaction_info["rewbuffer"])
            lenbuffer.extend(interaction_info["lenbuffer"])
            ep_infos.extend(interaction_info["ep_infos"])
            collection_time = time.time() - start
            # reward = self.evaluate(env)
            # rewbuffer.append(reward)
            locs = {
                # **dynamic_loss,
                # **behavior_loss,
                "losses": {**dynamic_loss, **behavior_loss},
                "rewbuffer": rewbuffer,
                "lenbuffer": lenbuffer,
                "ep_infos": ep_infos,
                "it": iteration,
                "collection_time": collection_time,
                "learn_time": learn_time,
                "num_learning_iterations": self.config.train_iterations,
            }
            self.log(locs)

            # save model every 10 iterations
            if (
                iteration + 1
            ) % self.config.pipe.save_interval == 0 and self.log_dir is not None:
                save_path = os.path.join(
                    self.log_dir,
                    "model_{}.pt".format(iteration),
                )
                self.save(save_path)
                print(f"{save_path=}")

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # mean_std = self.alg.actor.std.mean()
        mean_std = self.actor.std.mean()
        fps = int(
            self.num_steps_per_env
            * self.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        ep_string = f""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                if self.log_dir is not None:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        if self.log_dir is not None:
            # self.writer.add_scalar(
            #     "Loss/learning_rate", self.alg.learning_rate, locs["it"]
            # )
            self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
            # self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
            self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
            self.writer.add_scalar(
                "Perf/collection time", locs["collection_time"], locs["it"]
            )
            self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
            if len(locs["rewbuffer"]) > 0:
                self.writer.add_scalar(
                    "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
                )
                self.writer.add_scalar(
                    "Train/mean_episode_length",
                    statistics.mean(locs["lenbuffer"]),
                    locs["it"],
                )
                self.writer.add_scalar(
                    "Train/mean_reward/time",
                    statistics.mean(locs["rewbuffer"]),
                    self.tot_time,
                )
                self.writer.add_scalar(
                    "Train/mean_episode_length/time",
                    statistics.mean(locs["lenbuffer"]),
                    self.tot_time,
                )

        str = f" \033[1m Learning iteration {locs['it']}/{self.config.train_iterations} \033[0m"

        log_string = (
            f"""{'#' * width}\n"""
            f"""{str.center(width, ' ')}\n\n"""
            f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
            f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
        )
        if len(locs["rewbuffer"]) > 0:
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
        if len(locs["lenbuffer"]) > 0:
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""

        for k, v in locs["losses"].items():
            log_string += f"""{f'Mean {k}:':>{pad}} {statistics.mean(v):.4f}\n"""
            prefix = "Loss"
            if self.log_dir is not None:
                self.writer.add_scalar(f"{prefix}/{k}", statistics.mean(v), locs["it"])

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        torch.save(
            {
                "transition_model_state_dict": self.rssm.transition_model.state_dict(),
                "reward_predictor_state_dict": self.reward_predictor.state_dict(),
                "recurrent_state_model_state_dict": self.rssm.recurrent_model.state_dict(),
                "encoder_state_dict": self.encoder.state_dict(),
                "decoder_state_dict": self.decoder.state_dict(),
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "model_optimizer_state_dict": self.model_optimizer.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "infos": infos,
            },
            path,
        )

    def set_up_pipeline(self, log_dir, config, args):
        import yaml
        from legged_gym.utils.helpers import class_to_dict

        self.log_dir = log_dir

        with open(os.path.join(log_dir, "config.yaml"), "w") as f:
            yaml.dump(dict(config), f)
        with open(os.path.join(log_dir, "args.yaml"), "w") as f:
            args.physics_engine = "physx"
            yaml.dump(vars(args), f)
        # save entire source code
        legged_gym_path = Path(__file__).resolve().parents[1]
        os.system(
            f"rsync -a --exclude='logs' --exclude='configs' --exclude='resources' --exclude='__pycache__' {legged_gym_path} {log_dir}"
        )

    def evaluate(self, env):
        return self.environment_interaction(env, self.config.num_evaluate, train=False)

    def dynamic_learning(self, data):
        prior, deterministic = self.rssm.recurrent_model_input_init(len(data.action))

        data.embedded_observation = self.encoder(data.observation)

        for t in range(1, self.config.batch_length):
            deterministic = self.rssm.recurrent_model(
                prior, data.action[:, t - 1], deterministic
            )
            prior_dist, prior = self.rssm.transition_model(deterministic)
            posterior_dist, posterior = self.rssm.representation_model(
                data.embedded_observation[:, t], deterministic
            )

            self.dynamic_learning_infos.append(
                priors=prior,
                prior_dist_means=prior_dist.mean,
                prior_dist_stds=prior_dist.scale,
                posteriors=posterior,
                posterior_dist_means=posterior_dist.mean,
                posterior_dist_stds=posterior_dist.scale,
                deterministics=deterministic,
            )

            prior = posterior

        infos = self.dynamic_learning_infos.get_stacked()
        losses = self._model_update(data, infos)
        return {
            "posteriors": infos.posteriors.detach(),
            "deterministics": infos.deterministics.detach(),
            "losses": losses,
        }

    def _model_update(self, data, posterior_info):
        reconstructed_observation_dist = self.decoder(
            torch.cat(
                (
                    posterior_info.deterministics,
                    posterior_info.posteriors,
                ),
                dim=-1,
            )
        )
        reconstruction_observation_loss = reconstructed_observation_dist.log_prob(
            data.observation[:, 1:]
        )
        if self.config.use_continue_flag:
            continue_dist = self.continue_predictor(
                posterior_info.posteriors, posterior_info.deterministics
            )
            continue_loss = self.continue_criterion(
                continue_dist.probs, 1 - data.done[:, 1:]
            )

        reward_dist = self.reward_predictor(
            posterior_info.posteriors, posterior_info.deterministics
        )
        reward_loss = reward_dist.log_prob(data.reward[:, 1:])

        prior_dist = create_normal_dist(
            posterior_info.prior_dist_means,
            posterior_info.prior_dist_stds,
            event_shape=1,
        )
        posterior_dist = create_normal_dist(
            posterior_info.posterior_dist_means,
            posterior_info.posterior_dist_stds,
            event_shape=1,
        )
        kl_divergence_loss = torch.mean(
            torch.distributions.kl.kl_divergence(posterior_dist, prior_dist)
        )
        kl_divergence_loss = torch.max(
            torch.tensor(self.config.free_nats).to(self.device), kl_divergence_loss
        )
        model_loss = (
            self.config.kl_divergence_scale * kl_divergence_loss
            - reconstruction_observation_loss.mean()
            - reward_loss.mean()
        )
        if self.config.use_continue_flag:
            model_loss += continue_loss.mean()

        self.model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(
            self.model_params,
            self.config.clip_grad,
            norm_type=self.config.grad_norm_type,
        )
        self.model_optimizer.step()
        losses = {
            "reconstruction_observation_loss": reconstruction_observation_loss.mean().item(),
            "reward_loss": reward_loss.mean().item(),
            "kl_divergence_loss": kl_divergence_loss.item(),
            "model_loss": model_loss.item(),
        }
        return losses

    def behavior_learning(self, states, deterministics):
        """
        #TODO : last posterior truncation(last can be last step)
        posterior shape : (batch, timestep, stochastic)
        """
        state = states.reshape(-1, self.stochastic_size)
        deterministic = deterministics.reshape(-1, self.deterministic_size)

        # continue_predictor reinit
        for t in range(self.config.horizon_length):
            action = self.actor(state, deterministic)
            deterministic = self.rssm.recurrent_model(state, action, deterministic)
            _, state = self.rssm.transition_model(deterministic)
            self.behavior_learning_infos.append(
                priors=state, deterministics=deterministic
            )

        losses = self._agent_update(self.behavior_learning_infos.get_stacked())
        return losses

    def _agent_update(self, behavior_learning_infos):
        predicted_rewards = self.reward_predictor(
            behavior_learning_infos.priors, behavior_learning_infos.deterministics
        ).mean
        values = self.critic(
            behavior_learning_infos.priors, behavior_learning_infos.deterministics
        ).mean

        if self.config.use_continue_flag:
            continues = self.continue_predictor(
                behavior_learning_infos.priors, behavior_learning_infos.deterministics
            ).mean
        else:
            continues = self.config.discount * torch.ones_like(values)

        lambda_values = compute_lambda_values(
            predicted_rewards,
            values,
            continues,
            self.config.horizon_length,
            self.device,
            self.config.lambda_,
        )

        actor_loss = -torch.mean(lambda_values)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            self.config.clip_grad,
            norm_type=self.config.grad_norm_type,
        )
        self.actor_optimizer.step()

        value_dist = self.critic(
            behavior_learning_infos.priors.detach()[:, :-1],
            behavior_learning_infos.deterministics.detach()[:, :-1],
        )
        value_loss = -torch.mean(value_dist.log_prob(lambda_values.detach()))

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            self.config.clip_grad,
            norm_type=self.config.grad_norm_type,
        )
        self.critic_optimizer.step()

        losses = {"actor_loss": actor_loss.item(), "value_loss": value_loss.item()}
        return losses

    @torch.no_grad()
    def environment_interaction(self, env, num_interaction_episodes, train=True):
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(1, dtype=torch.float)
        cur_episode_length = torch.zeros(1, dtype=torch.float)
        score_lst = []

        for epi in range(num_interaction_episodes):
            posterior, deterministic = self.rssm.recurrent_model_input_init(1)
            action = torch.zeros(1, self.action_size).to(self.device)

            observation = env.reset()
            embedded_observation = self.encoder(
                torch.from_numpy(observation).float().to(self.device)
            )

            score = 0
            done = False

            while not done:
                deterministic = self.rssm.recurrent_model(
                    posterior, action, deterministic
                )
                embedded_observation = embedded_observation.reshape(1, -1)
                _, posterior = self.rssm.representation_model(
                    embedded_observation, deterministic
                )
                action = self.actor(posterior, deterministic).detach()

                if self.discrete_action_bool:
                    buffer_action = action.cpu().numpy()
                    env_action = buffer_action.argmax()

                else:
                    buffer_action = action.cpu().numpy()[0]
                    env_action = buffer_action

                next_observation, reward, done, info = env.step(env_action)
                if train:
                    self.buffer.add(
                        observation, buffer_action, reward, next_observation, done
                    )
                score += reward
                embedded_observation = self.encoder(
                    torch.from_numpy(next_observation).float().to(self.device)
                )
                observation = next_observation

                # Book keeping
                if "episode" in info:
                    ep_infos.append(info["episode"])
                cur_reward_sum += reward
                cur_episode_length += 1
                if done:
                    rewbuffer.append(cur_reward_sum.item())
                    lenbuffer.append(cur_episode_length.item())
                    cur_reward_sum = torch.zeros(1, dtype=torch.float)
                    cur_episode_length = torch.zeros(1, dtype=torch.float)
                # new_ids = (done > 0).nonzero(as_tuple=False)
                # rewbuffer.extend(cur_reward_sum[new_ids].tolist())
                # lenbuffer.extend(cur_episode_length[new_ids].tolist())
                # cur_reward_sum[new_ids] = 0
                # cur_episode_length[new_ids] = 0

                if done:
                    if train:
                        self.num_total_episode += 1
                        self.writer.add_scalar(
                            "training score", score, self.num_total_episode
                        )
                    else:
                        score_lst.append(score)
                    break

        if not train:
            evaluate_score = sum(score_lst) / len(score_lst)
            print("evaluate score : ", evaluate_score)
            self.writer.add_scalar("test score", evaluate_score, self.num_total_episode)
            return evaluate_score

        return {
            "rewbuffer": rewbuffer,
            "lenbuffer": lenbuffer,
            "ep_infos": ep_infos,
            "score": np.mean(score_lst),
        }

    @torch.no_grad()
    def do_inference(self, obs, num_actions, device=None):
        device = device if device is not None else self.device
        batch_size = obs.shape[0]
        if not hasattr(self, "did_one_iter"):
            self.did_one_iter = True
            # TODO: do init fresh once or at every subsequent inference step?
            self.prev_action = torch.zeros(batch_size, num_actions).to(device)
            _, self.prev_deterministic = self.rssm.recurrent_model_input_init(
                batch_size
            )

        embedded_observation = self.encoder(obs.to(device))
        _, posterior = self.rssm.representation_model(
            embedded_observation, self.prev_deterministic
        )

        deterministic = self.rssm.recurrent_model(
            posterior, self.prev_action, self.prev_deterministic
        )
        embedded_observation = embedded_observation.reshape(batch_size, -1)
        _, posterior = self.rssm.representation_model(
            embedded_observation, deterministic
        )
        action = self.actor(posterior, deterministic).detach()

        self.prev_deterministic = deterministic
        self.prev_action = action

        return action

    def get_inference_policy(self, device=None):
        self.actor.eval()
        if device is not None:
            self.actor.to(device)
        return self.do_inference
