import torch
import os
from maddpg.encoder_decoder import ADHOC, TEAMMATE
import time
import torch.distributions as D
import numpy as np

class ODITS:
    def __init__(self, args, agent_id):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0

        # create the network
        self.actor_network = ADHOC(args, agent_id)
        self.teammate_network = TEAMMATE(args, agent_id)

        # # build up the target network
        # self.actor_target_network = Actor(args, agent_id)
        # self.critic_target_network = Critic(args)

        # load the weights into the target networks
        # self.actor_target_network.load_state_dict(self.adhoc_network.state_dict())
        # self.critic_target_network.load_state_dict(self.teammate_network.state_dict())

        # create the optimizer
        self.adhoc_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.teammate_optim = torch.optim.Adam(self.teammate_network.parameters(), lr=self.args.lr_critic)

        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'adhoc_agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # 加载模型
        if os.path.exists(self.model_path + '/adhoc_params.pkl'):
            self.actor_network.load_state_dict(torch.load(self.model_path + '/adhoc_params.pkl'))
            self.critic_network.load_state_dict(torch.load(self.model_path + '/teammate_params.pkl'))
            print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
                                                                          self.model_path + '/adhoc_params.pkl'))
            print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
                                                                           self.model_path + '/teammate_params.pkl'))

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.teammate_network.parameters(), self.teammate_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)


    def get_b(o_next, r, o, u):
        return torch.cat((o_next, r, o, u))
    # update the network
    def train(self, transitions, other_agents):
        # torch.autograd.set_detect_anomaly(True)
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        r = transitions['r_%d' % self.agent_id]  # 训练时只需要自己的reward
        o, u, o_next = [], [], []  # 用来装每个agent经验中的各项
        for agent_id in range(self.args.n_agents):
            o.append(transitions['o_%d' % agent_id])
            u.append(transitions['u_%d' % agent_id])
            o_next.append(transitions['o_next_%d' % agent_id])
            
        # calculate the target Q value function
        u_next = []
        # begin = time.time()
        with torch.no_grad():
            # 得到下一个状态对应的动作
            index = 0
            o_teammate = None
            a_teammate = None

            for agent_id in range(self.args.n_agents):
                if agent_id == self.agent_id:
                    # print("train", o_next[agent_id].size(), u[agent_id].size())
                    action, encoded_z = self.actor_network(o_next[agent_id], u[agent_id])
                    u_next.append(action)
                else:
                    # 因为传入的other_agents要比总数少一个，可能中间某个agent是当前agent，不能遍历去选择动作
                    action = other_agents[index].policy.actor_network(o_next[agent_id])
                    u_next.append(action)
                    if o_teammate is None:
                        o_teammate = o_next[agent_id]
                        a_teammate = u[agent_id]
                        # print(o_next[agent_id].size(), u[agent_id].size())
                    else:
                        o_teammate = torch.cat((o_teammate,o_next[agent_id]),1)
                        a_teammate = torch.cat((a_teammate,u[agent_id]),1)
                    # a_teammate.append(u[agent_id])
                    index += 1

            q_next = self.teammate_network(o_teammate, a_teammate, u_next[self.agent_id])[0].detach()

            target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()
        # print("with",time.time() - begin)
        # the q loss
        action, encoded_z = self.actor_network(o[self.agent_id], u[self.agent_id])
        o_teammate = None
        a_teammate = None
        for i in range(self.args.n_agents):
            if i == self.agent_id:
                continue
            if o_teammate is None:
                o_teammate = o[agent_id]
                a_teammate = u[agent_id]
                # print(o_next[agent_id].size(), u[agent_id].size())
            else:
                o_teammate = torch.cat((o_teammate,o[agent_id]),1)
                a_teammate = torch.cat((a_teammate,u[agent_id]),1)
        q_value, encoded_c = self.teammate_network(o_teammate, a_teammate, u[self.agent_id].unsqueeze(1))
        # print("second", time.time() - begin)
        # q_value = self.critic_network(o, u)
        # the actor loss
        # 重新选择联合动作中当前agent的动作，其他agent的动作不变
        # begin = time.time()
        u[self.agent_id] = self.actor_network(o[self.agent_id], u[self.agent_id])[0].squeeze(1)
        action, encoded_z = self.actor_network(o[self.agent_id], u[self.agent_id])
        q_value, encoded_c = self.teammate_network(o_teammate, a_teammate, u[self.agent_id].unsqueeze(1))
        # print("3rd", time.time() - begin)
        self.teammate_optim.zero_grad()
        self.adhoc_optim.zero_grad()
        
        # begin = time.time()
        # critic_loss.backward()

        z = encoded_z.clone()
        c = encoded_c.clone()
        z[:, -1:] = torch.clamp(torch.exp(z[:, -1:]), min=0.002)
        c[:, -1:] = torch.clamp(torch.exp(c[:, -1:]), min=0.002)

        #                                       1 as latent space dim
        gaussian_proxy = D.Normal(z[:, :1], (z[:, 1:]) ** (1 / 2))
        # latent_z = gaussian_proxy.rsample()
        gaussian_team = D.Normal(c[:, :1], (c[:, 1:]) ** (1 / 2))
        # latent_c = gaussian_team.rsample()
        critic_loss = (target_q - q_value).pow(2).mean()
        loss = gaussian_team.entropy().sum(dim=-1).mean() * 0.0001 + D.kl_divergence(gaussian_proxy, gaussian_team).sum(dim=-1).mean() * 0.0001   # CE = H + KL
        loss = torch.clamp(loss, max=2e3) + critic_loss
        # critic_loss.backward()
        loss.backward()
        self.teammate_optim.step()
        self.adhoc_optim.step()
        # print("bp", time.time() - begin)
        
        

        self._soft_update_target_network()
        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        self.train_step += 1

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor_network.state_dict(), model_path + '/' + num + '_actor_params.pkl')
        torch.save(self.critic_network.state_dict(),  model_path + '/' + num + '_critic_params.pkl')


