import torch
import torch.nn as nn
import torch.nn.functional as F

class ADHOC(nn.Module):
    """Online Ad Hoc Teamwork Under Partial Observability

    Args:
        nn (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    def __init__(self, args, agent_id):
        super(ADHOC, self).__init__()
        
        self.args = args
        # print(args)
        
        self.agent_id = agent_id
        self.train_step = 0
        # self.dim_actions = args.dim_actions
        self.num_actions =  args.action_shape[agent_id]
        self.obs_size = args.obs_shape[agent_id]
        #on paper 10 for save the city, other 1
        self.z = 1
        
        
        #proxy
        self.p_fc1 = nn.Linear(args.obs_shape[agent_id], 128)
        self.p_bn1 = nn.BatchNorm1d(128)
        self.p_LRU1 = nn.LeakyReLU()
        self.p_LRU2 = nn.LeakyReLU()
        self.p_fc2 = nn.Linear(128, 2 * self.z)
        self.p_bn2 = nn.BatchNorm1d(2 * self.z)
        
        self.proxy_decoder = nn.Sequential(
            nn.Linear(2*self.z,64 * (self.num_actions + 1))
            # ,
            # nn.BatchNorm1d(64 * (self.num_actions + 1)),
        )
        
        self.m_weight = nn.Linear(64 * ( self.num_actions + 1), 64 * self.num_actions)
        self.m_bias = nn.Linear(64 * (self.num_actions + 1), self.num_actions)
        """
        w1 = torch.abs(self.m_weight(output of proxy encoder))
        b1 = self.m_bias(output of proxy encoder)
        w1 = w1.view(-1, 1, self.action?)
        b1 = b1.view(-1, 1, self.action?)
        output = torch.nn.functionl.elu(torch.hmm(output of gru, w1) + b1)
        """
        
        self.marginal = nn.Sequential(
            nn.Linear(self.num_actions + self.obs_size, 64),
            nn.ReLU())
        self.marginal_gru = nn.GRU(64,64)
        self.marginal_fc =  nn.Linear(64, self.num_actions)
        
                
    def forward(self, b, o_a):
        #local transition oit, rt-1, ai t-1, oi t-1
        # print(b, o_a)
        
        # o_a = torch.tensor(o_a, dtype=torch.float32).unsqueeze(0)
        
        
        tau = torch.cat([b, o_a], dim = 1)
        # print(b)
        # bi = x
        #proxy
        tmp = self.p_fc1(b)
        # print(obs[:,0,:].size())
        # tmp = self.p_bn1(tmp)
        tmp = self.p_LRU1(tmp)
        tmp = self.p_fc2(tmp)
        # encoded_z = self.p_bn2(tmp)
        encoded_z = self.p_LRU2(tmp)
        
        decoded = self.proxy_decoder(encoded_z)
        
        w1 = torch.abs(self.m_weight(decoded))
        w1 = w1.view(-1,64,5)
        b1 = self.m_bias(decoded)
        
        marginal_state = self.marginal(tau)
        # print(tau.size(), marginal_state.size())
        marginal_state, hh = self.marginal_gru(marginal_state)
        marginal_state = marginal_state.unsqueeze(1)
        # w1 = w1.unsqueeze(0)
        # print(marginal_state.size(), w1.size(), b1.size())
        # marginal_state = marginal_state.view(-1,1, self.)
        b1 = b1.unsqueeze(1)
        action = torch.bmm(marginal_state, w1)        
        action = F.elu(action + b1)
        
        # print(self.arg)

        return action, encoded_z


class TEAMMATE(nn.Module):
    """Online Ad Hoc Teamwork Under Partial Observability

    Args:
        nn (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    def __init__(self, args, agent_id):
        super(TEAMMATE, self).__init__()
        
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0
        # self.dim_actions = args.dim_actions
        self.num_actions =  args.action_shape[agent_id]
        self.obs_size = args.obs_shape[agent_id]
        #on paper 10 for save the city, other 1
        self.c = 1
        
        
        #proxy
        self.team_encoder = nn.Sequential(
            nn.Linear((self.num_actions + self.obs_size) * (args.n_agents - 1), 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 2 * self.c),
            nn.BatchNorm1d(2 * self.c),
            nn.LeakyReLU()
        )
        self.t_fc1 = nn.Linear((self.num_actions + self.obs_size) * (args.n_agents - 1), 128)
        # self.t_bn1 = nn.BatchNorm1d(128)
        # self.t_LRU1 = nn.LeakyReLU()
        # self.t_fc2 = nn.Linear(128, 2 * self.c)
        # self.t_bn2 = nn.BatchNorm1d(2 * self.c)
        
        self.team_decoder = nn.Sequential(
            nn.Linear(2*self.c,16768),
            nn.BatchNorm1d(16768)
        )
        self.int_fc1 = nn.Linear(self.num_actions, 128)
        self.q_pred = nn.Linear(128, 1)
        
        
        self.m_w1 = nn.Linear(16768, 128 * self.num_actions)
        self.m_b1 = nn.Linear(16768, 128)
        
        self.m_w2 = nn.Linear(16768, 128)
        self.m_b2 = nn.Linear(16768, 1)
        self.ReLu = nn.ReLU()
        """
        w1 = torch.abs(self.m_weight(output of proxy encoder))
        b1 = self.m_bias(output of proxy encoder)
        w1 = w1.view(-1, 1, self.action?)
        b1 = b1.view(-1, 1, self.action?)
        output = torch.nn.functionl.elu(torch.hmm(output of gru, w1) + b1)
        """

                
    def forward(self, s, a_, ui):
        
        #local transition oit, rt-1, ai t-1, oi t-1
        
        bi = torch.cat((s,a_), 1)
        
        # bi = bi.unsqueeze(1)
        
        # out = self.t_fc1(bi)
        # print(out.size())
        encoded_c = self.team_encoder(bi)
        #proxy
        
        
        decoded = self.team_decoder(encoded_c)
        decoded = torch.abs(decoded)
        
         
        w1 = self.m_w1(decoded)
        b1 = self.m_b1(decoded)
        w1 = w1.view(self.args.batch_size,self.num_actions,-1)
        b1 = b1.unsqueeze(1)
        w2 = self.m_w2(decoded)
        w2 = w2.view(self.args.batch_size,128,-1)
        b2 = self.m_b2(decoded)
        b2 = b2.unsqueeze(1)

        
        # fc1 = self.int_fc1(ui)
        
        # print(ui.size(),fc1.size(), w1.size(), b1.size())
        fc1 = F.elu(torch.bmm(ui, w1) + b1)
        fc1 = self.ReLu(fc1)
        # print(fc1.size(), w2.size(), b2.size())
        q = F.elu(torch.bmm(fc1, w2) + b2)
        
        return q, encoded_c