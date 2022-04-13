import torch
import torch.nn as nn
import torch.nn.functional as F


# define the actor network
class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.max_action = args.high_action
        self.device = "cpu"
        self.input_size = args.obs_shape[agent_id]
        self.hidden_size = 64
        self.num_layers = 1
        self.output_size = args.action_shape[agent_id]
        self.num_directions = 1  # 单向LSTM
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True,
                            bidirectional=False)
        self.linear = nn.Sequential(
            nn.Linear(self.hidden_size, self.output_size),
            nn.Tanh()
        )

    def forward(self, x):
        #print("idea1MADDPG")
        #print("\n x.shape", x.shape)
        batch_size = len(x)  # 获取batch-size
        #print("batch-size:", batch_size)
        x = x.reshape(1, batch_size, self.input_size)  #
        #print("reshape x, x.reshape:", x.shape)
        h_0 = torch.randn(self.num_directions * self.num_layers, \
                          batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_directions * self.num_layers, \
                          batch_size, self.hidden_size).to(self.device)

        # input_seq = input_seq.view(self.batch_size, seq_len, 1)  # (5, 30, 1)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(x)  #, (h_0, c_0) output(1, 30, 64)
        output = output.reshape(batch_size, self.hidden_size)
        #print("the shape of output:", output.shape)
        actions = self.max_action * self.linear(output)
        #print("the shape of actions:", actions.shape)
        return actions


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.max_action = args.high_action
        self.critic_net = nn.Sequential(
            nn.Linear(sum(args.obs_shape) + sum(args.action_shape), 64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action):
        state = torch.cat(state, dim=1)
        for i in range(len(action)):
            action[i] /= self.max_action
        action = torch.cat(action, dim=1)
        x = torch.cat([state, action], dim=1)
        return self.critic_net(x)
