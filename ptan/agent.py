"""
Agent is something which converts states into actions and has state
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F

import ptan.actions as ptan_acts


class BaseAgent:
    """
    Abstract Agent interface
    """

    def initial_state(self):
        """
        Should create initial empty state for the agent. It will be called for the start of the episode
        :return: Anything agent want to remember
        """
        return None

    def __call__(self, states, agent_states):
        """
        Convert observations and states into actions to take
        :param states: list of environment states to process
        :param agent_states: list of states with the same length as observations
        :return: tuple of actions, states
        """
        assert isinstance(states, list)
        assert isinstance(agent_states, list)
        assert len(agent_states) == len(states)

        raise NotImplementedError


def default_states_preprocessor(states):
    """
    Convert list of states into the form suitable for model. By default we assume Variable
    :param states: list of numpy arrays with states
    :return: Variable
    """
    if len(states) == 1:
        np_states = np.expand_dims(states[0], 0)
    else:
        np_states = np.stack(states)

    return torch.tensor(np_states)


def float32_preprocessor(states):
    np_states = np.array(states, dtype=np.float32)
    return torch.tensor(np_states)


class DQNAgent(BaseAgent):
    """
    DQNAgent is a memoryless DQN agent which calculates Q values
    from the observations and  converts them into the actions using action_selector
    """

    def __init__(self, dqn_model, action_selector, device="cpu", preprocessor=default_states_preprocessor):
        self.dqn_model = dqn_model
        self.action_selector = action_selector
        self.preprocessor = preprocessor
        self.device = device

    @torch.no_grad()
    def __call__(self, states, agent_states=None):
        if agent_states is None:
            agent_states = [None] * len(states)
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        q_v = self.dqn_model(states)
        q = q_v.data.cpu().numpy()
        actions = self.action_selector(q)
        return actions, agent_states


class TargetNet:
    """
    Wrapper around model which provides copy of it instead of trained weights
    """

    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def alpha_sync(self, alpha):
        """
        Blend params of target net with params from the model
        :param alpha:
        """
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.target_model.load_state_dict(tgt_state)


class PolicyAgent(BaseAgent):
    """
    Policy agent gets action probabilities from the model and samples actions from it
    """

    # TODO: unify code with DQNAgent, as only action selector is differs.
    def __init__(self, model, action_selector=ptan_acts.ProbabilityActionSelector(), device="cpu",
                 apply_softmax=False, preprocessor=default_states_preprocessor, ):
        self.model = model
        self.action_selector = action_selector
        self.device = device
        self.apply_softmax = apply_softmax
        self.preprocessor = preprocessor

    @torch.no_grad()
    def __call__(self, states, agent_states=None):
        """
        Return actions from given list of states
        :param states: list of states
        :return: list of actions
        """
        if agent_states is None:
            agent_states = [None] * len(states)
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        probs_v = self.model(states)
        if self.apply_softmax:
            probs_v = F.softmax(probs_v, dim=1)
        probs = probs_v.data.cpu().numpy()
        actions = self.action_selector(probs)
        return np.array(actions), agent_states


class ActorCriticAgent(BaseAgent):
    """
    Policy agent which returns policy and value tensors from observations. Value are stored in agent's state
    and could be reused for rollouts calculations by ExperienceSource.
    """

    def __init__(self, model, action_selector=ptan_acts.ProbabilityActionSelector(), device="cpu",
                 apply_softmax=False, preprocessor=default_states_preprocessor, ):
        self.model = model
        self.action_selector = action_selector
        self.device = device
        self.apply_softmax = apply_softmax
        self.preprocessor = preprocessor

    @torch.no_grad()
    def __call__(self, states, agent_states=None):
        """
        Return actions from given list of states
        :param states: list of states
        :return: list of actions
        """
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        probs_v, values_v = self.model(states)
        if self.apply_softmax:
            probs_v = F.softmax(probs_v, dim=1)
        probs = probs_v.data.cpu().numpy()
        actions = self.action_selector(probs)
        agent_states = values_v.data.squeeze().cpu().numpy().tolist()
        return np.array(actions), agent_states


if __name__ == "__main__":

    # 创建一个包含不同类型数据的NumPy数组
    np_states = np.array([1, 2.5, "3", True, None], dtype=object)
    nps_states = np.array([1, 2.5, "a string", True, None], dtype=object)
    print("np_states:", np_states)
    # as_states = nps_states.astype(np.float32)
    # print("as_states:", as_states)

    # 创建一个指定长度和类型的数组
    _states = np.zeros(len(np_states), dtype=np.float32)

    # 将NumPy数组中的对象转换为浮点数类型（或其他标准数值类型）
    for i in range(len(np_states)):
        if isinstance(np_states[i], (int, float, bool)):  # 检查对象是否为整数、浮点数或布尔值
            np_states[i] = float(np_states[i])  # 将对象转换为浮点数
        else:
            np_states[i] = 0.0

    print("float np_states:", np_states)

    # 将NumPy数组转换为PyTorch张量
    torch_tensor = torch.tensor(np_states.astype(np.float32))

    # 检查转换后的张量
    print("torch_tensor:", torch_tensor)

    # torch_tensor2 = torch.from_numpy(np_states)
    # print("torch_tensor2:", torch_tensor2)

    va_states = np.vstack(np_states).astype(np.float32)
    print("va_states:", va_states)
    torch_tensor3 = torch.tensor(va_states)
    print("torch_tensor3:", torch_tensor3)

    # 创建一个数组列表，数组是列表的元素
    ls_states = [np_states]
    # 对数组列表沿指定维度（0表示外维）扩展维度
    dm_states = np.expand_dims(ls_states[0], 0)
    dm_states2 = np.expand_dims(ls_states, axis=0)
    print("len(ls_states):", len(ls_states))
    print("ls_states:", ls_states)
    print("ls_states[0]:", ls_states[0])
    print("len(ls_states[0]):", len(ls_states[0]))
    print("dm_states:", dm_states)
    print("dm_states2:", dm_states2)

    # 创建一维数组，
    np_states2 = np.array([5, 6, 7, False, ""], dtype=object)
    # 创建数组列表
    ls_states2 = [np_states, np_states2]
    # 将数组列表进行堆迭
    st_states = np.stack(ls_states2)
    print("ls_states2:", ls_states2)
    print("st_states:", st_states)
