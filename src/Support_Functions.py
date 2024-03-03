#------------------------------------------------------------------------------
# 0 - IMPORT DES LIBRAIRIES ET MODULES
#------------------------------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np
import random
from itertools import product
from tqdm import tqdm




#------------------------------------------------------------------------------
# 1 - DEFINITION DES FONCTIONS
#------------------------------------------------------------------------------
def greedy_action(model, state):
    device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = model(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()
    
def EpsilonGreedy_action(env, model, state, epsilon=0):
    if np.random.rand() < epsilon:
        action = env.action_space.sample()
    else:
        action = greedy_action(model, state)
    return action

def Fill_Buffer(env, state, buffer, n_sample, model='None', epsilon=0):
    '''
    Ici, l'argument "state" correspond au premier état à partir duquel on 
    commence à remplir le buffer. Pour initialiser un buffer, on peut faire
    en amont de l'appel de la fonction:
        "state, _ = env.reset()"
    '''
    # Initialisation
    state_bound_min = [1e6]*env.observation_space.shape[0]
    state_bound_max = [0]*env.observation_space.shape[0]
    if model=='None':
        epsilon = 1     # Action entièrement aléatoire
    
    for _ in range(n_sample):
        # Création de l'échantillon
        action = EpsilonGreedy_action(env, model, state, epsilon)
        next_state, reward, done, trunc, _ = env.step(action)
        buffer.append(state, action, reward, next_state, done)
        
        # Calcul des bornes
        for s in range(env.observation_space.shape[0]):
            if state[s]>state_bound_max[s]:
                state_bound_max[s] = state[s]
            if state[s]<state_bound_min[s]:
                state_bound_min[s] = state[s]
        
        # Gestion des transitions
        if done or trunc:
            state, _ = env.reset()
        else:
            state = next_state
            

    return buffer, state_bound_min, state_bound_max

def Normalize_States(State, Lower_bounds, Upper_bounds):
    Lower_bounds    = torch.tensor(Lower_bounds).to(State.device)
    Upper_bounds    = torch.tensor(Upper_bounds).to(State.device)
    Norm_state      = (State - Lower_bounds) / (Upper_bounds - Lower_bounds)
    
    return Norm_state
    
def run_model(env,model,max_step,epsilon=0, state_ini='None'):
    # Initialisation
    
    device  = "cuda" if next(model.parameters()).is_cuda else "cpu"
    buffer  = ReplayBuffer(max_step, device)
    state   = state_ini
    if state_ini=='None':
        state,_ = env.reset()
    
    histo = []
    episode_cum_reward = 0
    
    # Réalisation des trajectoires
    for _ in range(max_step):
        # Transition
        action = EpsilonGreedy_action(env, model, state, epsilon)
        next_state, reward, done, trunc, _ = env.step(action)
        buffer.append(state, action, reward, next_state, done)
        episode_cum_reward += reward
        histo += [episode_cum_reward]
        
        # Création de l'échantillon (1/2)
        if done or trunc:
            state, _ = env.reset()
            episode_cum_reward = 0
        else:
            state = next_state
    
    return buffer, histo
#------------------------------------------
# FONCTIONS CHEATEES - NON UTILISEES -
if False:     
    def optimal_1_step_action(env,state):
        '''
        On calcul l'action optimale, i.e. qui maximise la récompense pondérée sur 
        un pas de temps. On ressort le tuplet au fromat du Buffer pour intégration
        si on le souhaite.'
        '''
        best_reward = 0
        for action in range(env.action_space.n):
            next_state, reward,done, trunc, _ = env.step(action)
            if reward > best_reward:
                out_tuple = (state,action,reward,next_state,done)
                best_reward = reward
        return out_tuple
    
    def optimal_1_step_buffer(env,state,buffer):
        '''
        Ici on fournit l'actin optimale sur un horizon de un pas de temps en chaque 
        état fournit dans le buffer. On fournit un buffer de même taille que le 
        buffer d'entrée dans lequel on aura substitué chaque tuplet par le tuplet
        correspondant à l'action optimale sur 1 pas de temps.'
        '''
        buffer_out = buffer
        for i in range(len(buffer.data)):       
            buffer_out.data[i] = optimal_1_step_action(env,buffer.data[0])    
        return buffer_out
    
    def optimal_N_step(env,init_state,N,gamma):
        '''
        On calcule la suite d'actions "N-optimales", i.e. qui maximise la récompense 
        gamma-pondérée sur N pas de temps en partant de l'état "state'
        On ressort une liste de N tuplet du Buffer pour intégration si on le souhaite.'
        '''
        best_reward = 0
        list_action = list(range(env.action_space.n))
        seq_list         = list(product(list_action, repeat=N))     
        # seq_list fournit la combinaiaosn exhaustive des séquences de N-actions
        # chaque séquence est un N-uplet et seq_list fournit la liste des ces séquences
        
        for seq in seq_list:
            cum_reward = 0
            for i in len(seq):
                pass
        return             
                

#------------------------------------------
# POUR EVALUATION
def Evaluate_Ref_Patient(env, Best_models,step_per_epi):
    """
    On réalise une trajectoire par modèle contenu dans 'Best_models'. 
    On fournit les buffer de chaque trajectoire et l'historique des gains'.
    Il faut que le nombre de 'step_per_epi' soit égal au nombre max d'itération
    avant remise à zéro de l'environnement'
    """
    BUFFER      = [None]*len(Best_models)
    HISTO       = [None]*len(Best_models)
    best_reward = 0
    ind         = 0
    for i in tqdm(range(len(Best_models))):
        BUFFER[i], HISTO[i] = run_model(env,Best_models[i],
                                        step_per_epi, epsilon=0, state_ini='None')
        if HISTO[i][-1] > best_reward:
            best_reward = HISTO[i][-1]
            ind = i
        print(f"Modèle #{i} | cumulated rewards : {HISTO[i][-1]:.4e}")
    
    print('Meilleur modèle sur patient de référence:')
    print(f"Modèle #{ind} | cumulated rewards : {HISTO[ind][-1]:.4e}")
    
    return BUFFER, HISTO, ind

def Evaluate_Rand_Patient(Env_List, Best_models, step_per_epi):
    """
    On réalise une trajectoire par modèle contenu dans 'Best_models' et par environnement
    contenu dans 'Env_List'. 
    On fournit les buffer de chaque trajectoire et l'historique des gains'.
    Il faut que le nombre de 'step_per_epi' soit égal au nombre max d'itération
    avant remise à zéro de l'environnement'
    """

    nb_patient  = len(Env_List)
    BUFFER      = [[] for _ in range(len(Best_models))]
    HISTO       = [[] for _ in range(len(Best_models))]
    REWARD      = [[] for _ in range(len(Best_models))]
    
    best_reward = 0
    ind         = 0

    for i in tqdm(range(len(Best_models))):
        tmp_buffer  = []
        tmp_histo   = []
        tmp_reward  = []
        for j in range(nb_patient):
            # env = TimeLimit(HIVPatient(domain_randomization=True),
            #                 max_episode_steps=step_per_epi)
            env = Env_List[j]
            buffer_result, histo_result = run_model(env, Best_models[i],
                                                    step_per_epi, epsilon=0, state_ini='None')
            tmp_buffer.append(buffer_result)
            tmp_histo.append(histo_result)
            tmp_reward.append(histo_result[-1])
    
        BUFFER[i]   = tmp_buffer
        HISTO[i]    = tmp_histo
        REWARD[i]   = tmp_reward
        print(f'Model #{i}  |  sum of cumulated rewards on random patient : {sum(REWARD[i]):.4e}')
        print(f'Model #{i}  | mean of cumulated rewards on random patient : {sum(REWARD[i]) / len(REWARD[i]):.4e}')
        print(f'Model #{i}  | min  of cumulated rewards on random patient : {min(REWARD[i]):.4e}')
        print(f'Model #{i}  | max  of cumulated rewards on random patient : {max(REWARD[i]):.4e}')

        if sum(REWARD[i]) > best_reward:
            best_reward = sum(REWARD[i])
            ind = i
    
    print(f'Meilleur modèle sur {nb_patient} patients:')
    print(f'Model #{i}  |  sum of cumulated rewards on random patient : {sum(REWARD[ind]):.4e}')
    print(f'Model #{i}  | mean of cumulated rewards on random patient : {sum(REWARD[ind]) / len(REWARD[ind]):.4e}')
    print(f'Model #{i}  | min  of cumulated rewards on random patient : {min(REWARD[ind]):.4e}')
    print(f'Model #{i}  | max  of cumulated rewards on random patient : {max(REWARD[ind]):.4e}')

    return BUFFER, HISTO, ind



#------------------------------------------------------------------------------
# 2 - DEFINITION DES CLASSES
#------------------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers=2, hidden_size=24):
        super(QNetwork, self).__init__()
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        # Couche d'entrée
        self.input_layer = nn.Linear(state_dim, hidden_size)
        # Couches cachées
        self.hidden_layers = nn.ModuleList()
        for _ in range(hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        # Couche de sortie
        self.output_layer = nn.Linear(hidden_size, action_dim)
    
    def forward(self, x):
        x = nn.functional.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = nn.functional.relu(layer(x))
        x = self.output_layer(x)
        return x
    
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity   = capacity
        self.data       = []
        self.index      = 0         # L'indice où l'on va insérer la nouvelle donnée
        self.device     = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)  # Création du nouvel emplacement
        self.data[self.index] = (s, a, r, s_, d)
        self.index = int((self.index + 1) % self.capacity)
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def concat(self, new_buffer):
        self.data += [new_buffer.data]
        self.capacity += len(new_buffer.data)
    def __len__(self):
        return len(self.data)







