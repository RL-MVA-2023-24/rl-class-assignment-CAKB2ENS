# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 00:00:45 2024

@author: Admin
"""

from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')
import os
import random
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import pickle
from tqdm import tqdm

#------------------------------------------------------------------------------
# Initialisation de l'environnement
from env_hiv import HIVPatient 
from gymnasium.wrappers import TimeLimit

from Support_Functions import greedy_action, EpsilonGreedy_action, Fill_Buffer, Normalize_States, run_model
from Support_Functions import Evaluate_Ref_Patient, Evaluate_Rand_Patient
from Support_Functions import QNetwork, ReplayBuffer

# env = TimeLimit(
#     env=HIVPatient(domain_randomization=False), max_episode_steps=200)



#------------------------------------------------------------------------------
class Agent_DQN_v3:
    def __init__(self):       
        self.Actor_acting_model         = []
        self.Actor_model_list           = []
        self.Actor_action_list          = []
        self.Actor_normaliztion_list    = []
        
        self.Actor_acting_model         = []
        self.Critic_model_list          = []
      

    def L_extract_config(self,config):
        # Création des variables de définition de la policy
        self.epsilon_max    = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min    = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop   = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay  = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step   = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        
        # Création des variables liées à l'apprentissage
        device      = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.memory = ReplayBuffer(buffer_size,device)
        
        self.gamma  = config['gamma'] if 'gamma' in config.keys() else 0.95
        
        lr              = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.criterion  = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        self.optimizer  = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        
        self.nb_gradient_steps      = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq     = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau      = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
        
        del device, buffer_size, lr
        
    def L_delete_config(self):
        # Suppression des variables de définition de la policy
        del self.epsilon_max, self.epsilon_min, self.epsilon_stop, self.epsilon_delay, self.epsilon_step
        
        # Suppression des variables de définition de l'apprentissage
        del self.memory, self.gamma,
        del self.criterion, self.optimizer, self.batch_size
        del self.nb_gradient_steps, self.update_target_strategy, self.update_target_freq, self.update_target_tau
        
    def L_save_training(self,path,Ref_name):
        torch.save(self.model_list_save,os.path.join(path,Ref_name+"_Model_List.pth"))
        with open(os.path.join(path,Ref_name+'_Batch_List.pkl'), 'wb') as f:
            pickle.dump(self.batch_list_save, f)
        with open(os.path.join(path,Ref_name+'_Bound_List.pkl'), 'wb') as f:
            pickle.dump(self.bound_list_save, f)
            
        print(f'sauvegarde : {os.path.join(path,Ref_name+"_Model_List.pth")}')
        print(f'sauvegarde : {os.path.join(path,Ref_name+"_Batch_List.pkl")}')
        print(f'sauvegarde : {os.path.join(path,Ref_name+"_Bound_List.pkl")}')
        
    def save(self,path):   
        pass

    
    def load(self): # RESTE A AFFECTER SES DONNEES A DES ATTRIBUTS OU DES VARIABLES LOCALES
        liste_chargee = torch.load(os.path.join(os.getcwd(),"Model_List.pth"))
        with open(os.path.join(os.getcwd(),'Batch_List.pkl'), 'wb') as f:
            ma_liste_chargee = pickle.load(f)
        with open(os.path.join(os.getcwd(),'Bound_List.pkl'), 'wb') as f:
            ma_liste_chargee = pickle.load(f)
            
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D   = self.memory.sample(self.batch_size)
            QYmax           = self.target_model(Y).max(1)[0].detach()
            update          = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA             = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss            = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def Update_QNet(self, step):
        if self.update_target_strategy == 'replace':
            if step % self.update_target_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())
        if self.update_target_strategy == 'ema':
            target_state_dict   = self.target_model.state_dict()
            model_state_dict    = self.model.state_dict()
            tau = self.update_target_tau
            for key in model_state_dict:
                target_state_dict[key] = tau*model_state_dict[key]\
                                        + (1-tau)*target_state_dict[key]
            self.target_model.load_state_dict(target_state_dict)
        

    def Train_DQN(self, env, config,
                  QNet, Target_QNet, 
                  Buffer, max_episode,
                  Ref_path,Ref_name,
                  buffer_bias=0):
        # 0 - INITIALISATION
        #---------------------------------------------------------------
        # 0.0 - Extraction de la configuration
        self.model          = QNet
        self.target_model   = Target_QNet
        self.L_extract_config(config)
        self.memory         = Buffer
        
        # 0.1 - Pour apprentissage
        episode_return      = []
        episode             = 0
        episode_cum_reward  = 0
        state, _            = env.reset()
        epsilon             = self.epsilon_max
        step                = -1
        # 0.2 - Pour les sorties
        best_score          = [1e7]
        histo_best_model    = []
        histo               = []
        # 0.3 - Pour enregistrement
        self.model_list_save     = []
        self.batch_list_save     = []
        self.bound_list_save     = []
        
                
        # 2 - APPRENTISSAGE : Recherche de Q(s,a)
        #---------------------------------------------------------------
        state, _        = env.reset()
        tmp_train_batch = ReplayBuffer(2200, 'cpu')
        while episode < max_episode:

            # 2.1 - Mise à jour de Epsilon
            step += 1
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
    
            # 2.2 - Incrément de la trajectoire et remplissage du Buffer
            action                  = EpsilonGreedy_action(env, self.model, state, epsilon)
            next_state, reward,\
                done, trunc, _      = env.step(action)
            episode_cum_reward      += reward
            histo                   += [episode_cum_reward]
            self.memory.append(state, action, reward, next_state, done)
            tmp_train_batch.append(state, action, reward, next_state, done)     # Pour la sauvegarde
            
            
            # 2.3 - Descente de gradient et mise à jour QNet
            for _ in range(self.nb_gradient_steps):
                self.gradient_step()
            self.Update_QNet(step)
            
            
            # 2.4 - Gestion des incréments : Fin d'épisode
            if done or trunc:
                episode_return.append(episode_cum_reward)
                # Affichage
                print("Episode ", '{:3d}'.format(episode),
                      ", epsilon ", '{:6.2f}'.format(epsilon),
                      ", buffer size ", '{:5d}'.format(len(self.memory)),
                      ", episode return ", '{:.4e}'.format(episode_cum_reward),  #{:4.1f}
                      sep='')
                
                # Rajout des échantillons dans le buffer si le modèle est le meilleur
                if (episode_cum_reward > best_score[-1]) and (buffer_bias!=0):                   
                    for _ in range(buffer_bias):
                        for i in range(len(tmp_train_batch.data)):
                            self.memory.append(tmp_train_batch.data[i][0],
                                               tmp_train_batch.data[i][1],
                                               tmp_train_batch.data[i][2],
                                               tmp_train_batch.data[i][3],
                                               tmp_train_batch.data[i][4])
                
                # Sauvegarde du modèle si meilleur
                if episode_cum_reward > best_score[-1]:
                    best_score          += [episode_cum_reward]
                    histo_best_model    += [histo]
                    
                    self.model_list_save += [deepcopy(self.model).to('cpu')]
                    self.batch_list_save += [tmp_train_batch]
                    
                    self.L_save_training(Ref_path,Ref_name)
                    print('New Best Model of cumulated rewards : {:.4e}'.format(episode_cum_reward))
                                        
                
                # Sauvegarde du dernier modèle
                if (episode+1) == max_episode:
                    self.model_list_save += [deepcopy(self.model).to('cpu')]
                    self.batch_list_save += [self.memory]
                    
                    self.L_save_training(Ref_path,Ref_name)
                    print('Last Model of cumulated rewards : {:.4e}'.format(episode_cum_reward))
                    
                # Gestion des incrémentations
                episode             += 1
                state, _            = env.reset()
                episode_cum_reward  = 0
                histo               = []
                tmp_train_batch     = ReplayBuffer(2200, 'cpu')
            else:
                state = next_state
        
        # 3 - MENAGE DANS LES VARIABLES
        #---------------------------------------------------------------
        del env, config, QNet, Target_QNet, max_episode, Ref_path, Ref_name
        del episode, step, epsilon 
        del episode_cum_reward, histo, tmp_train_batch
        del action, reward, state,next_state, done, trunc
        del self.model, self.target_model
        del self.bound_list_save
        # self.L_delete_config()
            
        print('APPRENTISSAGE : Fin')
        return self.model_list_save, self.batch_list_save, self.memory,\
            episode_return, best_score, histo_best_model
        
                
    def Train_DQN_MultiEnv(self, ENV, config,
                  QNet, Target_QNet, 
                  Buffer, max_episode,
                  Ref_path,Ref_name,
                  buffer_bias=0):
        # 0 - INITIALISATION
        #---------------------------------------------------------------
        # 0.0 - Extraction de la configuration
        self.model          = QNet
        self.target_model   = Target_QNet
        self.L_extract_config(config)
        self.memory         = Buffer
        
        # 0.1 - Pour apprentissage
        episode_return      = []
        episode             = 0
        episode_cum_reward  = 0
        flag_env            = 0
        env                 = ENV[0]
        state, _            = env.reset()
        epsilon             = self.epsilon_max
        step                = -1
        # 0.2 - Pour les sorties
        best_score          = [1e7]
        histo_best_model    = []
        histo               = []
        # 0.3 - Pour enregistrement
        self.model_list_save     = []
        self.batch_list_save     = []
        self.bound_list_save     = []
        
                
        # 2 - APPRENTISSAGE : Recherche de Q(s,a)
        #---------------------------------------------------------------
        tmp_train_batch = ReplayBuffer(2200, 'cpu')
        while episode < max_episode:

            # 2.1 - Mise à jour de Epsilon
            step += 1
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
    
            # 2.2 - Incrément de la trajectoire et remplissage du Buffer
            action                  = EpsilonGreedy_action(env, self.model, state, epsilon)
            next_state, reward,\
                done, trunc, _      = env.step(action)
            episode_cum_reward      += reward
            histo                   += [episode_cum_reward]
            self.memory.append(state, action, reward, next_state, done)
            tmp_train_batch.append(state, action, reward, next_state, done)     # Pour la sauvegarde
            
            
            # 2.3 - Descente de gradient et mise à jour QNet
            for _ in range(self.nb_gradient_steps):
                self.gradient_step()
            self.Update_QNet(step)
            
            
            # 2.4 - Gestion des incréments : Fin d'épisode
            if done or trunc:
                episode_return.append(episode_cum_reward)
                # Affichage
                print("Episode ", '{:3d}'.format(episode),
                      ", epsilon ", '{:6.2f}'.format(epsilon),
                      ", buffer size ", '{:5d}'.format(len(self.memory)),
                      ", episode return ", '{:.4e}'.format(episode_cum_reward),  #{:4.1f}
                      sep='')
                
                
                # Rajout des échantillons dans le buffer si le modèle est le meilleur
                if (episode_cum_reward > best_score[-1]) and (buffer_bias!=0):                   
                    for _ in range(buffer_bias):
                        for i in range(len(tmp_train_batch.data)):
                            self.memory.append(tmp_train_batch.data[i][0],
                                               tmp_train_batch.data[i][1],
                                               tmp_train_batch.data[i][2],
                                               tmp_train_batch.data[i][3],
                                               tmp_train_batch.data[i][4])
                
                # Sauvegarde du modèle si meilleur
                if episode_cum_reward > best_score[-1]:
                    best_score          += [episode_cum_reward]
                    histo_best_model    += [histo]
                    
                    self.model_list_save += [deepcopy(self.model).to('cpu')]
                    self.batch_list_save += [tmp_train_batch]
                    
                    self.L_save_training(Ref_path,Ref_name)
                    print('New Best Model of cumulated rewards : {:.4e}'.format(episode_cum_reward))
                    
                
                # Sauvegarde du dernier modèle
                if (episode+1) == max_episode:
                    self.model_list_save += [deepcopy(self.model).to('cpu')]
                    self.batch_list_save += [self.memory]
                    
                    self.L_save_training(Ref_path,Ref_name)
                    print('Last Model of cumulated rewards : {:.4e}'.format(episode_cum_reward))
                
                # Changement d'environnement
                flag_env = (flag_env + 1) % len(ENV)
                env      = ENV[flag_env]
                
                # Gestion des incrémentations
                episode             += 1
                state, _            = env.reset()
                episode_cum_reward  = 0
                histo               = []
                tmp_train_batch     = ReplayBuffer(2200, 'cpu')
            else:
                state = next_state
        
        # 3 - MENAGE DANS LES VARIABLES
        #---------------------------------------------------------------
        del ENV, config, QNet, Target_QNet, max_episode, Ref_path, Ref_name
        del episode, step, epsilon 
        del episode_cum_reward, histo, tmp_train_batch
        del action, reward, state,next_state, done, trunc
        del self.model, self.target_model
        del self.bound_list_save
        # self.L_delete_config()
            
        print('APPRENTISSAGE : Fin')
        return self.model_list_save, self.batch_list_save, self.memory,\
            episode_return, best_score, histo_best_model

    


#******************************************************************************
#******************************************************************************
#                ZONE D'APPRENTISSAGE : PATIENT DE REFERENCE
#******************************************************************************
#******************************************************************************
print('________________________________________________________________________')
print('APPRENTISSAGE SUR PATIENT DE REFERENCE')
print('________________________________________________________________________')
if False:

    # 1.1 - Création et définition des objets : Environnements
    nbr_episod      = 350       # Nombre de trajectoires pour un environnement
    step_per_epi    = 200       # Nombre de pas de chaque trajectoire
    env_PREF        = TimeLimit(HIVPatient(domain_randomization=False),
                                max_episode_steps=step_per_epi)
    
    # 1.2 - Création et définition des objets : Modèle et Agent  
    device  = 'cpu'
    Q_Net   = QNetwork(state_dim    = env_PREF.observation_space.shape[0], 
                       action_dim   = env_PREF.action_space.n, 
                       hidden_layers= 1, 
                       hidden_size  =128).to(device)
    Target_QNet = deepcopy(Q_Net)
    Agent_P_ref = Agent_DQN_v3()
    
    config_PREF  = {'learning_rate': 0.005,     # 0.005
                    'gamma': 0.95,
                    'buffer_size': 100000,
                    'epsilon_min': 0.01,
                    'epsilon_max': 0.9,
                    'epsilon_decay_period': step_per_epi*50,   # step_per_epi*40/(len(ENV) Je veux avoir une 1%-greedy policy après 50 trajectoires
                    'epsilon_delay_decay' : step_per_epi,      # Une décroissance à chaque trajectoire (200 steps)
                    'batch_size': 400,
                    'gradient_steps': 4,
                    'update_target_strategy': 'ema',            # 'ema' or 'replace'
                    'update_target_freq': 50,                   # step_per_epi/(4*(len(ENV))) Mise à jour du 'Target Network' tous les 1/4 d'épisode (50 step)
                    'update_target_tau': 0.005,
                    'criterion': torch.nn.SmoothL1Loss()}
    
    Buffer_P_ref = ReplayBuffer(100000, device)
    state_ini,_=env_PREF.reset()
    Buffer_P_ref,_,_ = Fill_Buffer(env_PREF, state_ini, 
                               Buffer_P_ref, n_sample = step_per_epi*5, 
                               model='None', epsilon=0)
    del state_ini
    
    # 1.2 - Apprentissage
    model_list, batch_list, memory,\
    episode_return, best_score, histo_best_model =\
            Agent_P_ref.Train_DQN(env_PREF, config_PREF,
                                  Q_Net, Target_QNet, 
                                  Buffer        = Buffer_P_ref, 
                                  max_episode   = nbr_episod,
                                  Ref_path      = os.getcwd(),
                                  Ref_name      = 'P_ref_Agent_v3',
                                  buffer_bias=3)
        

# 1.3 - Sélection du meilleur modèle parmi ceux enregistrés
print('________________________________________________________________________')
print('EVALUATION DES APPRENTISSAGE SUR LE PATIENT DE REFERENCE')
print('________________________________________________________________________')
print('\n')
print('------------------------------------------------------------------------')
print('Test des modèles enregistrés : Evaluation sur le patient de référence')
if False:
    Best_models = torch.load(os.path.join(os.getcwd(),'P_ref_Agent_v3'+"_Model_List.pth"))
    # Best_models = [torch.load(os.path.join(os.getcwd(),'Model_v1_PatientRef_E500.pth'))]
    step_per_epi = 200
    env     = TimeLimit(HIVPatient(domain_randomization=False), 
                        max_episode_steps=step_per_epi)
    Buffer_P_Ref, Histo_P_Ref, ind_ref = Evaluate_Ref_Patient(env,
                                                          Best_models,
                                                          step_per_epi)


print('------------------------------------------------------------------------')
print('Test des modèles enregistrés : Evaluation sur une population aléatoire')
if False:
    Best_models = [Best_models[ind_ref]]         # car les fonctions demandent des listes
    # Best_models = torch.load(os.path.join(os.getcwd(),'P_ref_Agent_v3'+"_Model_List.pth"))
    # Best_models = [torch.load(os.path.join(os.getcwd(),'Model_v1_PatientRef_E500.pth'))]
    step_per_epi = 200
    nb_patient = 15
    ENV = [None]*nb_patient
    for i in range(len(ENV)):
        ENV[i] = TimeLimit(HIVPatient(domain_randomization=True),
                           max_episode_steps=step_per_epi)
    
    BUFFER, HISTO, ind_rand = Evaluate_Rand_Patient(ENV,
                                               Best_models,
                                               step_per_epi)







#******************************************************************************
#******************************************************************************
#                ZONE D'APPRENTISSAGE : PATIENTS ALEATOIRES
#******************************************************************************
#******************************************************************************
print('________________________________________________________________________')
print('APPRENTISSAGE SUR PATIENTS ALEATOIRES')
print('________________________________________________________________________')
if False:
    # 1.1 - Création et définition des objets : Environnements
    nbr_episod      = 300       # Nombre de trajectoires pour un environnement
    step_per_epi    = 200       # Nombre de pas de chaque trajectoire
    nb_env          = 10
    ENV = [None]*nb_env
    for i in range(nb_env):
        ENV[i] = TimeLimit(HIVPatient(domain_randomization=True),
                           max_episode_steps=step_per_epi)
    
    # Ajout de xx patients de référence
    for i in range(5):
        ENV.append(TimeLimit(HIVPatient(domain_randomization=False),max_episode_steps=step_per_epi))
    nb_env = len(ENV)
    
    
    # 1.2 - Création et définition des objets : Modèle et Agent  
    device  = 'cpu'
    Q_Net   = Best_models[0]    # car maintenant on demande un unique modèle
    # Q_Net   = torch.load(os.path.join(os.getcwd(),'Model_v1_PatientRef_E500.pth'))
    # Q_Net   = QNetwork(state_dim    = ENV[0].observation_space.shape[0], 
    #                  action_dim     = ENV[0].action_space.n, 
    #                  hidden_layers  =1, 
    #                  hidden_size    =128).to(device)
    Target_QNet = deepcopy(Q_Net)
    config_RAND  = {'learning_rate': 0.001, #0.005
                    'gamma': 0.95,
                    'buffer_size': 100000,
                    'epsilon_min': 0.01,
                    'epsilon_max': 0.5,     #1
                    'epsilon_decay_period': step_per_epi*50,   # step_per_epi*40/(len(ENV) Je veux avoir une 1%-greedy policy après 50 trajectoires
                    'epsilon_delay_decay' : step_per_epi,      # Une décroissance à chaque trajectoire (200 steps)
                    'batch_size': 400,
                    'gradient_steps': 4,
                    'update_target_strategy': 'ema',            # 'ema' or 'replace'
                    'update_target_freq': 50,                   # step_per_epi/(4*(len(ENV))) Mise à jour du 'Target Network' tous les 1/4 d'épisode (50 step)
                    'update_target_tau': 0.005,
                    'criterion': torch.nn.SmoothL1Loss()}
    
    Random_Agent = Agent_DQN_v3()
    
    #------------------------------------------------------------------------------
    # 2.1 - Boucle d'apprentissage sur une configuration "MultiEnv"
    Buffer_P_random = ReplayBuffer(config_RAND['buffer_size'], device)
    print('Remplissage du Buffer')
    for i in tqdm(range(nb_env)):
        tmp_env         = ENV[i]
        state_ini,_     = tmp_env.reset()
        Buffer_P_random,_,_ = Fill_Buffer(tmp_env, state_ini,
                                      Buffer_P_random, n_sample = step_per_epi,
                                      model='None', epsilon=0)

    Best_models_R, Buffers_R,memory_R,\
    episode_return_R, best_score_R, histo_best_model_R,\
            = Random_Agent.Train_DQN_MultiEnv(ENV, config_RAND,
                                            Q_Net, Target_QNet, 
                                            Buffer        = Buffer_P_random, 
                                            max_episode   = nbr_episod,
                                            Ref_path      = os.getcwd(),
                                            Ref_name      = 'Random_Agent_v3_from_P_ref_Agent',
                                            buffer_bias=4)
    

# 1.3 - Sélection du meilleur modèle parmi ceux enregistrés
print('________________________________________________________________________')
print('EVALUATION DES APPRENTISSAGE SUR PATIENTS ALEATOIRES')
print('________________________________________________________________________')
print('\n')
print('------------------------------------------------------------------------')
print('Test des modèles enregistrés : Evaluation sur le patient de référence')
if False:
    Best_models     = torch.load(os.path.join(os.getcwd(),'Random_Agent_v3_from_P_ref_Agent'+"_Model_List.pth"))
    # Best_models     = torch.load(os.path.join(os.getcwd(),'Random_Agent_v3_from_E500'+"_Model_List.pth"))
    # Best_models = torch.load(os.path.join(os.getcwd(),'Random_Agent_v3_2'+"_Model_List.pth"))
    # Best_models = torch.load(os.path.join(os.getcwd(),'Random_Agent_v3'+"_Model_List.pth"))
    # Best_models = [torch.load(os.path.join(os.getcwd(),'Model_v1_PatientRef_E500.pth'))]
    step_per_epi = 200
    env     = TimeLimit(HIVPatient(domain_randomization=False), 
                        max_episode_steps=step_per_epi)
    Buffer_P_Ref, Histo_P_Ref, ind_ref = Evaluate_Ref_Patient(env,
                                                          Best_models,
                                                          step_per_epi)   


print('------------------------------------------------------------------------')
print('Test des modèles enregistrés : Evaluation sur une population aléatoire')
if False:
    Best_models = [Best_models[ind_ref]]
    # Best_models = torch.load(os.path.join(os.getcwd(),'Random_Agent_v3_2'+"_Model_List.pth"))
    # Best_models = [torch.load(os.path.join(os.getcwd(),'Model_v1_PatientRef_E500.pth'))]
    step_per_epi = 200
    nb_patient = 15
    ENV = [None]*nb_patient
    for i in range(len(ENV)):
        ENV[i] = TimeLimit(HIVPatient(domain_randomization=True),
                           max_episode_steps=step_per_epi)
    
    BUFFER, HISTO, ind_rand = Evaluate_Rand_Patient(ENV,
                                               Best_models,
                                               step_per_epi)






if False:
    from evaluate import evaluate_HIV, evaluate_HIV_population
    class ProjectAgent:   
    
        def __init__(self):
            self.model = []
            self.device = 'cpu'
        
        def act(self, observation, use_random=False):
            return self.greedy_action(observation)
    
        def save(self, path):
            pass
    
        def load(self):
            # self.model = torch.load(os.path.join(os.getcwd(),'Model_v1_PatientRef_E500.pth'), map_location=self.device)
            self.model = torch.load(os.path.join(os.getcwd(),'Random_Agent_v3_Model_List.pth'), map_location=self.device)
            self.model = self.model[32]
    
        def greedy_action(self, observation):
            with torch.no_grad():
                Q = self.model(torch.Tensor(observation).unsqueeze(0).to('cpu'))
                return torch.argmax(Q).item()
    
    agent = ProjectAgent()
    agent.load()
    score_agent: float = evaluate_HIV(agent=agent, nb_episode=1)
    print(f"Evaluation avec la procédure GitHub : evaluate_HIV              | score : {score_agent:.4e}")
    score_agent_dr: float = evaluate_HIV_population(agent=agent, nb_episode=15)
    print(f"Evaluation avec la procédure GitHub : evaluate_HIV_population   | score : {score_agent_dr:.4e}")
    



















