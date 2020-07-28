# -*- coding: utf-8 -*-
"""
Created on Mon May 11 18:03:13 2020

@author: César
"""
import matplotlib.pyplot as plt

import random
import torch
import numpy as np

from datetime import datetime
from argparse import ArgumentParser

from libs.neuralnetwork import MLP, CNN

from utils.decay_schedule import LinearDecaySchedule
from utils.params_manager import ParamsManager
from utils.experience_memory import Experience, ExperienceMemory

import envs.gameboy as GameBoy

# Parseador de argumentos
args = ArgumentParser("DeepQLearning")
args.add_argument("--params-file", help = "Path del fichero json de parámetros",
                  default = "parameters.json", metavar = "PFILE")
args.add_argument("--rom", help = "Rom de Gameboy disponible en la carpeta roms",
                  default = "blue.gb", metavar = "ROM")
args.add_argument("--test", help = "Modo de testing para jugar sin aprender. Por defecto está desactivado",
                  action = "store_true", default = False)
args.add_argument("--output-dir", help = " Directorio para almacenar los output. Por defecto = ./trained_models/results",
                  default = "./trained_models/results")
args = args.parse_args()

# Parámetros globales
manager = ParamsManager(args.params_file)

# Ficheros de logs acerca de la configuración de las ejecuciones
summary_filename_prefix = manager.get_agent_params()["summary_filename_prefix"]
summary_filename = summary_filename_prefix + args.rom  + datetime.now().strftime("%y-%m-%d-%H-%M-%S")

# Exportación de la configuración del agente y el emulador durante cada ejecución
#manager.export_agent_params(summary_filename + "/" + "agent_params.json")
#manager.export_emulator_params(summary_filename + "/" + "emu_params.json")

# Contador global de las ejecuciones
global_step_num = 0

# Habilitar la semilla aleatoria para poder reproducir el experimento a posteriori
seed = manager.get_agent_params()["seed"]
torch.manual_seed(seed)
np.random.seed(seed)

class DeepQLearner(object):
    def __init__(self, obs_shape, action_shape, hidden_shape, params):
        
        self.params = params
        self.gamma = self.params["gamma"]
        self.delta = self.params["delta"]
        self.learning_rate = self.params["learning_rate"]
        self.best_mean_reward = -float("inf")
        self.best_reward = -float("inf")
        self.training_steps_completed = 0
        self.action_shape = action_shape
        
        self.Q = CNN(obs_shape, action_shape, hidden_shape)   
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr = self.learning_rate)

        self.policy = self.epsilon_greedy_Q
        self.epsilon_max = self.params["epsilon_max"]
        self.epsilon_min = self.params["epsilon_min"]
        self.epsilon_decay = LinearDecaySchedule(initial_value = self.epsilon_max,
                                                 final_value = self.epsilon_min,
                                                 max_steps = self.params["epsilon_decay_final_step"])
        
        self.memory = ExperienceMemory(self.params["memory"])
        
        self.total_trainings = 0
        self.step_num = 0           
        
    def get_action(self, obs):
        return self.policy(obs)
   
    def epsilon_greedy_Q(self, obs):
        self.step_num += 1
        if random.random() < self.epsilon_decay(self.step_num) and not self.params["test"]:
            action = random.choice([a for a in range(self.action_shape)])
        else:
            action = np.argmax(self.Q(obs).detach().numpy())
        return action
        
    def learn(self, obs, action, reward, next_obs, done):
        if done:
            td_target = reward + torch.tensor(0.0, requires_grad = True)
        else:
            td_target = reward + self.gamma * torch.max(self.Q(next_obs))
        td_error = torch.nn.functional.mse_loss(self.Q(obs)[0][action], td_target)
        #print(td_target.item(), self.Q(obs)[action].item(), td_error.item())
        #print(reward, td_target.item(), td_error.item())
        self.Q_optimizer.zero_grad()
        td_error.backward()
        self.Q_optimizer.step()
        
    def replay_experience(self, batch_size = None):
        """
        Vuelve a jugar usando la experiencia aleatoria almacenada
        :param batch_size: Tamaño de la muestra a tomar de la memoria
        :return:
        """
        batch_size = batch_size if batch_size is not None else self.params["replay_batch_size"]
        experience_batch = self.memory.sample(batch_size)
        self.learn_from_batch_experience(experience_batch)
        self.training_steps_completed += 1
        #print("Replaying {} episodes".format(batch_size))
        
    def learn_from_batch_experience(self, experiences):
        """ 
        Actualiza la red neuronal profunda en base a lo aprendido en el conjunto de experiencias anteriores
        :param experiencias: fragmento  de recuerdos anteriores
        :return:
        """
        batch_xp = Experience(*zip(*experiences))
        obs_batch = np.array(batch_xp.obs)
        obs_batch = obs_batch
        action_batch = np.array(batch_xp.action)
        reward_batch = np.array(batch_xp.reward)
        next_obs_batch = np.array(batch_xp.next_obs)
        done_batch = np.array(batch_xp.done)
        
        td_target = reward_batch + ~done_batch * \
                    np.tile(self.gamma, len(next_obs_batch)) * \
                    self.Q(next_obs_batch).detach().max(1)[0].data.numpy()
        td_target = torch.from_numpy(td_target)
        action_idx = torch.from_numpy(action_batch)
        td_error = torch.nn.functional.mse_loss(self.Q(obs_batch).gather(1, action_idx.view(-1,1).long()),
                                                td_target.float().unsqueeze(1))
        self.Q_optimizer.zero_grad()
        td_error.mean().backward()
        self.Q_optimizer.step()  
  
    def save(self, env_name):
        file_name = self.params["save_dir"] + "DQL_" + env_name + ".ptm"
        agent_state = {"Q": self.Q.state_dict(),
                       "best_mean_reward": self.best_mean_reward,
                       "best_reward": self.best_reward,
                       "total_trainings": self.total_trainings}
        torch.save(agent_state, file_name)
        print("NN guardada en: ", file_name)
    
    def load(self, env_name):
        file_name = self.params["load_dir"] + "DQL_" + env_name + ".ptm"
        agent_state = torch.load(file_name, map_location = lambda storage, loc: storage)
        self.Q.load_state_dict(agent_state["Q"])
        #self.Q.eval()
        self.best_mean_reward = agent_state["best_mean_reward"]
        self.best_reward = agent_state["best_reward"]
        self.total_trainings = agent_state["total_trainings"]
        print("NN cargada desde: {} \nMejor recompensa media: {:.3f}\nMejor recompensa: {:.3f}\nTrains: {}".
              format(file_name, self.best_mean_reward, self.best_reward, self.total_trainings))

if __name__ == "__main__":
    
    # Carga de parámetros de los objetivos
    goal_params = manager.get_goal_params()
    goals = []
    for key, value in goal_params.items():
        goals.append(key)
    
    hidden_params = manager.get_hidden_params()
    
    # Carga de parámetros del emulador
    emu_conf = manager.get_emulator_params()
    
    # Nombres de cada red neuronal
    for goal in goals:
        emu_conf["MPL_" + goal] = args.rom + "." + goal
    emu_conf["MPL_strategy"] = args.rom + ".strategy"
    
    # Chequeo de modo TEST
    if args.test:
        emu_conf["test"] = True
    
    # Chequeo de existencia del juego
    gameboy_rom = False
    for game in GameBoy.get_games_list():
        if game in args.rom:
            gameboy_rom = True
    
    # Inicio del emulador
    if gameboy_rom:
        emulator = GameBoy.initialize_emulator(emu_conf)
        emulator.stop(save = False)
    else:
        print("Error: No se encuentra el ROM")
    
    # Carga de parámetros de la estrategia
    #strategy_params = manager.get_strategy_params()
    #strategy_params["test"] = args.test
    #strategy = DeepQLearner(len(goals), len(goals), [len(goals)], strategy_params)
    #if strategy_params["load_trained_model"]:
    #    try:
    #        strategy.load(emu_conf["MPL_strategy"])
    #    except FileNotFoundError:
    #        print("ERROR: No existe una NN entrenada para la estrategia. Creando una nueva.")
    
    # Carga de parámetros de los agentes para cada objetivo
    agent_params = manager.get_agent_params()
    agent_params["test"] = args.test
    agent = []
    previous_checkpoint_mean_ep_rew = []
    
    # Observaciones
    obs_shape = [3, 144, 160]
    # Acciones
    action_shape = 9 # 8 botones disponibles más 1 para la inacción
    num_act = ["none", "up", "down", "left", "right", "A", "B", "select", "start"]
    
    # Creación de un agente para cada objetivo
    for goal in goals:
        agent.append(DeepQLearner(obs_shape, action_shape, hidden_params[goal], agent_params))
        previous_checkpoint_mean_ep_rew.append(agent[-1].best_mean_reward)
        if agent_params["load_trained_model"]:
            try:
                agent[-1].load(emu_conf["MPL_"+goal])
                previous_checkpoint_mean_ep_rew[-1] = agent[-1].best_mean_reward
            except FileNotFoundError:
                print("ERROR: No existe una NN entrenada para {}. Creando una nueva.".format(goal))
    
    # Inicialización de las recompensas por episodio
    epi_obj = np.zeros(len(goals))
    mean_episode_reward = np.zeros(len(goals))
    total_episode_reward = np.zeros(len(goals))
    num_improved_episodes_before_checkpoint = np.zeros(len(goals))
    episode = 0
    state_count = 0
    saved_total_reward = 0.0
    state_save_freq = agent_params["state_save_freq"]
    state_back_freq = agent_params["state_back_freq"]
    register = []
    
    # Bucle de episodios
    while global_step_num < agent_params["max_training_steps"]:
        # Definición de las clases emulador, observador, actuador y recompensador
        emulator = GameBoy.reset_emulator(emulator, emu_conf)
        observer = GameBoy.Observer(emulator)
        actuator = GameBoy.Actuator(emulator, agent_params)
        rewarder = GameBoy.Rewarder(emulator, goal_params)
        
        obs = observer.observation() # Primer observación del episodio actual
        total_reward = 0.0 # Reinicio de la recompensa total del episodio
        done = False # Bajada de bandera de finalización del episodio
        step = 0 # Puesta a cero de los pasos del episodio
        dones = rewarder.is_done # Visualización de finalización de objetivos
        """
        if episode % state_back_freq == 0: # Reinicio del save state cada state_back_freq episodios
            state_count = 0
            total_reward = 0.0
        if state_count > 0: # Carga del penúltimo state guardado y la recompensa acumulada en ese momento
            GameBoy.load_state(emulator, emu_conf, state_count - 1)
            print("load state {} ...".format(state_count-1))
            total_reward = saved_total_reward
        """
        GameBoy.load_state(emulator, emu_conf, 6)
        print("load state {} ...".format(6))
        
        # Bucle dentro del episodio
        while not done:
            
            # Estrategia
            dones = rewarder.is_done
            #obj = strategy.get_action(dones)
            obj = 0
            #print(obj)
            """
            if (step + 1) % state_save_freq == 0: # Guardado del state cada state_save_freq steps
                GameBoy.save_state(emulator, emu_conf, state_count)
                print("saved state {} ...".format(state_count))
                state_count += 1
                saved_total_reward = total_reward
            """
            # Acción
            action = agent[obj].get_action(obs)
            actuator.step(num_act[action])
                        
            # Observación luego de la acción
            next_obs = observer.observation()
            next_dones = rewarder.is_done
                
            # Recompensas
            reward = rewarder.reward()[obj]# + punish
            total_reward += reward
            total_episode_reward[obj] = total_reward
            done = rewarder.is_done[obj]
            
            # Visualización
            print("Episode: {}. Step: {}/{:.0f}.".format(episode, global_step_num, agent_params["max_training_steps"]))
            print("Reward. Best: {:.3f}, Mean: {:.3f}, Actual: {:.3f}.".
                  format(agent[obj].best_reward, mean_episode_reward[obj], total_reward))
            print("+-----+-----+-----+-----+-----+-----+-----+-----+-----+")
            print("|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|".
                  format(agent[obj].Q(obs)[0][0].item(), agent[obj].Q(obs)[0][1].item(), agent[obj].Q(obs)[0][2].item(),
                         agent[obj].Q(obs)[0][3].item(), agent[obj].Q(obs)[0][4].item(), agent[obj].Q(obs)[0][5].item(),
                         agent[obj].Q(obs)[0][6].item(), agent[obj].Q(obs)[0][7].item(), agent[obj].Q(obs)[0][8].item()))
            print("+-----+-----+-----+-----+-----+-----+-----+-----+-----+")
            print(" " * 6 * action, "|", num_act[action])
            
            # Aprendizaje del agente
            #agent[obj].learn(obs, action, reward, next_obs, done)
            
            # Guardado en la memoria
            agent[obj].memory.store(obs.squeeze(), action, reward, next_obs.squeeze(), done)
            
            # Aprendizaje de la estrategia
            #strategy.learn(dones, obj, reward, next_dones, done)
            
            # Preparación para el próximo paso
            obs = next_obs
            dones = next_dones
            step += 1
            global_step_num += 1    
         
            if done:
                episode += 1
                epi_obj[obj] += 1
                agent[obj].total_trainings += 1
                mean_episode_reward[obj] = (mean_episode_reward[obj] * (epi_obj[obj] - 1) +
                                            total_episode_reward[obj]) / epi_obj[obj]
                
                # Actualización del best_reward
                if total_reward > agent[obj].best_reward:
                    agent[obj].best_reward = total_reward
                
                # Si la recompensa promedio supera a la mejor recompensa promedio anterior cuento
                if mean_episode_reward[obj] > previous_checkpoint_mean_ep_rew[obj]:
                    num_improved_episodes_before_checkpoint[obj] += 1
                # Si esa cuenta es mayor que el save_freq actualizo la mejor recompensa media
                if num_improved_episodes_before_checkpoint[obj] >= agent_params["save_freq"]:
                    previous_checkpoint_mean_ep_rew[obj] = mean_episode_reward[obj]
                    agent[obj].best_mean_reward = mean_episode_reward[obj]
                    agent[obj].save(emu_conf["MPL_"+goals[obj]])
                    num_improved_episodes_before_checkpoint[obj] = 0
                
                # Si el número de episodios es múltiplo de nn_save_freq guardo la red nuronal
                #if episode % agent_params["nn_save_freq"] == 0:
                #    agent[obj].save(emu_conf["MPL_"+goals[obj]])
                    #strategy.save(emu_conf["MPL_strategy"])
                               
                print("Episodio #{} finalizado con {} iteraciones. Train #{}\nRecompensa = {:.3f}, Media = {:.3f}, Mejor = {:.3f}".
                      format(episode, step+1, agent[obj].total_trainings, total_reward,
                              mean_episode_reward[obj], agent[obj].best_reward))
                print("Episodios antes del checkpoint: {} / {}".
                      format(int(num_improved_episodes_before_checkpoint[obj]), agent_params["save_freq"]))
                print("Pasos globales {} de {}. {} %\n".
                      format(global_step_num, int(agent_params["max_training_steps"]),
                             100 * global_step_num / agent_params["max_training_steps"]))
                register.append([step+1, total_reward, mean_episode_reward[obj], agent[obj].best_reward])
                
                # Si hay suficientes episodios guardados en la memoria replay experience
                if agent[obj].memory.__len__() >= 2 * agent_params["replay_start_size"] and not args.test:
                    print("Replaying episodes...")
                    agent[obj].replay_experience()
                    agent[obj].save(emu_conf["MPL_"+goals[obj]])
                    
                #new_window = np.zeros((72,80))
                #for i in range(360):
                #    new_window[4 * (i // 20):4 * (i // 20 + 1), 4 * (i % 20):4 * (i % 20 + 1)] = obs[i*16:(i+1)*16].reshape(4,4)
                #plt.imshow(new_window, cmap = "coolwarm")
                #plt.figure()
            
            if all(dones):
                break
                
    #for i in range(len(goals)):
    #    agent[i].save(emu_conf["MPL_"+goals[i]])
    #strategy.save(emu_conf["MPL_strategy"])
    emulator.stop(save = False)
    for i in range(4):
        plt.plot(np.array(register)[:,i])
        plt.figure()
 