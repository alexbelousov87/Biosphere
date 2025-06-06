from random import random, choice, uniform, shuffle, choices
from numpy import percentile,median,mean,arange
from copy import deepcopy
import numpy as np
from itertools import chain
import argparse
import csv
from datetime import datetime
import time
import math
import h5py
import os
import pickle
import zlib
import sys
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community import asyn_lpa_communities
from networkx.algorithms.community import modularity
import copy
import graph_tool.all as gt

print('Start', flush=True)
print(sys.argv, flush=True)

start_time = time.time()
stop_time = 1170

### ВОПРОСЫ
# 1) добавление вида и жизнь без добавления -- должны варьироваться
# например, на одно добавление, Х циклов без добавления -- это новый параметр
#

parser = argparse.ArgumentParser()

parser.add_argument('-self', '--points', type =float, required=True, dest='INFL',
             help='Сила самовлияния')
parser.add_argument('-env', '--sets', type =float, required=True, dest='FORA',
             help='Сила влияния среды')
parser.add_argument('-sos', type =int, required=True, dest='SOS',
             help='Параметр мутации')
parser.add_argument('-mu', type =float, required=True, dest='MU',
             help='Коэффициент мутации первого организма')
parser.add_argument('-path', '--pathern', type =int, required=True, dest='PATH',
             help='Каталог хранения данных')
parser.add_argument('-POPMAX', type =int, required=True, dest='POPMAX',
             help='Максимальная численность')
parser.add_argument('-sol', type =float, required=True, dest='SOL',
             help='Степень')
parser.add_argument('-num', type =int, required=True, dest='NUM',
             help='Номер эксперимента')


args = parser.parse_args()
value_INFL = args.INFL
value_FORA = args.FORA
value_PATH = args.PATH
value_MU = args.MU
value_SOS = args.SOS
value_POPMAX = args.POPMAX
value_SOL = args.SOL
value_NUM = args.NUM

print('PM: ', value_INFL, value_FORA, value_MU, value_SOS, value_POPMAX, value_SOL, flush=True)

#####################################
### ОСНОВНЫЕ ВАРЬИРУЕМЫЕ ПАРАМЕТРЫ
#####################################

const_first_org_mu = value_MU
const_initial_speed_of_speciation = value_SOS

const_first_org_self_infl = value_INFL
const_env_quality = value_FORA

const_max_pop_size = value_POPMAX # максимальная численность в биосфере
speed_of_life = value_SOL


#####################################
### ПАРАМЕТРЫ ПО УМОЛЧАНИЮ, все значения можно менять
#####################################

#const_num_of_experiments = 10**6 # сколько всего повторений делаем
const_time_to_live = 10**5 # 5000 # максимальное время, в течение которого мы отслеживаем жизнь биосферы
const_max_num_of_species = 5000 # максимальное число видов, по достижении которого эксперимент останавливается (по техн. причинам)
                                              # поскольку в текущей схеме порождается 1 вид на 1 шаге, макс число видов не превысит числа шагов


const_speed_of_speed_change = 0.5 # постоянная скорость изменения скорости изменения коэффициентов

const_born_threshold = 2 # численность вида, после которой он может начать порождать разделяясь на 2
                         # (до этого новый вид полностью забирает старый)

const_initial_abundance = 1 # численность новообразованного вида
const_die_less_then = const_initial_abundance # вид умирает, если его численность ниже этого значения


## раздел констант, определяющих пределы вариации влияния видов друг на друга
const_self_min = -1 + 1 / 1000
const_self_max =  - 1/1000 # !! это важный параметр, который неплохо было бы менять !!
const_other_min = -1 + 1 / 1000
const_other_max = 1 - 1 / 1000
const_env_infl_min = -1 + 1 / 1000
const_env_infl_max = 1 - 1 / 1000


## подготовим названия параметров
# mu -- скорость мутации
# infl -- сила, с которой вид влияет .....
# k1_* -- параметры, определяющие силу изменения параметров mu p и infl потомка при изменении генотипа предка на 1
raw_param_names = ['mu','infl','SoS','env', 'SoL']
param_names=raw_param_names[:]
for c in raw_param_names:
    param_names.append('k1_'+c)




## масштабирующие константы -- на какое максимальное значение может произойти сдвиг при мутации
# (обычно берется равным возможному разбросу значений параметра)
# k1_* -- набор коэффициентов, показывающих насколько будет отличаться
# влияние и другие параметры мутировавшего потомка от соответствующих параметров данного родителя, если
# их геномы отличаются на 1
# ("равномерная непрерывность" функции влияния по переменной геному)
const_max={}
const_min={}

const_min['mu'] = 1
const_max['mu'] = 20
const_min['infl'] = const_other_min
const_max['infl'] = const_other_max
const_min['env'] = const_env_infl_min
const_max['env'] = const_env_infl_max
const_min['SoS'] = 0
const_max['SoS'] = 5
const_min['SoL'] = 1.1
const_max['SoL'] = 3

for c in raw_param_names:
    const_min['k1_' + c] = 0
    const_max['k1_' + c] = 1




##############################
# КЛАСС ПОПУЛЯЦИЯ
##############################
class population:

    def __init__(self, param_time_to_live = const_time_to_live,
                 param_speed_of_speed_change = const_speed_of_speed_change,
                 param_initial_abundance = const_initial_abundance,
                 param_die_less_then = const_die_less_then,
                 param_self_min = const_self_min,
                 param_self_max = const_self_max,
                 param_other_min = const_other_min,
                 param_other_max = const_other_max,
                 param_first_org_mu = const_first_org_mu,
                 param_first_org_self_infl = const_first_org_self_infl,
                 param_biosphere_lim = const_max_num_of_species
                 ):

        # все "константы" эксперимента могу варьироваться при создании новой биосферы (класса population)
        self.param_time_to_live = param_time_to_live
        self.param_speed_of_speed_change = param_speed_of_speed_change
        self.param_initial_abundance = param_initial_abundance
        self.param_die_less_then = param_die_less_then
        self.param_self_min = param_self_min
        self.param_self_max = param_self_max
        self.param_other_min = param_other_min
        self.param_other_max = param_other_max
        self.param_first_org_mu = param_first_org_mu
        self.param_first_org_self_infl = param_first_org_self_infl
        self.param_biosphere_lim = param_biosphere_lim
        self.input_params = [const_first_org_self_infl, const_env_quality, const_first_org_mu, const_initial_speed_of_speciation, const_max_pop_size, speed_of_life]

        self.orgs = [] # список организмов популяции
        self.time = 0 # итерация жизни популяции
        self.single_species_gen = '' # служебный параметр, играет роль когда нужно оценить 1-stab
        self.genome_counter = 0
        self.do_time_scaling = False # служеный параметр, показыващий будет ли кто-то порождать новый вид или нет
        self.A = [] # вектор численности видов
        self.INFL = [] # матрица влияния
        self.raw_total_num_of_new_species = 0 # служебная переменная, хранящая количесвто планируемых новых видов
                                              # в схеме без time-scale

        # хэш всяких логов
        self.loghash={}
        self.loghash['abundance'] = []
        self.loghash['poroditeli'] = []
        self.loghash['num_of_species'] = []
        self.loghash['min_spec_abund'] = []
        self.loghash['25perc_spec_abund'] = []
        self.loghash['50perc_spec_abund'] = []
        self.loghash['75perc_spec_abund'] = []
        self.loghash['max_spec_abund'] = []
        self.loghash['die_times'] = []
        self.loghash['life_time'] = []
        self.loghash['max_infl'] = []
        self.loghash['min_infl'] = []
        self.loghash['max_infl_self'] = []
        self.loghash['min_infl_self'] = []
        self.loghash['max_infl_env'] = []
        self.loghash['min_infl_env'] = []
        self.loghash['change_abund'] = []
        self.loghash['size_spec'] = []
        self.loghash['entropy'] = []
        self.loghash['entropy_norm'] = []
        self.loghash['modularity'] = []
        self.loghash['num_community'] = []
        self.loghash['max_community'] = []        
        self.loghash['Length_of_existence_end'] = [0]
        self.loghash['Length_of_dominance_end'] = [0]
        self.loghash['Length_of_dominance_first_species'] = [0]
        self.loghash['Phenotype_Feature_Portrays'] = [[], [], [], [], []]

        self.INF = np.array([])
        self.A = np.array([])
        self.indx = []
        self.old_orgs_gen = [] # массив, содержащий виды на предыдущем шаге
        self.old_org_abund = [] # массив, содержащий численность видов на предыдущем шаге
        self.old_max_org_abund = []
        self.old_min_org_abund = []
        self.lifetime = 0

    ##################
    # жизнь популяции
    def live(self):

        for self.time in range(self.param_time_to_live):

            # растем
            self.born_new_species()

            # влияем
            self.change_abundance()

            # умираем
            self.remove_low_abundant()

            # пишем логи
            self.logs()

            # проверка условий досрочной остановки эксперимента
            # если после роста
            # общая численность биосферы слишком велика или
            # число видов ней слишком велико для вычислений или
            # вся популяция вымерла
            check_stop = self.stop_experiment_conditions()
            if check_stop != 'Continue': return check_stop

        return 'Success'

    # проверка условий досрочной остановки эксперимента
    def stop_experiment_conditions(self):

        tot_abund = sum([org.abund for org in self.orgs])

        if tot_abund > const_max_pop_size:
            return 'Too large'

        if len(self.orgs) > self.param_biosphere_lim:
            return 'Too many species'

        if len(self.orgs) == 0:
            return 'Ceased'

        if (time.time()-start_time)/60 > stop_time:
            f = open(r'./input_files/' + str(value_PATH) + '/pm_' + str(value_POPMAX) + '_' +  str(value_NUM) + '.txt', 'wb')
            pickle.dump(popul, f)
            f.close()
            exit()
            return 'Time out'

        if self.one_stab() == True:
            return '1-stabilized'

        return 'Continue'

    ##################
    # выбираем один организм и размножаем его с мутацией, добавляя в популяцию
    def born_new_species(self):

        # если еще никого нет
        if len(self.orgs)==0:

            # генерим первый
            self.raw_total_num_of_new_species = 1
            born_species(popul=self)

        # если популяция уже непуста, выбираем кто будет порождать
        else:

            # сколько видов будет всего порождено в схеме без time-scale
            self.raw_total_num_of_new_species = 0

            for org in self.orgs:

                # сколько видов породил бы организм по схеме без time-scale
                org.raw_num_of_new_species = 10 ** org.param['SoS'] * org.abund # поделить на const_max_size численность биосферы
                self.raw_total_num_of_new_species += org.raw_num_of_new_species # закомментировать

            # веса для выбора кто именно будет порождать
            weights = [org.raw_num_of_new_species / self.raw_total_num_of_new_species for org in self.orgs] # закомментировать

            # выбираем 1 порождающего
            born_species(self, choices(self.orgs, weights=weights)[0]) # ввнести в верхний цикл for org in self.orgs

    ##################
    # взаимное влияние, выражающееся в изменении численности видов биосферы
    def change_abundance(self):

        speedup_abundance = 1

        # изменяем численность
        bAe = speedup_abundance * ((np.inner(self.INF[:,1:], self.A))/const_max_pop_size)
        aAe = speedup_abundance * self.INF[:,0]
        Ae = aAe + bAe
        self.A *= (1 + (np.power([org.param['SoL'] for org in self.orgs], Ae) - 1) / (self.raw_total_num_of_new_species/const_max_pop_size)) # закомментировать

        for org in self.orgs:
            org.abund_diff = self.A[org.index] - org.abund
            org.abund = self.A[org.index]


    ##################
    # смерть малочисленных видов
    def remove_low_abundant(self):
        arr = []
        arr1 = []
        new_orgs = []
        count = 0
        for org in self.orgs:
            if org.abund >= self.param_die_less_then:
                new_orgs.append(org)
            else:
                start_time = time.time()
                self.loghash['life_time'].append(self.time-org.born_time)
                self.loghash['die_times'].append(self.time)
                self.INF = np.delete(np.delete(self.INF, (org.index-count), axis=0), ((org.index-count)+1), axis=1)
                self.A = np.delete(self.A, (org.index-count), axis=0)
                count += 1
        for i in range(len(new_orgs)):
            new_orgs[i].index = i
        self.orgs = deepcopy(new_orgs)

    ##################
    # пишем логи
    def logs(self):

        p1 = []
        p2 = []
        p3 = []
        p4 = []
        p5 = []
        Matrix = []

        a = [org.abund for org in self.orgs]

        self.loghash['Length_of_dominance_end'] = [0]
        self.loghash['Length of dominance_first_species'] = [0]

        # Modularity

        Weigths_matrix = copy.deepcopy(self.INF[:,1:]) # матрица весов для расчета modularity
        Adjacency_matrix = copy.deepcopy(self.INF[:,1:]) # матрица смежности

        for raw in range(len(Weigths_matrix)):
            for col in range(len(Weigths_matrix[raw])):
                if Weigths_matrix[raw][col] > 0 and raw != col:
                    Matrix.append((raw, col, Weigths_matrix[raw][col]))
        
        graph = gt.Graph()
        ewt = graph.new_ep("double")
        
        if Matrix != []:
            graph.add_edge_list(Matrix, eprops=[ewt])
            state = gt.minimize_blockmodel_dl(graph, state=gt.ModularityState)
            Modularity = state.modularity()
            num_blocks, max_block = state.get_B()
            self.loghash['modularity'].append(Modularity)
            self.loghash['num_community'].append(num_blocks)
            self.loghash['max_community'].append(max(max_block[1]))
        else:
            self.loghash['modularity'].append(0)   

        sum_sp = 0
        max_abund = 0
        for org in self.orgs:
            if org.abund > max_abund:
                self.loghash['Length_of_existence_end'][0] = self.time-org.born_time
                max_abund = org.abund
                dominance_spec = org


            p1.append(org.param['SoS'])
            p2.append(org.param['SoL'])
            p3.append(org.param['mu'])
            p4.append(popul.INF[org.index][org.index+1])
            p5.append(popul.INF[org.index][0])

            sum_sp += -(float(org.abund) / float(sum(a))) * math.log2(
                float(org.abund) / float(sum(a)))
        self.loghash['Phenotype_Feature_Portrays'][0] = p1
        self.loghash['Phenotype_Feature_Portrays'][1] = p2
        self.loghash['Phenotype_Feature_Portrays'][2] = p3
        self.loghash['Phenotype_Feature_Portrays'][3] = p4
        self.loghash['Phenotype_Feature_Portrays'][4] = p5

        self.loghash['entropy'].append(sum_sp)
        if len(a) == 1:
            self.loghash['entropy_norm'].append(sum_sp/(math.log2(2)))
        elif len(a)!= 0:
            self.loghash['entropy_norm'].append(sum_sp / (math.log2(float(len(a)))))

        if dominance_spec.born_time == 0:
            self.loghash['Length_of_dominance_first_species'][0] += 1

        dominance_spec.dominance = dominance_spec.dominance + 1
        self.loghash['Length_of_dominance_end'][0] = dominance_spec.dominance

        if len(a)>0:
            self.loghash['abundance'].append(sum(a)) # общая чисденность биосферы
            self.loghash['num_of_species'].append(len(a)) # число видов в биосфере

    def output(self):

        string = str(self.input_params[0]) + ';' + str(self.input_params[1]) + ';' + str(self.input_params[2]) + ';' + str(self.input_params[3]) + ';' + \
                 str(self.input_params[4]) + ';' + str(self.input_params[5]) + ';' + ' '.join(map(str, np.array([self.time]))) + ';' + ' '.join(
            map(str, np.asarray([org.abund for org in self.orgs]))) + ';' + ' '.join(
            map(str, np.asarray(self.loghash['abundance']))) + ';' + ' '.join(
            map(str, np.asarray(self.loghash['num_of_species']))) + ';' + ' '.join(
            map(str, np.asarray(self.loghash['life_time']))) + ';' + ' '.join(
            map(str, np.asarray(self.loghash['die_times']))) + ';' + ' '.join(
            map(str, np.asarray(self.loghash['entropy']))) + ';' + ' '.join(
            map(str, np.asarray(self.loghash['entropy_norm']))) + ';' + ' '.join(
            map(str, np.asarray(self.loghash['Phenotype_Feature_Portrays'][0]))) + ';' + ' '.join(
            map(str, np.asarray(self.loghash['Phenotype_Feature_Portrays'][1]))) + ';' + ' '.join(
            map(str, np.asarray(self.loghash['Phenotype_Feature_Portrays'][2]))) + ';' + ' '.join(
            map(str, np.asarray(self.loghash['Phenotype_Feature_Portrays'][3]))) + ';' + ' '.join(
            map(str, np.asarray(self.loghash['Phenotype_Feature_Portrays'][4]))) + ';' + ' '.join(
            map(str, np.asarray(self.loghash['Length_of_dominance_end']))) + ';' + ' '.join(
            map(str, np.asarray(self.loghash['Length_of_dominance_first_species']))) + ';' + ' '.join(
            map(str, np.asarray(self.loghash['Length_of_existence_end']))) + ';' + ' '.join(
            map(str, np.asarray(self.loghash['modularity']))) + ';' + ' '.join(
            map(str, np.asarray(self.loghash['num_community']))) + ';' + ' '.join(
            map(str, np.asarray(self.loghash['max_community'])))
        return string
        
    def one_stab(self):

        a = [org.abund for org in self.orgs]
        g = [org.gen for org in self.orgs]

        trigger = 1

        # Проверяем различие видов на текущем и предыдущем шагах и изменение численности видов.
        if len(g) == len(self.old_orgs_gen):
            if self.old_orgs_gen == g[:]:
                for i in range(len(a)):
                    #if math.fabs(1 - ((self.old_max_org_abund[i]-self.old_min_org_abund[i])/a[i])) < 0.01:
                    if math.fabs(1 - (a[i]/self.old_max_org_abund[i])) < 0.01 and math.fabs(1 - (a[i]/self.old_min_org_abund[i])) < 0.01:
                        self.old_max_org_abund[i] = max(self.old_max_org_abund[i], a[i])
                        self.old_min_org_abund[i] = min(self.old_min_org_abund[i], a[i])
                    else:
                        trigger = 0
                        break
            else:
                trigger = 0
        else:
            trigger = 0

        if trigger == 0:
            self.old_orgs_gen = g[:]
            self.old_max_org_abund = a[:]
            self.old_min_org_abund = a[:]
            self.lifecount = 0
            return False

        elif trigger == 1:
            self.lifecount = self.lifecount + 1
            if self.lifecount > 100:
                return True

remember_first_gen = ''
##############################
# КЛАСС ОРГАНИЗМ
##############################
class born_species:
    def __init__(self, popul=None, parent=None):
        ##############
        # все параметры вида просто для удобства в одном месте

        self.gen = ''
        self.plan_to_speciation = [] # массив, хранящий план вида на видообразование
        self.param = {}
        self.infl = {} # массив, содержащий силы влияния всех других на этот вид
        self.env_infl = const_env_quality # как влияет окружающая среда на организм
        self.abund_diff = 0
        self.abund_diff_old = 0
        self.index = 0
        self.dominance = 0
        self.num_species = 0 # параметр, показывающий сколько видов планирует породить организм на конкретном шаге (промежуточный параметр - M(org))
        if parent is None: self.born_time = 0 # запоминаем дату рождения, если это самый первый организм
        else: self.born_time = popul.time # запоминаем дату рождения во всех иных случаях

        # генотипы выдаем последовательно -- это просто номера
        self.gen = str(popul.genome_counter)
        popul.genome_counter += 1

        ##############
        # если это первый в мире организм, то генерим его
        if parent is None:

            # случайно генерим группу параметров k1
            for s in raw_param_names:
                self.param['k1_' + s] = uniform(const_min['k1_' + s], const_max['k1_' + s])

            # остальные параметры задаем вручную (они приходят снаружи)
            self.param['mu'] = popul.param_first_org_mu
            self.param['SoS'] = const_initial_speed_of_speciation
            self.param['SoL'] = speed_of_life

            # создаем влияние первого в мире вида на самого себя влияние среды на него
            popul.INF = np.array([[const_env_quality, popul.param_first_org_self_infl]], dtype=float)

            # численность начального вида
            self.abund = const_initial_abundance
            popul.A = np.array([popul.param_initial_abundance], dtype=float)
            self.index = len(popul.A)-1

            # добавляем первый организм в популяцию
            popul.orgs.append(self)

        ##############
        # если это чей-то потомок, то наследуем с мутациями
        else:

            # зададим стандарнтную численность нового вида и вычтем ее из вида предкового
            # (численность предкового вида при этом должна быть не меньше const_initial_abundance
            # это мы должны проверить при выборе этого предкового вида)
            # если у предка хватает численности породить и остаться живым, делаем это
            if popul.A[parent.index] >= const_born_threshold:

                popul.A[parent.index] -= popul.param_initial_abundance
                popul.A = np.concatenate((popul.A, [popul.param_initial_abundance]))

                parent.abund -= popul.param_initial_abundance
                self.abund = popul.param_initial_abundance

                # номер нового вида в матрице влияний и векторе численности
                self.index = len(popul.A)-1

                # добавляем нулевые ячейки правым столбцом и нижней строкой, далее будем заполнять эти ячейки
                popul.INF = np.vstack((popul.INF , np.zeros((1, len(popul.INF [0])), dtype=popul.INF.dtype)))
                popul.INF = np.hstack((popul.INF , np.zeros((popul.INF.shape[0], 1), dtype=popul.INF.dtype)))

                ####
                # промутируем предковые параметры
                for s in param_names:

                    if s != 'infl' and s != 'env':
                        ## сдвиг основных параметров потомка относительно родителя
                        # меняем мы один из базовых параметров или меняем скорость изменения базовых параметров
                        if s in raw_param_names: speed_of_change = parent.param['k1_' + s]
                        else: speed_of_change = popul.param_speed_of_speed_change

                        # рассчитываем величину сдвига параметра s в зависимости от:
                        # 1) размер отличия генотипа предка от потомка
                        # 2) скорости изменения данного параметра (размер изменений на "единицу" разницы между геномами)
                        # 3) нормировка с учетом допустимого разброса данного параметра
                        #    (т.е. перевод из процентов в абсолютные значения)
                        sdvi = (parent.param['mu'] / 100) * \
                               speed_of_change * \
                               (const_max[s] - const_min[s])

                        # запишем значение нового параметра
                        self.param[s] = self.vary(parent.param[s],sdvi,const_min[s],const_max[s])

                ########
                # рассчитываем влияние каждого из других видов на этот вид,
                # и этого вида на каждый из других, исходя из влияния его предка
                for org in popul.orgs:

                    # рассчитываем величину сдвига параметра infl в зависимости от:
                    # 1) скорость изменения параметра infl при изменении генома на единицу
                    # 2) разница в геномах
                    # 3) нормировка с учетом допустимого разброса infl (здесь -- 2)
                    #    (т.е. перевод из процентов в абсолютные значения)
                    sdvi = self.param['k1_infl'] * \
                           (parent.param['mu'] / 100) * 2 # [-1,1] -- макс диапазон изменений для infl


                    # запишем влияние организма org на новичка
                    popul.INF[self.index][org.index + 1] = self.vary(popul.INF[parent.index][org.index + 1], sdvi,
                                                                     popul.param_other_min,popul.param_other_max)

                    # запишем влияние новичка на организм org
                    popul.INF[org.index][self.index + 1] = self.vary(popul.INF[org.index][parent.index + 1], sdvi,
                                                                     popul.param_other_min,popul.param_other_max)

                ########
                # рассчитываем влияние вида на самого себя
                # 1) скорость изменения параметра infl при изменении генома на единицу
                # 2) разница в геномах с предком
                # 3) нормировка с учетом допустимого разброса infl (здесь -- 1)
                #    (т.е. перевод из процентов в абсолютные значения)
                sdvi = self.param['k1_infl'] * \
                       (parent.param['mu'] / 100) * 1 # [-1,0] -- макс диапазон изменений для self_infl

                # самовлияние
                popul.INF[self.index][self.index+1] = self.vary(popul.INF[parent.index][parent.index+1], sdvi,
                                                                popul.param_self_min, popul.param_self_max)

                ########
                # влияние среды на новичка
                popul.INF[self.index][0] = self.vary(popul.INF[parent.index][0], sdvi, const_env_infl_min,
                                                     const_env_infl_max)

                # записываем новый организм в популяцию
                popul.orgs.append(self)


    # служебная процедура сдвига параметра и проверки попадания в лимиты после сдвига
    def vary(self,base,sdvig,min_lim,max_lim):

        tmp = base + uniform(-sdvig, sdvig) # сдвигаем
        tmp = min(tmp, max_lim)  # убедимся, что оно не поднимется выше xmax
        tmp = max(tmp, min_lim)  # и не опустится ниже xmin

        return tmp

############################################################
#
# MAIN CYCLE
#
############################################################
ok = False

infl = np.array([value_INFL])
fora = np.array([value_FORA])
mu = np.array([value_MU])
sos = np.array([value_SOS])
popmax = np.array([value_POPMAX])
sol = np.array([value_SOL])

why_stop = 0

start_time = time.time()
popul = population()
ok =  popul.live()
string = popul.output()

if ok == "Ceased":
    why_stop = 1
elif ok == "Too large": 
    why_stop = 2
elif ok == "Too many species": 
    why_stop = 3
elif ok == "Success": 
    why_stop = 4
elif ok == "1-stabilized": 
    why_stop = 5
elif ok == "Time out": 
    why_stop = 6

string = string + ';' + str(why_stop)

long_text_compressed = zlib.compress(string.encode('utf-8'))
try:
  f = open('./data_out/' + str(value_PATH) + '/pm_' + str(value_POPMAX) + '_' +  str(value_NUM) + '.txt', 'wb')
  f.write(long_text_compressed)
  f.close()
except:
  print("что то с записью в файл", flush = True)

print('Ok')
