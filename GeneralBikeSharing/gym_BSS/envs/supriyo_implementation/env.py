import gym
import numpy as np
import os.path
from gym import error, spaces, warnings
from gym.utils import seeding


class BSSEnv(gym.Env):

    def __init__(self, nzones=95, ntimesteps=100, data_dir=None, data_set_name='actual_data_art', scenarios=list(range(1, 500))):
        super().__init__()
        self.scenarios = list(range(1, 500))
        self.nzones = nzones
        self.ntimesteps = ntimesteps
        #self.scenarios = scenarios
        # print(scenarios)
        self.data_set_name = data_set_name
        """
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "default_data")
           """
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__))

        self.data_dir = data_dir
        self.__read_data()
        self.capacities = np.array(self.__cp)
        self.starting_allocation = np.array(self.__ds)
        # print(self.starting_allocation)
        self.max_demand = 100
        self.metadata = {
            'render.modes': [],
            'nzones': self.nzones,
            'ntimesteps': self.ntimesteps,
            'nbikes': self.nbikes,
            'capacities': self.capacities,
            'data_dir': self.data_dir,
            'scenarios': self.scenarios
        }
        self.observation_space = spaces.Box(low=np.array([0] * (2 * self.nzones + 1)), high=np.array(
            [self.max_demand] * self.nzones + list(self.capacities) + [self.ntimesteps]), dtype=np.float32)
        self.action_space = spaces.Box(low=np.zeros(
            [self.nzones]), high=self.capacities, dtype=np.float32)
        self._scenario = 20
        self.seed(None)
        """
    def __read_data(self):
        self.__read_capacity_and_starting_allocation(
            os.path.join(self.data_dir, "demand_bound_artificial_60.txt"))
        self.__read_zone_distances(os.path.join(
            self.data_dir, "RawData", "distance_zone.txt"))
        self.__read_demand_data(self.scenarios, os.path.join(
            self.data_dir, "DemandScenarios", self.data_set_name, "DemandScenarios1", "demand_scenario_{scenario}.txt"))
        """

    def __read_data(self):
        self.__read_capacity_and_starting_allocation(os.path.join(
            self.data_dir, "demand_bound.txt"))
        self.__read_zone_distances(os.path.join(
            self.data_dir, "distance_zone.txt"))
        self.__read_demand_data(self.scenarios, os.path.join(
            self.data_dir, "demand_scenario", "demand_scenario_{scenario}"))

    def __read_capacity_and_starting_allocation(self, filename):
        f = open(filename)
        line = f.readline()  # 95
        self.nzones = int(line)
        print(self.nzones)

        self.__cp = [0 for k in range(self.nzones)]
        self.__ds = [[0.0 for k in range(self.nzones)] for j in range(
            self.ntimesteps + 1)]  # Distribution is zones
        line = f.readline()  # 15,15,15,....19
        line = line.strip(" \n")
        line = line.split(" ")  # 分割數字成['15','15','15',....]
        for s in range(self.nzones):
            self.__cp[s] = int(line[s])

        line = f.readline()
        line = line.strip(" \n")
        line = line.split(" ")  # line= '7','7','7',......'9' 也是95個

        self.nbikes = 0
        for s in range(0, self.nzones):
            self.__ds[0][s] = int(line[s])
            self.nbikes = self.nbikes + self.__ds[0][s]
        # print(self.nbikes)
        f.close()

    def __read_demand_data(self, scenarios, filename_unformatted):
        self.demand_data = {}
        # print(scenarios) scenarios=21~60
        for scenario in scenarios:
            flow = [[[0.0 for k in range(self.nzones)] for j in range(
                self.nzones)] for i in range(self.ntimesteps)]  # Known Flow
            f2 = open(filename_unformatted.format(scenario=scenario))
            for i in range(self.ntimesteps):
                for j in range(self.nzones):
                    line = f2.readline()
                    line = line.strip(" \r\n")
                    line = line.split(" ")
                    # print(line)
                    for k in range(self.nzones):
                        if line[k] == '':
                            break
                        flow[i][j][k] = float(line[k])
            f2.close()
            self.demand_data[scenario] = flow

    def __read_zone_distances(self, filename):
        self.__dis = [[0.0 for k in range(self.nzones)]
                      for i in range(self.nzones)]
        f2 = open(filename)
        line = f2.readline()
        ma = 0          # max distance
        T = 0
        for T in range(self.nzones):
            line = line.strip(' \r\n ')
            line = line.split(" ")
            for i in range(self.nzones):
                self.__dis[T][i] = float(line[i])  # /10000.0
                if(self.__dis[T][i] > ma):
                    ma = self.__dis[T][i]
            line = f2.readline()
        f2.close()
        # print(ma)
        for i in range(self.nzones):
            self.__dis[i][i] = 0  # 自己到自己=0

        self.__mindis = [[-1 for k in range(self.nzones)]
                         for i in range(self.nzones)]
        # initial mindis=-1
        for i in range(self.nzones):
            sortindex = sorted(
                range(len(self.__dis[i])), key=lambda k: self.__dis[i][k])  # 排序i到其他人的距離當index
            for j in range(self.nzones):
                self.__mindis[i][j] = sortindex[j]  # sort過的距離(按照小到大排)

    def seed(self, seed=None):
        if seed is None:
            seed = seeding.create_seed(max_bytes=4)
        self.__nprandom = np.random.RandomState(seed)
        return [seed]

    def _get_observation(self):
        if self.__t == 0:
            # initial demand_2d = 0
            demand_2d = np.zeros(shape=[self.nzones, self.nzones])
        else:
            demand_2d = np.array(self.__fl[self.__t - 1])
        assert list(demand_2d.shape) == [self.nzones, self.nzones]
        demand_1d = np.sum(demand_2d, axis=1)
        alloc = np.array(self.__ds[self.__t])  # 如何分配
        obs = np.concatenate([demand_1d, alloc, [self.__t]])  # 放在一起 = state
        assert list(obs.shape) == list(self.observation_space.shape)
        return obs

    # reset allocation to initial [0][0~95]有值 其他0
    def __reset_allocation(self):
        self.__ds = list(self.starting_allocation)

    def __reset_flow(self, scenario):
        #self.__fl = self.demand_data[scenario]

        # modify into random demand

        self.__fl = {}
        demand_flow = [[[0.0 for k in range(self.nzones)] for j in range(
            self.nzones)] for i in range(self.ntimesteps)]
        for i in range(self.ntimesteps):
            for j in range(self.nzones):
                for k in range(self.nzones):
                    demand_flow[i][j][k] = float(
                        np.random.randint(5, 25)) if j != k else 0.0
        self.__fl = demand_flow

        ####################################################
        self.__xfl = [[[0.0 for k in range(self.nzones)] for j in range(
            self.nzones)] for i in range(self.ntimesteps)]  # Actual computed Flow
        self.__tfl1 = [[0.0 for k in range(self.nzones)]
                       for j in range(self.ntimesteps)]

        for t in range(0, self.ntimesteps):
            for s in range(0, self.nzones):
                for s1 in range(0, self.nzones):
                    self.__tfl1[t][s] = self.__tfl1[t][s] + self.__fl[t][s][s1]

    def reset(self):
        # pick up a day at random
        self._scenario = self.scenarios[self.__nprandom.randint(
            len(self.scenarios))]
        # self._scenario = self._scenario + 1
        # print("demand scenario is:", self._scenario)
        self.__reset_allocation()
        self.__reset_flow(self._scenario)

        self.__yp = [[0.0 for k in range(self.nzones)]
                     for j in range(self.ntimesteps)]
        self.__yn = [[0.0 for k in range(self.nzones)]
                     for j in range(self.ntimesteps)]
        self.__t = 0

        return self._get_observation()

    def __set_yp_yn_from_action(self, action):
        if action is None:
            warnings.warn(
                "no action was provided. taking default action of not changing allocation")
        else:
            action = np.array(action)
            if not(hasattr(action, 'shape') and list(action.shape) == list(self.action_space.shape)):
                raise error.InvalidAction(
                    'action shape must be as per env.action_space.shape. Provided action was {0}'.format(action))
            if np.round(np.sum(action)) != self.nbikes:
            # if abs(np.sum(action) - self.nbikes) > 1:
                raise error.InvalidAction(
                    'Dimensions of action must sum upto env.metadata["nbikes"]. Provided action was {0} with sum {1}'.format(action, sum(action)))
            if np.any(action < -1e-6):
                raise error.InvalidAction(
                    'Each dimension of action must be positive. Provided action was {0}'.format(action))
            if np.any(action > self.capacities + 1e-6):
                raise error.InvalidAction(
                    'Individual dimensions of action must be less than respective dimentions of env.metadata["capacities"]. Provided action was {0}'.format(self.capacities - action))
            #print("action: ", action)
            #print("current_alloc", self.__ds[self.__t])
            # print("settime",self.__t)
            alloc_diff = action - np.array(self.__ds[self.__t])
            yn = alloc_diff * (alloc_diff > 0)
            yp = - alloc_diff * (alloc_diff < 0)
            # print(self.__t)
            self.__yp[self.__t] = list(yp)
            self.__yn[self.__t] = list(yn)

    def __calculate_lost_demand_new_allocation(self):
        full_lost = 0.0
        iteration = self.__t
       # print("calculatetime",iteration)
        moving_cost = (sum(self.__yp[iteration])+sum(self.__yn[iteration]))/2
        # assert abs(sum(self.__yp[iteration]) - sum(self.__yn[iteration])) < 1e-6, "sum(yp)={0}\nsum(yn)={1}\nyp={2}\nyn={3}".format(
        #     sum(self.__yp[iteration]), sum(self.__yn[iteration]), self.__yp[iteration], self.__yn[iteration])
        assert np.all(np.array(self.__yp[iteration]) >= -0.0)
        assert np.all(np.array(self.__yn[iteration]) >= -0.0)
        # before_reallocation=760
        before_reallocation = sum(self.__ds[iteration])
        # print("test",self.__ds[iteration+1])   #until here equal to action!! ds
        # print("ds[i]",self.__ds[iteration])

        #print("Sum before reallocation:", before_reallocation)
        for s in range(self.nzones):
            # and ((yn[iteration][s]-yp[iteration][s])<=cp[s]-ds[iteration][s])):
            # print(self.__yp[iteration][s],"testing")
            # print(self.__yn[iteration][s])
            if((self.__ds[iteration][s] >= (self.__yp[iteration][s] - self.__yn[iteration][s]))):
                self.__ds[iteration][s] = self.__ds[iteration][s] - \
                    (self.__yp[iteration][s] - self.__yn[iteration][s])
            # elif((self.__yn[iteration][s] - self.__yp[iteration][s]) > self.__cp[s] - self.__ds[iteration][s]):
            #     self.__ds[iteration][s] = self.__cp[s]
            else:
                # print(self.__yp[iteration][s],"testing")
                # print(self.__yn[iteration][s])
                self.__ds[iteration][s] = 0.0
        # print("test",self.__ds[iteration+1])   #until here equal to action!! ds

        for s in range(self.nzones):
            for s1 in range(self.nzones):
                # if(self.__tfl1[iteration][s] <= self.__ds[iteration][s]):
                #     self.__xfl[iteration][k][s][s1] = self.__fl[iteration][k][s][s1]
                # else:
                if(self.__tfl1[iteration][s] > 0):
                    # print(self.__tfl1[iteration][s]==sum(self.__fl[iteration][s])) 恆TRUE
                    self.__xfl[iteration][s][s1] = min(self.__ds[iteration][s], sum(
                        self.__fl[iteration][s])) * (self.__fl[iteration][s][s1] / (self.__tfl1[iteration][s] * 1.0))
            # print("before",self.__xfl[iteration][s])
            # print("test",self.__tfl1[iteration][s])
            # print("violate",self.__ds[iteration][s]<sum(self.__fl[iteration][s]))
            if(self.__tfl1[iteration][s] > 0 and (self.__ds[iteration][s]) < sum(self.__fl[iteration][s])):
                # print("before",self.__xfl[iteration][s])
                self.__xfl[iteration][s] = np.rint(self.__xfl[iteration][s])
                # print("modify",self.__xfl[iteration][s])

                for find_last in range(self.nzones-1, -1, -1):
                    if(self.__xfl[iteration][s][find_last] != 0):
                        self.__xfl[iteration][s][find_last] = self.__ds[iteration][s] - \
                            sum(self.__xfl[iteration][s][0:find_last])
                        break
                #print("after rint",self.__xfl[iteration][s])
            # print("after",self.__xfl[iteration][s])
                    # self.__fl[iteration][s][s1] / (self.__tfl1[iteration][s] normalize?
        # print("test2",self.__ds[iteration+1])

        for i in range(self.nzones):
            self.__ds[iteration + 1][i] = self.__ds[iteration][i] - \
                min(self.__ds[iteration][i], sum(self.__fl[iteration][i]))
        # print("test3",self.__ds[iteration+1])
        for z in range(self.nzones):
            for z1 in range(self.nzones):
                if(sum(self.__fl[iteration][z1]) > 0):
                    # (1.0*min(ds[iteration][z1],sum(fl[iteration][z1]))*fl[timstep][z1][z])/sum(fl[iteration][z1])
                    self.__ds[iteration + 1][z] = self.__ds[iteration + 1][z] + \
                        self.__xfl[iteration][z1][z]
        # print("test4",self.__ds[iteration+1])

        flag = 0

        after_reallocation = sum(self.__ds[iteration + 1])
       # print("before allocate",self.__ds[iteration+1])
        # print("Sum after reallocation:", sum(self.__ds[iteration + 1]))
        # assert abs(after_reallocation - before_reallocation) < 1e-6, "This is where the bug is. sum before reallocation={0}. sum after reallocation={1}\nallocation_before={2}\nallocation_after={3}".format(
        #     before_reallocation, after_reallocation, self.__ds[iteration], self.__ds[iteration + 1])
        while(flag == 0):
            for s in range(self.nzones):
                if(self.__ds[iteration + 1][s] > self.__cp[s]):
                    #print("readjusting for zone", s)
                    for s1 in self.__mindis[s]:
                        if((self.__ds[iteration + 1][s] - self.__cp[s]) <= (self.__cp[s1] - self.__ds[iteration + 1][s1])):
                            self.__ds[iteration + 1][s1] = self.__ds[iteration +
                                                                     1][s1] - self.__cp[s] + self.__ds[iteration + 1][s]
                            full_lost += self.__ds[iteration +
                                                   1][s] - self.__cp[s]
                            self.__ds[iteration + 1][s] = self.__cp[s]
                            break
                        elif(((self.__cp[s1] - self.__ds[iteration + 1][s1]) > 0) and ((self.__ds[iteration + 1][s] - self.__cp[s]) > (self.__cp[s1] - self.__ds[iteration + 1][s1]))):
                            self.__ds[iteration + 1][s] = self.__ds[iteration + 1][s] - \
                                (self.__cp[s1] - self.__ds[iteration + 1][s1])
                            full_lost += self.__cp[s1] - \
                                self.__ds[iteration + 1][s1]
                            self.__ds[iteration + 1][s1] = self.__cp[s1]
            after_readjustment_sum = sum(self.__ds[iteration + 1])
            assert abs(after_reallocation - after_readjustment_sum) < 1e-6, "This is where the bug is. starting_sum={0}. After readjustment, sum={1}".format(
                after_reallocation, after_readjustment_sum)

            flag = 1
            for s in range(self.nzones):
                assert self.__ds[iteration +
                                 1][s] <= self.__cp[s], "I am stuck. Something is wrong. Readjustment should have finished in one pass"
       # print("test4",self.__ds[iteration+1])
       # print("ds[i]",self.__ds[iteration])
        lost_call = 0
        revenue = 0
        for s in range(self.nzones):
            for s1 in range(self.nzones):
                revenue += self.__xfl[iteration][s][s1]
                lost_call += self.__fl[iteration][s][s1] - \
                    self.__xfl[iteration][s][s1]
        return lost_call, full_lost, moving_cost, revenue

    def step(self, action):
        # modify yp and yn here according to action
        self.__set_yp_yn_from_action(action)
        lost_demand, full_lost_demand, moving_cost_demand, revenue = self.__calculate_lost_demand_new_allocation()
        r = -(lost_demand + full_lost_demand+2*moving_cost_demand)
        # print("lost_call",lost_demand,"full",full_lost_demand)
        self.__t += 1
        done = self.__t >= self.ntimesteps
        return self._get_observation(), r, done, {"lost_demand_pickup": lost_demand, "lost_demand_dropoff": full_lost_demand, "revenue": revenue, "scenario": self._scenario}

    def render(self, mode='human', close=False):
        if not close:
            raise NotImplementedError(
                "This environment does not support rendering")
