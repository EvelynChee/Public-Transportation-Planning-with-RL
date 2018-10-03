import numpy as np

class Station:
    """Common base class for all bus stations. Each day consists of 1000
    unit time. In each hour of the day, the station might have higher passenger
    arrival rate during the last 15 minutes of the hour (peak hour).

    Args:
        peak_rate (int): An integer >= 1 indicating the mean time
                         interval in between the arrival between two
                         passengers during peak hour.
        non_peak_rate (int): An integer >= 1 indicating the mean time
                             interval in between the arrival between two
                             passengers during non-peak hour.
        max_passenger (int): Maximum number of passengers that the station
                             could contain.

    """

    def __init__(self, peak_rate, non_peak_rate, max_passengers):
        self.peak = peak_rate
        self.non_peak = non_peak_rate
        self.passengers = 0             # Current number of passengers at the station
        self.buses = None               # Current bus at the station if available
        self.max = max_passengers

    def add_passenger(self, time):
        """Adds passengers to bus stops randomly based on the arrival rate unless
        the maximum is reached. The rate depends on the current time.

        Args:
            time (int): Current time.

        Returns:
            Initial total number of passengers at the station.

        """
        # Higher passenger arrival rate during last 15mins of every hour
        if 90 <= (time % 120) < 120:
            rate = self.peak
        else:
            rate = self.non_peak

        total = self.passengers
        self.passengers = min(self.max, self.passengers+np.random.binomial(1,1/rate))
        return total

    def empty(self):
        """Checks if there is a bus picking up passengers at the station.

        Returns:
            True if there is a bus, False otherwise.

        """
        return self.buses == None

    def incoming(self, bus):
        """Bus coming in to the station to pick up the passengers.

        Args:
           bus (Bus): Incoming bus.

        Returns:
           Total number of passengers picked up.

        """
        self.buses = bus
        n = self.passengers
        self.passengers = 0
        bus.boarding(n)
        return n

    def outgoing(self, bus):
        """Removes bus from the station if it is currently at the station.

        Args:
            bus (Bus): Bus leaving the station.

        """
        if self.buses == bus:
            self.buses = None

class Bus:
    """Common base class for all buses.

    Args:
        deploy_cost (float): Cost of deploying the bus.
        boarding_rate (int): Number of passengers boarding the bus per unit time.

    """
    def __init__(self, deploy_cost, boarding_rate):
        self.waiting = 0                # Total time that the bus still needs to wait at a station
        self.location = 0               # Current location of the bus in the route
        self.departed = False           # Indicates if the bus has departed from terminal
        self.deploy_cost = deploy_cost
        self.boarding_rate = boarding_rate

    def depart(self):
        """Deploys the bus from terminal.

        Returns:
            The cost of deploying the bus.
        """
        self.departed = True
        return -1 * self.deploy_cost

    def arrive(self):
        """Bus arrives back at the terminal and waiting for deployment.

        """
        self.departed = False
        self.location = 0

    def forward(self):
        """Bus moves forward by one unit in the route.

        """
        self.location += 1

    def boarding(self, passengers):
        """Bus picking up passengers at a station. The time needed for passengers
        to board the bus depends on the number of passengers and boarding rate.

        Args:
           passengers (int): Number of passengers that needs to be picked up.

        """
        self.waiting += np.ceil(passengers / self.boarding_rate)

class BusLine:
    """Common base class for the bus route and contains Stations and Buses.
    Buses travel in one direction and in a loop. Only one terminal available.
    Buses operate for 1920 unit time in each day (equivalent to 16 hours).

    Args:
        distance (int): Distance of the bus route.
        waiting_cost (float): Cost for each waiting passenger per unit time.
        overtake_penalty (float): Cost for a bus overtaking another.
        pickup_reward (float): Reward for picking up a passenger.

    """
    def __init__(self, distance, waiting_cost, overtake_penalty, no_bus_penalty, pickup_reward):
        self.time = 0                                   # Current time of the day
        self.map = np.zeros(distance, dtype=Station)    # Map of bus route indicating locations of Station
        self.buses = []                                 # List of buses ready for deployment
        self.N = 0                                      # Number of stations
        self.M = 0                                      # Number of buses
        self.D = distance
        self.waiting_cost = waiting_cost
        self.overtake_penalty = overtake_penalty
        self.no_bus_penalty = no_bus_penalty
        self.pickup_reward = pickup_reward

    def add_station(self, peak_rate, non_peak_rate, location, max_passengers):
        """Adds station to bus route if no station exists at that location and the
        location is within the bus route.

        Args:
            peak_rate (int): An integer >= 1 indicating the mean time
                             interval in between the arrival between two
                             passengers during peak hour.
            non_peak_rate (int): An integer >= 1 indicating the mean time
                                 interval in between the arrival between two
                                 passengers during non-peak hour.
            location (int): Location of the station to be added.
            max_passengers (int): Maximum number of passengers that the station
                                  could contain.

        """
        if location >= len(self.map):
            print('Location is out of bound.')
        elif isinstance(self.map[location], Station):
            print('Bus station already exists at current location.')
        else:
            self.map[location] = Station(peak_rate, non_peak_rate, max_passengers)
            self.N += 1

    def add_bus(self, deploy_cost, boarding_rate):
        """Adds a bus which is ready to be deployed for this bus route at terminal.

        Args:
            deploy_cost (float): Cost of deploying this bus.
            boarding_rate (int): Number of passengers that can board this bus
                                 per unit time.

        """
        self.buses.append(Bus(deploy_cost, boarding_rate))
        self.M += 1

    def get_feature(self):
        """Gets the current representation of the environment. Information on
        the number of passengers at each bus station and the distance of the
        nearest bus from the terminal will be given.

        Returns:
            An array where the first element is the current time, the next
            N elements indicates the the number of passengers at each of the
            N bus stops and the last element indicates the distance of the
            nearest bus from the terminal. Values are normalized such that
            they are all in between 0 and 1.

        """
        feature = np.zeros(self.N+2)

        # First element is the current time out of one day.
        feature[0] = self.time/1920

        # To obtain the first N elements, the number of passengers at all
        # stations in the route are obtained.
        i = 1
        total_max = 0   # Total maximum passengers that the bus route can have
        for loc in self.map:
            if isinstance(loc, Station):
                feature[i] = loc.passengers
                total_max += loc.max
                i += 1

        # Divides the first N elements by the total maximum passengers that the
        # N stations could contain. Each element indicates the proportion of
        # passengers at that station out of the total maximum.
        feature[1:-1] /= total_max

        # The nearest bus from the terminal is located and only the distance of
        # the deployed buses are considered. If all buses are currently at the
        # terminal, then the nearest distance would be D+1.
        feature[-1] = self.D+1
        for bus in self.buses:
            if bus.location > 0:
                feature[-1] = min(feature[-1], bus.location)

        # Normalizing the last element to be in between 0 and 1 by scaling it
        # with the distance of the bus route.
        feature[-1] /= self.D + 1

        return feature.reshape((1,self.N+2))

    def move_forward(self):
        """Moving forward the system by one time step, which includes passengers
        arriving at the stations and buses that are deployed moving
        forward by one unit.

        Returns:
            The toal reward received during the one unit time.

        """
        # Each day has 16 operating hours and 1 unit time = 0.5 mins.
        # Hence, 0 <= time <= 1920.
        self.time = (self.time+1)%1920


        reward = 0

        # Only buses that have departed from the terminal are considered.
        for bus in self.buses:
            if bus.departed:
                # If bus is picking up passengers at a bus station, then decrease
                # the waiting time by one unit time.
                if bus.waiting > 0:
                    bus.waiting -= 1
                else:
                    # If bus is located at a bus station, then it will be leaving the
                    # station since all passengers have boarded the bus.
                    if bus.location != 0:
                        old_loc = self.map[bus.location-1]
                        if isinstance(old_loc, Station):
                            old_loc.outgoing(bus)

                    # Bus will move forward by one unit distance.
                    bus.forward()

                    # Check if bus arrives at the terminal.
                    if bus.location == len(self.map)+1:
                        bus.arrive()
                    else:
                        # If bus arrives at a bus station, then it will pick up passengers
                        # if there is no other bus at that station currently. It will
                        # overtake the other bus if otherwise.
                        new_loc = self.map[bus.location-1]
                        if isinstance(new_loc, Station):
                            if new_loc.empty():
                                reward += new_loc.incoming(bus) * self.pickup_reward
                            else:
                                reward -= self.overtake_penalty

        # Passengers are added to each bus station.
        for s in self.map:
            if isinstance(s, Station):
                reward -= s.add_passenger(self.time) * self.waiting_cost

        return reward

    def take_action(self, n=0):
        """Bus operator takes an action. n=0 is not deploying any bus and n=1 is to
        deploy one bus from terminal into the route. After taking action,
        everything move forward by one unit time accordingly.

        Args:
            n (int, optional): Action to be taken. Default 0.

        Returns:
            The total reward received after taking the action.

        """
        reward = 0

        if n == 1:
            # Only deploy one bus which is currently at the terminal.
            for bus in self.buses:
                if not bus.departed:
                    reward += bus.depart()
                    reward += self.move_forward()
                    return reward
            # Penalized for deploying a bus when there is none.
            reward -= self.no_bus_penalty

        reward += self.move_forward()
        return reward
