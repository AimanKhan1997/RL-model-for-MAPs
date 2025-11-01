import gymnasium as gym
import numpy as np
import traci
import math
import stable_baselines3 as sb3
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from gym.spaces import MultiDiscrete, Box

BATTERY_CAPACITY = 640 # battery capacity of electric vehicles in Wh
BATTERY_CAPACITY_POD = 2000 # battery capacity of charging pods in Wh
LOW_BATTERY_THRESHOLD = 20 # 20% battery capacity
SLOWDOWN_SPEED_ZERO_RANGE = 3 # reduced speed for vehicles with zero remaining range
WIRELESS_POD_POWER_RATING = 18000  # W
#CHARGE_RATE = WIRELESS_POD_POWER_RATING / 3600  # Wh per second
DURATION = 80  # seconds
#total_energy_charged = 0
elec_consumption = 0
total_energy_delivered_ini = 0
#CHARGING_DISTANCE_THRESHOLD = 40  # meters
Max_charge_for_EVs= 80
# fixed-size limits for SB3 compatibility
MAX_EVS = 75       # max EV slots we encode in obs (choose >= expected active EVs)
MAX_PODS = 60      # max Pods
MAX_PARKINGS = 12  # number of parking areas
CHARGE_LEVELS = 9  # 0..8 meaning 0%,10%,...,80%
NOOP_EV = MAX_EVS  # index reserved for "no EV chosen"
NOOP_POD = MAX_PODS
timestamps = []
speeds = []
k_ev = 4
k_pod = 4
k_global = 2 + MAX_PARKINGS
# Dictionary to track the last speed and time for each charging pod
charging_pod_speeds = {}
# Define your custom environment that integrates SUMO and the charging pod simulation
class ChargingPodEnv(gym.Env):
    def __init__(self, sumo_config_file, parking_areas_list, load_state_file=None):
        super(ChargingPodEnv, self).__init__()
        # Define action and observation space
        self.action_space = gym.spaces.Discrete(3)  # 3 actions: assign pod, stop charging, slow down, charge_pod
        obs_len = k_global + MAX_EVS * k_ev + MAX_PODS * k_pod
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32)
        # Initialize SUMO and other variables
        self.sumo_config_file = sumo_config_file
        self.parking_areas_list = parking_areas_list
        self.episode_count = 0  # ✅ track episode numbers
        self.current_step = 0
        self.total_energy_charged = 0
        self.elec_consumption = 0
        self.load_state_file = load_state_file
        self.done_time = 700  # Simulation timestep at which training ends
        self.warm_up_time= 500
        self.assigned_charging_pod_for_electric_veh = {}
        self.CHARGE_RATE=18000/3600  # Wh per second
        self.pod_stall_timers = {}  # key: pod_id, value: timestep count of being stalled
        self.STALL_THRESHOLD = 60  # e.g., if pod is stalled for 60 steps, take action
        self.CHARGING_DISTANCE_THRESHOLD = 40  # meters
        self.zero_range_vehicles = set()
        self.low_battery_vehicles = set()
        self.zero_range_pods = set()
        self.episode_reward = 0
        self.current_step = 0
        self.pod_lane_positions = {}
        self.ev_lane_positions = {}
        self._last_penalty_time = -1  # To track the last penalty time for parking area occupancy
        # bookkeeping for assignments -> used to compute approach rewards
        self.assignment_time_steps = {}  # mapped by vehicle_id
        self.assignment_start_distance = {}  # vehicle_id -> distance at moment of assignment
        self.assignment_started_charging = {}  # vehicle_id -> bool (has reached charging distance)
        self.slow_speed= 6.5  # Speed to slow down low battery vehicles
        self.pod_last_distances = {}  # pod_id -> last distance reading
        self.pod_total_distances = {}  # pod_id -> total traveled distance
        self.episode_zero_range = 0
        self.episode_energy = 0.0
        self.station_energy = 0.0
        self.total_energy_delivered = 0.0
        self.total_energy_delivered_ini=0.0
        self.total_energy_charged = 0.0
        self.actual_energy_delivered=0.0
        self.parking_areas = ["pa_0", "pa_4", "pa_5", "pa_7", "pa_8", "pa_10", "pa_11", "pa_3", "pa_0", "pa_4", "pa_5", "pa_6","pa_2", "pa_3","pa_0", "pa_1","pa_2","pa_3", "pa_0", "pa_1","pa_9","pa_10", "pa_11", "pa_3"]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # This sets self.np_random
        if hasattr(self, "episode_zero_range"):
            print(f"[Episode {self.episode_count} finished] Zero-range={self.episode_zero_range}, eff={(self.total_energy_charged / max(1, self.actual_energy_delivered)) * 100:.2f}")
        self.episode_count += 1
        self.current_step = 0
        self.station_energy = 0.0
        self.episode_reward = 0
        self.episode_zero_range = 0
        self.episode_energy = 0.0
        self.total_energy_delivered = 0.0
        self.total_energy_delivered_ini = 0.0
        self.actual_energy_delivered=0.0
        self._last_penalty_time = -1
        self.pod_last_distances.clear()
        self.pod_total_distances.clear()
        self.assigned_charging_pod_for_electric_veh.clear()
        self.zero_range_vehicles.clear()
        self.low_battery_vehicles.clear()
        self.zero_range_pods.clear()
        self.assignment_time_steps.clear()
        self.assignment_start_distance.clear()  # vehicle_id -> distance at moment of assignment
        self.assignment_started_charging.clear()
        self.pod_lane_positions.clear()
        self.ev_lane_positions.clear()
        self.pod_stall_timers.clear()
        self.total_energy_charged=0.0
        # if SUMO connection exists, close it first
        try:
            if traci.isLoaded():
                traci.close()
        except Exception:
            pass
        # if traci.isLoaded():
        #     traci.close()
        if self.load_state_file:
            traci.start(["sumo", "-c", self.sumo_config_file, "--load-state", self.load_state_file])
        else:
            traci.start(["sumo", "-c", self.sumo_config_file])
        obs = self._get_state()

        # Defensive check: no NaN or Inf in obs
        if not np.all(np.isfinite(obs)):
            print(f"[Warning] reset() returned invalid obs: {obs}")
            obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

        return obs, {}

    def _safe_zero_obs(self):
        # return an observation shaped like _get_state(), but safe when TRACI is down
        # Example: if obs size is 8 (4 scalars + 4 parking occupancy)
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def step(self, action):
        try:
            traci.simulationStep()
        except traci.exceptions.FatalTraCIError as e:
            print("[SUMO] Connection closed during simulationStep():", e)
            # prepare info to log episode end
            info ={ "episode" : {'r': self.episode_reward, 'l': self.current_step},
            "bad_exit" : True,
            "zero_range": 0,
            "eff": 0}  # <-- ensure key always exists
            # make sure SUMO is closed on our side (it likely already is)
            try:
                if traci.isLoaded():
                    traci.close()
            except Exception:
                pass
            # return safe observation and mark episode terminated so SB3 will call reset()
            return self._safe_zero_obs(), 0.0, True, False, info
        #traci.simulationStep()  # Advance the simulation by one step
        self.current_step += 1
        reward = self._apply_action(action)
        step_dist_penalty = 0.0

        for ev_id in traci.vehicle.getIDList():
            if traci.vehicle.getTypeID(ev_id) == "ElectricVehicle":
                #self.ev_lane_positions[ev_id] = traci.vehicle.getLanePosition(ev_id)
                soc = (float(traci.vehicle.getParameter(ev_id, "device.battery.actualBatteryCapacity"))/ BATTERY_CAPACITY) * 100
                if soc <= 0:
                    reward += self.handle_zero_range_vehicle(ev_id)
                elif soc <= LOW_BATTERY_THRESHOLD:
                    reward -= 0.5
                # elif soc > LOW_BATTERY_THRESHOLD:
                #     reward += 1
            # else:
            #     self.ev_lane_positions[ev_id] = None
            if ev_id in self.assigned_charging_pod_for_electric_veh:
                self._start_charging(ev_id, self.assigned_charging_pod_for_electric_veh[ev_id])
        for pod_id in traci.vehicle.getIDList():
            if traci.vehicle.getTypeID(pod_id) == "ChargingPod":
                try:
                    current_distance = traci.vehicle.getDistance(pod_id)
                    last_distance = self.pod_last_distances.get(pod_id, current_distance)
                    if traci.simulation.getTime() >= self.warm_up_time:
                        elec_consumption= traci.vehicle.getElectricityConsumption(pod_id)
                        self.episode_energy+= elec_consumption
                    #reward -= (max(0,elec_consumption)/max(1,elec_consumption)) * 0.1  # Penalize electricity consumption
                    #print(f"reward after elec consumption penalty: {reward} for pod {pod_id} with elec consumption {elec_consumption}")
                    soc = (float(traci.vehicle.getParameter(pod_id, "device.battery.actualBatteryCapacity"))/ BATTERY_CAPACITY_POD) * 100
                    if soc <= 0:
                        reward -= 2  # Penalty for pod with zero battery
                        self._stop_charging(ev_id, pod_id)
                        self.zero_range_pods.add(pod_id)
                    elif soc <= 10:
                        reward -= 1
                        self._stop_charging(ev_id, pod_id)
                    traveled = max(0, current_distance - last_distance)
                    step_dist_penalty += 1e-6 * traveled
                    #reward -= step_dist_penalty
                    #print(f"reward after distance penalty: {reward} for pod {pod_id} with traveled {traveled}")
                    # Update cumulative distance
                    self.pod_total_distances[pod_id] = self.pod_total_distances.get(pod_id, 0) + traveled
                    # Store for next step
                    self.pod_last_distances[pod_id] = current_distance
                    if traci.vehicle.getRoadID(pod_id) == "E3":
                        traci.vehicle.setRouteID(pod_id, "r_1")
                    ##FORCE TO PARKING AREA IF NOT ASSIGNED
                    if traci.vehicle.getRoadID(pod_id) in ["E7", "E1", "E2", "E10",
                                                               "E11"] and pod_id not in self.assigned_charging_pod_for_electric_veh.values() \
                            and not traci.vehicle.isStoppedParking(pod_id) and traci.vehicle.getLanePosition(
                        pod_id) < 430:
                        # Map edge to corresponding parking area
                        edge_to_parking = {"E7": "pa_6", "E1": "pa_1", "E2": "pa_2", "E10": "pa_9",
                                           "E11": "pa_10"}  # Update mapping if parking area IDs differ
                        parking_area = edge_to_parking.get(traci.vehicle.getRoadID(pod_id))
                        next_stops = traci.vehicle.getNextStops(pod_id)
                        if next_stops:
                            next_stop_id = next_stops[0][2]
                            if parking_area and next_stop_id != parking_area:
                                traci.vehicle.setParkingAreaStop(pod_id, parking_area, duration=90000)
                                print(f" → Moved {pod_id} to available lot {parking_area} forcefully")
                                try:
                                    traci.vehicle.rerouteParkingArea(pod_id, parking_area)
                                    # forced_pods.add(vehicle_id)
                                except traci.exceptions.TraCIException as e:
                                    print(f"Error rerouting parking area for {pod_id}: {e}")
                except traci.exceptions.TraCIException:
                    continue
                #self.pod_lane_positions[pod_id] = traci.vehicle.getLanePosition(pod_id)
            #else:
                #self.pod_lane_positions[pod_id] = None
        self.handle_electric_vehicles()
        reward -= 1 * len(self.zero_range_vehicles)
        self.update_pod_routes(self.assigned_charging_pod_for_electric_veh)
        ##Calculate energy charged by charging stations
        for charging_station in traci.chargingstation.getIDList():
            if traci.simulation.getTime() == self.warm_up_time:
                energy_charged = float(
                    traci.simulation.getParameter(charging_station, "chargingStation.totalEnergyCharged"))
                self.total_energy_delivered_ini += energy_charged
            if traci.simulation.getTime() == self.done_time:
                energy_charged = float(
                    traci.simulation.getParameter(charging_station, "chargingStation.totalEnergyCharged"))
                self.total_energy_delivered += energy_charged
                self.actual_energy_delivered = self.total_energy_delivered - self.total_energy_delivered_ini
                efficiency = self.total_energy_charged / self.actual_energy_delivered
                print(f"Energy charged by charging station {charging_station} is {energy_charged} Wh.")
                print(f"Total energy delivered by charging stations: {self.actual_energy_delivered} Wh.")
                print(f"energy charged {self.total_energy_charged} Wh and energy consumed {self.episode_energy} Wh")
                print(f"Overall efficiency: {efficiency * 100:.2f} %")
                print(f"pods with zero range: {self.zero_range_pods}")
        # Apply penalty if any parking area has zero occupancy
        sim_time = int(traci.simulation.getTime())
        terminated = sim_time >= self.done_time
        truncated= False
        info = {}
        self.episode_reward += reward
        if terminated:
            eff = (self.total_energy_charged / max(1, self.total_energy_delivered)) * 100
            info['episode'] = {'r': self.episode_reward, 'l': self.current_step}
            info['zero_range'] = self.episode_zero_range
            info['eff'] = eff
            # Strong reward if no EV reached zero range
            if self.episode_zero_range == 0:
                reward += 10  # big bonus for safety

            # Reward efficiency (scale carefully!)
            reward += 10 * eff  # weight by efficiency
            #reward -= self.episode_energy * 0.001  # penalize high energy consumption
        #reward = np.clip(reward, -100, 100)  # keep reward bounded
        # Defensive check on obs
        obs = self._get_state()
        if not np.all(np.isfinite(obs)):
            print(f"[Warning] step() produced invalid obs at step {self.current_step}: {obs}")
            obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

        # Defensive check on reward
        if not np.isfinite(reward):
            print(f"[Warning] Non-finite reward at step {self.current_step}: {reward}")
            reward = 0.0  # fallback safe value

        return obs, reward, terminated, truncated, info

    def close(self):
        if traci.isLoaded():
            traci.close()

    def _apply_action(self, action):
        active_vehicles = traci.vehicle.getIDList()
        # 1. update position caches (safe: ignore TraCI exceptions)
        for vid in active_vehicles:
            try:
                vtype = traci.vehicle.getTypeID(vid)
                # update lane positions for later use
                if vtype == "ChargingPod":
                    try:
                        self.pod_lane_positions[vid] = traci.vehicle.getLanePosition(vid)
                    except traci.exceptions.TraCIException:
                        self.pod_lane_positions.pop(vid, None)
                elif vtype == "ElectricVehicle":
                    try:
                        self.ev_lane_positions[vid] = traci.vehicle.getLanePosition(vid)
                    except traci.exceptions.TraCIException:
                        self.ev_lane_positions.pop(vid, None)
            except traci.exceptions.TraCIException:
                # vehicle disappeared between getIDList and getTypeID — skip
                continue
        candidate_evs = [vid for vid in traci.vehicle.getIDList()
                         if traci.vehicle.getTypeID(vid) == "ElectricVehicle"]
        reward = 0.0
        for vehicle_id in list(self.low_battery_vehicles):
            assign_pod = self.assigned_charging_pod_for_electric_veh.get(vehicle_id)
            if action == 0:
                if vehicle_id in self.low_battery_vehicles and vehicle_id not in self.assigned_charging_pod_for_electric_veh:
                    reward += self.handle_low_battery_vehicle(vehicle_id)
                else:
                    reward = -1
            elif action == 1:
                if vehicle_id is not None and assign_pod is not None:
                    soc_ev = float(traci.vehicle.getParameter(vehicle_id, "device.battery.actualBatteryCapacity"))
                    soc_percent = (soc_ev / BATTERY_CAPACITY) * 100
                    pod_pos = traci.vehicle.getPosition(assign_pod)
                    if soc_percent >= Max_charge_for_EVs:
                        reward += self._stop_charging(vehicle_id, assign_pod)
                    elif 60 < soc_percent < Max_charge_for_EVs and 430 < traci.vehicle.getLanePosition(assign_pod) < 450 :
                        reward += self._stop_charging(vehicle_id, assign_pod)
                    else:
                        reward -= 1  # Penalize stopping too early
                else:
                    reward -= 1  # Penalize invalid stop
            elif action == 2:
                if vehicle_id is not None:
                    reward += self.redistribute_pods()
        return reward

    def redistribute_pods(self):
        reward = 0.0
        parking_areas = ["pa_0", "pa_4", "pa_5", "pa_7", "pa_8", "pa_10", "pa_11", "pa_3", "pa_0", "pa_4", "pa_5",
                         "pa_6", "pa_2", "pa_3", "pa_0", "pa_1", "pa_2", "pa_3", "pa_0", "pa_1", "pa_9", "pa_10",
                         "pa_11", "pa_3"]
        n = len(parking_areas)
        for i in range(n):
            # Create mapping from parking area IDs to their edge IDs
            pa_to_edge = {
                pa_id: traci.parkingarea.getLaneID(pa_id).split('_')[0]
                for pa_id in parking_areas
            }
            current_area = parking_areas[i]
            prev_area = parking_areas[(i - 1) % n]
            current_edge = pa_to_edge[current_area]
            prev_edge = pa_to_edge[prev_area]

            current_occupants = traci.parkingarea.getVehicleIDs(current_area)
            prev_occupants = traci.parkingarea.getVehicleIDs(prev_area)

            # --- Vehicle density estimation (number of EVs on same or nearby edges) ---
            try:
                nearby_evs = [
                    ev for ev in traci.vehicle.getIDList()
                    if traci.vehicle.getTypeID(ev) == "ElectricVehicle"
                       and traci.vehicle.getRoadID(ev) in [current_edge, prev_edge]
                ]
                #print(f" number of evs at egde {current_edge} and {prev_edge} are {len(nearby_evs)}")
            except traci.TraCIException:
                nearby_evs = []

            if ((len(current_occupants) <=2 and len(prev_occupants) > 4) or (len(prev_occupants) >= 8)) and len(nearby_evs) >= 2:
                max_battery = -1
                pod = None
                for pod_id in prev_occupants:
                    if traci.vehicle.getTypeID(pod_id) == "ChargingPod":
                        battery = float(traci.vehicle.getParameter(pod_id, "device.battery.actualBatteryCapacity"))
                        if battery > max_battery:
                            max_battery = battery
                            pod = pod_id

                # 5. Check for opportunistic EV charging along the way
                nearest_ev, dist_to_ev = self._find_nearest_ev(pod)
                if nearest_ev not in self.low_battery_vehicles:
                    if nearest_ev and nearest_ev not in self.assigned_charging_pod_for_electric_veh:
                        soc_ev = float(traci.vehicle.getParameter(nearest_ev, "device.battery.actualBatteryCapacity"))
                        self.low_battery_vehicles.add(nearest_ev)
                        reward += 1  # partial reward for useful intermediate charging
                        #print(f"Pod {pod} will opportunistically charge EV {nearest_ev} with soc {(soc_ev / BATTERY_CAPACITY) * 100:.2f}% along the way")
                        continue
                    # 6. Move the pod to target parking area
                    try:
                        traci.vehicle.resume(pod)
                        traci.vehicle.setParkingAreaStop(pod, current_area, duration=90000)
                        reward += 0.5 # small reward for balancing occupancy
                        #print(f"Pod {pod} moved from {prev_area} to {current_area} to balance occupancy")
                    except traci.TraCIException as e:
                        print(f"⚠️ Could not move pod {pod} to: {e}")

        # 7. Penalize imbalance
        # imbalance = max(occupancies.values()) - min(occupancies.values())
        # reward -= 0.1 * imbalance

        return reward

    def _find_nearest_ev(self, pod_id):
        try:
            pod_edge = traci.vehicle.getRoadID(pod_id)
            pod_pos = traci.vehicle.getPosition(pod_id)
        except traci.exceptions.TraCIException:
            return None, float('inf')

        nearest_ev = None
        lowest_soc = 80
        min_distance = float('inf')

        for ev_id in traci.vehicle.getIDList():
            try:
                if traci.vehicle.getTypeID(ev_id) != "ElectricVehicle":
                    continue

                ev_edge = traci.vehicle.getRoadID(ev_id)
                if ev_edge != pod_edge:  # only same-edge EVs
                    continue

                soc = float(
                    traci.vehicle.getParameter(ev_id, "device.battery.actualBatteryCapacity")) / BATTERY_CAPACITY * 100
                pos_ev = traci.vehicle.getPosition(ev_id)
                dist = traci.simulation.getDistance2D(pod_pos[0], pod_pos[1], pos_ev[0], pos_ev[1])
                # Skip EVs outside SOC window
                if soc < 25 or soc > 80 or traci.vehicle.getLanePosition(ev_id) >=840:
                    continue

                if soc < lowest_soc or (soc == lowest_soc and dist < min_distance):
                    lowest_soc = soc
                    min_distance = dist
                    nearest_ev = ev_id

            except traci.exceptions.TraCIException:
                continue

        return nearest_ev, min_distance

    def handle_electric_vehicles(self):
        """
        Loop through all electric vehicles, check their SOC, and handle them accordingly.
        If their SOC is below the threshold, add them to the low_battery_vehicles list and assign them to the nearest pod.
        If their SOC is zero, handle them as zero-range vehicles.
        After charging, if SOC is back to normal, update their behavior.
        """
        active_vehicles = traci.vehicle.getIDList()
        SOC_MARGIN = 5  # Margin above the low battery threshold to consider charging
        for vehicle_id in active_vehicles:
            vehicle_type = traci.vehicle.getTypeID(vehicle_id)
            if vehicle_type == "ElectricVehicle":
                # if traci.vehicle.getRoadID(vehicle_id) == "E3":
                #     traci.vehicle.setRouteID(vehicle_id, "r_0_1")
                try:
                    # Get the SOC of the vehicle
                    battery_capacity = float(
                        traci.vehicle.getParameter(vehicle_id, "device.battery.actualBatteryCapacity"))
                    soc = (battery_capacity / BATTERY_CAPACITY) * 100
                    # If SOC is below the threshold, handle the vehicle
                    if soc < LOW_BATTERY_THRESHOLD:# :
                        if vehicle_id not in self.low_battery_vehicles:
                            self.low_battery_vehicles.add(vehicle_id)
                            #print(f"Electric Vehicle {vehicle_id} has low SOC: {soc:.2f}% at time {traci.simulation.getTime()}")
                            traci.vehicle.slowDown(vehicle_id, self.slow_speed, duration=10)
                            traci.vehicle.changeLane(vehicle_id, 1, duration=9999)
                            traci.vehicle.setColor(vehicle_id, (255, 0, 0))  # Red color for low battery
                    elif LOW_BATTERY_THRESHOLD <= soc <= (LOW_BATTERY_THRESHOLD + SOC_MARGIN) and traci.vehicle.getLanePosition(vehicle_id) >460 and traci.vehicle.getLanePosition(vehicle_id) < 530:
                        if vehicle_id not in self.low_battery_vehicles:
                            self.low_battery_vehicles.add(vehicle_id)
                            #print(f"Electric Vehicle {vehicle_id} has low SOC: {soc:.2f}% at time {traci.simulation.getTime()}")
                            traci.vehicle.slowDown(vehicle_id, self.slow_speed, duration=10)
                            traci.vehicle.changeLane(vehicle_id, 1, duration=9999)
                            traci.vehicle.setColor(vehicle_id, (255, 0, 0))  # Red color for low battery
                    # If SOC is zero, handle the zero-range vehicle
                    elif soc == 0:
                        self.handle_zero_range_vehicle(vehicle_id)
                    # If the vehicle's SOC is back to normal (greater than threshold), update the vehicle's behavior
                    elif soc >= LOW_BATTERY_THRESHOLD and vehicle_id  not in self.assigned_charging_pod_for_electric_veh:
                        self.handle_full_battery_vehicle(vehicle_id)
                except traci.exceptions.TraCIException:
                    continue

    def handle_full_battery_vehicle(self, vehicle_id):
        """
        Handle electric vehicles that have a full battery or have surpassed the low battery threshold.
        This includes resuming normal behavior such as lane changes, speed, and color changes.
        """
        # Vehicle has sufficient SOC, change it back to normal operation
        traci.vehicle.setColor(vehicle_id, (255, 255, 255))  # Set color to white (normal state)
        #traci.vehicle.setSpeed(vehicle_id, 15)  # Resume normal speed (e.g., 15 m/s)
        traci.vehicle.changeLane(vehicle_id, 0, duration=80)  # Change lane back to the right lane if needed
        if vehicle_id in self.zero_range_vehicles:
            self.zero_range_vehicles.discard(vehicle_id)
        #print(f"Electric Vehicle {vehicle_id} is back to normal operation with SOC > {LOW_BATTERY_THRESHOLD}")

    def handle_zero_range_vehicle(self, vehicle_id):
        """
        Handle electric vehicles with zero SOC by marking them as zero-range vehicles,
        slowing them down, and setting their color to red to indicate they need assistance.
        """
        reward=0
        if vehicle_id not in self.zero_range_vehicles:
            self.zero_range_vehicles.add(vehicle_id)
            self.episode_zero_range += 1  # count unique zero-range events this episode
            print(f"Electric Vehicle {vehicle_id} has zero SOC at time {traci.simulation.getTime()}")
            traci.vehicle.slowDown(vehicle_id, SLOWDOWN_SPEED_ZERO_RANGE, duration=0)  # Reduce speed to 3 m/s
            traci.vehicle.setColor(vehicle_id, (255, 0, 0))  # Red color for zero SOC
            reward-= 2  # Penalty for zero-range vehicle
        return reward

    def handle_low_battery_vehicle(self, vehicle_id):
        """
        Assigns the nearest and most capable (highest-SOC) charging pod to an EV with low battery.
        Criteria:
            - EV SOC below threshold
            - Pod on same or next edge
            - Pod not already assigned
        Returns:
            float: reward for this assignment
        """
        reward = 0.0
        try:
            edge_id = traci.vehicle.getRoadID(vehicle_id)
            route = traci.vehicle.getRoute(vehicle_id)
            current_index = route.index(edge_id)
            next_edge_id = route[current_index + 1] if current_index + 1 < len(route) else None
        except (ValueError, traci.TraCIException):
            next_edge_id = None

        # Visual feedback for low SOC
        traci.vehicle.slowDown(vehicle_id, self.slow_speed, duration=10)
        traci.vehicle.changeLane(vehicle_id, 1, duration=9999)
        traci.vehicle.setColor(vehicle_id, (255, 0, 0))

        nearest_charging_pod_id = None
        best_score = -float('inf')  # composite score combining distance and SOC
        min_distance = float('inf')
        try:
            ev_pos = traci.vehicle.getPosition(vehicle_id)
            ev_lane_pos = self.ev_lane_positions.get(vehicle_id, 0)
        except traci.TraCIException:
            return 0.0
        # Iterate over all pods
        for pod_id in traci.vehicle.getIDList():
            if traci.vehicle.getTypeID(pod_id) != "ChargingPod":
                continue
            # Skip if already assigned
            if pod_id in self.assigned_charging_pod_for_electric_veh.values():
                continue
            try:
                pod_edge = traci.vehicle.getRoadID(pod_id)
                if pod_edge not in [edge_id, next_edge_id]:
                    continue  # only same or next edge pods
                pod_pos = traci.vehicle.getPosition(pod_id)
                dist = traci.simulation.getDistance2D(ev_pos[0], ev_pos[1], pod_pos[0], pod_pos[1])
                # if ev_lane_pos <= 461:  # optional lane constraint
                #     continue
                # Compute pod SOC (normalized)
                pod_soc = (
                        float(traci.vehicle.getParameter(pod_id, "device.battery.actualBatteryCapacity"))
                        / BATTERY_CAPACITY_POD * 100
                )
                if pod_soc < 50:
                    continue  # skip low-SOC pods
                # --- Composite Scoring ---
                # Closer distance = higher score, higher SOC = higher score
                # weight_distance and weight_soc control balance
                weight_distance = 0.6
                weight_soc = 0.4
                normalized_distance = max(0.0, 1.0 - (dist / 200.0))  # within 200m max
                score = weight_distance * normalized_distance + weight_soc * (pod_soc / 100.0)
                #print(f"Pod {pod_id}: dist={dist:.1f}, normalized_dist= {normalized_distance:.1f}, pod_SOC={pod_soc:.1f}%, score={score:.3f}")
                # Optional same-edge bonus
                if pod_edge == edge_id:
                    score *= 1.1
                # Track best candidate
                if score > best_score and ev_lane_pos > 461 and pod_edge == edge_id:
                    best_score = score
                    nearest_charging_pod_id = pod_id
                    min_distance = dist
            except traci.TraCIException:
                continue
        # --- Assign chosen pod ---
        if nearest_charging_pod_id :
            self.assigned_charging_pod_for_electric_veh[vehicle_id] = nearest_charging_pod_id
            ev_soc = (
                    float(traci.vehicle.getParameter(vehicle_id, "device.battery.actualBatteryCapacity"))
                    / BATTERY_CAPACITY * 100
            )

            # Reward increases for urgent assignments (low-SOC EVs)
            urgency_reward = max(0, 50 - ev_soc) / 5.0
            distance_reward = max(0.0, (1.0 - (min_distance / 200.0))) * 2.0
            soc_bonus = best_score * 3.0

            reward += 1#urgency_reward + distance_reward + soc_bonus

            #print(f"✅ EV {vehicle_id} assigned → Pod {nearest_charging_pod_id} "f"(dist={min_distance:.1f}, pod_SOC={pod_soc:.1f}%)")
        else:
            reward -= 0.5  # mild penalty if no suitable pod found
        return reward

    def _get_state(self):
        active_evs = [v for v in traci.vehicle.getIDList() if traci.vehicle.getTypeID(v) == "ElectricVehicle"]
        active_pods = [v for v in traci.vehicle.getIDList() if traci.vehicle.getTypeID(v) == "ChargingPod"]

        ev_slots = active_evs[:MAX_EVS] + [None] * (MAX_EVS - len(active_evs))
        pod_slots = active_pods[:MAX_PODS] + [None] * (MAX_PODS - len(active_pods))

        # global features
        n_evs_norm = len(active_evs) / MAX_EVS
        n_pods_norm = len(active_pods) / MAX_PODS

        # parking occupancies
        occup = []
        all_parking_ids = traci.parkingarea.getIDList()
        # ensure list is always the same length
        pa_slots = list(self.parking_areas_list[:MAX_PARKINGS])
        if len(pa_slots) < MAX_PARKINGS:
            pa_slots += [None] * (MAX_PARKINGS - len(pa_slots))

        for pa in pa_slots:
            if pa is None or pa not in all_parking_ids:
                occ = 0.0
            else:
                occ = len(traci.parkingarea.getVehicleIDs(pa))
            occup.append(occ / (MAX_PODS + 1))  # normalize
        # for pa in self.parking_areas_list[:MAX_PARKINGS]:
        #     occ = len(traci.parkingarea.getVehicleIDs(pa)) if pa in traci.parkingarea.getIDList() else 0
        #     occup.append(occ / (MAX_PODS + 1))  # normalize

        ev_feats = []
        for ev in ev_slots:
            if ev is None:
                ev_feats += [0.0, 0.0, 0.0, 0.0]  # soc, lanepos, dist_to_nearest_pod, assigned_flag
            else:
                soc = float(traci.vehicle.getParameter(ev, "device.battery.actualBatteryCapacity")) / BATTERY_CAPACITY
                lanepos = traci.vehicle.getLanePosition(ev) / 1000.0  # scale
                # compute distance to nearest pod (if any)
                min_dist = 10000
                for pod in pod_slots:
                    if pod is None: continue
                    try:
                        pos_ev = traci.vehicle.getPosition(ev)
                        pos_p = traci.vehicle.getPosition(pod)
                        d = traci.simulation.getDistance2D(pos_ev[0], pos_ev[1], pos_p[0], pos_p[1])
                        if d < min_dist: min_dist = d
                    except traci.TraCIException:
                        continue
                min_dist_norm = min_dist / 1000.0
                assigned = 1.0 if ev in self.assigned_charging_pod_for_electric_veh else 0.0
                ev_feats += [soc, lanepos, min_dist_norm, assigned]

        pod_feats = []
        for pod in pod_slots:
            if pod is None:
                pod_feats += [0.0, 0.0, 0.0, 0.0]  # soc, lanepos, is_stopped, assigned_flag
            else:
                soc = float(
                    traci.vehicle.getParameter(pod, "device.battery.actualBatteryCapacity")) / BATTERY_CAPACITY_POD
                lanepos = traci.vehicle.getLanePosition(pod) / 1000.0
                is_stopped = 1.0 if traci.vehicle.isStoppedParking(pod) else 0.0
                assigned_flag = 1.0 if pod in self.assigned_charging_pod_for_electric_veh.values() else 0.0
                pod_feats += [soc, lanepos, is_stopped, assigned_flag]

        obs = np.array([n_evs_norm, n_pods_norm] + occup + ev_feats + pod_feats, dtype=np.float32)
        return obs

    def share_energy(self, charging_pod_id, vehicle_id):
        """Share energy between a charging pod and an electric vehicle."""
        try:
            # Get the current battery capacity of the charging pod
            actual_battery_capacity_pod = float(
                traci.vehicle.getParameter(charging_pod_id, "device.battery.actualBatteryCapacity"))
            # Get the current battery capacity of the vehicle
            actual_battery_capacity = float(
                traci.vehicle.getParameter(vehicle_id, "device.battery.actualBatteryCapacity"))
            # The pod can only share as much energy as it has
            transferable_energy = min(self.CHARGE_RATE, actual_battery_capacity_pod)

            if transferable_energy <= 0:  # Pod empty, no energy to share
                self._stop_charging(vehicle_id, charging_pod_id)
                return 0.0
            new_energy_charging_pod = max(0.0, actual_battery_capacity_pod - transferable_energy)
            new_energy_ev = min(BATTERY_CAPACITY, actual_battery_capacity + transferable_energy)

            # Update the battery levels
            traci.vehicle.setParameter(charging_pod_id, "device.battery.actualBatteryCapacity", new_energy_charging_pod)
            traci.vehicle.setParameter(vehicle_id, "device.battery.actualBatteryCapacity", new_energy_ev)
            # Optionally track total energy charged for monitoring purposes
            if traci.simulation.getTime() >= self.warm_up_time:  # Steady state starts after 1000 seconds
                #global total_energy_charged
                self.total_energy_charged += transferable_energy
            return transferable_energy  # Return the amount of energy shared
        except traci.exceptions.TraCIException as e:
            print(f"Error sharing energy between vehicle {vehicle_id} and charging pod {charging_pod_id}: {str(e)}")
            return False
    # def share_energy(self,charging_pod_id, vehicle_id):
    #     """Share energy between a charging pod and an electric vehicle."""
    #     try:
    #         # Get the current battery capacity of the charging pod
    #         actual_battery_capacity_pod = float(
    #             traci.vehicle.getParameter(charging_pod_id, "device.battery.actualBatteryCapacity"))
    #         # Get the current battery capacity of the vehicle
    #         actual_battery_capacity = float(
    #             traci.vehicle.getParameter(vehicle_id, "device.battery.actualBatteryCapacity"))
    #         new_energy_charging_pod = max(0.0, actual_battery_capacity_pod - self.CHARGE_RATE)
    #         new_energy_ev = min(BATTERY_CAPACITY, actual_battery_capacity + self.CHARGE_RATE)
    #
    #         # Update the battery levels
    #         traci.vehicle.setParameter(charging_pod_id, "device.battery.actualBatteryCapacity", new_energy_charging_pod)
    #         traci.vehicle.setParameter(vehicle_id, "device.battery.actualBatteryCapacity", new_energy_ev)
    #         # Optionally track total energy charged for monitoring purposes
    #         if traci.simulation.getTime() >= self.warm_up_time:  # Steady state starts after 1000 seconds
    #             #global total_energy_charged
    #             self.total_energy_charged += self.CHARGE_RATE
    #         return self.CHARGE_RATE  # Return the amount of energy shared
    #     except traci.exceptions.TraCIException as e:
    #         print(f"Error sharing energy between vehicle {vehicle_id} and charging pod {charging_pod_id}: {str(e)}")
    #         return False

    def update_pod_routes(self, assigned_charging_pod_for_electric_veh):
        """
        Update pod routes dynamically only if:
        1. The pod is assigned to an EV.
        2. The EV is approaching a junction edge (divergence point).
        3. The pod's upcoming edge differs from the EV's actual route branch.
        """

        for ev_id, pod_id in self.assigned_charging_pod_for_electric_veh.items():
            if ev_id not in traci.vehicle.getIDList() or pod_id not in traci.vehicle.getIDList():
                continue

            try:
                ev_edge = traci.vehicle.getRoadID(ev_id)
                pod_edge = traci.vehicle.getRoadID(pod_id)
                ev_route = traci.vehicle.getRoute(ev_id)
                pod_route = traci.vehicle.getRoute(pod_id)

                if not ev_route or not pod_route:
                    continue
                # pod_index = find_last_index(pod_route, pod_edge)
                # Use SUMO's live route index for pod (since route may have changed)
                try:
                    pod_index = traci.vehicle.getRouteIndex(pod_id)
                    ev_index = traci.vehicle.getRouteIndex(ev_id)
                except traci.TraCIException:
                    pod_index = pod_route.index(pod_edge) if pod_edge in pod_route else -1
                    ev_index = ev_route.index(ev_edge) if ev_edge in ev_route else -1
                if ev_index == -1:
                    continue  # Edge not found properly

                # Determine next edge for both
                ev_next_edge = ev_route[ev_index + 1] if ev_index + 1 < len(ev_route) else None
                pod_next_edge = pod_route[pod_index + 1] if pod_index + 1 < len(pod_route) else None

                # Build transitions (prev → next)
                ev_transition = (ev_edge, ev_next_edge)
                pod_transition = (pod_edge, pod_next_edge)
                # print(f"EV {ev_id} on {ev_edge}→{ev_next_edge}, Pod {pod_id} on {pod_edge}→{pod_next_edge}")

                # Define critical divergence transitions (these are where EV may change path)
                DIVERGENCE_TRANSITIONS = {("E6", "E7"), ("E6", "E8"), ("E0", "E1"), ("E0", "E5"), ("E1", "E2"),
                                          ("E1", "E10")}

                # Only check rerouting if both vehicles are at or approaching a divergence
                if ev_transition in DIVERGENCE_TRANSITIONS and pod_transition in DIVERGENCE_TRANSITIONS:
                    # print(f"At divergence: EV {ev_id} on {ev_transition}, Pod {pod_id} on {pod_transition}")
                    # If the transition direction differs (e.g., EV: E6→E7, Pod: E6→E8)
                    if ev_transition != pod_transition:
                        new_route = ev_route[ev_index:]
                        # --- 2. Update route and reroute to the new parking area ---
                        traci.vehicle.setParameter(pod_id, "has.route", "false")
                        traci.vehicle.setRoute(pod_id, new_route)
                        #print(f"🔀 Updated route for pod {pod_id} to follow EV {ev_id}")

            except traci.TraCIException as e:
                print(f"⚠️ Could not update route for pod {pod_id}: {e}")

    def _start_charging(self, vehicle_id, charging_pod_id):
        """
        Continuously share energy between the vehicle and the charging pod as long as they are within the charging distance threshold.
        Also, check less frequently, scale reward based on energy transferred, and include a timeout for safety.
        """
        reward = 0
        timeout_limit = 80  # Timeout after 50 steps (to prevent infinite loops)
        energy_shared = 0  # To keep track of the energy transferred
        initial_distance = traci.vehicle.getDistance(charging_pod_id)
        initial_energy = traci.vehicle.getElectricityConsumption(charging_pod_id)
        pos_ev = traci.vehicle.getPosition(vehicle_id)
        pos_pod = traci.vehicle.getPosition(charging_pod_id)
        distance = traci.simulation.getDistance2D(pos_ev[0], pos_ev[1], pos_pod[0], pos_pod[1])
        battery_capacity_percentage = (float(traci.vehicle.getParameter(vehicle_id,
                                                                        "device.battery.actualBatteryCapacity")) / BATTERY_CAPACITY)*100
        # Resume the charging pod if it is stopped, so it can move towards the EV
        if vehicle_id and charging_pod_id in self.assigned_charging_pod_for_electric_veh.values():
            if (traci.vehicle.isStoppedParking(charging_pod_id) and charging_pod_id in self.assigned_charging_pod_for_electric_veh.values() and traci.vehicle.getRoadID(charging_pod_id)== traci.vehicle.getRoadID(vehicle_id)
                    and self.ev_lane_positions[vehicle_id] > 461):
                traci.vehicle.resume(charging_pod_id)  # Resume the charging pod if it's parked
                #print(f"Charging pod {charging_pod_id} resumed.")
            if distance <= self.CHARGING_DISTANCE_THRESHOLD:
                if not self.assignment_started_charging.get(vehicle_id, False):
                    start_d = self.assignment_start_distance.get(vehicle_id, distance)
                    #print(f"Vehicle {vehicle_id} starting distance to pod {charging_pod_id}: {start_d:.1f}m")
                    approach_dist = max(0.0, start_d - distance)  # how much pod closed the gap
                    reward += 2.5#max(0.0, (1.0 - distance / max(1.0, start_d))) * 5.0
                    #print(f"Vehicle {vehicle_id} initial distance bonus reward, distance {distance:.1f}m, reward {reward:.2f}")
                    self.assignment_started_charging[vehicle_id] = True
                    self.assignment_time_steps[vehicle_id] = 0
                reward += 1.5 # Small reward for being within charging distance
                traci.vehicle.changeLane(charging_pod_id, 1, duration=700)
                traci.vehicle.setColor(vehicle_id, (0, 255, 0))  # Change color to green to indicate charging
                energy_shared += self.share_energy(charging_pod_id, vehicle_id)# Share energy between the vehicle and charging pod
                scaled_reward = energy_shared * 0.5  # Adjust multiplier as needed
                reward += scaled_reward
                if traci.vehicle.getLanePosition(vehicle_id) != 0:
                    lane_distance = (traci.vehicle.getLanePosition(charging_pod_id) -
                                     traci.vehicle.getLanePosition(vehicle_id))
                    if lane_distance < -100:
                        traci.vehicle.slowDown(charging_pod_id, 2, duration=0)
                        traci.vehicle.changeLane(charging_pod_id, 2, duration=700)
                    if lane_distance > 0 or distance >= 50:
                        traci.vehicle.slowDown(charging_pod_id, 2, duration=0)
                        traci.vehicle.changeLane(charging_pod_id, 2, duration=700)
                else:
                    traci.vehicle.slowDown(charging_pod_id, 7, duration=0)
                next_stops = traci.vehicle.getNextStops(charging_pod_id)
                if next_stops:
                    next_stop_id = next_stops[0][2]
                    next_stop_lane = next_stops[0][0]
                    edge_id = traci.lane.getEdgeID(next_stop_lane)  # Get the edge ID associated with the lane
                    # print(f"Next stop ID: {next_stop_id} and edge ID: {edge_id}")
                    next_stop_position = traci.parkingarea.getEndPos(next_stop_id)
                    electric_veh_lane_position = traci.vehicle.getLanePosition(vehicle_id)
                    pod_id = traci.vehicle.getRoadID(charging_pod_id)
                    veh_id= traci.vehicle.getRoadID(vehicle_id)
                    if pod_id == edge_id or veh_id == edge_id:
                        # Calculate the distance to the next stop
                        distance_to_next_stop = traci.simulation.getDistance2D(
                            traci.vehicle.getLanePosition(charging_pod_id), 0, next_stop_position,0)
                        distance_park= abs(next_stop_position - electric_veh_lane_position)
                        #print(f"check {distance_to_next_stop} and distance {distance_park} for pod {charging_pod_id} and ev {vehicle_id}.")
                        if distance_to_next_stop < 80  or distance_park < 120  and battery_capacity_percentage < 80:
                            traci.vehicle.setParkingAreaStop(charging_pod_id, next_stop_id, duration=0)
                            #print(f"Parking skip for pod {charging_pod_id} at stop {next_stop_id}, time {traci.simulation.getTime()}")
            # if battery_capacity_percentage >= 80:
            #     reward -= (battery_capacity_percentage - 80) * 2  # Penalty grows as it overcharges
                #self._stop_charging(vehicle_id,charging_pod_id)
            elif distance > self.CHARGING_DISTANCE_THRESHOLD and battery_capacity_percentage < 80:# keep checking distance and maintian it
                #traci.vehicle.setColor(vehicle_id, (255, 255, 255))  # White color
                lane_pos_ev = traci.vehicle.getLanePosition(vehicle_id)
                lane_pos_pod = traci.vehicle.getLanePosition(charging_pod_id)
                self.assignment_time_steps[vehicle_id] = self.assignment_time_steps.get(vehicle_id, 0) + 1

                if lane_pos_pod > lane_pos_ev and traci.vehicle.getRoadID(charging_pod_id) == traci.vehicle.getRoadID(vehicle_id):
                    # Charging pod is ahead → slow down
                    #print(f"Pod {charging_pod_id} is ahead of EV {vehicle_id} and slowing down.")
                    traci.vehicle.slowDown(charging_pod_id, 1, duration=10)
                elif lane_pos_pod < lane_pos_ev and self.assignment_time_steps[vehicle_id] < timeout_limit:
                    if traci.vehicle.getSpeed(charging_pod_id)==1:
                        traci.vehicle.changeLane(charging_pod_id, 1, duration=700)  # Try lane alignment
                        print(f"Pod {charging_pod_id} is behind EV {vehicle_id} and changing lane.")
                    # Pod is behind or nearby → resume or speed up if needed
                    #print(f"Pod {charging_pod_id} is behind or adjacent to EV {vehicle_id}.")
                    traci.vehicle.slowDown(charging_pod_id, 25, duration=10)  # Or use resume if needed
                    next_stops = traci.vehicle.getNextStops(charging_pod_id)
                    if next_stops:
                        next_stop_id = next_stops[0][2]
                        next_stop_lane = next_stops[0][0]
                        edge_id = traci.lane.getEdgeID(next_stop_lane)  # Get the edge ID associated with the lane
                        # print(f"Next stop ID: {next_stop_id} and edge ID: {edge_id}")
                        next_stop_position = traci.parkingarea.getEndPos(next_stop_id)
                        electric_veh_lane_position = traci.vehicle.getLanePosition(vehicle_id)
                        pod_id = traci.vehicle.getRoadID(charging_pod_id)
                        if pod_id == edge_id:
                            # Calculate the distance to the next stop
                            distance_to_next_stop = traci.simulation.getDistance2D(
                                traci.vehicle.getLanePosition(charging_pod_id), 0, next_stop_position, 0)
                            distance_park= abs(next_stop_position - electric_veh_lane_position)
                            # print(f"check {distance_to_next_stop} and id {next_stop_id} and pod {charging_pod_id}.")
                            if distance_to_next_stop < 80  or distance_park < 120 and battery_capacity_percentage < 80:
                                traci.vehicle.setParkingAreaStop(charging_pod_id, next_stop_id, duration=0)
                                #print(f"Parking skip_2 {charging_pod_id} at stop {next_stop_id} and pos {next_stop_position}")
                traci.vehicle.changeLane(charging_pod_id, 2, duration=700)  # Try lane alignment

                if self.assignment_time_steps[vehicle_id] >= timeout_limit:
                    self._stop_charging(vehicle_id, charging_pod_id)
                    print(f"Charging pod {charging_pod_id} timed out after {timeout_limit} steps.")
                    self.assignment_time_steps[vehicle_id] = 0
                    reward -= 1  # Penalty for timeout
                return reward
        return 0

    def _charging_reward(self, soc):
        """
        Returns reward based on SOC (percentage).
        Peak reward between 75% and 80%.
        Penalizes both undercharging (<50%) and overcharging (>80%).
        """
        MIN_STOP_CAPACITY = 50
        MAX_STOP_CAPACITY = 80
        # Strong penalty if stopping too early
        if soc < MIN_STOP_CAPACITY:
            return -1
        # Reward grows until 75
        if soc < 75:
            return 2  # linear growth from 50 to 75
        # Plateau peak between 75–80
        if 75 <= soc <= MAX_STOP_CAPACITY:
            return 2.5  # max reward region
        # Penalty for overshooting >80
        if soc > MAX_STOP_CAPACITY:
            return -0.6#-3 * (soc - MAX_STOP_CAPACITY)
        return 0

    def _stop_charging(self, vehicle_id, charging_pod_id):
        """
        Stops the charging process, releases the assignment, resets visual indicators,
        and redirects the charging pod to the next parking stop if available.
        """
        MIN_STOP_CAPACITY = 50  # Minimum % before stopping allowed
        try:
            battery_capacity_percentage = (float(traci.vehicle.getParameter(vehicle_id,
                                                                            "device.battery.actualBatteryCapacity")) / BATTERY_CAPACITY) * 100
            if vehicle_id in self.assigned_charging_pod_for_electric_veh:
                del self.assigned_charging_pod_for_electric_veh[vehicle_id] # Remove assignment
            if vehicle_id in self.assignment_start_distance:
                del self.assignment_start_distance[vehicle_id]
            if vehicle_id in self.assignment_started_charging:
                del self.assignment_started_charging[vehicle_id]
            #print(f"Charging ended for vehicle {vehicle_id} at SOC {battery_capacity_percentage:.2f}%")
            # Redirect to next stop if not already parked
            #if not traci.vehicle.isStoppedParking(charging_pod_id):
            next_stops = traci.vehicle.getNextStops(charging_pod_id)
            if next_stops:
                next_stop_id = next_stops[0][2]
                traci.vehicle.setParkingAreaStop(charging_pod_id, next_stop_id, duration=90000)
                #print(f"Charging pod {charging_pod_id} redirected to parking area {next_stop_id}.")
            reward = self._charging_reward(battery_capacity_percentage)
            #print(f"Stop charging reward: {reward:.2f} for vehicle {vehicle_id} at SOC {battery_capacity_percentage:.2f}%")
            self.low_battery_vehicles.discard(vehicle_id)  # Remove from low battery list
            # Lane change and visual reset
            traci.vehicle.changeLane(charging_pod_id, 2, duration=80)
            traci.vehicle.setColor(vehicle_id, (255, 255, 255))  # White = not charging
            return reward

        except traci.exceptions.TraCIException as e:
            print(f"Exception during stopping charge: {e}")
            return -10  # Optionally penalize unexpected failure

   # def handle_low_battery_vehicle(self, vehicle_id):
    #     """
    #     Handle electric vehicles with low battery by assigning them to the nearest charging pod.
    #     Args:
    #         vehicle_id (str): The ID of the electric vehicle.
    #     """
    #     edge_id = traci.vehicle.getRoadID(vehicle_id)
    #     route = traci.vehicle.getRoute(vehicle_id)
    #     traci.vehicle.slowDown(vehicle_id, self.slow_speed, duration=10)
    #     traci.vehicle.changeLane(vehicle_id, 1, duration=9999)
    #     traci.vehicle.setColor(vehicle_id, (255, 0, 0))  # Red color for low battery
    #     try:
    #         current_index = route.index(edge_id)
    #         next_edge_id = route[current_index + 1] if current_index + 1 < len(route) else None
    #     except ValueError:
    #         next_edge_id = None
    #     nearest_charging_pod_id = None
    #     min_distance = float('inf')
    #     reward = 0
    #     # Look for the nearest available charging pod on the same edge
    #     for charging_pod_id in traci.vehicle.getIDList():
    #         if traci.vehicle.getTypeID(charging_pod_id) == "ChargingPod" and traci.vehicle.getRoadID(
    #                 charging_pod_id) == edge_id and charging_pod_id not in self.assigned_charging_pod_for_electric_veh.values():# and traci.vehicle.isStoppedParking(charging_pod_id):
    #             pod_edge = traci.vehicle.getRoadID(charging_pod_id)
    #             if pod_edge in [edge_id, next_edge_id]:
    #                 try:
    #                     charging_pod_position = traci.vehicle.getPosition(charging_pod_id)  # Get the charging pod position
    #                     electric_veh_position = traci.vehicle.getPosition(vehicle_id)  # Get the vehicle position
    #                     distance_to_electric_veh = traci.simulation.getDistance2D(electric_veh_position[0],
    #                                                                                   electric_veh_position[1],
    #                                                                                   charging_pod_position[0],
    #                                                                                   charging_pod_position[1])
    #                     electric_veh_lane_position = self.ev_lane_positions[vehicle_id]
    #                     if distance_to_electric_veh < min_distance and electric_veh_lane_position > 461:
    #                         nearest_charging_pod_id = charging_pod_id
    #                         min_distance = distance_to_electric_veh
    #                         pod_soc = float(traci.vehicle.getParameter(charging_pod_id,
    #                                                                    "device.battery.actualBatteryCapacity")) / BATTERY_CAPACITY_POD
    #                         # same-edge bonus
    #                         same_edge = (traci.vehicle.getRoadID(vehicle_id) == traci.vehicle.getRoadID(charging_pod_id))
    #                         same_edge_bonus = 1.2 if same_edge else 1.0
    #                         # normalized distance reward
    #                         max_dist = 200.0  # tune: typical max distance expected
    #                         assign_reward = max(0.0, (1.0 - (distance_to_electric_veh / max_dist))) * (
    #                                 1.0 + pod_soc) * same_edge_bonus * 9.0
    #                         #reward += assign_reward
    #                         #print(f"Assign reward: {assign_reward:.2f} (dist: {distance_to_electric_veh:.1f}, pod_soc: {pod_soc:.2f}, same_edge: {same_edge})")
    #                 except traci.exceptions.TraCIException:
    #                     continue
    #
    #     # Assign the EV to the nearest charging pod if one is found
    #     if nearest_charging_pod_id:
    #         self.assigned_charging_pod_for_electric_veh[vehicle_id] = nearest_charging_pod_id
    #         #print(f"Electric Vehicle {vehicle_id} assigned to Charging Pod {nearest_charging_pod_id} at time {traci.simulation.getTime()}")
    #         soc = float(
    #             traci.vehicle.getParameter(vehicle_id, "device.battery.actualBatteryCapacity")) / BATTERY_CAPACITY * 100
    #         urgency_reward = max(0, 50 - soc) / 5  # e.g., up to +10 when SOC very low
    #         reward += 1#(1 + urgency_reward)
    #     return reward
