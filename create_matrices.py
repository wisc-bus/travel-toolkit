import pandas as pd
from math import radians, sin, cos, asin, sqrt, atan2, pi, ceil, floor
from collections import defaultdict
import numpy as np
from datetime import datetime, timedelta
from collections import namedtuple
import gtfs_kit as gk
import argparse
import pickle
from joblib import dump

class Node:
    def __init__(self, trip_id, route_short_name, stop_sequence, stop_id, stop_x, stop_y, arrival_time, departure_time, max_walking_distance, node_id=-1):
        self.trip_id = trip_id
        self.route_short_name = route_short_name
        self.stop_sequence = stop_sequence
        self.stop_id = stop_id
        self.stop_x = stop_x
        self.stop_y = stop_y
        self.arrival_time = arrival_time
        self.departure_time = departure_time
        self.walking_distance = max_walking_distance
        self.children = []
        self.children_ids = set()
        self.id = node_id
        self.index = -1

    def distance(self, other):
        return sqrt((other.stop_x - self.stop_x)**2 + (other.stop_y - self.stop_y)**2)

    def __str__(self):
        return f"({self.trip_id}, {self.route_short_name}, {self.stop_sequence}, {self.stop_id}, {self.arrival_time}, {self.departure_time}, {self.walking_distance}, {self.stop_x}, {self.stop_y})"

    def __repr__(self):
        rv = self.__str__()
        rv += "\nChildren:\n"
        for child in self.children:
            rv += f"  cost: {child.cost} "
            rv += f"  type: {child.type} "
            child = child.node
            rv += str(child)
            rv += "\n"
        return rv

    def __lt__(self, other):
        # always retain sequence here
        return False


class NodeCostPair:
    def __init__(self, node, cost, type):
        self.node = node
        self.cost = cost  # the cost here means the walking distance
        self.type = type

class MatrixGen:
    def __init__(self, segments):
        self.segments = segments
        self.points = []
        # walking and riding enroute nodes prioritized based on which has less enroute nodes (less travel_minutes)
        for s in segments:
            # we want to include 0 min travel times
            assert s.travel_minutes >= 0
            self.points.append(s.point1)
            self.points.append(s.point2)
            # differentiate between walk and ride stops
            if s.enroute_type == "walk":
                for enroute in range(1, s.travel_minutes):
                    self.points.append(f"{s.point1}_to_{s.point2}_walk_progress_{enroute}")
            # walking enroute node
            else:
                for enroute in range(1, s.travel_minutes):
                    self.points.append(f"{s.point1}_to_{s.point2}_ride_progress_{enroute}")
                
        self.points = sorted(set(self.points))

    def gen_1min_matrix(self, start_minute, as_dataframe=False):
        # if matrix[A,B] is 1, you can get from point A at time start_minute to B at start_minute+1
        #
        # 1's on diagonal because you can stay where you are.
        # TODO: actually, you can't stay where you are if you're enroute...
        #
        # TODO: bool?
        matrix = np.zeros((len(self.points), len(self.points)), dtype=np.uint8)

        # Don't set 1 on diagonal of enroute stops
        for i in range(len(self.points)):
            if "progress" not in self.points[i]:
                matrix[i, i] = 1

        for s in self.segments:
            if s.start_minute > start_minute:
                # it's in the future
                continue
            # special case where 6:04 to 6:05 matrix needs to capture 6:04 to 6:04 interval
            elif s.start_minute + timedelta(minutes=s.travel_minutes) == start_minute and s.start_minute == start_minute:
                matrix[self.points.index(s.point1), self.points.index(s.point2)] = 1
            elif s.start_minute + timedelta(minutes=s.travel_minutes) <= start_minute:
                # it's in the past
                continue

            point1 = (
                s.point1 if s.start_minute == start_minute
                else f"{s.point1}_to_{s.point2}_{s.enroute_type}_progress_{int((start_minute - s.start_minute).total_seconds() / 60)}"
            )
            point2 = (
                s.point2 if s.start_minute + timedelta(minutes=s.travel_minutes) == start_minute + timedelta(minutes=1) or s.start_minute == start_minute
                else f"{s.point1}_to_{s.point2}_{s.enroute_type}_progress_{int((start_minute - s.start_minute).total_seconds() / 60)}"
            )  
            matrix[self.points.index(point1), self.points.index(point2)] = 1

        if as_dataframe:
            comb_matrix = pd.DataFrame(matrix, index=self.points, columns=self.points)
        return comb_matrix


def is_day_valid(day):
    # return the valid calender in a specific day
    return (day == 1)


def get_valid_stopTime(df, start_time, elapse_time):
    start_time = pd.to_timedelta(start_time)
    end_time = start_time + pd.to_timedelta(elapse_time)
    return df[(df['arrival_time'] > start_time) & (df['arrival_time'] < end_time)]


def clean_up_data(stops_df, trips_df, stopTimes_df, calendar_df, start_time, elapse_time, day):
    # get valid service_ids
    calendar_df['start_date'] = pd.to_datetime(
    calendar_df['start_date'], format='%Y%m%d')
    calendar_df['end_date'] = pd.to_datetime(
    calendar_df['end_date'], format='%Y%m%d')
    
    calendar_filtered_df = calendar_df[is_day_valid(calendar_df[day])]
    
    service_ids = calendar_filtered_df["service_id"].tolist()
    
    # get valid trips
    trips_df = trips_df[trips_df["service_id"].isin(service_ids)]
    
    # get valid stop_times
    stopTimes_filtered_df = trips_df.merge(stopTimes_df, on="trip_id")
    stopTimes_merged_df = (stopTimes_filtered_df.merge(stops_df, on="stop_id")[
    ["service_id", "trip_id", "route_id", "stop_id", "stop_sequence", "arrival_time", "departure_time", "stop_lon", "stop_lat"]].rename(columns={"stop_lon": "stop_x", "stop_lat": "stop_y"}))
    
    #get stop_times within the time frame
    stopTimes_merged_df['arrival_time'] = pd.to_timedelta(
    stopTimes_merged_df['arrival_time'])
    
    stopTimes_merged_df['departure_time'] = pd.to_timedelta(
    stopTimes_merged_df['departure_time'])
    
    #add trip_delays
    for (trip_id, delay) in []:
        stopTimes_merged_df.loc[stopTimes_merged_df["trip_id"] == trip_id, "arrival_time"] += pd.to_timedelta(delay)
    
    stopTimes_final_df = get_valid_stopTime(stopTimes_merged_df, start_time, elapse_time).sort_values(by="arrival_time")

    return stopTimes_final_df


def create_nodes(curr_nodes, stopTimes_final_df, trip_node_dict, stop_node_dict, route_node_dict, max_edge_walking_distance):
    for index, row in stopTimes_final_df.iterrows():
        node = Node(row["trip_id"], row["route_id"], row["stop_sequence"], row["stop_id"], row["stop_x"],
                    row["stop_y"], row["arrival_time"], row["departure_time"], max_edge_walking_distance, index)
        curr_nodes.append(node)
        trip_node_dict[row["trip_id"]].append(node)
        stop_node_dict[row["stop_id"]].append(node)
        route_node_dict[row["route_id"]].append(node)

    # gen edges
    # direct sequence
    # forms connection in each trip, with start being the first.
    for trip_id, nodes in trip_node_dict.items():
        for i in range(len(nodes)-1):
            start = nodes[i]
            end = nodes[i+1]
            nodeCostPair = NodeCostPair(end, 0, "RIDE")
            start.children.append(nodeCostPair)
            start.children_ids.add(end.id)

def construct_segments(curr_nodes, TripSegment):
    segments = []
    for node in curr_nodes:
        for child in node.children:  
            segments.append(TripSegment(str(node.stop_id), str(child.node.stop_id), node.departure_time, int((child.node.arrival_time - node.departure_time).total_seconds() // 60), "ride"))

    segments = list(set(segments))
    return segments


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return R * c  # Distance in meters


def compute_distances(points, max_edge_walking_distance):
    distance_dict = {}

    # Compute pairwise distances and store in the dictionary
    for i in range(len(points)):
        for j in range(i + 1, len(points)):  # Avoid duplicate pairs
            dist = haversine(points[i][0], points[i][1], points[j][0], points[j][1])

            # only consider walkable stops given the maximum walking distance
            if dist <= max_edge_walking_distance:
                # very unlikely two pairwise distances will be exactly the same and get overwritten
                if dist not in distance_dict:
                    distance_dict[dist] = (points[i][2], points[j][2])

    return distance_dict

def add_walking_segments(segments, distance_dict, start_time, avg_walking_speed, TripSegment):
    for dist in distance_dict:
        segments.append(TripSegment(str(distance_dict[dist][0]), str(distance_dict[dist][1]), pd.to_timedelta(start_time), ceil(dist / (avg_walking_speed * 60)), "walk"))


def time_to_timedelta(t_str):
    h, m, s = map(int, t_str.split(":"))
    return timedelta(hours=h, minutes=m, seconds=s)


def construct_matrices(segments, start_time, elapse_time):
    all_min_matrices = []
    matrixGen = MatrixGen(segments)

    # start and end times
    start_td = time_to_timedelta(start_time)
    elapse_td = time_to_timedelta(elapse_time)
    end_td = start_td + elapse_td
    
    # find matrix for each minute in range
    step = timedelta(minutes=1)
    
    # Create an iterator
    current_time = start_td
    while current_time <= end_td:
        all_min_matrices.append(matrixGen.gen_1min_matrix(current_time, as_dataframe=True))
        current_time += step

    return all_min_matrices


def main():
    # parse arguments and show examples of possible inputs with help
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="minneapolis_gtfs.zip")
    parser.add_argument("day", help="monday")
    parser.add_argument("start_time", help="12:00:00")
    parser.add_argument("elapse_time", help="00:05:00")
    parser.add_argument("avg_walking_speed", help="1.4")
    args = parser.parse_args()

    # create local variables
    data_path = args.data_path
    day = args.day
    start_time = args.start_time
    elapse_time = args.elapse_time
    avg_walking_speed = float(args.avg_walking_speed)
    max_edge_walking_distance = avg_walking_speed * pd.to_timedelta(elapse_time).total_seconds()

    # use gtfs parser
    feed = gk.read_feed(data_path, dist_units="m")
    print("parsed data")

    stops_df = feed.stops
    trips_df = feed.trips
    stopTimes_df = feed.stop_times
    calendar_df = feed.calendar

    # have dataframe in a format we want
    stopTimes_final_df = clean_up_data(stops_df, trips_df, stopTimes_df, calendar_df, start_time, elapse_time, day)
    print("cleaned up data")

    curr_nodes = []
    # gen nodes
    trip_node_dict = defaultdict(list)
    stop_node_dict = defaultdict(list)
    route_node_dict = defaultdict(list)

    # creating nodes gives us an organized way of showing which stops we can ride to from each stop
    create_nodes(curr_nodes, stopTimes_final_df, trip_node_dict, stop_node_dict, route_node_dict, max_edge_walking_distance)
    print("created nodes")

    TripSegment = namedtuple("TripSegment", ["point1", "point2", "start_minute", "travel_minutes", "enroute_type"])
    # add each child node as a riding segment so it can be used when the matrices are constructed
    segments = construct_segments(curr_nodes, TripSegment)
    print("created segments")

    all_stops_info = []

    for _, row in stops_df.iterrows():
        all_stops_info.append((row.stop_lat, row.stop_lon, row.stop_id))

    # pairwise distances between nodes allow us to see which nodes are within walking distance
    distance_dict = compute_distances(all_stops_info, max_edge_walking_distance)
    print("pairwise distances computed")

    # add each pairwise distance as a walking segment so it can be used when the matrices are constructed
    add_walking_segments(segments, distance_dict, start_time, avg_walking_speed, TripSegment)
    print("walking segments added")

    # there is a matrix constructed for each minute between start_time and elapse_time showing which stops we can
    # get to using walking or riding. In the next tool, using matrix multiplication will reveal which stops are reachable across a time range
    all_min_matrices = construct_matrices(segments, start_time, elapse_time)
    print("matrices constructed")

    # write all the matrices to a pickle file for later use
    dump(all_min_matrices, "matrix_each_time.pkl")
    print("matrices saved to pickle file")


if __name__ == "__main__":
    main()
