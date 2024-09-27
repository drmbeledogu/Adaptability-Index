import numpy as np
import json
import re
import requests
import py7zr
import os
from tqdm import tqdm
from copy import deepcopy
from collections import Counter
from datasets import load_dataset


class SportVUCleaner:
    def __init__(self, rim_locations: dict | None=None):
        """
        Intializes the SportVUCleaner class

        Args:
            rim_locations (dict | None, optional): Dictionary specifing the x, y, z location of the left and right rims. Defaults to None.
        """
        if rim_locations is None:
            self._rim_locations = {
                'left': {
                    'x': 5.25,
                    'y': 25,
                    'z': 10
                },
                'right': {
                    'x': 88.75,
                    'y': 25,
                    'z': 10
                }
            }
        else:
            self._rim_locations = rim_locations

    def get_all_available_game_names(self) -> list[str]:
        """
        Fetches all available file names for the 2015-2016 SportVU dataset

        Returns:
            list[str]: List of the game file names
        """
        _URL = "https://github.com/linouk23/NBA-Player-Movements/raw/master/data/2016.NBA.Raw.SportVU.Game.Logs"
        res = requests.get(_URL)
        text = res.text
        json_pattern = r'{"items":*\[.*?\]'
        json_match = re.findall(json_pattern, text, re.DOTALL)
        ITEMS = json.loads(json_match[0]+"}")['items']
        return [item['name'] for item in ITEMS]

    def import_sportvu_game_data(self, name: str, loader_script_path: str='nba_tracking_data_loader.py', keep_in_memory: bool=False, verbose: bool=False) -> object:
        """
        Imports raw SportVU + NBA play-by-play data.
        Modified from the script here https://huggingface.co/datasets/dcayton/nba_tracking_data_15_16

        Args:
            name (str): Name of the file in the raw repo
            loader_script_path (str, optional): Location of the data loader script path. Defaults to 'nba_tracking_data_loader.py'.
            keep_in_memory (bool, optional): Whether to load the dataset into memory. Defaults to False.
            verbose (bool, optional): Print status to terminal. Defaults to False.

        Returns:
            object: HF dataset object containing the raw SportVU + NBA Play-by-Play data game data
        """
        if verbose:
            print(f"Starting dataset load for game: {name}")
        dataset = load_dataset(path=loader_script_path, name=name, keep_in_memory=keep_in_memory, trust_remote_code=True)
        hf_data = dataset['train']
        
        if verbose:
            print(f"Loaded sportvu tracking data from huggingface for {hf_data} games")
        
        return hf_data
    
    def remove_duplicate_events(self, raw_hf_data: object, verbose: bool=False) -> list[dict]:
        """
        Removes duplicate events for a game in the raw dataset from the Huggingface pull

        Args:
            raw_hf_data (object): Raw huggingface data for a singular game
            verbose (bool, optional): Print status to terminal. Defaults to False.

        Returns:
            list[dict]: List of dictionaries containing the de-duplicated SportVU + Play-by-Play data
        """
        if verbose:
            print(f"Deduplicating {len(raw_hf_data)} events")
        
        deduped_data = []
        continuous_events = {
            'indices': [],
            'event_type': []
        }
        for i in range(len(raw_hf_data)-1):
            current_type = raw_hf_data[i]['event_info']['type']
            
            # If the continuous events list is empty, add the current event to it
            if len(continuous_events['indices']) == 0:
                continuous_events['indices'].append(i)
                continuous_events['event_type'].append(current_type)
            try:
                #Get start and end of the current and next event
                start_timestamp1 = raw_hf_data[i]['moments'][0]['timestamp']
                end_timestamp1 = raw_hf_data[i]['moments'][-1]['timestamp']
                start_timestamp2 = raw_hf_data[i+1]['moments'][0]['timestamp']
                end_timestamp2 = raw_hf_data[i+1]['moments'][-1]['timestamp']
            except:
                continue
            
            # If next event and current event are duplicates, add the next event to the continuous events list
            if (start_timestamp1 == start_timestamp2) and (end_timestamp1 == end_timestamp2):
                next_type = raw_hf_data[i+1]['event_info']['type']
                continuous_events['indices'].append(i+1)
                continuous_events['event_type'].append(next_type)
            
            # If the next event is not a duplicate
            else:
                if len(continuous_events['indices']) > 1:
                    
                    # If the continuous event includes a made shot, use it as the event data for posession purposes
                    if 1 in continuous_events['event_type']:
                        proper_index = continuous_events['indices'][continuous_events['event_type'].index(1)]
                        deduped_data.append(raw_hf_data[proper_index])
                    
                    # If the continuous eevnt does not include a made shot but includes a missed shot, use it as the event data for posession purposes
                    elif 2 in continuous_events['event_type']:
                        proper_index = continuous_events['indices'][continuous_events['event_type'].index(2)]
                        deduped_data.append(raw_hf_data[proper_index])
                
                # If there is only one event in the continuous events list
                else:
                    # If the event is a shot, add it to the cleaned data
                    if raw_hf_data[i]['event_info']['type'] in [1, 2]:
                        deduped_data.append(raw_hf_data[i])
                
                # Reset the continuous events list
                continuous_events = {
                    'indices': [],
                    'event_type': []
                }
        if verbose:
            print(f"Deduplicated {len(raw_hf_data)} events to {len(deduped_data)} events")
        
        return deduped_data
    
    def _euclidean_distance_2d(self, point1: dict[str, float], point2: dict[str, float]) -> float:
        """
        Calculate the Euclidean distance between two points in 2D space.

        Args:
            point1 (dict[str, float]): First point must contain 'x' and 'y' as keys
            point2 (dict[str, float]): Second point must contain 'x' and 'y' as keys

        Returns:
            float: Distance between the points
        """
        return np.sqrt((point1['x'] - point2['x'])**2 + (point1['y'] - point2['y'])**2)

    def _euclidean_distance_3d(self, point1: dict[str, float], point2: dict[str, float]) -> float:
        """
        Calculate the Euclidean distance between two points in 3D space.

        Args:
            point1 (dict[str, float]): First point must contain 'x', 'y', and 'z' as keys
            point2 (dict[str, float]): Second point must contain 'x', 'y', and 'z' as keys

        Returns:
            float: Distance between the points
        """
        return np.sqrt((point1['x'] - point2['x'])**2 + (point1['y'] - point2['y'])**2 + (point1['z'] - point2['z'])**2)

    def determine_offensive_side(self, event: dict, radius: float) -> str:
        """
        Determine which side of the court contained more frames with the basketball coordinates
        within a certain radius of the rim location.

        Args:
            event (dict): SportVU event
            radius (float): The radius within which to check the ball's proximity to the rim.

        Returns:
            str: Which side was the offensive side - 'left' or 'right'
        """
        left_count = 0
        right_count = 0

        moments = event['moments']
        ball_positions = [{'x': moment['ball_coordinates']['x'], 'y': moment['ball_coordinates']['y'], 'z': moment['ball_coordinates']['z']} for moment in moments]
        
        # Calculate distance of the ball to the left and right rim over the entire moment. Pick which ever side has more frames where the ball is closer to that rim
        for ball_pos in ball_positions:
            left_distance = self._euclidean_distance_3d(ball_pos, self._rim_locations['left'])
            right_distance = self._euclidean_distance_3d(ball_pos, self._rim_locations['right'])
            
            if left_distance <= radius:
                left_count += 1
            if right_distance <= radius:
                right_count += 1
        if left_count > right_count:
            return 'left'
        else:
            return 'right'

    def assign_offensive_side(self, events: list[dict], shot_attempt_radius: float=3) -> list[dict]:
        """
        Assign the offensive side to each row in the dataset, given the entire game data

        Args:
            events (list[dict]): List of SportVU events
            shot_attempt_radius (float, optional): Radius around the rim to consider for side assignment. Defaults to 3.

        Returns:
            list[dict]: List of events with new field of 'offensive side'
        """
        
        # Split the data into quarters
        data_quarters = {
            i: [event for event in events if event['moments'][0]['quarter'] == i]
            for i in range(1, 5)
        }

        # Extract the home and away team ids
        home_team_id = events[0]['home']['teamid']
        away_team_id = events[0]['visitor']['teamid']
        cleaned_data_w_sides = []
        for quarter, rows in data_quarters.items():
            side_counts = {
                home_team_id: [],
                away_team_id: []
            }
            # Determine the offensive side for each event in the quarter
            for row in rows:
                possession_team_id = row['event_info']['possession_team_id']
                side = self.determine_offensive_side(row=row, radius=shot_attempt_radius)
                side_counts[possession_team_id].append(side)
            
            # Count the number of times each team was on the left and right side
            home_team_counts = Counter(side_counts[home_team_id])

            # Assign the side for which most possessions occured on for each team
            if home_team_counts['left'] > home_team_counts['right']:
                home_team_side = 'left'
                away_team_side = 'right'
            else:
                home_team_side = 'right'
                away_team_side = 'left'

            # Add the correct court side to the event data
            for row in rows:
                possession_team_id = row['event_info']['possession_team_id']
                if possession_team_id == home_team_id:
                    row['event_info']['offensive_side'] = home_team_side
                else:
                    row['event_info']['offensive_side'] = away_team_side
                cleaned_data_w_sides.append(row)
        
        return cleaned_data_w_sides

    def get_offensive_frames(self, event: list[dict]) -> dict:
        """
        Get the frames for an event that correspond to only the half court offense.
        Defined as all 5 offensive players are on the offensive half

        Args:
            event (list[dict]): SportVU event data

        Returns:
            list[dict]: SportVU event data with only offesnive frames
        """
        new_event = deepcopy(event)
        offensive_frames = []
        offensive_side = event['event_info']['offensive_side']
        offensive_team_id =  event['event_info']['possession_team_id']
        moments = event['moments']
        for moment in moments:
            player_xs = [player['x'] for player in moment['player_coordinates'] if player['teamid'] == offensive_team_id]
            
            # Check to see if all players are on the correct side of the court
            if offensive_side == 'left':
                if all([x <= 47 for x in player_xs]):
                    offensive_frames.append(moment)
            else:
                if all([x >= 47 for x in player_xs]):
                    offensive_frames.append(moment)
        
        new_event['moments'] = offensive_frames
        return new_event
    
    def remove_event_overlap(self, deduped_event_data: list[dict]) -> list[dict]:
        """
        Remove overlap across events by filtering for only frames in the halfcourt offense

        Args:
            deduped_event_data (list[dict]): List of events that have already been deduped

        Returns:
            list[dict]: Events with only offensive frames
        """
        deduped_event_data_w_sides = self.assign_offensive_side(deduped_event_data)
        no_overlap_data = []
        for event in deduped_event_data_w_sides:
            offensive_frames = self.get_offensive_frames(event)
            if len(offensive_frames['moments']) > 0:
                no_overlap_data.append(offensive_frames)
        return no_overlap_data
    
    def save_cleaned_data(self, cleaned_data: list[dict], filename: str, folder_name: str="./clean_data", verbose: bool=False):
        """
        Saves the cleaed data

        Args:
            cleaned_data (list[dict]): Deduped and overlap removed SportVu data for a game
            filename (str): name of the file when saved
            folder_name (str, optional): Directory to save the file to. Defaults to "./clean_data".
            verbose (bool, optional): Print status to terminal. Defaults to False.
        """
        # Serialize the list of dictionaries to a JSON string (in-memory)
        json_filepath = f"{filename[:-3]}.json"
        with open(json_filepath, 'w') as json_file:
            json.dump(cleaned_data, json_file)

        # Specify the subdirectory and ensure it exists'
        os.makedirs(folder_name, exist_ok=True)
        archive_file_path = os.path.join(folder_name, filename) 

        # Create the .7z archive and add the in-memory JSON data
        with py7zr.SevenZipFile(archive_file_path, 'w') as archive:
            archive.write(json_filepath)
        
        os.remove(json_filepath)
        
        if verbose:
            print(f"Successfully saved {filename}")

if __name__ == "__main__":
    sportvu_cleaner = SportVUCleaner()
    
    # Get all available game file names from repo
    game_names = sportvu_cleaner.get_all_available_game_names()

    # Filter for games that have not been completed
    output_folder = "./clean_game_data"
    completed = os.listdir(output_folder)
    game_names = [name for name in game_names if name not in completed]

    for game_name in tqdm([game_names[0]]):
        raw_data = sportvu_cleaner.import_sportvu_game_data(name=game_name)
        deduped_data = sportvu_cleaner.remove_duplicate_events(raw_hf_data=raw_data, verbose=True)
        no_overlap_data = sportvu_cleaner.remove_event_overlap(deduped_data=deduped_data)
        sportvu_cleaner.save_cleaned_data(cleaned_data=no_overlap_data, filename=game_name, folder_name=output_folder)
