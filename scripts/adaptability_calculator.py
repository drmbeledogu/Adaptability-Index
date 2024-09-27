import numpy as np
import pandas as pd
import os
import json
from entropy_estimators import entropyd, entropy, midd, micd, cmicd
from collections import Counter
import py7zr
from tqdm import tqdm

class AdaptabilityCalculator():
    
    def __init__(self):
        pass

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

    def read_game_data(self, game_name, folder_name="clean_game_data") -> dict:
        """
        Get the sportvu event data for a game. Meant to be used for games that have been compressed and cleaned.
        
        Parameters:
        - game_name: The name of the file containing the game data
        - folder_name: The name of the folder where all the game data is held
        
        Returns:
        - Dict that contains the sportvu json event data.
        """
        archive_file_path = os.path.join(folder_name, game_name) 

        # Open the .7z file and extract the JSON file
        with py7zr.SevenZipFile(archive_file_path, mode='r') as archive:
            # Extract all the files to memory
            extracted_files = archive.readall()

        # Get the JSON file content (assuming it is named 'my_data_file.json')
        json_file_name = f"{game_name[:-3]}.json"
        json_data = extracted_files[json_file_name].read().decode('utf-8')  # Decode bytes to string

        # Convert JSON string back to a list of dictionaries
        clean_game_data = json.loads(json_data)

        return clean_game_data

    def read_game_data(self, game_name: str, folder_name: str="./clean_game_data") -> dict:
        """
        Get the sportvu event data for a game. Meant to be used for games that have been compressed and cleaned.

        Args:
            game_name (str): The name of the file containing the game data
            folder_name (str, optional): The name of the folder where all the game data is held. Defaults to "./clean_game_data".

        Returns:
            dict: Dict that contains the sportvu json event data.
        """
        archive_file_path = os.path.join(folder_name, game_name) 

        # Open the .7z file and extract the JSON file
        with py7zr.SevenZipFile(archive_file_path, mode='r') as archive:
            # Extract all the files to memory
            extracted_files = archive.readall()

        # Get the JSON file content (assuming it is named 'my_data_file.json')
        json_file_name = f"{game_name[:-3]}.json"
        json_data = extracted_files[json_file_name].read().decode('utf-8')  # Decode bytes to string

        # Convert JSON string back to a list of dictionaries
        clean_game_data = json.loads(json_data)

        return clean_game_data

    def is_in_moment(self, player_id: int, moment: dict) -> bool:
        """
        Check if a player is in an a moment.

        Args:
            player_id (int): The player's ID to check for.
            moment (dict): The data for the moment

        Returns:
            bool: True if the player was in the moment, False otherwise.
        """
        return any([player_id == oncourt_player['playerid'] for oncourt_player in moment['player_coordinates']])

    def get_team_id(self, player_id: int, event: dict) -> int | None:
        """
        Get the team ID for a player in an event.

        Args:
            player_id (int): The player's ID to check for.
            event (dict): The data for the event.

        Returns:
             int | None: Team ID if player is found else None
        """
        home_players = event['home']['players']
        away_players = event['visitor']['players']
        home_player_ids = [player['playerid'] for player in home_players]
        away_player_ids = [player['playerid'] for player in away_players]
        
        if player_id in home_player_ids:
            return event['home']['teamid']
        elif player_id in away_player_ids:
            return event['visitor']['teamid']
        else:
            return None

    def get_lineup_from_moment(self, team_id: int, moment: dict) -> frozenset:
        """
        Get the lineup for a team in a moment.

        Args:
            team_id (int): The team ID for the lineup of interest.
            moment (dict): The data for the moment.

        Returns:
            frozenset: Set of player IDs representing the lineup for the team.
        """

        lineup = [player['playerid'] for player in moment['player_coordinates'] if player['teamid'] == team_id]
        return frozenset(lineup)

    def _convert_court_coordinates(self, x: float, y: float) -> tuple[float, float]:
        """
        Convert the x and y coordinates of a player to the court coordinates on one side.

        Args:
            x (float): The x-coordinate of the player.
            y (float): y-coordinate of the player.

        Returns:
            tuple[float, float]: Tuple of the x and y coordinates in the court coordinates.
        """
        return 94-x, 50-y

    def get_player_coordinates(self, player_id: int, all_player_coordinates: list[dict]) -> dict | None:
        """
        Get the coordinates of a player from all available player coordinates

        Args:
            player_id (int): ID of the player of interest
            all_player_coordinates (list[dict]): All player coordinates for the moment

        Returns:
            dict | None: Coordinates of the player of interest of they're found. None otherwise
        """
        for player in all_player_coordinates:
            if player['playerid'] == player_id:
                return {'x': player['x'], 'y': player['y'], 'z': player['z']}
        return None  # Return None if the player_id is not found
    
    def player_has_ball(self, player_coordinates: dict, ball_coordinates: dict, radius: float=3, height: float=3) -> bool:
        """
        Determine if the player is in possession of the ball

        Args:
            player_coordinates (dict): Coordinates of the player. Must containt keys 'x', 'y', 'z'
            ball_coordinates (dict): Coordinates of the ball. Must containt keys 'x', 'y', 'z'
            radius (float, optional): The radius allowed around a player to be considered to have possession of the ball. Defaults to 3.
            height (float, optional): The height above the player for which they can be considered in possession of the ball. Defaults to 3.

        Returns:
            bool: True if the player has the ball and False otherwise
        """
        ball_plane_distance = self._euclidean_distance_2d(player_coordinates, ball_coordinates)
        ball_height_distance = ball_coordinates["z"] - player_coordinates["z"]

        if ball_plane_distance <= radius and ball_height_distance <= height:
            return True
        else:
            return False

    def extract_position_vector(self, player_id: int, moment: dict) -> list[float]:
        """
        Extract the position vector to be used in a list of position vectors from the moment data

        Args:
            player_id (int): The player's ID to extract position vector for
            moment (dict): The moment to extract the position vector for

        Returns:
            list[float]: A list of the x, y coordinates of the player
        """
        
        player_coordinates = self.get_player_coordinates(player_id=player_id, all_player_coordinates=moment['player_coordinates'])
        if player_coordinates['x'] > 47: # Convert coordinates if on wrong side of the court
            x, y = self._convert_court_coordinates(player_coordinates['x'], player_coordinates['y'])
        else:
            x, y = player_coordinates['x'], player_coordinates['y']
        return [x, y]

    def get_vectors_from_moment(self, player_id: int, team_id: int, moment: dict, include_ball: bool=True) -> tuple[frozenset, list, bool | None]:
        """
        Extract lineup set, position_vector, and ball_possession boolean from a moment

        Args:
            player_id (int): The player's ID to extract the relevant information for
            team_id (int): The player's team ID
            moment (dict): The moment to extract the relevant information for
            include_ball (bool, optional): Boolean to extract ball possession info or not. Defaults to True.

        Raises:
            Exception: Player ID not found in moment
            Exception: Full lineup is not on the court
            Exception: No shot clock in data

        Returns:
            tuple[frozenset, list, bool | None]: Tuple of the frozenset of the lineup, the position vector, and the boolean for ball possession
        """
        if not self.is_in_moment(player_id, moment):
            raise Exception(f"Player ID {player_id} is not in the moment")
        if len(moment['player_coordinates']) != 10:
            raise Exception(f"Moment does not include all players")
        if moment['shot_clock'] is None:
            raise Exception(f"Moment happens with empty value for shot clock")
        
        lineup = self.get_lineup_from_moment(team_id, moment)  # Already returns a frozenset
        position_vector = self.extract_position_vector(player_id, moment)

        if include_ball:
            player_coordinates = self.get_player_coordinates(player_id, moment['player_coordinates'])
            ball_possession = self.player_has_ball(player_coordinates, ball_coordinates=moment['ball_coordinates'])
        else:
            ball_possession = None
        
        return lineup, position_vector, ball_possession
    
    def filter_vectors_for_minimum(
            self,
            lineups: list[frozenset],
            position_vectors: list,
            lineup_stats: dict,
            ball_possessions: list[bool] | None=None,
            min_games: int | None=None,
            min_events: int | None=None,
            min_moments: int | None=None
        ) -> tuple[list[frozenset], list, list[bool] | None]:
        """
        Filter raw vectors extracted from moments to meet minimum games, events or moments thresholds per lineup

        Args:
            lineups (list[frozenset]): List of lineups for each moment
            position_vectors (list): List of position vectors for each moment
            lineup_stats (dict): Lineup statistics containing the set of games, number of events, and number of moments. Dict must contain `games'(set), 'events'(int), 'moments'(int)
            ball_possessions (list[bool] | None, optional): List of booleans for ball possesion. Can be None if not to be included in calculation. Defaults to None.
            min_games (int | None, optional): Minimum number of games a lineup must have to be included. Defaults to None.
            min_events (int | None, optional): Minimum number of events a lineup must have to be included. Defaults to None.
            min_moments (int | None, optional): Minimum number of moments a lineup must have to be included. Defaults to None.

        Returns:
            tuple[list[frozenset], list, list[bool] | None]: Tuple containing the lineups, position vectors, and possesions booleans (if appliacable) that meet minimum thresholds for each lineup
        """
        # Apply filtering based on cumulative statistics for each lineup
        filtered_lineups, filtered_position_vectors = [], []
        if ball_possessions is not None:
            filtered_ball_possessions = []
        else:
            filtered_ball_possessions = None

        for i, lineup in enumerate(lineups):
            lineup_data = lineup_stats[lineup]

            # Apply filters based on the total counts across games, events, and moments for the lineup
            if (min_games is not None and len(lineup_data["games"]) < min_games):
                continue  # Skip if minimum games threshold is not met
            if (min_events is not None and lineup_data["events"] < min_events):
                continue  # Skip if minimum events threshold is not met
            if (min_moments is not None and lineup_data["moments"] < min_moments):
                continue  # Skip if minimum moments threshold is not met
            
            # If all conditions are met, retain the lineup and corresponding position vector
            filtered_lineups.append(lineup)
            filtered_position_vectors.append(position_vectors[i])
            if ball_possessions is not None:
                filtered_ball_possessions.append(ball_possessions[i])
        
        return filtered_lineups, filtered_position_vectors, filtered_ball_possessions

    def get_vectors_from_events(
            self,
            player_id: int,
            events: list[dict],
            include_ball: bool=True,
            min_games: int | None=None,
            min_events: int | None=None,
            min_moments: int | None=None
        ) -> tuple[list[frozenset], list[list], list[bool] | None, dict]:
        """
        Get lineups, position vectors, and possession booleans (if applicable) from a list of events.
        Provides option to filter some lineups out based on minimum games/evets/moments

        Args:
            player_id (int): The player's ID to extract the relevant information for
            events (list[dict]): The events to extract the relevant information for
            include_ball (bool, optional): Boolean to extract ball possession info or not. Defaults to True.
            min_games (int | None, optional): Minimum number of games a lineup must have to be included. Defaults to None.
            min_events (int | None, optional): Minimum number of events a lineup must have to be included. Defaults to None.
            min_moments (int | None, optional): Minimum number of moments a lineup must have to be included. Defaults to None.

        Returns:
            tuple[list[frozenset], list[list], list[bool] | None, dict]: Tuple of the frozenset of the lineup, the position vector, and the boolean for ball possession as well as the lineup stats
        """
        position_vectors, lineups = [], []
        if include_ball:
            ball_possessions = []
        else:
            ball_possessions = None
        lineup_stats = {}  # To track cumulative games, events, and moments for each lineup
        
        for event in tqdm(events, desc="Event analysis loop", leave=False):
            team_id = self.get_team_id(player_id, event)
            if team_id is not None:
                if event['event_info']['possession_team_id'] == team_id:
                    moments_in_event = 0
                    for moment in event['moments']:
                        if not self.is_in_moment(player_id, moment):
                            continue
                        if len(moment['player_coordinates']) != 10:
                            continue
                        if moment['shot_clock'] is None:
                            continue
                        
                        # Get data for mutual information calculation
                        lineup, position_vector, ball_possession = self.get_vectors_from_moment(player_id, team_id, moment, include_ball=include_ball)
                        lineups.append(lineup)
                        position_vectors.append(position_vector)
                        if include_ball:
                            ball_possessions.append(ball_possession)

                        # Initialize lineup statistics if not already present
                        if lineup not in lineup_stats:
                            lineup_stats[lineup] = {"games": set(), "events": 0, "moments": 0}

                        # Update lineup statistics
                        lineup_stats[lineup]["moments"] += 1
                        moments_in_event += 1

                    # Track events and games only if valid moments found in the event
                    if moments_in_event > 0:
                        lineup_stats[lineup]["events"] += 1
                        lineup_stats[lineup]["games"].add(event['gameid'])

        # Apply filtering based on cumulative statistics for each lineup
        filtered_lineups, filtered_position_vectors, filtered_ball_possessions = self.filter_vectors_for_minimum(lineups, position_vectors, lineup_stats, ball_possessions=ball_possessions, min_games=min_games, min_events=min_events, min_moments=min_moments)

        return filtered_lineups, filtered_position_vectors, filtered_ball_possessions, lineup_stats

    def compute_info_theory_results_from_vectors(
            self,
            position_vectors: list[list],
            lineups: list[frozenset],
            ball_possessions: list[bool]=None,
            mi_metric: str="euclidean"
        ) -> dict:
        """
        Compute information theoretic measures from vectors.
        Computes lineup entropy, position entropy, mutual information between lineup and position/ball possesion(if applicable)
        Also computes some other useful values like number of samples and number of unique lineups

        Args:
            position_vectors (list[list]): List of [x, y] position vectors corresponding to moments
            lineups (list[frozenset]): List of lineeups corresponding to moments
            ball_possessions (list[bool], optional): List of booleans determining ball possession corresponding to moments. Defaults to None.
            mi_metric (str, optional): Distance metric to be used in entropy/mutual info calcs. Defaults to "euclidean".

        Returns:
            dict: Result of information theoretic quantities
        """
        if ball_possessions is not None: # Compute Position/ball possesion and lineup MI
            lineup_ball_possesion_mutual_info = midd(lineups, ball_possessions, base=np.e)
            conditional_mutual_info = cmicd(position_vectors, lineups, ball_possessions, k=3, base=np.e, metric=mi_metric)
            final_mutual_info = lineup_ball_possesion_mutual_info + conditional_mutual_info
        else: # Compute Positionand lineup MI
            final_mutual_info = micd(position_vectors, lineups, k=3, base=np.e, metric=mi_metric)

        # Compute Lineup entropy
        lineup_entropy = entropyd(lineups, base=np.e)

        # Compute position_vector entropy
        position_entropy = entropy(position_vectors, k=3, base=np.e, metric=mi_metric)

        # Compute % of LE explained by MI
        pct_shared = final_mutual_info/lineup_entropy

        results = {
            "mutual_info": final_mutual_info,
            "position_entropy": position_entropy,
            "lineup_entropy": lineup_entropy,
            "mi_pct_lineup_entropy": pct_shared,
            "moment_count": len(lineups),
            "unique_lineups": len(list(Counter(lineups).keys()))
        }

        return results
    
    def compute_info_theory_results_from_events(
            self,
            player_id: int,
            events: list[dict],
            include_ball: bool=True,
            min_games: int | None=None,
            min_events: int | None=None,
            min_moments: int | None=None,
            mi_metric: str="euclidean"
        ) -> dict:
        """
        Compute information theoretic measures from events.
        Computes lineup entropy, position entropy, mutual information between lineup and position/ball possesion(if applicable)
        Also computes some other useful values like number of samples, unique lineups, events, and games

        Args:
            player_id (int): The player's ID to compute the information for
            events (list[dict]): The events to compute the information for
            include_ball (bool, optional): Boolean to use ball possession info or not. Defaults to True.
            min_games (int | None, optional): Minimum number of games a lineup must have to be included. Defaults to None.
            min_events (int | None, optional): Minimum number of events a lineup must have to be included. Defaults to None.
            min_moments (int | None, optional): Minimum number of moments a lineup must have to be included. Defaults to None.
            mi_metric (str, optional): Distance metric to be used in entropy/mutual info calcs. Defaults to "euclidean".

        Returns:
            dict: Result of information theoretic quantities
        """
        lineups, position_vectors, ball_possessions, lineup_stats = self.get_vectors_from_events(player_id, events, include_ball=include_ball, min_games=min_games, min_events=min_events, min_moments=min_moments)

        # Compute final results using the filtered data
        if include_ball:
            results = self.compute_info_theory_results_from_vectors(position_vectors=position_vectors, lineups=lineups, ball_possessions=ball_possessions, mi_metric=mi_metric)
        else:
            results = self.compute_info_theory_results_from_vectors(position_vectors=position_vectors, lineups=lineups, ball_possessions=None, mi_metric=mi_metric)
        
        results["player_id"] = player_id
        results["event_count"] = sum(lineup_stats[l]["events"] for l in set(lineups))
        results["game_count"] = len(set().union(*(lineup_stats[l]["games"] for l in lineups)))

        return results
    
    def compute_info_theory_results_from_games(
            self,
            player_id: int,
            games: list[str],
            include_ball: bool=True,
            min_games: int | None=None,
            min_events: int | None=None,
            min_moments: int | None=None,
            mi_metric: str="euclidean",
            folder_name: str="./clean_game_data"
        ) -> dict:
        """
        Compute information theoretic measures from games.
        Computes lineup entropy, position entropy, mutual information between lineup and position/ball possesion(if applicable)
        Also computes some other useful values like number of samples, unique lineups, events, and games

        Args:
            player_id (int): The player's ID to extract the compute the information for
            games (list[str]): The games to compute the information for. These are file names and assumes they are zipped
            include_ball (bool, optional): Boolean to use ball possession info or not. Defaults to True.
            min_games (int | None, optional): Minimum number of games a lineup must have to be included. Defaults to None.
            min_events (int | None, optional): Minimum number of events a lineup must have to be included. Defaults to None.
            min_moments (int | None, optional): Minimum number of moments a lineup must have to be included. Defaults to None.
            mi_metric (str, optional): Distance metric to be used in entropy/mutual info calcs. Defaults to "euclidean".
            folder_name (str, optional): The name of the folder where the game data is held. Defaults to "./clean_game_data".

        Returns:
            dict: Result of information theoretic quantities
        """
        events = []
        for game_name in tqdm(games, desc="Game download loop", leave=False):
            game_data = self.read_game_data(game_name, folder_name=folder_name)
            if self.get_team_id(player_id, game_data[0]) is not None:
                events.extend(game_data)
        results = self.compute_info_theory_results_from_events(player_id, events, include_ball=include_ball, min_games=min_games, min_events=min_events, min_moments=min_moments, mi_metric=mi_metric)
        return results
    
if __name__ == "__main__":
    adaptability_calculator = AdaptabilityCalculator()
    result_columns = ["mutual_info", "position_entropy", "lineup_entropy", "mi_pct_lineup_entropy", "moment_count", "unique_lineups", "player_id", "event_count", "game_count"]

    # Get players to run analaysis on
    players_of_interest = pd.read_csv("./auxilliary_data/players_of_interest.csv")
    player_ids = players_of_interest["Player ID"].to_list()
    
    # Retrieve results that have already been computed
    output_filepath = "./results/adaptability_calc_results_ind_conditional.csv"
    if os.path.exists(output_filepath):
        results = pd.read_csv(output_filepath)
    else:
        results = pd.DataFrame(columns=result_columns)
    
    # Run calc on player_ids that have not been run
    player_ids = [player_id for player_id in player_ids if player_id not in results["player_id"].to_list()]
    
    # Get full list of games
    games = os.listdir("./clean_game_data")

    # Run calc on all players
    for player_id in tqdm(player_ids, desc="Player loop"):
        abbrev_team = players_of_interest[players_of_interest["Player ID"] == player_id]["Team"].iloc[0]
        player_games = [game for game in games if abbrev_team in game]
        # try:
        new_row = pd.DataFrame([adaptability_calculator.compute_info_theory_results_from_games(player_id=player_id, games=player_games, include_ball=True, min_games=None, min_events=None, min_moments=10_000, mi_metric="euclidean")])
        # except:
        #     new_row_data = {col: pd.NA for col in result_columns}
        #     new_row_data["player_id"] = player_id
        #     new_row = pd.DataFrame([new_row_data])
        results = pd.concat([results, new_row], ignore_index=True)
        results.to_csv(output_filepath, index=False)
