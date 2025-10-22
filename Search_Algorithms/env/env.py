"""
Search Environment for Fundamentals of AI Course
University of Isfahan - Fall 2025

Course Instructor: Professor Dr. Hossein Karshenas
Teaching Assistants:
    - Mahdi Mahdieh
    - Younes Rad
    - Pooya Esfandany
    - Danial Shafiei

This environment implements a grid-based search problem where an agent must navigate
through various terrains while avoiding enemies and collecting targets. The project
focuses on implementing and comparing search algorithms like Breadth-First Search,
Depth-Limited Search, Uniform-Cost Search, and particularly A* with different heuristic
functions.

Key Components:
- GameState: Represents the current state of the game (agent position, targets, step count)
- GridWorld: Main game environment that loads maps and handles visualization
- Search Algorithms: Implementation of various uninformed and informed search strategies

Environment Features:
- Multiple terrain types with different costs (grass, bushes, rocks)
- Moving enemy with predefined path
- Target collection objective
- Pygame-based visualization
- Performance metrics tracking
- User-defined text-based maps.

Usage:
    from environment import play
    score = play("map1", a_star_search, delay=500)

Note: This code is for educational purposes in the Fundamentals of AI course.
"""

import pygame
import numpy as np
import os
import time

NODE_EXPANSION_COST = 0.01
GRASS_PASSING_COST = 5
BUSH_PASSING_COST = 100
ENEMY_COLLISION_PENALTY = 10000
GRID_SIZE_REWARD = 10
TARGET_REWARD = 200
MAX_ACTIONS = 200

DIRECTIONS = {
    'U': (-1, 0),
    'D': (1, 0),
    'L': (0, -1),
    'R': (0, 1)
}

# Global reference for the current game (simplified approach)
_current_game = None


def play(map_name, search_algorithm, delay=1000):
    global _current_game
    _current_game = GridWorld(map_name)
    return _current_game.run_game(search_algorithm, delay)


class GameState:
    __slots__ = ('_agent_pos', '_targets', '_step', '_grid_shape', '_enemy_path', '_original_grid', '_grid_world_ref')

    def __init__(self, agent_position, targets, step, grid_shape, enemy_path, original_grid, grid_world_ref):
        self._agent_pos = agent_position
        self._targets = frozenset(targets)
        self._step = step
        self._grid_shape = grid_shape
        self._enemy_path = enemy_path
        self._original_grid = original_grid
        self._grid_world_ref = grid_world_ref

    def __hash__(self):
        return hash((self._agent_pos, self._targets, self._step))

    def __eq__(self, other):
        return (self._agent_pos == other.get_agent_position() and
                list(self._targets) == other.get_targets_frozenset() and
                self._step == other.get_step_count())

    def get_targets_frozenset(self):
        return self._targets

    def action_to_position(self, action):
        dr, dc = DIRECTIONS[action]
        return self._agent_pos[0] + dr, self._agent_pos[1] + dc

    def _update_targets(self, new_position):
        new_targets = set(self._targets)
        if new_position in new_targets:
            new_targets.remove(new_position)
        return frozenset(new_targets)

    def get_grid_size(self):
        return self._grid_shape

    def get_enemy_position(self):
        if self._enemy_path:
            return self._enemy_path[self._step % len(self._enemy_path)]
        return None

    def get_enemy_next_position(self):
        if self._enemy_path:
            return self._enemy_path[(self._step + 1) % len(self._enemy_path)]
        return None

    def get_enemy_path(self):
        return self._enemy_path

    def get_agent_position(self):
        return self._agent_pos

    def get_terrain_cost(self, position):
        r, c = position
        cost = 0
        cell_type = self._original_grid[r, c]
        if cell_type in ['T', 'A', 'G', 'R', 'E']:
            cost += GRASS_PASSING_COST
        if r < 0 or r >= self._grid_shape[0] or c < 0 or c >= self._grid_shape[1]:
            cost += GRASS_PASSING_COST
        elif cell_type == 'B':
            cost += BUSH_PASSING_COST
        if self.get_enemy_position() == position:
            cost += ENEMY_COLLISION_PENALTY
        return cost

    def get_targets_positions(self):
        return list(self._targets)


    def get_step_count(self):
        return self._step

    def is_goal_state(self):
        return len(self._targets) == 0 and self._agent_pos != self.get_enemy_position()

    def is_collision_state(self):
        return self._agent_pos == self.get_enemy_position() or self._agent_pos == self.get_enemy_next_position()



    def get_all_successors(self):
        successors = []

        for action, (dr, dc) in DIRECTIONS.items():
            new_r, new_c = self._agent_pos[0] + dr, self._agent_pos[1] + dc
            old_r, old_c = self._agent_pos[0], self._agent_pos[1]
            new_pos = (new_r, new_c)

            if not (0 <= new_r < self._grid_shape[0] and
                    0 <= new_c < self._grid_shape[1]):
                new_pos = (old_r, old_c)
            elif self._original_grid[new_r, new_c] == 'R':
                new_pos = (old_r, old_c)

            new_targets = self._update_targets(new_pos)

            cost = self.get_terrain_cost(new_pos)

            new_state = GameState(
                new_pos,
                new_targets,
                self._step + 1,
                self._grid_shape,
                self._enemy_path,
                self._original_grid,
                self._grid_world_ref
            )
            successors.append((action, cost, new_state))
        return successors

    def get_successors(self):
        successors = []

        for action, (dr, dc) in DIRECTIONS.items():
            new_r, new_c = self._agent_pos[0] + dr, self._agent_pos[1] + dc

            new_pos = (new_r, new_c)

            if not (0 <= new_r < self._grid_shape[0] and
                    0 <= new_c < self._grid_shape[1]):
                continue
            elif self._original_grid[new_r, new_c] == 'R':
                continue

            new_targets = self._update_targets(new_pos)

            cost = self.get_terrain_cost(new_pos)

            new_state = GameState(
                new_pos,
                new_targets,
                self._step + 1,
                self._grid_shape,
                self._enemy_path,
                self._original_grid,
                self._grid_world_ref
            )
            successors.append((action, cost, new_state))

        return successors

    def get_bushes_positions(self):
        return self._grid_world_ref.get_bushes_positions()


    def get_rocks_positions(self):
        return self._grid_world_ref.get_rocks_positions()


    def is_bush_position(self, pos):
        return self._grid_world_ref.is_bush_position(pos)


    def is_rock_position(self, pos):
        self._grid_world_ref.is_rock_position(pos)


class CountingGameState:
    __slots__ = ('_state', '_counter')

    def __init__(self, state, counter):
        self._state = state
        self._counter = counter

    def get_all_successors(self):
        successors = self._state.get_all_successors()
        self._counter.increment_more(len(successors))

        counted_successors = []
        for action, cost, state in successors:
            counted_state = CountingGameState(state, self._counter)
            counted_successors.append((action, cost, counted_state))

        return counted_successors

    def get_successors(self):
        successors = self._state.get_successors()
        self._counter.increment_more(len(successors))

        counted_successors = []
        for action, cost, state in successors:
            counted_state = CountingGameState(state, self._counter)
            counted_successors.append((action, cost, counted_state))

        return counted_successors

    def get_expanded_nodes(self):
        return self._counter.get_count()

    def __getattr__(self, name):
        return getattr(self._state, name)

    def __hash__(self):
        return self._state.__hash__()

    def __eq__(self, other):
        if isinstance(other, CountingGameState):
            return self._state == other._state
        return self._state == other


class CallCounter:
    def __init__(self):
        self._count = 0

    def increment(self):
        self._count += 1

    def increment_more(self, count):
        if count < 0:
            raise ValueError('count must be positive')
        self._count += count

    def get_count(self):
        return self._count


class GridWorld:
    def __init__(self, map_name, cell_size = 80):
        self._cell_size = cell_size

        self._map_name = map_name
        self._original_grid = None
        self._grid_data = None
        self._initial_state = None
        self._enemy_path = []
        self._expanded_nodes = 0
        self._total_cost = 0
        self._targets_count = 0

        # Pygame assets
        self._screen = None
        self._images = {}
        self._font = None
        self._pygame_initialized = False


        # Loading elements
        self._bushes = []
        self._rocks = []
        self._load_map(map_name)
        self._load_assets()






    def reset(self):
        self._original_grid = None
        self._grid_data = None
        self._initial_state = None
        self._expanded_nodes = 0
        self._total_cost = 0

        # Pygame assets
        self._pygame_initialized = False
        self._load_map(self._map_name)


    def _load_map(self, map_name):
        map_path = f"env/src/maps/{map_name}.txt"
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Map file not found: {map_path}")

        with open(map_path, 'r') as f:
            content = f.read().strip()

        parts = content.split('\n\n')
        grid_lines = [line for line in parts[0].split('\n') if line.strip()]
        path_str = parts[1] if len(parts) > 1 else ""

        self._original_grid = np.array([list(line) for line in grid_lines])
        self._grid_data = self._original_grid.copy()

        rows, cols = self._original_grid.shape
        agent_pos = None
        enemy_pos = None
        targets = []
        bushes = []
        rocks = []
        enemy_count = 0
        targets_count = 0
        for r in range(rows):
            for c in range(cols):
                cell = self._original_grid[r, c]
                if cell == 'A':
                    agent_pos = (r, c)
                elif cell == 'E':
                    enemy_count += 1
                    enemy_pos = (r, c)
                elif cell == 'T':
                    targets.append((r, c))
                    targets_count += 1
                elif cell == 'B':
                    bushes.append((r, c))
                elif cell == 'R':
                    rocks.append((r, c))

        self._bushes_positions = frozenset(bushes)
        self._rocks_positions = frozenset(rocks)
        self._targets_count = targets_count

        if agent_pos is None:
            raise ValueError("No agent starting position found")
        if enemy_count > 1:
            raise ValueError("Too many enemies found")

        self._enemy_path = self._parse_enemy_path(path_str, enemy_pos)

        self._initial_state = GameState(
            agent_pos,
            targets,
            0,
            self._original_grid.shape,
            self._enemy_path,
            self._original_grid, self  # Pass reference to grid data
        )

    def _parse_enemy_path(self, path_str, start_pos):
        if start_pos is None or path_str == '':
            return None
        path = [start_pos]
        current_pos = start_pos

        for move in path_str:
            if move not in DIRECTIONS:
                raise ValueError("Invalid move")

            dr, dc = DIRECTIONS[move]
            new_r, new_c = current_pos[0] + dr, current_pos[1] + dc

            if (0 <= new_r < self._original_grid.shape[0] and
                    0 <= new_c < self._original_grid.shape[1] and
                    self._original_grid[new_r, new_c] != 'R'):
                current_pos = (new_r, new_c)
                path.append(current_pos)

        return path

    def _load_assets(self):
        try:
            if not pygame.get_init():
                pygame.init()

            icon_path = "env/src/icons/"

            self._images = {
                'grass': self._load_image(icon_path + "grass.png", 'grass'),
                'rock': self._load_image(icon_path + "rock.png", 'rock'),
                'bush': self._load_image(icon_path + "bush.png", 'bush'),
                'target': self._load_image(icon_path + "star.png", 'target'),
                'agent': self._load_image(icon_path + "bird.png"),
                'enemy': self._load_image(icon_path + "pig.png")
            }

            self._font = pygame.font.Font(None, 24)

        except Exception as e:
            print(f"Warning: Could not load assets, using fallback colors: {e}")
            self._images = {}

    def _load_image(self, path, element_type = None):
        try:
            if os.path.exists(path):
                image = pygame.image.load(path)
                if element_type == 'grass':
                    image = pygame.transform.scale(image, (self._cell_size, 2 * self._cell_size))
                elif element_type == 'rock':
                    image = pygame.transform.scale(image, (self._cell_size, 1.2 * self._cell_size))
                elif element_type == 'bush':
                    image = pygame.transform.scale(image, (self._cell_size, 1.4 * self._cell_size))
                elif element_type == 'target':
                    image = pygame.transform.scale(image, (self._cell_size, 1.5 * self._cell_size))
                else:
                    image = pygame.transform.scale(image, (self._cell_size, self._cell_size))
                return image
        except:
            raise FileNotFoundError(f"Image file not found: {path}")

    def _initialize_pygame(self):
        if self._pygame_initialized:
            return

        pygame.init()
        rows, cols = self._original_grid.shape
        screen_width = cols * self._cell_size
        screen_height = rows * self._cell_size + 30 # HUD

        self._screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption(f"Search Environment - {self._map_name}")
        self._pygame_initialized = True

    def _draw(self, state):
        if not self._pygame_initialized:
            self._initialize_pygame()

        self._screen.fill((0, 0, 0))

        rows, cols = self._original_grid.shape
        agent_pos = state.get_agent_position()
        enemy_pos = state.get_enemy_position()
        bushes = state.get_bushes_positions()
        targets = state.get_targets_positions()
        rocks = state.get_rocks_positions()

        for r in range(rows):
            for c in range(cols):

                bush_rect = pygame.Rect(c * self._cell_size, (r - .6) * self._cell_size, self._cell_size, self._cell_size)
                self._screen.blit(self._images['grass'], bush_rect)

                if (r, c) in bushes:
                    bush_rect = pygame.Rect(c * self._cell_size,
                                            (r-.2) * self._cell_size,
                                            self._cell_size,
                                            self._cell_size)
                    self._screen.blit(self._images['bush'], bush_rect)

                if (r, c) in targets:
                    target_rect = pygame.Rect(c * self._cell_size,
                                              (r - .3) * self._cell_size,
                                              self._cell_size,
                                              self._cell_size)
                    self._screen.blit(self._images['target'], target_rect)

                if (r, c) == agent_pos:
                    agent_rect = pygame.Rect(c * self._cell_size,
                                             (r-.1) * self._cell_size,
                                             self._cell_size,
                                             self._cell_size)
                    self._screen.blit(self._images['agent'], agent_rect)

                if (r, c) == enemy_pos:
                    enemy_rect = pygame.Rect(c * self._cell_size,
                                             (r-.1) * self._cell_size,
                                             self._cell_size,
                                             self._cell_size)
                    self._screen.blit(self._images['enemy'], enemy_rect)

                if (r, c) in rocks:
                    rock_rect = pygame.Rect(c * self._cell_size,
                                            (r - .2) * self._cell_size,
                                            self._cell_size,
                                            self._cell_size)
                    self._screen.blit(self._images['rock'], rock_rect)

        self._draw_hud(state)

        pygame.display.flip()


    def _draw_hud(self, state):
        rows, cols = self._original_grid.shape
        hud_y = rows * self._cell_size

        hud_rect = pygame.Rect(0, hud_y, cols * self._cell_size, 50)
        pygame.draw.rect(self._screen, (30, 30, 30), hud_rect)
        pygame.draw.line(self._screen, (100, 100, 100),
                         (0, hud_y),
                         (cols * self._cell_size, hud_y),
                         2)

        if self._font:
            targets_text = self._font.render(f"Targets: {len(state.get_targets_positions())}", True, (255, 255, 255))
            self._screen.blit(targets_text, (self._cell_size * 0.1 , hud_y + 10))

            cell_cost = state.get_terrain_cost(state.get_agent_position())
            steps_text = self._font.render(f"Steps: {state.get_step_count()} (Cost: {cell_cost})", True, (255, 255, 255))
            self._screen.blit(steps_text, (self._cell_size * 1.6, hud_y + 10))

    def run_game(self, search_algorithm, delay=1000):
        self._initialize_pygame()
        clock = pygame.time.Clock()

        print(f"Running {search_algorithm.__name__} on {self._map_name}...")
        start_time = time.time()
        initial_state = CountingGameState(self._initial_state, CallCounter())
        actions = search_algorithm(initial_state)
        self._expanded_nodes = initial_state.get_expanded_nodes()
        search_time = time.time() - start_time

        print(f"Search completed in {search_time:.2f}s, found {len(actions)} actions")

        current_state = self._initial_state

        running = True
        action_index = 0

        while running and action_index < len(actions):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            if not running:
                break

            action = actions[action_index]
            new_r, new_c = current_state.action_to_position(action)
            cost = current_state.get_terrain_cost((new_r, new_c))
            self._total_cost += cost
            action_index += 1

            found = False
            for act, cost, next_state in current_state.get_successors():
                if act == action:
                    current_state = next_state
                    found = True
                    break

            if not found:
                print(f"Invalid action: {action} at step {current_state.get_step_count()}")
                break

            self._draw(current_state)
            pygame.time.wait(delay)

            if current_state.is_goal_state():
                print("Success!")
                break

            if current_state.is_collision_state():
                print("Game Over!")
                break

            clock.tick(60)

        final_score = self._calculate_score(current_state)

        self._show_results(final_score, current_state, search_time, len(actions))

        waiting = True
        wait_start = pygame.time.get_ticks()
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    waiting = False
            if pygame.time.get_ticks() - wait_start > 5000:
                waiting = False
            pygame.time.wait(100)

        pygame.quit()
        return final_score

    def _calculate_score(self, state):
        score = 0

        # score -= self._expanded_nodes * NODE_EXPANSION_COST
        # r, c = state.get_grid_size()
        # score += ((r * c) ** 2) * GRID_SIZE_REWARD if state.is_goal_state() else 0
        score += self._targets_count * TARGET_REWARD if state.is_goal_state() else 0
        score -= ENEMY_COLLISION_PENALTY if state.is_collision_state() else 0
        score -= self._total_cost
        return int(score)

    def _show_results(self, score, final_state, search_time, actions_found):
        if not self._pygame_initialized:
            return

        rows, cols = self._original_grid.shape
        result_surface = pygame.Surface((cols * self._cell_size, rows * self._cell_size))
        result_surface.fill((0, 0, 0))
        result_surface.set_alpha(220)

        self._screen.blit(result_surface, (0, 0))

        if self._font:
            if final_state.is_goal_state():
                title = "MISSION SUCCESS!"
                color = (0, 255, 0)
            elif final_state.is_collision_state():
                title = "GAME OVER - COLLISION!"
                color = (255, 0, 0)
            else:
                title = "MISSION INCOMPLETE"
                color = (255, 255, 0)

            title_font = pygame.font.Font(None, 48)
            title_text = title_font.render(title, True, color)
            title_rect = title_text.get_rect(center=(cols * self._cell_size // 2,
                                                     rows * self._cell_size // 2 - 80))
            self._screen.blit(title_text, title_rect)

            stats = [
                f"Final Score: {score}",
                f"Expanded Nodes: {self._expanded_nodes}",
                f"Search Time: {search_time:.2f}s",
                f"Actions Found: {actions_found}",
                f"Steps Taken: {final_state.get_step_count()}",
                f"Targets Remaining: {len(final_state.get_targets_positions())}",
                "",
                "Press ESC to exit or wait 5 seconds"
            ]

            for i, stat in enumerate(stats):
                stat_text = self._font.render(stat, True, (255, 255, 255))
                stat_rect = stat_text.get_rect(center=(cols * self._cell_size // 2,
                                                       rows * self._cell_size // 2 - 20 + i * 25))
                self._screen.blit(stat_text, stat_rect)

        pygame.display.flip()

        print(f"Expanded Nodes: {self._expanded_nodes}")
        print(f"Score: {score}")


    def get_bushes_positions(self):
        return self._bushes_positions

    def get_rocks_positions(self):
        return self._rocks_positions

    def is_bush_position(self, pos):
        return pos in self._bushes_positions

    def is_rock_position(self, pos):
        return pos in self._rocks_positions
