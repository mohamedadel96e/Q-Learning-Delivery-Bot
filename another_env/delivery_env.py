"""Custom Gymnasium environment for an urban delivery robot."""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register, registry

import config


class DeliveryEnv(gym.Env):
    """A retro grid-world delivery task with buildings and service stops."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    ACTIONS = {
        0: np.array([0, -1], dtype=np.int8),  # Up
        1: np.array([0, 1], dtype=np.int8),  # Down
        2: np.array([-1, 0], dtype=np.int8),  # Left
        3: np.array([1, 0], dtype=np.int8),  # Right
    }
    ACTION_NAMES = {
        0: "up",
        1: "down",
        2: "left",
        3: "right",
        4: "pickup",
        5: "dropoff",
        6: "wait",
    }

    STEP_REWARD = config.STEP_REWARD
    ILLEGAL_REWARD = config.ILLEGAL_REWARD
    PICKUP_REWARD = config.PICKUP_REWARD
    DROPOFF_REWARD = config.DROPOFF_REWARD
    RED_LIGHT_PENALTY = config.RED_LIGHT_PENALTY
    YELLOW_LIGHT_PENALTY = config.YELLOW_LIGHT_PENALTY

    TRAFFIC_LIGHT_CYCLE = config.TRAFFIC_LIGHT_CYCLE_STEPS
    TRAFFIC_LIGHT_PHASES = config.TRAFFIC_LIGHT_PHASES  # green=0, yellow=1, red=2

    def __init__(
        self,
        render_mode: str | None = None,
        grid_size: int = config.GRID_SIZE,
        max_steps: int = config.MAX_STEPS_PER_EPISODE,
    ) -> None:
        super().__init__()

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.render_mode = render_mode
        self.grid_size = grid_size
        self.max_steps = max_steps

        self.buildings = {
            (2, 2),
            (2, 3),
            (3, 2),
            (3, 3),
            (6, 2),
            (6, 3),
            (7, 2),
            (7, 3),
            (2, 6),
            (2, 7),
            (3, 6),
            (3, 7),
            (6, 6),
            (6, 7),
            (7, 6),
            (7, 7),
        }
        self.building_blocks = (
            (2, 2, 2, 2),
            (6, 2, 2, 2),
            (2, 6, 2, 2),
            (6, 6, 2, 2),
        )

        self.pickup_locations = ((1, 1), (1, 8), (8, 1), (8, 8))
        self.dropoff_locations = ((0, 5), (5, 0), (5, 9), (9, 5))
        self.depot_position = (0, 0)
        self.planter_locations = (
            (1, 2),
            (4, 2),
            (5, 3),
            (8, 3),
            (1, 6),
            (4, 7),
            (5, 6),
            (8, 7),
        )
        self.streetlight_locations = (
            (1, 0),
            (4, 1),
            (5, 1),
            (8, 0),
            (0, 4),
            (9, 4),
            (1, 9),
            (8, 9),
        )
        # Traffic-light intersection cells
        self.traffic_light_cells = {(4, 4), (5, 4), (4, 5), (5, 5)}

        self.valid_start_locations = tuple(
            (x, y)
            for y in range(self.grid_size)
            for x in range(self.grid_size)
            if (x, y) not in self.buildings
        )

        # 7 actions: 4 moves + pickup + dropoff + wait
        self.action_space = spaces.Discrete(7)

        # State encodes: position × carrying × pickup_id × dropoff_id × traffic_phase
        self.observation_space = spaces.Discrete(
            self.grid_size
            * self.grid_size
            * 2
            * len(self.pickup_locations)
            * len(self.dropoff_locations)
            * self.TRAFFIC_LIGHT_PHASES
        )

        self.robot_position = np.array([0, 0], dtype=np.int8)
        self.pickup_index = 0
        self.dropoff_index = 0
        self.carrying_package = False
        self.traffic_phase = 0  # 0=green, 1=yellow, 2=red
        self.delivered = False
        self.steps_taken = 0
        self.last_event = "reset"

        self.window_size = 600
        self.cell_size = self.window_size // self.grid_size
        self.window = None
        self.clock = None
        self._pygame = None

    @property
    def pickup_position(self) -> tuple[int, int]:
        return self.pickup_locations[self.pickup_index]

    @property
    def dropoff_position(self) -> tuple[int, int]:
        return self.dropoff_locations[self.dropoff_index]

    def _encode_state(self) -> int:
        x, y = (int(v) for v in self.robot_position)
        state = x
        state = state * self.grid_size + y
        state = state * 2 + int(self.carrying_package)
        state = state * len(self.pickup_locations) + self.pickup_index
        state = state * len(self.dropoff_locations) + self.dropoff_index
        state = state * self.TRAFFIC_LIGHT_PHASES + self.traffic_phase
        return int(state)

    def decode_state(self, state: int) -> dict[str, Any]:
        """Decode a discrete state index for debugging and presentations."""
        traffic_phase = state % self.TRAFFIC_LIGHT_PHASES
        state //= self.TRAFFIC_LIGHT_PHASES
        dropoff_index = state % len(self.dropoff_locations)
        state //= len(self.dropoff_locations)
        pickup_index = state % len(self.pickup_locations)
        state //= len(self.pickup_locations)
        carrying = bool(state % 2)
        state //= 2
        y = state % self.grid_size
        x = state // self.grid_size
        return {
            "x": int(x),
            "y": int(y),
            "carrying_package": carrying,
            "pickup_index": int(pickup_index),
            "dropoff_index": int(dropoff_index),
            "traffic_phase": int(traffic_phase),
        }

    def _get_info(self) -> dict[str, Any]:
        phase_names = {0: "green", 1: "yellow", 2: "red"}
        return {
            "robot_position": tuple(int(v) for v in self.robot_position),
            "pickup_position": self.pickup_position,
            "dropoff_position": self.dropoff_position,
            "carrying_package": self.carrying_package,
            "is_success": self.delivered,
            "steps_taken": self.steps_taken,
            "last_event": self.last_event,
            "traffic_phase": phase_names[self.traffic_phase],
        }

    def _is_inside_grid(self, position: np.ndarray) -> bool:
        x, y = (int(v) for v in position)
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def _is_building(self, position: np.ndarray | tuple[int, int]) -> bool:
        x, y = (int(v) for v in position)
        return (x, y) in self.buildings

    def _is_valid_position(self, position: np.ndarray) -> bool:
        return self._is_inside_grid(position) and not self._is_building(position)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        super().reset(seed=seed)

        options = options or {}
        self.pickup_index = int(
            options.get(
                "pickup_index",
                self.np_random.integers(len(self.pickup_locations)),
            )
        )
        self.dropoff_index = int(
            options.get(
                "dropoff_index",
                self.np_random.integers(len(self.dropoff_locations)),
            )
        )

        if "start_position" in options:
            start = tuple(options["start_position"])
            if start not in self.valid_start_locations:
                raise ValueError(f"Invalid start position: {start}")
        elif options.get("random_start", False):
            blocked_service_cells = {self.pickup_position, self.dropoff_position}
            valid_starts = [
                position
                for position in self.valid_start_locations
                if position not in blocked_service_cells
            ]
            start = valid_starts[int(self.np_random.integers(len(valid_starts)))]
        else:
            start = self.depot_position

        self.robot_position = np.array(start, dtype=np.int8)

        self.carrying_package = False
        self.delivered = False
        self.steps_taken = 0
        self.traffic_phase = 0
        self.last_event = "reset"

        if self.render_mode == "human":
            self.render()

        return self._encode_state(), self._get_info()

    def _advance_traffic_light(self) -> None:
        """Cycle the traffic light phase every TRAFFIC_LIGHT_CYCLE steps."""
        if self.steps_taken % self.TRAFFIC_LIGHT_CYCLE == 0:
            self.traffic_phase = (self.traffic_phase + 1) % self.TRAFFIC_LIGHT_PHASES

    def _is_on_traffic_light(self) -> bool:
        return tuple(int(v) for v in self.robot_position) in self.traffic_light_cells

    def step(self, action: int) -> tuple[int, int, bool, bool, dict[str, Any]]:
        action = int(action)
        reward = self.STEP_REWARD
        terminated = False
        self.steps_taken += 1
        self.last_event = self.ACTION_NAMES.get(action, "unknown")

        # Advance the traffic light cycle
        self._advance_traffic_light()

        if action == 6:
            # Wait action — stay in place, only costs the base step reward
            self.last_event = "wait"
        elif action in self.ACTIONS:
            next_position = self.robot_position + self.ACTIONS[action]
            if self._is_valid_position(next_position):
                self.robot_position = next_position.astype(np.int8)
                # Apply traffic-light penalties when entering an intersection
                if self._is_on_traffic_light():
                    if self.traffic_phase == 2:  # red
                        reward += self.RED_LIGHT_PENALTY
                        self.last_event = "ran_red_light"
                    elif self.traffic_phase == 1:  # yellow
                        reward += self.YELLOW_LIGHT_PENALTY
                        self.last_event = "ran_yellow_light"
            else:
                reward = self.ILLEGAL_REWARD
                self.last_event = "blocked"
        elif action == 4:
            if not self.carrying_package and tuple(self.robot_position) == self.pickup_position:
                self.carrying_package = True
                reward = self.PICKUP_REWARD
                self.last_event = "picked_up"
            else:
                reward = self.ILLEGAL_REWARD
                self.last_event = "illegal_pickup"
        elif action == 5:
            if self.carrying_package and tuple(self.robot_position) == self.dropoff_position:
                self.carrying_package = False
                self.delivered = True
                terminated = True
                reward = self.DROPOFF_REWARD
                self.last_event = "delivered"
            else:
                reward = self.ILLEGAL_REWARD
                self.last_event = "illegal_dropoff"
        else:
            reward = self.ILLEGAL_REWARD
            self.last_event = "invalid_action"

        truncated = self.steps_taken >= self.max_steps and not terminated

        if self.render_mode == "human":
            self.render()

        return self._encode_state(), int(reward), terminated, truncated, self._get_info()

    def render(self, mode: str | None = None) -> np.ndarray | None:
        render_mode = mode or self.render_mode
        if render_mode is None:
            return None

        self._ensure_pygame(render_mode)
        frame = self._render_frame()

        if render_mode == "human":
            self.window.blit(frame, frame.get_rect())
            self._pygame.event.pump()
            self._pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None

        if render_mode == "rgb_array":
            return np.transpose(
                self._pygame.surfarray.array3d(frame),
                axes=(1, 0, 2),
            )

        raise ValueError(f"Unsupported render mode: {render_mode}")

    def close(self) -> None:
        if self._pygame is not None:
            if self.window is not None:
                self._pygame.display.quit()
            self._pygame.quit()
        self.window = None
        self.clock = None
        self._pygame = None

    def _ensure_pygame(self, render_mode: str) -> None:
        if self._pygame is None:
            import pygame

            self._pygame = pygame
            pygame.init()

        if self.window is None and render_mode == "human":
            self.window = self._pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
            self._pygame.display.set_caption("DeliveryBot-v0")

        if self.clock is None:
            self.clock = self._pygame.time.Clock()

    def _render_frame(self):
        pygame = self._pygame
        surface = pygame.Surface((self.window_size, self.window_size))

        colors = {
            "road": (42, 45, 51),
            "road_dark": (31, 34, 40),
            "road_speck": (57, 61, 70),
            "lane": (238, 202, 91),
            "sidewalk": (190, 196, 204),
            "sidewalk_alt": (174, 182, 193),
            "paver": (148, 157, 169),
            "curb": (225, 230, 235),
            "building": (70, 89, 126),
            "building_alt": (82, 105, 146),
            "building_shadow": (35, 43, 61),
            "window_lit": (255, 226, 132),
            "window_dim": (139, 174, 205),
            "roof": (49, 61, 86),
            "grid": (32, 35, 41),
            "robot": (255, 126, 41),
            "robot_loaded": (255, 156, 52),
            "package": (255, 214, 64),
            "dropoff": (64, 226, 122),
            "dropoff_dark": (25, 130, 70),
            "crosswalk": (231, 235, 240),
            "depot": (68, 160, 255),
            "depot_dark": (35, 75, 122),
            "planter": (84, 104, 76),
            "leaf": (58, 166, 88),
            "leaf_dark": (30, 108, 66),
            "lamp": (255, 230, 145),
            "lamp_post": (78, 84, 94),
            "signal_red": (231, 72, 86),
            "signal_yellow": (245, 188, 66),
            "signal_green": (70, 205, 123),
            "outline": (22, 24, 29),
        }

        road_rows = {0, 4, 5, 9}
        road_cols = {0, 4, 5, 9}
        service_cells = {self.pickup_position, self.dropoff_position, self.depot_position}

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                rect = self._cell_rect(x, y)
                if self._is_road_cell(x, y, road_rows, road_cols):
                    self._draw_road_cell(surface, rect, x, y, colors)
                else:
                    self._draw_sidewalk_cell(surface, rect, x, y, colors)

                if x in road_cols and y in road_rows:
                    self._draw_crosswalk(surface, rect, colors["crosswalk"])

        self._draw_lane_markings(surface, road_rows, road_cols, colors)
        self._draw_curbs(surface, road_rows, road_cols, colors)

        for index, block in enumerate(self.building_blocks):
            self._draw_building_block(surface, block, colors, index)

        self._draw_depot(surface, colors)

        for position in self.planter_locations:
            if position not in service_cells:
                self._draw_planter(surface, position, colors)

        self._draw_traffic_signals(surface, colors)

        for position in self.streetlight_locations:
            if position not in service_cells:
                self._draw_streetlight(surface, position, colors)

        self._draw_dropoff(surface, colors)

        if not self.carrying_package and not self.delivered:
            self._draw_package(surface, self.pickup_position, colors)

        self._draw_robot(surface, colors)

        for i in range(self.grid_size + 1):
            offset = i * self.cell_size
            pygame.draw.line(
                surface,
                colors["grid"],
                (0, offset),
                (self.window_size, offset),
                1,
            )
            pygame.draw.line(
                surface,
                colors["grid"],
                (offset, 0),
                (offset, self.window_size),
                1,
            )

        return surface

    def _cell_rect(self, x: int, y: int):
        return self._pygame.Rect(
            x * self.cell_size,
            y * self.cell_size,
            self.cell_size,
            self.cell_size,
        )

    def _is_road_cell(
        self,
        x: int,
        y: int,
        road_rows: set[int],
        road_cols: set[int],
    ) -> bool:
        return x in road_cols or y in road_rows

    def _draw_road_cell(
        self,
        surface,
        rect,
        x: int,
        y: int,
        colors: dict[str, tuple[int, int, int]],
    ) -> None:
        pygame = self._pygame
        pygame.draw.rect(surface, colors["road"], rect)
        pygame.draw.rect(surface, colors["road_dark"], rect, width=1)

        for index in range(4):
            dot_x = rect.left + ((x * 19 + y * 7 + index * 13) % self.cell_size)
            dot_y = rect.top + ((x * 11 + y * 23 + index * 17) % self.cell_size)
            pygame.draw.circle(surface, colors["road_speck"], (dot_x, dot_y), 1)

    def _draw_sidewalk_cell(
        self,
        surface,
        rect,
        x: int,
        y: int,
        colors: dict[str, tuple[int, int, int]],
    ) -> None:
        pygame = self._pygame
        base = colors["sidewalk"] if (x + y) % 2 == 0 else colors["sidewalk_alt"]
        pygame.draw.rect(surface, base, rect)

        tile = max(12, self.cell_size // 3)
        for offset in range(tile, self.cell_size, tile):
            pygame.draw.line(
                surface,
                colors["paver"],
                (rect.left + offset, rect.top),
                (rect.left + offset, rect.bottom),
                1,
            )
            pygame.draw.line(
                surface,
                colors["paver"],
                (rect.left, rect.top + offset),
                (rect.right, rect.top + offset),
                1,
            )

    def _draw_lane_markings(
        self,
        surface,
        road_rows: set[int],
        road_cols: set[int],
        colors: dict[str, tuple[int, int, int]],
    ) -> None:
        pygame = self._pygame
        dash = self.cell_size // 3
        gap = self.cell_size // 4

        for y in road_rows:
            for x in range(self.grid_size):
                if x in road_cols:
                    continue
                rect = self._cell_rect(x, y)
                center_y = rect.centery
                start = rect.left + gap
                pygame.draw.line(
                    surface,
                    colors["lane"],
                    (start, center_y),
                    (min(start + dash, rect.right - gap), center_y),
                    3,
                )

        for x in road_cols:
            for y in range(self.grid_size):
                if y in road_rows:
                    continue
                rect = self._cell_rect(x, y)
                center_x = rect.centerx
                start = rect.top + gap
                pygame.draw.line(
                    surface,
                    colors["lane"],
                    (center_x, start),
                    (center_x, min(start + dash, rect.bottom - gap)),
                    3,
                )

    def _draw_curbs(
        self,
        surface,
        road_rows: set[int],
        road_cols: set[int],
        colors: dict[str, tuple[int, int, int]],
    ) -> None:
        pygame = self._pygame
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self._is_road_cell(x, y, road_rows, road_cols):
                    continue

                rect = self._cell_rect(x, y)
                neighbors = {
                    "left": (x - 1, y),
                    "right": (x + 1, y),
                    "top": (x, y - 1),
                    "bottom": (x, y + 1),
                }
                if neighbors["left"][0] >= 0 and self._is_road_cell(*neighbors["left"], road_rows, road_cols):
                    pygame.draw.line(surface, colors["curb"], rect.topleft, rect.bottomleft, 3)
                if neighbors["right"][0] < self.grid_size and self._is_road_cell(*neighbors["right"], road_rows, road_cols):
                    pygame.draw.line(surface, colors["curb"], rect.topright, rect.bottomright, 3)
                if neighbors["top"][1] >= 0 and self._is_road_cell(*neighbors["top"], road_rows, road_cols):
                    pygame.draw.line(surface, colors["curb"], rect.topleft, rect.topright, 3)
                if neighbors["bottom"][1] < self.grid_size and self._is_road_cell(*neighbors["bottom"], road_rows, road_cols):
                    pygame.draw.line(surface, colors["curb"], rect.bottomleft, rect.bottomright, 3)

    def _draw_building_block(
        self,
        surface,
        block: tuple[int, int, int, int],
        colors: dict[str, tuple[int, int, int]],
        index: int,
    ) -> None:
        pygame = self._pygame
        x, y, width, height = block
        rect = pygame.Rect(
            x * self.cell_size,
            y * self.cell_size,
            width * self.cell_size,
            height * self.cell_size,
        )
        shadow = rect.move(7, 9).inflate(-10, -10)
        body = rect.inflate(-14, -14)
        facade = colors["building"] if index % 2 == 0 else colors["building_alt"]

        pygame.draw.rect(surface, colors["building_shadow"], shadow, border_radius=8)
        pygame.draw.rect(surface, facade, body, border_radius=7)
        pygame.draw.rect(surface, colors["outline"], body, width=2, border_radius=7)

        roof = body.inflate(-20, -20)
        roof.height = max(16, roof.height // 4)
        roof.top = body.top + 10
        pygame.draw.rect(surface, colors["roof"], roof, border_radius=5)

        window_w = max(8, self.cell_size // 5)
        window_h = max(7, self.cell_size // 6)
        start_x = body.left + 16
        start_y = body.top + self.cell_size // 2
        step_x = window_w + 10
        step_y = window_h + 9

        row = 0
        y_pos = start_y
        while y_pos + window_h < body.bottom - 10:
            col = 0
            x_pos = start_x
            while x_pos + window_w < body.right - 10:
                lit = (row + col + index) % 3 == 0
                color = colors["window_lit"] if lit else colors["window_dim"]
                pygame.draw.rect(
                    surface,
                    color,
                    pygame.Rect(x_pos, y_pos, window_w, window_h),
                    border_radius=2,
                )
                x_pos += step_x
                col += 1
            y_pos += step_y
            row += 1

    def _draw_depot(
        self,
        surface,
        colors: dict[str, tuple[int, int, int]],
    ) -> None:
        pygame = self._pygame
        rect = self._cell_rect(*self.depot_position)
        pad = pygame.Rect(0, 0, self.cell_size - 14, self.cell_size - 14)
        pad.center = rect.center

        pygame.draw.rect(surface, colors["depot_dark"], pad, border_radius=6)
        pygame.draw.rect(surface, colors["depot"], pad.inflate(-12, -12), border_radius=4)
        pygame.draw.line(surface, colors["curb"], pad.midleft, pad.midright, 3)
        pygame.draw.line(surface, colors["curb"], pad.midtop, pad.midbottom, 3)

    def _draw_planter(
        self,
        surface,
        position: tuple[int, int],
        colors: dict[str, tuple[int, int, int]],
    ) -> None:
        pygame = self._pygame
        rect = self._cell_rect(*position)
        base = pygame.Rect(0, 0, self.cell_size // 2, self.cell_size // 4)
        base.center = (rect.centerx, rect.centery + self.cell_size // 7)
        pygame.draw.rect(surface, colors["planter"], base, border_radius=4)

        leaf_centers = (
            (base.centerx - 10, base.top),
            (base.centerx, base.top - 8),
            (base.centerx + 10, base.top),
        )
        for index, center in enumerate(leaf_centers):
            leaf_color = colors["leaf"] if index != 1 else colors["leaf_dark"]
            pygame.draw.circle(surface, leaf_color, center, self.cell_size // 8)

    def _draw_traffic_signals(
        self,
        surface,
        colors: dict[str, tuple[int, int, int]],
    ) -> None:
        """Draw traffic signals with the currently active phase highlighted."""
        pygame = self._pygame
        # Dimmed versions of each light when not active
        dim_red = (80, 30, 30)
        dim_yellow = (80, 65, 25)
        dim_green = (25, 70, 45)

        # Active colors based on current phase
        red_color = colors["signal_red"] if self.traffic_phase == 2 else dim_red
        yellow_color = colors["signal_yellow"] if self.traffic_phase == 1 else dim_yellow
        green_color = colors["signal_green"] if self.traffic_phase == 0 else dim_green

        for x, y in ((4, 4), (5, 4), (4, 5), (5, 5)):
            rect = self._cell_rect(x, y)
            box = pygame.Rect(0, 0, 12, 28)
            box.center = (rect.centerx, rect.centery)
            pygame.draw.rect(surface, colors["outline"], box, border_radius=3)
            pygame.draw.circle(surface, red_color, (box.centerx, box.top + 6), 4)
            pygame.draw.circle(surface, yellow_color, box.center, 4)
            pygame.draw.circle(surface, green_color, (box.centerx, box.bottom - 6), 4)

            # Glow effect for the active light
            glow = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
            if self.traffic_phase == 0:
                pygame.draw.circle(glow, (*colors["signal_green"], 40), (box.centerx, box.bottom - 6), 10)
            elif self.traffic_phase == 1:
                pygame.draw.circle(glow, (*colors["signal_yellow"], 40), box.center, 10)
            else:
                pygame.draw.circle(glow, (*colors["signal_red"], 40), (box.centerx, box.top + 6), 10)
            surface.blit(glow, (0, 0))

    def _draw_streetlight(
        self,
        surface,
        position: tuple[int, int],
        colors: dict[str, tuple[int, int, int]],
    ) -> None:
        pygame = self._pygame
        rect = self._cell_rect(*position)
        base_x = rect.left + self.cell_size // 4
        base_y = rect.top + self.cell_size // 5

        glow = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
        pygame.draw.circle(glow, (*colors["lamp"], 46), (base_x, base_y), self.cell_size // 3)
        surface.blit(glow, (0, 0))

        pygame.draw.line(
            surface,
            colors["lamp_post"],
            (base_x, base_y + 5),
            (base_x, base_y + self.cell_size // 2),
            3,
        )
        pygame.draw.circle(surface, colors["lamp"], (base_x, base_y), 5)

    def _draw_crosswalk(self, surface, rect, color: tuple[int, int, int]) -> None:
        stripe_width = max(3, self.cell_size // 14)
        gap = max(5, self.cell_size // 8)
        start_x = rect.left + gap
        while start_x < rect.right - gap:
            stripe = self._pygame.Rect(
                start_x,
                rect.top + self.cell_size // 3,
                stripe_width,
                self.cell_size // 3,
            )
            self._pygame.draw.rect(surface, color, stripe)
            start_x += stripe_width + gap

    def _draw_dropoff(self, surface, colors: dict[str, tuple[int, int, int]]) -> None:
        x, y = self.dropoff_position
        rect = self._cell_rect(x, y)
        pulse = (math.sin(self.steps_taken * 0.65) + 1.0) / 2.0
        size = int(self.cell_size * (0.48 + 0.16 * pulse))
        target = self._pygame.Rect(0, 0, size, size)
        target.center = rect.center

        pygame = self._pygame
        glow = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
        pygame.draw.circle(
            glow,
            (*colors["dropoff"], int(42 + 34 * pulse)),
            rect.center,
            int(self.cell_size * (0.42 + 0.12 * pulse)),
        )
        surface.blit(glow, (0, 0))
        pygame.draw.rect(surface, colors["dropoff_dark"], target.inflate(10, 10), width=3)
        pygame.draw.rect(surface, colors["dropoff"], target, width=4)
        pygame.draw.line(surface, colors["dropoff"], target.midtop, target.midbottom, 2)
        pygame.draw.line(surface, colors["dropoff"], target.midleft, target.midright, 2)

    def _draw_package(
        self,
        surface,
        position: tuple[int, int],
        colors: dict[str, tuple[int, int, int]],
    ) -> None:
        rect = self._cell_rect(*position)
        package = self._pygame.Rect(0, 0, self.cell_size // 2, self.cell_size // 2)
        package.center = rect.center
        self._pygame.draw.rect(surface, colors["outline"], package.inflate(4, 4))
        self._pygame.draw.rect(surface, colors["package"], package)
        self._pygame.draw.rect(surface, (255, 232, 116), package.inflate(-8, -8))
        self._pygame.draw.line(
            surface,
            (178, 132, 32),
            package.midleft,
            package.midright,
            2,
        )

    def _draw_robot(self, surface, colors: dict[str, tuple[int, int, int]]) -> None:
        rect = self._cell_rect(*tuple(int(v) for v in self.robot_position))
        body = self._pygame.Rect(0, 0, self.cell_size - 20, self.cell_size - 20)
        body.center = rect.center
        color = colors["robot_loaded"] if self.carrying_package else colors["robot"]

        pygame = self._pygame
        pygame.draw.rect(surface, colors["outline"], body.inflate(6, 6), border_radius=7)
        pygame.draw.rect(surface, color, body, border_radius=6)
        pygame.draw.rect(surface, (255, 184, 88), body.inflate(-14, -16), border_radius=4)

        eye = pygame.Rect(0, 0, 6, 6)
        eye.center = (body.centerx + 8, body.centery - 8)
        pygame.draw.rect(surface, (255, 244, 210), eye, border_radius=2)
        pygame.draw.line(
            surface,
            (255, 244, 210),
            (body.right - 3, body.centery),
            (body.right + 5, body.centery),
            2,
        )

        wheel_y = body.bottom - 3
        pygame.draw.circle(surface, colors["outline"], (body.left + 8, wheel_y), 5)
        pygame.draw.circle(surface, colors["outline"], (body.right - 8, wheel_y), 5)
        pygame.draw.circle(surface, (85, 91, 102), (body.left + 8, wheel_y), 2)
        pygame.draw.circle(surface, (85, 91, 102), (body.right - 8, wheel_y), 2)

        if self.carrying_package:
            package = pygame.Rect(0, 0, self.cell_size // 3, self.cell_size // 4)
            package.center = (body.centerx, body.top - 3)
            pygame.draw.rect(surface, colors["outline"], package.inflate(4, 4))
            pygame.draw.rect(surface, colors["package"], package)


if "DeliveryBot-v0" not in registry:
    register(
        id="DeliveryBot-v0",
        entry_point="delivery_env:DeliveryEnv",
        max_episode_steps=150,
    )
