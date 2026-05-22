"""Custom Gymnasium environment for an urban delivery robot."""

from __future__ import annotations

import math
from itertools import permutations
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
    }

    STEP_REWARD = config.STEP_REWARD
    ILLEGAL_REWARD = config.ILLEGAL_REWARD
    PICKUP_REWARD = config.PICKUP_REWARD
    DROPOFF_REWARD = config.DROPOFF_REWARD

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
        # Planters (grass) — each pair is at least 3 Manhattan distance
        # apart (i.e. 2 empty squares between them).  These are impassable.
        self.planter_locations = (
            (1, 3),
            (3, 1),
            (1, 7),
            (3, 8),
            (6, 1),
            (8, 3),
            (6, 8),
            (8, 6),
        )
        self.grass_cells = set(self.planter_locations)

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
        self.valid_start_locations = tuple(
            (x, y)
            for y in range(self.grid_size)
            for x in range(self.grid_size)
            if (x, y) not in self.buildings
            and (x, y) not in self.grass_cells
        )

        self.action_space = spaces.Discrete(6)

        # All possible pickup-to-dropoff permutations (4! = 24)
        self._all_perms = list(permutations(range(len(self.dropoff_locations))))
        self._num_perms = len(self._all_perms)  # 24
        self.perm_index = 0

        # 4 packages × 3 statuses × 24 permutations
        # State = position (100) × pkg_statuses (81) × perm (24) = 194,400
        self._status_combos = 3 ** len(self.pickup_locations)
        self.observation_space = spaces.Discrete(
            self.grid_size * self.grid_size * self._status_combos * self._num_perms
        )

        self.robot_position = np.array([0, 0], dtype=np.int8)
        self.num_packages = len(self.pickup_locations)  # always 4
        self.max_carry = config.MAX_CARRY
        # 0 = waiting at pickup, 1 = carried, 2 = delivered
        self.package_status = [0] * self.num_packages
        self.delivered = False
        self.steps_taken = 0
        self.last_event = "reset"

        # Per-package colors for rendering
        self.package_colors = [
            (255, 214, 64),   # Gold
            (64, 224, 208),   # Teal
            (255, 105, 180),  # Pink
            (144, 238, 144),  # Lime
        ]

        self.window_size = 600
        self.cell_size = self.window_size // self.grid_size
        self.window = None
        self.clock = None
        self._pygame = None

    @property
    def num_carrying(self) -> int:
        return sum(1 for s in self.package_status if s == 1)

    @property
    def num_delivered(self) -> int:
        return sum(1 for s in self.package_status if s == 2)

    def _pkg_dropoff(self, pkg_id: int) -> tuple[int, int]:
        """Return the dropoff location for *pkg_id* under the current permutation."""
        return self.dropoff_locations[self._all_perms[self.perm_index][pkg_id]]

    def _encode_pkg_status(self) -> int:
        code = 0
        for s in self.package_status:
            code = code * 3 + s
        return code

    def _decode_pkg_status(self, code: int) -> list[int]:
        statuses = []
        for _ in range(self.num_packages):
            statuses.append(code % 3)
            code //= 3
        return list(reversed(statuses))

    def _encode_state(self) -> int:
        x, y = (int(v) for v in self.robot_position)
        pos = x * self.grid_size + y
        return (pos * self._status_combos + self._encode_pkg_status()) * self._num_perms + self.perm_index

    def decode_state(self, state: int) -> dict[str, Any]:
        """Decode a discrete state index for debugging and presentations."""
        perm_idx = state % self._num_perms
        state //= self._num_perms
        pkg_code = state % self._status_combos
        pos = state // self._status_combos
        y = pos % self.grid_size
        x = pos // self.grid_size
        return {
            "x": int(x),
            "y": int(y),
            "package_status": self._decode_pkg_status(pkg_code),
            "perm_index": int(perm_idx),
        }

    def _get_info(self) -> dict[str, Any]:
        status_names = {0: "waiting", 1: "carried", 2: "delivered"}
        return {
            "robot_position": tuple(int(v) for v in self.robot_position),
            "is_success": self.delivered,
            "steps_taken": self.steps_taken,
            "last_event": self.last_event,
            "num_carrying": self.num_carrying,
            "num_delivered": self.num_delivered,
            "package_status": [
                {
                    "id": i,
                    "pickup": self.pickup_locations[i],
                    "dropoff": self._pkg_dropoff(i),
                    "status": status_names[self.package_status[i]],
                }
                for i in range(self.num_packages)
            ],
        }

    def _is_inside_grid(self, position: np.ndarray) -> bool:
        x, y = (int(v) for v in position)
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def _is_building(self, position: np.ndarray | tuple[int, int]) -> bool:
        x, y = (int(v) for v in position)
        return (x, y) in self.buildings

    def _is_grass(self, position: np.ndarray | tuple[int, int]) -> bool:
        x, y = (int(v) for v in position)
        return (x, y) in self.grass_cells

    def _is_valid_position(self, position: np.ndarray) -> bool:
        return (
            self._is_inside_grid(position)
            and not self._is_building(position)
            and not self._is_grass(position)
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        super().reset(seed=seed)

        options = options or {}

        # All 4 packages start as waiting; shuffle the dropoff permutation.
        self.package_status = [0] * self.num_packages
        self.perm_index = int(self.np_random.integers(self._num_perms))

        if "start_position" in options:
            start = tuple(options["start_position"])
            if start not in self.valid_start_locations:
                raise ValueError(f"Invalid start position: {start}")
        elif options.get("random_start", False):
            valid_starts = [
                p for p in self.valid_start_locations
                if p not in set(self.pickup_locations)
            ]
            start = valid_starts[int(self.np_random.integers(len(valid_starts)))]
        else:
            start = self.depot_position

        self.robot_position = np.array(start, dtype=np.int8)

        self.delivered = False
        self.steps_taken = 0
        self.last_event = "reset"

        if self.render_mode == "human":
            self.render()

        return self._encode_state(), self._get_info()

    def step(self, action: int) -> tuple[int, int, bool, bool, dict[str, Any]]:
        action = int(action)
        reward = self.STEP_REWARD
        terminated = False
        self.steps_taken += 1
        self.last_event = self.ACTION_NAMES.get(action, "unknown")

        pos = tuple(int(v) for v in self.robot_position)

        if action in self.ACTIONS:
            next_position = self.robot_position + self.ACTIONS[action]
            if self._is_valid_position(next_position):
                self.robot_position = next_position.astype(np.int8)
            else:
                reward = self.ILLEGAL_REWARD
                self.last_event = "blocked"
        elif action == 4:
            # Pickup: find a waiting package at the robot's current cell.
            picked = False
            if self.num_carrying < self.max_carry:
                for i in range(self.num_packages):
                    if self.package_status[i] == 0 and self.pickup_locations[i] == pos:
                        self.package_status[i] = 1
                        reward = self.PICKUP_REWARD
                        self.last_event = f"picked_up_{i}"
                        picked = True
                        break
            if not picked:
                reward = self.ILLEGAL_REWARD
                self.last_event = "illegal_pickup"
        elif action == 5:
            # Dropoff: deliver a carried package whose dropoff matches.
            dropped = False
            for i in range(self.num_packages):
                if self.package_status[i] == 1 and self._pkg_dropoff(i) == pos:
                    self.package_status[i] = 2
                    reward = self.DROPOFF_REWARD
                    self.last_event = f"delivered_{i}"
                    dropped = True
                    break
            if not dropped:
                reward = self.ILLEGAL_REWARD
                self.last_event = "illegal_dropoff"

            # Check if all packages are delivered.
            if self.num_delivered >= self.num_packages:
                self.delivered = True
                terminated = True
                self.last_event = "all_delivered"
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
        # Collect all active service cells for planter exclusion
        service_cells = set(self.pickup_locations) | set(self.dropoff_locations) | {self.depot_position}

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

        for position in self.streetlight_locations:
            if position not in service_cells:
                self._draw_streetlight(surface, position, colors)

        # Draw dropoff markers for non-delivered packages
        for i in range(self.num_packages):
            if self.package_status[i] != 2:  # not delivered
                bright = self.package_status[i] == 1  # carried = bright
                self._draw_dropoff_marker(
                    surface, self._pkg_dropoff(i),
                    self.package_colors[i], bright, colors,
                )

        # Draw waiting packages at their pickup locations
        for i in range(self.num_packages):
            if self.package_status[i] == 0:
                self._draw_package(surface, self.pickup_locations[i], self.package_colors[i])

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

    def _draw_dropoff_marker(
        self,
        surface,
        position: tuple[int, int],
        color: tuple[int, int, int],
        bright: bool,
        colors: dict[str, tuple[int, int, int]],
    ) -> None:
        """Draw a crosshair dropoff marker at *position* in the given color."""
        x, y = position
        rect = self._cell_rect(x, y)
        pulse = (math.sin(self.steps_taken * 0.65) + 1.0) / 2.0
        size = int(self.cell_size * (0.48 + 0.16 * pulse)) if bright else int(self.cell_size * 0.40)
        target = self._pygame.Rect(0, 0, size, size)
        target.center = rect.center

        pygame = self._pygame
        if bright:
            glow = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
            pygame.draw.circle(
                glow, (*color, int(42 + 34 * pulse)),
                rect.center, int(self.cell_size * (0.42 + 0.12 * pulse)),
            )
            surface.blit(glow, (0, 0))

        dark = tuple(max(0, c - 50) for c in color)
        pygame.draw.rect(surface, dark, target.inflate(10, 10), width=3)
        pygame.draw.rect(surface, color, target, width=4)
        pygame.draw.line(surface, color, target.midtop, target.midbottom, 2)
        pygame.draw.line(surface, color, target.midleft, target.midright, 2)

    def _draw_package(
        self,
        surface,
        position: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        rect = self._cell_rect(*position)
        package = self._pygame.Rect(0, 0, self.cell_size // 2, self.cell_size // 2)
        package.center = rect.center
        lighter = tuple(min(255, c + 40) for c in color)
        self._pygame.draw.rect(surface, (22, 24, 29), package.inflate(4, 4))
        self._pygame.draw.rect(surface, color, package)
        self._pygame.draw.rect(surface, lighter, package.inflate(-8, -8))
        self._pygame.draw.line(
            surface,
            tuple(max(0, c - 80) for c in color),
            package.midleft,
            package.midright,
            2,
        )

    def _draw_robot(self, surface, colors: dict[str, tuple[int, int, int]]) -> None:
        rect = self._cell_rect(*tuple(int(v) for v in self.robot_position))
        body = self._pygame.Rect(0, 0, self.cell_size - 20, self.cell_size - 20)
        body.center = rect.center
        loaded = self.num_carrying > 0
        color = colors["robot_loaded"] if loaded else colors["robot"]

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

        # Draw small colored squares on top for each carried package
        carried = [i for i in range(self.num_packages) if self.package_status[i] == 1]
        pkg_w = self.cell_size // 4
        pkg_h = self.cell_size // 5
        start_x = body.centerx - (len(carried) * (pkg_w + 2)) // 2
        for idx, pkg_id in enumerate(carried):
            pkg_rect = pygame.Rect(start_x + idx * (pkg_w + 2), body.top - 6, pkg_w, pkg_h)
            pygame.draw.rect(surface, colors["outline"], pkg_rect.inflate(3, 3))
            pygame.draw.rect(surface, self.package_colors[pkg_id], pkg_rect)


if "DeliveryBot-v0" not in registry:
    register(
        id="DeliveryBot-v0",
        entry_point="delivery_env:DeliveryEnv",
        max_episode_steps=150,
    )
