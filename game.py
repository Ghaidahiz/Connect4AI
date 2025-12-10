import pygame
import sys
import numpy as np
import pandas as pd
import torch
from model import Model, get_board_from_flat, get_next_open_row, check_win
import math
import random
import time

class Connect4GUI:
    def __init__(self):
        pygame.init()
        
        # Game Constants
        self.ROWS = 6
        self.COLS = 7
        self.FPS = 60
        
        # Get screen info for full screen
        screen_info = pygame.display.Info()
        self.SCREEN_WIDTH = screen_info.current_w
        self.SCREEN_HEIGHT = screen_info.current_h
        
        # Calculate cell size based on screen dimensions
        self.CELL_SIZE = min(
            int(self.SCREEN_WIDTH * 0.9) // self.COLS,
            int(self.SCREEN_HEIGHT * 0.7) // (self.ROWS + 1)
        )
        
        # Ensure minimum cell size
        self.CELL_SIZE = max(self.CELL_SIZE, 60)
        
        self.RADIUS = self.CELL_SIZE // 2 - 5
        self.WIDTH = self.COLS * self.CELL_SIZE
        self.HEIGHT = (self.ROWS + 1) * self.CELL_SIZE
        
        self.BACKGROUND_TOP = (24, 28, 60)  
        self.BACKGROUND_BOTTOM = (10, 12, 32) 
        self.BOARD_COLOR = (22, 36, 71)
        self.BOARD_HIGHLIGHT = (31, 64, 104)
        self.PLAYER_COLOR = (208, 32, 47)  
        self.AI_COLOR = (255, 209, 0)  
        self.ACCENT_COLOR = (78, 205, 196) 
        self.TEXT_COLOR = (230, 230, 230)  
        self.SHADOW_COLOR = (35, 35, 55) 
        # Initialize screen
        try:
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.FULLSCREEN)
            print("Running in FULLSCREEN mode")
        except:
            self.screen = pygame.display.set_mode((self.WIDTH + 100, self.HEIGHT + 100))
            print("Running in WINDOWED mode")
        
        pygame.display.set_caption("Connect 4 - AI Challenge")
        
        # Center the board on screen
        self.board_offset_x = (self.SCREEN_WIDTH - self.WIDTH) // 2
        self.board_offset_y = (self.SCREEN_HEIGHT - self.HEIGHT) // 2
        
        # Initialize fonts scaled to screen size
        font_base = max(24, self.CELL_SIZE // 3)
        self.title_font = pygame.font.SysFont('ByteBounce', int(font_base * 2.0), bold=False)
        self.score_font = pygame.font.SysFont('ByteBounce', int(font_base * 1.0), bold=False)
        self.button_font = pygame.font.SysFont('ByteBounce', int(font_base * 0.9), bold=False)
        self.turn_font = pygame.font.SysFont('ByteBounce', int(font_base * 0.8), bold=False)
        
        # Initialize model - DON'T override seeds here
        print("Loading AI Model...")
        try:
            self.model = Model()
            print("Model loaded successfully!")
            print("Note: Model uses deterministic seeds for reproducibility")
            print("To make it non-deterministic, modify seeds in model.py")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
        
        # Game state
        self.reset_game()
        
        # Animation variables
        self.falling_piece = None
        self.fall_speed = 1200  # Pixels per second
        self.pulse_animation = 0
        self.pulse_speed = 0.05
        self.last_time = time.time()
        
        # Particle system for effects
        self.particles = []
        
    def get_last_open_row(self, board, col):
        """Get the last available row from the BOTTOM (Connect 4 gravity)"""
        for r in range(self.ROWS - 1, -1, -1):  # Start from bottom row
            if board[r][col] == 0:
                return r
        return -1
    
    def create_gradient_surface(self, width, height, color1, color2, vertical=True):
        """Create a gradient surface"""
        gradient = pygame.Surface((width, height))
        if vertical:
            for y in range(height):
                ratio = y / height
                r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
                g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
                b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
                pygame.draw.line(gradient, (r, g, b), (0, y), (width, y))
        else:
            for x in range(width):
                ratio = x / width
                r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
                g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
                b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
                pygame.draw.line(gradient, (r, g, b), (x, 0), (x, height))
        return gradient
    
    def draw_rounded_rect(self, surface, color, rect, radius):
        """Draw a rounded rectangle"""
        x, y, width, height = rect
        
        # Draw rounded rect
        pygame.draw.rect(surface, color, rect, border_radius=radius)
        
        # Add subtle highlight
        highlight = pygame.Surface((width - 20, 8), pygame.SRCALPHA)
        highlight.fill((255, 255, 255, 30))
        surface.blit(highlight, (x + 10, y + 10))
    
    def draw_particle(self, x, y, color, radius):
        """Draw a glowing particle"""
        surface = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
        
        # Create glow effect
        for i in range(3, 0, -1):
            alpha = 50 // i
            pygame.draw.circle(
                surface,
                (*color, alpha),
                (radius * 2, radius * 2),
                radius * i // 2
            )
        
        # Draw main particle
        pygame.draw.circle(surface, (*color, 200), (radius * 2, radius * 2), radius)
        self.screen.blit(surface, (x - radius * 2, y - radius * 2))
    
    def create_particles(self, x, y, color, count=20):
        """Create particle explosion effect"""
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 6)
            size = random.uniform(2, 6)
            life = random.uniform(20, 40)
            
            self.particles.append({
                'x': x,
                'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'color': color,
                'size': size,
                'life': life,
                'max_life': life
            })
    
    def update_particles(self):
        """Update particle system"""
        for particle in self.particles[:]:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['vy'] += 0.3  # Gravity
            particle['life'] -= 1
            
            if particle['life'] <= 0:
                self.particles.remove(particle)
            else:
                # Fade out
                alpha = int(255 * (particle['life'] / particle['max_life']))
                self.draw_particle(particle['x'], particle['y'], particle['color'], particle['size'])
    
    def create_gradient_highlight(self, width, height, color):
        """Create a highlight with gradient transparency from center to edges"""
        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        
        # Create horizontal gradient
        center_x = width // 2
        max_alpha = 30  # Maximum transparency at center
        
        for x in range(width):
            # Normalized distance from center (0 to 1)
            distance = abs(x - center_x) / center_x
            
            # Quadratic falloff for smooth edges
            alpha = int(max_alpha * (1 - distance) * (1 - distance))
            
            if alpha > 0:
                # Draw vertical line with gradient alpha
                pygame.draw.line(surface, (*color, alpha), (x, 0), (x, height))
        
        # Add vertical gradient (darker at bottom)
        for y in range(height):
            vertical_factor = 1.0 - (y / height) * 0.3  # 30% darker at bottom
            for x in range(width):
                current_color = surface.get_at((x, y))
                if current_color[3] > 0:  # If not transparent
                    new_alpha = int(current_color[3] * vertical_factor)
                    surface.set_at((x, y), (*color[:3], new_alpha))
        
        return surface

    def draw_column_highlight(self, col, pulse_intensity):
        """Draw a beautiful column highlight with soft edges and glow"""
        x = self.board_offset_x + col * self.CELL_SIZE
        y = self.board_offset_y + self.CELL_SIZE
        width = self.CELL_SIZE
        height = self.HEIGHT - self.CELL_SIZE
        
        # Base color with pulse effect
        base_color = self.ACCENT_COLOR
        pulse_factor = 0.2 + 0.8 * pulse_intensity
        highlight_color = tuple(int(c * pulse_factor) for c in base_color)
        
        # Draw gradient highlight
        highlight_surface = self.create_gradient_highlight(width, height, highlight_color)
        self.screen.blit(highlight_surface, (x, y))
        
        # Add glow effect
        self.draw_column_glow(x, y, width, height, highlight_color, pulse_intensity)

    def draw_column_glow(self, x, y, width, height, color, pulse_intensity):
        """Add a glowing outline to the column"""
        # Outer glow (very soft)
        glow_size = 10
        glow_surface = pygame.Surface((width + glow_size * 2, height + glow_size * 2), pygame.SRCALPHA)
        
        # Draw glow as a rounded rectangle
        glow_rect = pygame.Rect(glow_size, glow_size, width, height)
        
        # Multiple layers for glow effect
        for i in range(3, 0, -1):
            layer_size = glow_size * i // 3
            layer_alpha = 15 // i * pulse_intensity
            
            # Draw rounded rectangle for glow layer
            temp_surface = pygame.Surface((width + layer_size * 2, height + layer_size * 2), pygame.SRCALPHA)
            pygame.draw.rect(temp_surface, (*color, layer_alpha), 
                            pygame.Rect(layer_size, layer_size, width, height),
                            border_radius=15)
            
            # Simple blur by scaling
            temp_surface = pygame.transform.smoothscale(
                pygame.transform.smoothscale(temp_surface, 
                                        (width // 2 + layer_size, height // 2 + layer_size)),
                (width + layer_size * 2, height + layer_size * 2)
            )
            
            # Composite onto glow surface
            glow_surface.blit(temp_surface, (glow_size - layer_size, glow_size - layer_size))
        
        # Draw the glow
        self.screen.blit(glow_surface, (x - glow_size, y - glow_size))
        
        # Add subtle inner highlight (top edge)
        top_highlight = pygame.Surface((width - 20, 3), pygame.SRCALPHA)
        top_highlight.fill((255, 255, 255, int(30 * pulse_intensity)))
        self.screen.blit(top_highlight, (x + 10, y + 5))

    def draw_column_preview(self, col):
        """Draw a preview piece at the top of the highlighted column"""
        preview_x = self.board_offset_x + col * self.CELL_SIZE + self.CELL_SIZE // 2
        preview_y = self.board_offset_y + self.CELL_SIZE // 2
        
        # Create preview piece with pulsing effect
        pulse = abs(math.sin(self.pulse_animation * 3)) * 0.3 + 0.7
        preview_color = tuple(int(c * 0.6 * pulse) for c in self.PLAYER_COLOR)
        
        # Draw preview piece with regular (sharper) drawing
        # Use smaller size for preview 
        self.draw_piece(preview_x, preview_y, preview_color, size_mult=0.8)
        
    def draw_soft_piece(self, x, y, color, size_mult=1.0):
        """Draw a piece with soft edges for preview"""
        radius = int(self.RADIUS * size_mult)
        
        # Create piece surface
        piece_surface = pygame.Surface((radius * 2 + 10, radius * 2 + 10), pygame.SRCALPHA)
        
        # Draw soft glow
        for i in range(5, 0, -1):
            glow_radius = radius + i * 2
            glow_alpha = 20 // i
            pygame.draw.circle(piece_surface, (*color, glow_alpha),
                            (radius + 5, radius + 5), glow_radius)
        
        # Draw main piece with gradient
        for i in range(radius, 0, -1):
            ratio = i / radius
            # Soft gradient from center
            r = int(color[0] * (0.5 + 0.5 * ratio))
            g = int(color[1] * (0.5 + 0.5 * ratio))
            b = int(color[2] * (0.5 + 0.5 * ratio))
            alpha = int(150 * ratio)  # More transparent at edges
            
            pygame.draw.circle(piece_surface, (r, g, b, alpha),
                            (radius + 5, radius + 5), i)
        
        self.screen.blit(piece_surface, (x - radius - 5, y - radius - 5))
        
    def draw_board(self):
            """Draw the game board with beautiful gradients"""
            # Draw full screen background gradient
            background = self.create_gradient_surface(
                self.SCREEN_WIDTH, self.SCREEN_HEIGHT, 
                self.BACKGROUND_TOP, self.BACKGROUND_BOTTOM
            )
            self.screen.blit(background, (0, 0))
            
            # Draw board background with rounded corners (centered)
            board_rect = pygame.Rect(
                self.board_offset_x - 20, 
                self.board_offset_y - 20, 
                self.WIDTH + 40, 
                self.HEIGHT + 40
            )
            self.draw_rounded_rect(self.screen, self.BOARD_COLOR, board_rect, 30)
            
            # Draw grid cells with glow effect
            for col in range(self.COLS):
                for row in range(self.ROWS):
                    x = self.board_offset_x + col * self.CELL_SIZE + self.CELL_SIZE // 2
                    y = self.board_offset_y + (row + 1) * self.CELL_SIZE + self.CELL_SIZE // 2
                    
                    # Draw cell shadow
                    shadow_radius = self.RADIUS + 4
                    shadow_surface = pygame.Surface((shadow_radius * 2, shadow_radius * 2), pygame.SRCALPHA)
                    pygame.draw.circle(shadow_surface, (0, 0, 0, 80), 
                                    (shadow_radius, shadow_radius), shadow_radius)
                    self.screen.blit(shadow_surface, (x - shadow_radius, y - shadow_radius))
                    
                    # Draw cell
                    pygame.draw.circle(self.screen, self.BOARD_HIGHLIGHT, (x, y), self.RADIUS + 4)
                    pygame.draw.circle(self.screen, self.BOARD_COLOR, (x, y), self.RADIUS)
                    
                    # Draw piece if exists
                    if self.board[row][col] == 1:  # Player
                        self.draw_piece(x, y, self.PLAYER_COLOR)
                    elif self.board[row][col] == -1:  # AI
                        self.draw_piece(x, y, self.AI_COLOR)
            
            # Draw falling piece animation
            if self.falling_piece:
                current_time = time.time()
                dt = current_time - self.falling_piece['start_time']
                
                col = self.falling_piece['col']
                target_row = self.falling_piece['target_row']
                color = self.falling_piece['color']
                
                # Calculate current position
                start_x = self.board_offset_x + col * self.CELL_SIZE + self.CELL_SIZE // 2
                start_y = self.board_offset_y + self.CELL_SIZE // 2  # Top of board
                target_y = self.board_offset_y + (target_row + 1) * self.CELL_SIZE + self.CELL_SIZE // 2
                distance = target_y - start_y
                fall_time = distance / self.fall_speed
                
                if dt < fall_time:
                    # Still falling
                    progress = dt / fall_time
                    current_y = start_y + distance * progress
                    
                    # Draw falling piece
                    self.draw_piece(start_x, current_y, color, falling=True)
                else:
                    # Landed
                    x = self.board_offset_x + col * self.CELL_SIZE + self.CELL_SIZE // 2
                    y = self.board_offset_y + (target_row + 1) * self.CELL_SIZE + self.CELL_SIZE // 2
                    
                    # Update board
                    self.board[target_row][col] = self.falling_piece['player']
                    
                    # Check for win/draw
                    self.check_game_state(target_row, col)
                    
                    # Create landing particle effect
                    self.create_particles(x, y, color, count=30)
                    
                    # Clear falling piece
                    self.falling_piece = None
                    
                    # Switch turns if game not over
                    if not self.game_over:
                        self.current_player *= -1
            
            # Draw column highlights on hover  <-- FIX INDENTATION HERE
            mouse_x, mouse_y = pygame.mouse.get_pos()  # <-- Remove extra indentation
            if not self.game_over and self.current_player == 1 and not self.falling_piece:
                # Adjust mouse position relative to board
                board_mouse_x = mouse_x - self.board_offset_x
                if 0 <= board_mouse_x < self.WIDTH:
                    col = board_mouse_x // self.CELL_SIZE
                    if 0 <= col < self.COLS:
                        # Check if column has space
                        row = self.get_last_open_row(self.board, col)
                        if row != -1:
                            # Draw soft column highlight with pulse effect
                            pulse = abs(math.sin(self.pulse_animation)) * 0.2 + 0.8
                            self.draw_column_highlight(col, pulse)
                            
                            # Draw preview piece
                            self.draw_column_preview(col)
                    
            # Update pulse animation
            self.pulse_animation += self.pulse_speed
            
            # Draw particles
            self.update_particles()
            
            # Draw UI elements
            self.draw_ui()
            
    def draw_piece(self, x, y, color, size_mult=1.0, falling=False):
        """Draw a game piece with gradient and glow effect"""
        radius = int(self.RADIUS * size_mult)
        
        if not falling:
            # Draw glow effect DIRECTLY to screen (not confined to surface)
            for i in range(3, 0, -1):
                glow_radius = radius + i * 3
                # Create glow color with transparency
                glow_color = (*color, 60 // i)
                
                # Draw multiple circles to create smoother glow
                for j in range(3):
                    current_glow_radius = glow_radius - j
                    current_alpha = max(10, 60 // i - j * 10)
                    glow_color_with_alpha = (*color, current_alpha)
                    
                    # Temporary surface for the glow circle
                    glow_surf_size = current_glow_radius * 2 + 10
                    glow_surface = pygame.Surface((glow_surf_size, glow_surf_size), pygame.SRCALPHA)
                    pygame.draw.circle(
                        glow_surface,
                        glow_color_with_alpha,
                        (current_glow_radius + 5, current_glow_radius + 5),
                        current_glow_radius
                    )
                    self.screen.blit(glow_surface, 
                                (x - current_glow_radius - 5, y - current_glow_radius - 5))
        
        # Create piece surface for the main piece
        piece_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        
        # Draw main piece with gradient
        # Draw main piece with improved gradient
        for i in range(radius, 0, -1):
            ratio = i / radius
            # Quadratic curve for smoother transition
            brightness = 0.8 + 0.2 * (ratio * ratio)  # Center: 1.0, Edge: 0.8
            r = int(color[0] * brightness)
            g = int(color[1] * brightness)
            b = int(color[2] * brightness)
            pygame.draw.circle(piece_surface, (r, g, b, 255), (radius, radius), i)
        self.screen.blit(piece_surface, (x - radius, y - radius))
        
            
    def draw_ui(self):
        """Draw user interface elements"""
        # Draw top bar with gradient
        top_bar_height = self.CELL_SIZE
        top_bar = pygame.Rect(0, 0, self.SCREEN_WIDTH, top_bar_height)
        top_gradient = self.create_gradient_surface(self.SCREEN_WIDTH, top_bar_height, 
                                                  self.BOARD_HIGHLIGHT, self.BOARD_COLOR)
        self.screen.blit(top_gradient, (0, 0))
        
        # Draw title with shadow
        title_text = "CONNECT 4"
        title_shadow = self.title_font.render(title_text, True, self.SHADOW_COLOR)
        title_main = self.title_font.render(title_text, True, self.ACCENT_COLOR)
        
        title_x = self.SCREEN_WIDTH // 2 - title_main.get_width() // 2
        title_y = top_bar_height // 2 - title_main.get_height() // 2
        
        self.screen.blit(title_shadow, (title_x + 4, title_y + 4))
        self.screen.blit(title_main, (title_x, title_y))
        
        # Draw scores
        player_score_text = f"Player: {self.player_score}"
        ai_score_text = f"AI: {self.ai_score}"
        
        player_score = self.score_font.render(player_score_text, True, self.PLAYER_COLOR)
        ai_score = self.score_font.render(ai_score_text, True, self.AI_COLOR)
        
        # Add background to scores
        player_bg = pygame.Surface((player_score.get_width() + 20, player_score.get_height() + 10))
        player_bg.fill(self.BOARD_COLOR)
        player_bg.set_alpha(200)
        self.screen.blit(player_bg, (15, 15))
        
        ai_bg = pygame.Surface((ai_score.get_width() + 20, ai_score.get_height() + 10))
        ai_bg.fill(self.BOARD_COLOR)
        ai_bg.set_alpha(200)
        self.screen.blit(ai_bg, (self.SCREEN_WIDTH - ai_score.get_width() - 35, 15))
        
        self.screen.blit(player_score, (25, 20))
        self.screen.blit(ai_score, (self.SCREEN_WIDTH - ai_score.get_width() - 25, 20))
        
        # Draw current turn indicator
        if not self.game_over:
            if self.falling_piece:
                if self.falling_piece['player'] == 1:
                    turn_text = "Your piece is falling..."
                else:
                    turn_text = "AI piece is falling..."
            else:
                turn_text = "Your Turn" if self.current_player == 1 else "AI Thinking..."
            
            turn_color = self.PLAYER_COLOR if self.current_player == 1 else self.AI_COLOR
            
            # Animate turn indicator
            pulse = abs(math.sin(self.pulse_animation * 2)) * 0.3 + 0.7
            animated_color = tuple(int(c * pulse) for c in turn_color)
            turn_display = self.turn_font.render(turn_text, True, animated_color)
            
            # Position at bottom of screen
            turn_x = self.SCREEN_WIDTH // 2 - turn_display.get_width() // 2
            turn_y = self.SCREEN_HEIGHT - 50
            
            self.screen.blit(turn_display, (turn_x, turn_y))
        
        # Draw game over message
        if self.game_over:
            # Dark overlay with blur effect
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            for i in range(3):  # Create multiple layers for blur effect
                overlay.fill((0, 0, 0, 60))
                self.screen.blit(overlay, (0, 0))
            
            # Result text
            if self.winner == 1:
                result_text = "VICTORY!"
                result_color = self.PLAYER_COLOR
            elif self.winner == -1:
                result_text = "AI WINS!"
                result_color = self.AI_COLOR
            else:
                result_text = "DRAW!"
                result_color = self.ACCENT_COLOR
            
            result = self.title_font.render(result_text, True, result_color)
            result_shadow = self.title_font.render(result_text, True, self.SHADOW_COLOR)
            
            result_x = self.SCREEN_WIDTH // 2 - result.get_width() // 2
            result_y = self.SCREEN_HEIGHT // 2 - 100
            
            self.screen.blit(result_shadow, (result_x + 3, result_y + 3))
            self.screen.blit(result, (result_x, result_y))
            
            # Play again button with hover effect
            button_width = 240
            button_height = 60
            button_rect = pygame.Rect(
                self.SCREEN_WIDTH // 2 - button_width // 2,
                self.SCREEN_HEIGHT // 2,
                button_width,
                button_height
            )
            
            # Check for hover
            mouse_pos = pygame.mouse.get_pos()
            is_hovered = button_rect.collidepoint(mouse_pos)
            
            button_color = self.ACCENT_COLOR if not is_hovered else self.PLAYER_COLOR
            self.draw_rounded_rect(self.screen, button_color, button_rect, 20)
            
            # Button text
            button_text = self.button_font.render("PLAY AGAIN", True, self.BACKGROUND_TOP)
            self.screen.blit(button_text, 
                           (self.SCREEN_WIDTH // 2 - button_text.get_width() // 2, 
                            self.SCREEN_HEIGHT // 2 + 20))
            
            # Instructions
            instruction_text = "Press R to restart anytime | ESC to exit"
            instruction = self.turn_font.render(instruction_text, True, self.TEXT_COLOR)
            self.screen.blit(instruction, 
                           (self.SCREEN_WIDTH // 2 - instruction.get_width() // 2, 
                            self.SCREEN_HEIGHT // 2 + 100))
    
    def board_to_dataframe(self):
        """Convert current board state to DataFrame for model prediction"""
        # Flatten the board (row-major order)
        flat_board = []
        for row in range(self.ROWS):
            for col in range(self.COLS):
                flat_board.append(self.board[row][col])
        
        # Create DataFrame with expected columns
        data = {
            'turn': [self.current_player],  # -1 for AI's turn
            **{f'p{i+1}': [flat_board[i]] for i in range(42)}
        }
        
        df = pd.DataFrame(data)
        return df
    
    def analyze_board_features(self):
        """Debug function to see what features the model would create"""
        if not self.model:
            return
        
        try:
            df = self.board_to_dataframe()
            
            # Let's manually check some of the features from model.py
            from model import create_features, count_threats, count_split_threes
            
            # Get the features for the current board
            features = create_features(df.iloc[0])
            
            print("\n=== MODEL FEATURE ANALYSIS ===")
            print(f"Current player (turn): {self.current_player}")
            print(f"Player odd threats: {features['my_odd_threats']}")
            print(f"Player even threats: {features['my_even_threats']}")
            print(f"AI odd threats: {features['opp_odd_threats']}")
            print(f"AI even threats: {features['opp_even_threats']}")
            print(f"Player split threes: {features['my_split_threes']}")
            print(f"AI split threes: {features['opp_split_threes']}")
            print(f"Center column pieces: {features['center_col_pieces']}")
            print(f"Winning moves available: {features['winning_moves']}")
            print(f"Blocking moves available: {features['blocking_moves']}")
            print(f"Suicide moves: {features['suicide_moves']}")
            print(f"Attack moves: {features['attack_moves']}")
            print(f"Fork moves: {features['fork_moves']}")
            print(f"Valid moves count: {features['valid_moves_count']}")
            print(f"Best move score: {features['best_move_score']}")
            print(f"Missed blocks: {features['missed_blocks']}")
            print("=============================\n")
            
        except Exception as e:
            print(f"Error analyzing features: {e}")
    
# Replace the entire ai_move() method and add helper methods:

    def ai_move(self):
        """Get AI move with FIXED suicide detection and defensive play"""
        if self.game_over or self.falling_piece:
            return None
        
        print("\n=== AI DECISION MAKING ===")
        
        # 1. Check if AI has immediate winning move
        ai_win_col = self.find_winning_move(-1)
        if ai_win_col is not None:
            row = self.get_last_open_row(self.board, ai_win_col)
            print(f"1. AI WINNING MOVE at column {ai_win_col}")
            return ai_win_col, row
        
        # 2. Check if player has immediate winning move (block it!)
        player_win_col = self.find_winning_move(1)
        if player_win_col is not None:
            row = self.get_last_open_row(self.board, player_win_col)
            print(f"2. BLOCKING player's winning move at column {player_win_col}")
            return player_win_col, row
        
        # 3. Check for suicide moves (BUG FIX - model.py has wrong logic)
        suicide_moves = self.find_suicide_moves(-1)
        print(f"3. Suicide moves detected: {suicide_moves}")
        
        # 4. Get valid non-suicide moves
        valid_moves = []
        for col in range(self.COLS):
            row = self.get_last_open_row(self.board, col)
            if row != -1 and col not in suicide_moves:
                valid_moves.append((col, row))
        
        # If all moves are suicide, we have to play one
        if not valid_moves and suicide_moves:
            print("WARNING: All moves are suicide!")
            for col in suicide_moves:
                row = self.get_last_open_row(self.board, col)
                valid_moves.append((col, row))
        
        # 5. Use model prediction for remaining valid moves
        if valid_moves and self.model:
            try:
                df = self.board_to_dataframe()
                predictions = self.model.predict(df)
                model_col = int(predictions[0])
                
                print(f"4. Model predicted column: {model_col}")
                
                # Check if model prediction is in valid moves
                valid_cols = [col for col, _ in valid_moves]
                if model_col in valid_cols:
                    row = self.get_last_open_row(self.board, model_col)
                    print(f"5. Using model's prediction: column {model_col}")
                    return model_col, row
            except Exception as e:
                print(f"Model error: {e}")
        
        # 6. Fallback: choose best move from valid options
        if valid_moves:
            best_col, best_row = self.choose_strategic_move(valid_moves)
            print(f"6. Fallback strategic move: column {best_col}")
            return best_col, best_row
        
        print("=== NO VALID MOVES FOUND ===")
        return None

    def find_winning_move(self, player):
        """Find immediate winning move for given player"""
        for col in range(self.COLS):
            row = self.get_last_open_row(self.board, col)
            if row != -1:
                # Try the move
                self.board[row][col] = player
                if check_win(self.board, player):
                    # Undo the move
                    self.board[row][col] = 0
                    return col
                # Undo the move
                self.board[row][col] = 0
        return None

    def find_suicide_moves(self, player):
        """CORRECT suicide move detection: moves that give opponent immediate win"""
        suicide_moves = []
        opponent = -player
        
        for col in range(self.COLS):
            row = self.get_last_open_row(self.board, col)
            if row == -1:
                continue
            
            # Temporarily make AI's move
            self.board[row][col] = player
            
            # Check if opponent can win immediately after this move
            opponent_can_win = False
            for opp_col in range(self.COLS):
                opp_row = self.get_last_open_row(self.board, opp_col)
                if opp_row != -1:
                    self.board[opp_row][opp_col] = opponent
                    if check_win(self.board, opponent):
                        opponent_can_win = True
                    # Undo opponent's test move
                    self.board[opp_row][opp_col] = 0
                    
                    if opponent_can_win:
                        break
            
            # Undo AI's test move
            self.board[row][col] = 0
            
            if opponent_can_win:
                suicide_moves.append(col)
        
        return suicide_moves

    def choose_strategic_move(self, valid_moves):
        """Choose strategic move from valid options"""
        if not valid_moves:
            return None
        
        # Score each move
        scored_moves = []
        for col, row in valid_moves:
            score = 0
            
            # Position scoring (center is best)
            if col == 3:
                score += 10
            elif col in [2, 4]:
                score += 8
            elif col in [1, 5]:
                score += 6
            else:
                score += 4
            
            # Test if this creates a win for AI
            self.board[row][col] = -1
            if check_win(self.board, -1):
                score += 1000  # Winning move
            self.board[row][col] = 0
            
            # Test if this creates a threat (3 in a row with open 4th)
            self.board[row][col] = -1
            threat_score = self.evaluate_threats(-1)
            score += threat_score * 5
            self.board[row][col] = 0
            
            # Penalize if it gives opponent easy win
            self.board[row][col] = -1
            opponent_threat = self.evaluate_threats(1)
            score -= opponent_threat * 3
            self.board[row][col] = 0
            
            scored_moves.append((col, row, score))
        
        # Choose highest score
        scored_moves.sort(key=lambda x: x[2], reverse=True)
        return scored_moves[0][0], scored_moves[0][1]

    def evaluate_threats(self, player):
        """Evaluate number of threats created by current board for given player"""
        threat_count = 0
        board = self.board
        
        # Check horizontal threats
        for r in range(self.ROWS):
            for c in range(self.COLS - 3):
                window = [board[r][c], board[r][c+1], board[r][c+2], board[r][c+3]]
                if window.count(player) == 3 and window.count(0) == 1:
                    # Check if the empty spot is playable
                    empty_idx = window.index(0)
                    empty_col = c + empty_idx
                    if self.get_last_open_row(board, empty_col) == r:
                        threat_count += 1
        
        # Check vertical threats
        for r in range(self.ROWS - 3):
            for c in range(self.COLS):
                window = [board[r][c], board[r+1][c], board[r+2][c], board[r+3][c]]
                if window.count(player) == 3 and window.count(0) == 1:
                    threat_count += 1
        
        # Check diagonal threats
        for r in range(self.ROWS - 3):
            for c in range(self.COLS - 3):
                # Down-right diagonal
                window = [board[r][c], board[r+1][c+1], board[r+2][c+2], board[r+3][c+3]]
                if window.count(player) == 3 and window.count(0) == 1:
                    threat_count += 1
        
        for r in range(3, self.ROWS):
            for c in range(self.COLS - 3):
                # Up-right diagonal
                window = [board[r][c], board[r-1][c+1], board[r-2][c+2], board[r-3][c+3]]
                if window.count(player) == 3 and window.count(0) == 1:
                    threat_count += 1
        
        return threat_count
    
    def check_game_state(self, row, col):
        """Check if the game has ended after a move"""
        # Check for win
        player = self.board[row][col]
        if check_win(self.board, player):
            self.winner = player
            self.game_over = True
            if self.winner == 1:
                self.player_score += 1
            else:
                self.ai_score += 1
            
            # Big particle explosion for win
            x = self.board_offset_x + col * self.CELL_SIZE + self.CELL_SIZE // 2
            y = self.board_offset_y + (row + 1) * self.CELL_SIZE + self.CELL_SIZE // 2
            color = self.PLAYER_COLOR if self.winner == 1 else self.AI_COLOR
            for _ in range(150):
                self.create_particles(x, y, color, count=8)
        
        # Check for draw
        elif np.all(self.board != 0):
            self.winner = 0
            self.game_over = True
    
    def start_move(self, col, player):
        """Start a move animation at the specified column"""
        if self.game_over or self.falling_piece:
            return False
        
        row = self.get_last_open_row(self.board, col)
        if row == -1:  # Column is full
            return False
        
        # Set falling piece animation
        color = self.PLAYER_COLOR if player == 1 else self.AI_COLOR
        
        self.falling_piece = {
            'col': col,
            'target_row': row,
            'color': color,
            'player': player,
            'start_time': time.time()
        }
        
        return True
    
    def reset_game(self):
        """Reset the game state"""
        self.board = np.zeros((self.ROWS, self.COLS), dtype=int)
        self.current_player = 1  # Player starts
        self.game_over = False
        self.winner = None
        self.player_score = 0
        self.ai_score = 0
        self.falling_piece = None
        self.last_time = time.time()
    
    def handle_click(self, pos):
        """Handle mouse click events"""
        x, y = pos
        
        if self.game_over:
            # Check if play again button was clicked
            button_width = 240
            button_height = 60
            button_rect = pygame.Rect(
                self.SCREEN_WIDTH // 2 - button_width // 2,
                self.SCREEN_HEIGHT // 2,
                button_width,
                button_height
            )
            if button_rect.collidepoint(x, y):
                self.reset_game()
                return
        
        if not self.game_over and self.current_player == 1 and not self.falling_piece:
            # Adjust mouse position relative to board
            board_x = x - self.board_offset_x
            if 0 <= board_x < self.WIDTH:
                col = board_x // self.CELL_SIZE
                if 0 <= col < self.COLS:
                    # Check if column has space
                    if self.get_last_open_row(self.board, col) != -1:
                        if self.start_move(col, 1):
                            print(f"Player chose column {col}")
    
    def update(self):
        """Update game state"""
        current_time = time.time()
        self.last_time = current_time
        
        # Update particles
        self.update_particles()
        
        # If it's AI's turn and no piece is falling
        if not self.game_over and not self.falling_piece and self.current_player == -1:
            # Small delay before AI moves for better UX
            if not hasattr(self, 'ai_thinking_start'):
                self.ai_thinking_start = time.time()
            
            thinking_time = time.time() - self.ai_thinking_start
            if thinking_time > 0.8:  # 800ms thinking time
                # Get AI move using the model's full capabilities
                ai_move = self.ai_move()
                if ai_move:
                    col, row = ai_move
                    self.start_move(col, -1)
                delattr(self, 'ai_thinking_start')
    
    def run(self):
        """Main game loop"""
        clock = pygame.time.Clock()
        running = True
        
        print("\n" + "="*70)
        print("CONNECT 4 - AI CHALLENGE (USING FULL MODEL FEATURES)")
        print("="*70)
        print(f"Screen Resolution: {self.SCREEN_WIDTH}x{self.SCREEN_HEIGHT}")
        print(f"Board Size: {self.WIDTH}x{self.HEIGHT}")
        print(f"Cell Size: {self.CELL_SIZE}")
        print("="*70)
        print("IMPORTANT: The model's create_features() function checks for:")
        print("• Winning moves (immediate wins)")
        print("• Blocking moves (opponent's immediate wins)")
        print("• Suicide moves (moves that give opponent immediate win)")
        print("• Attack moves (creates threats)")
        print("• Fork moves (creates multiple threats)")
        print("• Threats count (odd/even playable threats)")
        print("="*70)
        print("Controls:")
        print("• Click in a column to drop your piece")
        print("• Press R to restart")
        print("• Press ESC to exit")
        print("="*70 + "\n")
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        print("Game restarted!")
                        self.reset_game()
                    elif event.key == pygame.K_ESCAPE:
                        running = False
            
            # Update game state
            self.update()
            
            # Draw everything
            self.draw_board()
            
            # Update display
            pygame.display.flip()
            clock.tick(self.FPS)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = Connect4GUI()
    game.run()
