# %%

import pygame
import sys
import numpy as np
import time
import math

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 800
BOARD_SIZE = 15
CELL_SIZE = WIDTH // BOARD_SIZE
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)

# Create the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Gomoku")

class GomokuAI:
    def __init__(self, depth=3):
        self.depth = depth
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.current_i = -1
        self.current_j = -1
        self.next_bound = {}
        self.board_value = 0
        self.turn = 0
        self.last_played = 0
        self.empty_cells = BOARD_SIZE * BOARD_SIZE
        self.pattern_dict = self.create_pattern_dict()
        self.cache = {}  # Add cache

    def ai_move(self):
        best_score = -math.inf
        best_move = None
        alpha = -math.inf
        beta = math.inf
        
        moves = list(self.child_nodes(self.next_bound))
        blocking_moves = self.find_blocking_moves()
        moves = blocking_moves + [move for move in moves if move not in blocking_moves]
        moves.sort(key=lambda move: self.evaluate_move(move[0], move[1]), reverse=True)
        
        for i, j in moves[:min(len(moves), 15)]:  # Consider top 15 moves, including blocking moves
            self.set_state(i, j, 1)
            self.update_bound(i, j, self.next_bound)
            score = self.minimax(self.depth - 1, False, alpha, beta)
            self.set_state(i, j, 0)
            
            if score > best_score:
                best_score = score
                best_move = (i, j)
            
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
        
        if best_move:
            self.set_state(best_move[0], best_move[1], 1)
            self.update_bound(best_move[0], best_move[1], self.next_bound)
            self.current_i, self.current_j = best_move
        
        return (self.current_i, self.current_j)

    def evaluate_move(self, i, j):
        score = 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1), (-1, -1), (1, 1)]
        for dir in directions:
            score += self.count_pattern(i, j, dir, 1, {}, True)
            opponent_score = self.count_pattern(i, j, dir, -1, {}, True)
            score += opponent_score * 1.2  # Give higher weight to blocking opponent's moves
        return score

    def find_blocking_moves(self):
        blocking_moves = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j] == 0:
                    if self.is_blocking_move(i, j):
                        blocking_moves.append((i, j))
        return blocking_moves

    def is_blocking_move(self, i, j):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1), (-1, -1), (1, 1)]
        for dir in directions:
            if self.count_consecutive(i, j, dir, -1) >= 3:
                return True
        return False

    def count_consecutive(self, i, j, direction, player):
        count = 0
        di, dj = direction
        for step in range(1, 5):  # Check up to 4 steps in each direction
            ni, nj = i + di * step, j + dj * step
            if 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE and self.board[ni][nj] == player:
                count += 1
            else:
                break
        return count

    def minimax(self, depth, is_maximizing, alpha, beta):
        board_hash = hash(self.board.tobytes())
        if board_hash in self.cache:
            return self.cache[board_hash]

        if depth == 0 or self.is_game_over():
            score = self.evaluate_board()
            self.cache[board_hash] = score
            return score

        if is_maximizing:
            max_eval = -math.inf
            for i, j in self.child_nodes(self.next_bound):
                self.set_state(i, j, 1)
                self.update_bound(i, j, self.next_bound)
                eval = self.minimax(depth - 1, False, alpha, beta)
                self.set_state(i, j, 0)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            self.cache[board_hash] = max_eval
            return max_eval
        else:
            min_eval = math.inf
            for i, j in self.child_nodes(self.next_bound):
                self.set_state(i, j, -1)
                self.update_bound(i, j, self.next_bound)
                eval = self.minimax(depth - 1, True, alpha, beta)
                self.set_state(i, j, 0)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            self.cache[board_hash] = min_eval
            return min_eval

    def is_game_over(self):
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j] != 0 and self.is_five(i, j, self.board[i][j]):
                    return True
        return self.empty_cells == 0

    def evaluate_board(self):
        score = 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1), (-1, -1), (1, 1)]
        for i in range(max(0, self.current_i - 2), min(BOARD_SIZE, self.current_i + 3)):
            for j in range(max(0, self.current_j - 2), min(BOARD_SIZE, self.current_j + 3)):
                if self.board[i][j] != 0:
                    for dir in directions:
                        score += self.count_pattern(i, j, dir, self.board[i][j], {}, True)
        
        # Add center preference
        center = BOARD_SIZE // 2
        score += (center - abs(self.current_i - center)) + (center - abs(self.current_j - center))
        
        return score

    def create_pattern_dict(self):
        return {
            (1, 1, 1, 1, 1): 100000,
            (0, 1, 1, 1, 1, 0): 10000,
            (0, 1, 1, 1, 0): 1000,
            (0, 1, 1, 0): 100,
            (0, 1, 0): 10
        }

    def is_valid(self, i, j, state=True):
        if i < 0 or i >= BOARD_SIZE or j < 0 or j >= BOARD_SIZE:
            return False
        if state:
            return self.board[i][j] == 0
        return True

    def set_state(self, i, j, state):
        if self.is_valid(i, j):
            if self.board[i][j] == 0 and state != 0:
                self.empty_cells -= 1
            elif self.board[i][j] != 0 and state == 0:
                self.empty_cells += 1
            self.board[i][j] = state
            self.last_played = state
        else:
            print(f"Invalid move: ({i}, {j})")

    def count_direction(self, i, j, xdir, ydir, state):
        count = 0
        for step in range(1, 5):
            if xdir != 0 and (j + xdir*step < 0 or j + xdir*step >= BOARD_SIZE):
                break
            if ydir != 0 and (i + ydir*step < 0 or i + ydir*step >= BOARD_SIZE):
                break
            if self.board[i + ydir*step][j + xdir*step] == state:
                count += 1
            else:
                break
        return count

    def is_five(self, i, j, state):
        directions = [
            [(-1, 0), (1, 0)],
            [(0, -1), (0, 1)],
            [(-1, 1), (1, -1)],
            [(-1, -1), (1, 1)]
        ]
        for dir in directions:
            count = 1
            for d in dir:
                step = 1
                while True:
                    next_i = i + d[0] * step
                    next_j = j + d[1] * step
                    if next_i < 0 or next_i >= BOARD_SIZE or next_j < 0 or next_j >= BOARD_SIZE or self.board[next_i][next_j] != state:
                        break
                    count += 1
                    step += 1
            if count >= 5:
                return True
        return False

    def child_nodes(self, bound):
        for pos in sorted(bound.items(), key=lambda el: el[1], reverse=True):
            yield pos[0]

    def update_bound(self, new_i, new_j, bound):
        played = (new_i, new_j)
        if played in bound:
            bound.pop(played)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1), (-1, -1), (1, 1)]
        for dir in directions:
            new_col = new_j + dir[0]
            new_row = new_i + dir[1]
            if self.is_valid(new_row, new_col) and (new_row, new_col) not in bound:
                bound[(new_row, new_col)] = 0

    def count_pattern(self, i_0, j_0, pattern, score, bound, flag):
        directions = [(1, 0), (1, 1), (0, 1), (-1, 1)]
        length = len(pattern)
        count = 0

        for dir in directions:
            if dir[0] * dir[1] == 0:
                steps_back = dir[0] * min(5, j_0) + dir[1] * min(5, i_0)
            elif dir[0] == 1:
                steps_back = min(5, j_0, i_0)
            else:
                steps_back = min(5, BOARD_SIZE-1-j_0, i_0)
            i_start = i_0 - steps_back * dir[1]
            j_start = j_0 - steps_back * dir[0]

            z = 0
            while z <= steps_back:
                i_new = i_start + z*dir[1]
                j_new = j_start + z*dir[0]
                index = 0
                remember = []
                while index < length and self.is_valid(i_new, j_new, state=False) and self.board[i_new][j_new] == pattern[index]:
                    if self.is_valid(i_new, j_new):
                        remember.append((i_new, j_new))
                    i_new = i_new + dir[1]
                    j_new = j_new + dir[0]
                    index += 1

                if index == length:
                    count += 1
                    for pos in remember:
                        if pos not in bound:
                            bound[pos] = 0
                        bound[pos] += flag*score
                    z += index
                else:
                    z += 1

        return count

    def evaluate(self, new_i, new_j, board_value, turn, bound):
        value_before = 0
        value_after = 0
        
        for pattern in self.pattern_dict:
            score = self.pattern_dict[pattern]
            value_before += self.count_pattern(new_i, new_j, pattern, abs(score), bound, -1)*score
            self.board[new_i][new_j] = turn
            value_after += self.count_pattern(new_i, new_j, pattern, abs(score), bound, 1) *score
            self.board[new_i][new_j] = 0

        return board_value + value_after - value_before

    def alpha_beta_pruning(self, depth, board_value, bound, alpha, beta, maximizing_player):
        if depth <= 0 or self.is_five(self.current_i, self.current_j, self.last_played):
            return board_value

        if maximizing_player:
            max_val = float('-inf')
            for child in self.child_nodes(bound):
                i, j = child[0], child[1]
                new_bound = dict(bound)
                new_val = self.evaluate(i, j, board_value, 1, new_bound)
                self.board[i][j] = 1
                self.update_bound(i, j, new_bound)
                eval = self.alpha_beta_pruning(depth-1, new_val, new_bound, alpha, beta, False)
                if eval > max_val:
                    max_val = eval
                    if depth == self.depth:
                        self.current_i = i
                        self.current_j = j
                        self.board_value = eval
                        self.next_bound = new_bound
                alpha = max(alpha, eval)
                self.board[i][j] = 0
                del new_bound
                if beta <= alpha:
                    break
            return max_val
        else:
            min_val = float('inf')
            for child in self.child_nodes(bound):
                i, j = child[0], child[1]
                new_bound = dict(bound)
                new_val = self.evaluate(i, j, board_value, -1, new_bound)
                self.board[i][j] = -1
                self.update_bound(i, j, new_bound)
                eval = self.alpha_beta_pruning(depth-1, new_val, new_bound, alpha, beta, True)
                if eval < min_val:
                    min_val = eval
                    if depth == self.depth:
                        self.current_i = i
                        self.current_j = j
                        self.board_value = eval
                        self.next_bound = new_bound
                beta = min(beta, eval)
                self.board[i][j] = 0
                del new_bound
                if beta <= alpha:
                    break
            return min_val

    def first_move(self):
        self.current_i, self.current_j = 7, 7
        self.set_state(self.current_i, self.current_j, 1)

def ai_move(ai):
    start_time = time.time()
    ai.alpha_beta_pruning(ai.depth, ai.board_value, ai.next_bound, float('-inf'), float('inf'), True)
    end_time = time.time()
    print('Finished alpha-beta pruning in:', end_time - start_time)

    if ai.is_valid(ai.current_i, ai.current_j):
        print(ai.current_i, ai.current_j)
        ai.set_state(ai.current_i, ai.current_j, 1)  # Always set AI move to 1 (black)
        ai.update_bound(ai.current_i, ai.current_j, ai.next_bound)
    else:
        print('Error: i and j not valid. Given:', ai.current_i, ai.current_j)
        bound_sorted = sorted(ai.next_bound.items(), key=lambda el: el[1], reverse=True)
        pos = bound_sorted[0][0]
        ai.current_i, ai.current_j = pos[0], pos[1]
        ai.set_state(ai.current_i, ai.current_j, 1)  # Always set AI move to 1 (black)
        ai.update_bound(ai.current_i, ai.current_j, ai.next_bound)
        print(ai.current_i, ai.current_j)

    return ai.current_i, ai.current_j

def check_human_move(ai, mouse_pos, player):
    move_i = mouse_pos[1] // CELL_SIZE
    move_j = mouse_pos[0] // CELL_SIZE
    
    if ai.is_valid(move_i, move_j):
        ai.board_value = ai.evaluate(move_i, move_j, ai.board_value, player, ai.next_bound)
        ai.set_state(move_i, move_j, player)
        ai.update_bound(move_i, move_j, ai.next_bound)
        return move_i, move_j
    return None

def draw_board(ai):
    screen.fill(GRAY)
    for i in range(BOARD_SIZE):
        pygame.draw.line(screen, BLACK, (i * CELL_SIZE, 0), (i * CELL_SIZE, HEIGHT), 1)
        pygame.draw.line(screen, BLACK, (0, i * CELL_SIZE), (WIDTH, i * CELL_SIZE), 1)
    
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if ai.board[row][col] == 1:
                pygame.draw.circle(screen, BLACK, (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 2 - 2)
            elif ai.board[row][col] == -1:
                pygame.draw.circle(screen, WHITE, (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 2 - 2)

def draw_menu():
    screen.fill(GRAY)
    font = pygame.font.Font(None, 36)
    title = font.render("Gomoku", True, BLACK)
    option1 = font.render("1. Play against AI", True, BLACK)
    option2 = font.render("2. Two Player Mode", True, BLACK)
    screen.blit(title, (WIDTH // 2 - 50, HEIGHT // 3))
    screen.blit(option1, (WIDTH // 2 - 100, HEIGHT // 2))
    screen.blit(option2, (WIDTH // 2 - 100, HEIGHT // 2 + 50))
    pygame.display.flip()

def main():
    ai = GomokuAI()
    game_mode = None
    game_over = False
    current_player = 1  # 1 for black (AI), -1 for white (human)

    while True:
        if game_mode is None:
            draw_menu()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        game_mode = "AI"
                        ai.first_move()
                        current_player = -1  # Human plays first after AI's first move
                    elif event.key == pygame.K_2:
                        game_mode = "2P"
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
        else:
            draw_board(ai)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    elif event.key == pygame.K_r and game_over:
                        # Reset the game and return to menu
                        ai = GomokuAI()
                        game_mode = None
                        game_over = False
                        current_player = 1
                        break
                elif event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                    if (game_mode == "AI" and current_player == -1) or game_mode == "2P":
                        mouse_pos = pygame.mouse.get_pos()
                        move = check_human_move(ai, mouse_pos, current_player)
                        if move:
                            if ai.is_five(move[0], move[1], current_player):
                                game_over = True
                            elif game_mode == "AI":
                                current_player = 1  # Switch to AI's turn
                            else:
                                current_player *= -1  # Switch players in 2P mode

            if game_mode == "AI" and current_player == 1 and not game_over:
                move = ai_move(ai)
                if move is not None:
                    if ai.is_five(move[0], move[1], 1):
                        game_over = True
                    else:
                        current_player = -1  # Switch back to human's turn
                else:
                    game_over = True  # If AI can't move, end the game

            if game_over:
                font = pygame.font.Font(None, 36)
                if current_player == 1:
                    text = font.render("Black wins!", True, BLACK)
                elif current_player == -1:
                    text = font.render("White wins!", True, BLACK)
                else:
                    text = font.render("It's a draw!", True, BLACK)
                screen.blit(text, (WIDTH // 2 - 50, HEIGHT // 2))
                
                restart_text = font.render("Press 'R' to return to menu", True, BLACK)
                screen.blit(restart_text, (WIDTH // 2 - restart_text.get_width() // 2, HEIGHT // 2 + 50))
                
                exit_text = font.render("Press 'Esc' to exit", True, BLACK)
                screen.blit(exit_text, (WIDTH // 2 - exit_text.get_width() // 2, HEIGHT // 2 + 100))
                
                pygame.display.flip()

            if ai.empty_cells == 0:
                game_over = True

if __name__ == "__main__":
    main()

# %%
