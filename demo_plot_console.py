import numpy as np
import time
import sys
import shutil

class CouplingMatrixAnimator:
    def __init__(self, matrix_size=10, color_mode=True):
        """
        Initialize the coupling matrix animator
        
        Args:
            matrix_size (int): Size of the square matrix
            color_mode (bool): Whether to use ANSI color codes
        """
        self.size = matrix_size
        self.color_mode = color_mode
        
        # Color palette for heatmap
        self.color_palette = [
            '\033[48;5;232m',  # darkest
            '\033[48;5;233m',
            '\033[48;5;234m',
            '\033[48;5;235m',
            '\033[48;5;236m',
            '\033[48;5;237m',
            '\033[48;5;238m',
            '\033[48;5;239m',
            '\033[48;5;240m',
            '\033[48;5;241m',
            '\033[48;5;242m',
            '\033[48;5;243m',
            '\033[48;5;244m',
            '\033[48;5;245m',
            '\033[48;5;246m',
            '\033[48;5;247m',
            '\033[48;5;248m',
            '\033[48;5;249m',
            '\033[48;5;250m',
            '\033[48;5;251m',
            '\033[48;5;252m',
            '\033[48;5;253m',
            '\033[48;5;254m',
            '\033[48;5;255m',  # lightest
            '\033[48;5;15m'    # bright white
        ]
        self.reset_color = '\033[0m'
    
    def generate_dynamic_matrix(self, mode='sync', seed=42):
        """
        Generate a dynamically evolving coupling matrix
        
        Args:
            mode (str): 'sync' or 'async' matrix evolution
            seed (int): Random seed for reproducibility
        
        Returns:
            generator of matrices
        """
        np.random.seed(seed)
        
        # Initial matrix
        matrix = np.zeros((self.size, self.size))
        
        for _ in range(100):  # Number of frames
            if mode == 'sync':
                # Synchronous update (whole matrix changes)
                matrix += np.random.normal(0, 0.1, (self.size, self.size))
                matrix = np.clip(matrix, 0, 1)
            else:
                # Asynchronous update (localized changes)
                i, j = np.random.randint(0, self.size, 2)
                matrix[i, j] += np.random.normal(0, 0.2)
                matrix[i, j] = np.clip(matrix[i, j], 0, 1)
            
            yield matrix
    
    def color_value(self, value):
        """Map a value to a color index"""
        if not self.color_mode:
            return ' '
        color_idx = min(int(value * len(self.color_palette)), len(self.color_palette) - 1)
        return self.color_palette[color_idx]
    
    def render_matrix(self, matrix):
        """
        Render matrix to console
        
        Args:
            matrix (np.ndarray): Coupling matrix to render
        """
        render_lines = []
        for row in matrix:
            line = ''
            for val in row:
                color = self.color_value(val)
                line += color + '  ' + self.reset_color
            render_lines.append(line)
        
        return '\n'.join(render_lines)
    
    def animate(self, mode='sync', seed=42, interval=0.1):
        """
        Animate the coupling matrix
        
        Args:
            mode (str): 'sync' or 'async' matrix evolution
            seed (int): Random seed
            interval (float): Time between frames
        """
        # Get terminal size
        terminal_columns, terminal_rows = shutil.get_terminal_size()
        
        # Check if terminal is large enough
        if terminal_columns < self.size * 2 or terminal_rows < self.size:
            print("Terminal too small for the matrix visualization.")
            return
        
        try:
            for matrix in self.generate_dynamic_matrix(mode, seed):
                # Clear screen
                sys.stdout.write('\033[2J')  # Clear entire screen
                sys.stdout.write('\033[H')   # Move cursor to top-left
                
                # Render matrix
                print(self.render_matrix(matrix))
                
                # Pause
                sys.stdout.flush()
                time.sleep(interval)
        
        except KeyboardInterrupt:
            print("\nAnimation stopped.")

def main():
    # Demo usage
    animator = CouplingMatrixAnimator(matrix_size=20)
    
    print("Synchronous Matrix Evolution:")
    animator.animate(mode='sync', interval=0.05)
    
    print("\nAsynchronous Matrix Evolution:")
    animator.animate(mode='async', interval=0.05)

if __name__ == '__main__':
    main()