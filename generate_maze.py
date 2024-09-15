import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_maze():
    maze_size = 10  # Size of the maze
    maze = np.ones((maze_size, maze_size), dtype=np.uint8) * 255  # White background
    
    # Draw a simple maze
    for i in range(maze_size):
        maze[i, 2] = 0
    for j in range(maze_size):
        maze[5, j] = 0
    maze[0, 0] = 0  # Start point
    maze[-1, -1] = 0  # Finish point
    
    return maze

def save_maze(image_path):
    maze = generate_maze()
    plt.figure(figsize=(5, 5))
    plt.imshow(maze, cmap='gray')
    plt.axis('off')
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    save_maze('sample_maze.png')
