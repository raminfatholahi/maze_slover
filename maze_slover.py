import cv2
import numpy as np
import argparse

def load_image(image_path):
    """ Load the maze image """
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def preprocess_image(image):
    """ Preprocess the image for maze solving """
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    return binary_image

def find_start_finish_points(image):
    """ Find the start and finish points of the maze """
    # Assuming the start point is the top-left corner and the finish is the bottom-right
    start_point = (0, 0)
    finish_point = (image.shape[1] - 1, image.shape[0] - 1)
    return start_point, finish_point

def find_shortest_path(binary_image, start, finish):
    """ Find the shortest path using A* algorithm """
    from queue import PriorityQueue

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(point):
        neighbors = []
        x, y = point
        if x > 0: neighbors.append((x - 1, y))
        if x < binary_image.shape[1] - 1: neighbors.append((x + 1, y))
        if y > 0: neighbors.append((x, y - 1))
        if y < binary_image.shape[0] - 1: neighbors.append((x, y + 1))
        return neighbors

    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, finish)}

    while not open_set.empty():
        current = open_set.get()[1]

        if current == finish:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for neighbor in get_neighbors(current):
            if binary_image[neighbor[1], neighbor[0]] == 255:  # Ensure it's a path
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, finish)
                    open_set.put((f_score[neighbor], neighbor))

    return []

def visualize_path(image, path):
    """ Visualize the path on the maze """
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for (x, y) in path:
        color_image[y, x] = [0, 0, 255]  # Mark path in red
    cv2.imshow("Maze Solver", color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Solve a maze using OpenCV.")
    parser.add_argument("image", help="Path to the maze image")
    args = parser.parse_args()

    image = load_image(args.image)
    binary_image = preprocess_image(image)
    start, finish = find_start_finish_points(binary_image)
    path = find_shortest_path(binary_image, start, finish)
    visualize_path(image, path)

if __name__ == "__main__":
    main()
