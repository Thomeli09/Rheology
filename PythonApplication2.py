from telnetlib import X3PAD
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Initialize main window
root = tk.Tk()
root.title("Spring Assembly Time Response")

# Lists to store inputs
force_time_list = []
node_list = [(0, 0.0, True)]  # Default node: (node number, position, is_fixed)
element_list = []
next_node_number = 1  # Start node numbering from 1

def on_closing():
    """
    Handles the closing event of the GUI. Asks the user for confirmation to quit,
    closes all matplotlib figures, and then quits and destroys the root window.
    """
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        # Close all matplotlib figures
        plt.close('all')
        
        # Quit the tkinter GUI safely
        root.quit()
        root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

def update_buttons():
    delete_force_time_button['state'] = 'normal' if force_time_list else 'disabled'
    plot_force_button['state'] = 'normal' if force_time_list else 'disabled'
    show_assembly_button['state'] = 'normal' if element_list else 'disabled'
    compute_and_store_button['state'] = 'normal' if force_time_list and element_list else 'disabled'
    delete_last_element_button['state'] = 'normal' if element_list else 'disabled'
    delete_node_button['state'] = 'normal' if len(node_list) > 1 else 'disabled'
    delete_element_button['state'] = 'normal' if element_list else 'disabled'
    show_data_button['state'] = 'normal' if node_list else 'disabled'

# Function to plot force vs time
def plot_force_time():
    if force_time_list:
        times, forces = zip(*force_time_list)
        plt.figure()
        plt.plot(times, forces, marker='o')
        plt.xlabel('Time')
        plt.ylabel('Force')
        plt.title('Force vs Time')
        plt.show()

# Function to add force-time pair
def add_force_time():
    try:
        time = float(time_entry.get())
        force = float(force_entry.get())
        if time < 0:
            raise ValueError("Time cannot be negative.")
        force_time_list.append((time, force))
        update_force_time_listbox()
        update_buttons()
    except ValueError as e:
        messagebox.showerror("Invalid input", str(e))

# Function to delete last force-time pair
def delete_last_force_time():
    if force_time_list:
        force_time_list.pop()
        update_force_time_listbox()
        update_buttons()

# Function to update listbox for force-time pairs
def update_force_time_listbox():
    force_time_listbox.delete(0, tk.END)
    for time, force in force_time_list:
        force_time_listbox.insert(tk.END, f"Time: {time}, Force: {force}")

# Function to add node
def add_node():
    global next_node_number
    try:
        x_position = float(x_position_entry.get())
        is_fixed = bool(is_fixed_var.get())
        node_list.append((next_node_number, x_position, is_fixed))
        next_node_number += 1
        update_node_listbox()
        update_buttons()
    except ValueError:
        messagebox.showerror("Invalid input", "Please enter valid numerical values.")

# Function to delete node by number
def delete_node():
    try:
        node_number = int(delete_node_number_entry.get())
        if node_number == 0:
            raise ValueError("Cannot delete the default node.")
        node_list[:] = [node for node in node_list if node[0] != node_number]
        update_node_listbox()
        update_buttons()
    except ValueError as e:
        messagebox.showerror("Invalid input", str(e))

# Function to update listbox for nodes
def update_node_listbox():
    node_listbox.delete(0, tk.END)
    for node_number, x_position, is_fixed in node_list:
        node_listbox.insert(tk.END, f"Node {node_number}: Position {x_position}, {'Fixed' if is_fixed else 'Free'}")

# Function to add element (spring or dashpot)
def add_element():
    try:
        element_type = element_type_var.get()
        node1 = int(node1_entry.get())
        node2 = int(node2_entry.get())
        value = float(value_entry.get())
        if value <= 0:
            raise ValueError(f"{element_type.capitalize()} value must be greater than 0.")
        if not any(node[0] == node1 for node in node_list) or not any(node[0] == node2 for node in node_list):
            raise ValueError("Both nodes must be created first.")
        element_list.append((element_type, node1, node2, value))
        update_element_listbox()
        update_buttons()
    except ValueError as e:
        messagebox.showerror("Invalid input", str(e))

# Function to delete last element
def delete_last_element():
    if element_list:
        element_list.pop()
        update_element_listbox()
        update_buttons()

# Function to delete element by number
def delete_element():
    try:
        element_number = int(delete_element_number_entry.get())
        if 1 <= element_number <= len(element_list):
            element_list.pop(element_number - 1)
            update_element_listbox()
            update_buttons()
        else:
            raise ValueError("Invalid element number.")
    except ValueError as e:
        messagebox.showerror("Invalid input", str(e))

# Function to update listbox for elements
def update_element_listbox():
    element_listbox.delete(0, tk.END)
    for i, (element_type, node1, node2, value) in enumerate(element_list, 1):
        element_listbox.insert(tk.END, f"{element_type.capitalize()} {i}: Node1: {node1}, Node2: {node2}, Value: {value}")

# Function to plot a spring
def plot_spring(x1, x2, label, line_color='black', line_width=1, text_color='black', text_size=10):
    """
    Plots a spring between two points with specified styling options.
    
    Parameters:
    - x1, x2: Coordinates of the two nodes (ensure x1 is less than x2 for proper orientation).
    - label: The label for the spring.
    - line_color: Color of the spring lines. Default is blue.
    - line_width: Width of the spring lines. Default is 1.
    - text_color: Color of the label text. Default is blue.
    - text_size: Size of the label text. Default is 10.
    """
    if x1 > x2:  # Ensure x1 is always less than x2 for correct orientation
        x1, x2 = x2, x1
        
    # Calculate the height of the spring
    spring_height = (x2 - x1) * 0.1  # 10% of the total length
    half_spring_height = spring_height / 2

    # Calculate thirds for the spring layout
    one_third = (x2 - x1) / 3
    two_thirds = 2 * one_third

    # Points for the spring
    num_points = 20  # Number of points for the zigzag part
    x_mid_start = x1 + one_third
    x_mid_end = x1 + two_thirds
    x_mid = np.linspace(x_mid_start, x_mid_end, num_points)

    # Generating piecewise straight lines for spring effect
    # Ensure segments are odd for the pattern to start and end at y=0
    segments = 13
    if segments % 2 == 1:
        segments += 1  # Increase segments by 1 if even to make it odd

    x_coords = []
    y_coords = []
    
    # Generate the x and y coordinates
    step = (x_mid_end - x_mid_start) / segments  # Determine the step size for x based on the number of segments
    for i in range(segments + 1):
        current_x = x_mid_start + i * step
        x_coords.append(current_x)
        # Determine y coordinate based on alternating pattern
        if i % 2 == 0:
            y_coords.append(0)
        else:
            y_coords.append(half_spring_height if (i % 4 == 1) else -half_spring_height)  # Alternate between positive and negative heights

    # Plot the ends of the spring as straight lines
    plt.plot([x1, x1 + one_third], [0, 0], color=line_color, linewidth=line_width)
    plt.plot([x1 + two_thirds, x2], [0, 0], color=line_color, linewidth=line_width)

    # Plot the central, piecewise straight "zigzag" part of the spring
    plt.plot(x_coords, y_coords, color=line_color, linewidth=line_width)

    # Label the spring
    plt.text((x1 + x2) / 2, 0.05 + half_spring_height, label, ha='center', color=text_color, fontsize=text_size)

# Function to plot a damper
def plot_damper(x1, x2, label, line_color='black', line_width=1, text_color='black', text_size=10):
    """
    Plots a realistic damper (like a piston) between two points.
    
    Parameters:
    - x1, x2: Coordinates of the two nodes (x1 should be less than x2 for proper orientation).
    - label: The label for the damper.
    - line_color: Color of the damper lines. Default is black.
    - line_width: Width of the damper and link lines. Default is 1.
    - text_color: Color of the label text. Default is black.
    - text_size: Size of the label text. Default is 10.
    """
    if x1 > x2:  # Ensure x1 is always less than x2 for correct damper orientation
        x1, x2 = x2, x1

    # Calculate the height of the damper head
    damper_height = (x2 - x1) * 0.1  # 10% of the total length
    half_damper_height = damper_height / 2
    
    # Plot the links to the damper
    x3 = x1 + (x2 - x1) / 3
    x4 = x1 + 2 * (x2 - x1) / 3
    plt.plot([x1, x3], [0, 0], color=line_color, linewidth=line_width)
    plt.plot([x4, x2], [0, 0], color=line_color, linewidth=line_width)
   
    # Plot the vertical parts of the damper
    plt.plot([x3, x3], [-half_damper_height, half_damper_height], color=line_color, linewidth=line_width)
    plt.plot([x4, x4], [-half_damper_height, half_damper_height], color=line_color, linewidth=line_width)

    # Plot the top and bottom lines of the damper
    plt.plot([x4, x3 - 0.5 * (x4 - x3)], [-half_damper_height, -half_damper_height], color=line_color, linewidth=line_width)
    plt.plot([x4, x3 - 0.5 * (x4 - x3)], [half_damper_height, half_damper_height], color=line_color, linewidth=line_width)

    # Add a rectangle to represent the fluid damper
    fluid_damper = plt.Rectangle((x3, -half_damper_height), x4 - x3, damper_height, color='grey', alpha=0.5)
    plt.gca().add_patch(fluid_damper)

    # Label the damper
    plt.text((x1 + x2) / 2, 0.05 + half_damper_height, label, ha='center', color=text_color, fontsize=text_size)

# Function to show spring and dashpot assembly
def show_assembly():
    """
    Displays an assembly diagram with springs and dampers, marking used nodes with dots.
    
    Parameters:
    - element_list: List of tuples (element_type, node1, node2, _)
    - node_list: List of tuples (node_number, x_position, _)
    """
    if element_list:
        # Determine unused nodes
        unused_nodes = {node[0] for node in node_list} - {node1 for _, node1, _, _ in element_list} - {node2 for _, _, node2, _ in element_list}
        
        plt.figure()
        for i, (element_type, node1, node2, _) in enumerate(element_list, 1):
            # Get positions for each node
            x1 = next(x for n, x, _ in node_list if n == node1)
            x2 = next(x for n, x, _ in node_list if n == node2)

            # Draw elements (springs or dampers)
            if element_type == 'spring':
                plot_spring(x1, x2, f"Spring {i}")
            else:
                plot_damper(x1, x2, f"Damper {i}")

        # Extract node positions and numbers for plotting
        node_positions = [x for _, x, _ in node_list]
        node_numbers = [n for n, _, _ in node_list]

        # Plot nodes using scatter
        plt.scatter(node_positions, [0]*len(node_positions), color='red', s=100, zorder=5)  # s is size of the dot

        # Label nodes
        for node_number, x_position in zip(node_numbers, node_positions):
            plt.text(x_position, 0.05, f"Node {node_number}", ha='center', va='bottom')

        plt.xlabel('Position')
        plt.title('Spring and Dashpot Assembly')
        plt.show()

# Function to compute and store data
def assemble_global_matrices(num_nodes, element_list):
    """
    Assemble the global stiffness and damping matrices.
    """
    K = np.zeros((num_nodes, num_nodes))
    C = np.zeros((num_nodes, num_nodes))

    for element_type, node1, node2, value in element_list:
        if element_type == 'spring':
            k = value
            c = 0
        elif element_type == 'dashpot':
            k = 0
            c = value
        K[node1, node1] += k
        K[node2, node2] += k
        K[node1, node2] -= k
        K[node2, node1] -= k

        C[node1, node1] += c
        C[node2, node2] += c
        C[node1, node2] -= c
        C[node2, node1] -= c

    return K, C

def apply_boundary_conditions(matrix, locked_nodes):
    """
    Apply boundary conditions to the stiffness and damping matrices.
    """
    for node in locked_nodes:
        matrix[node, :] = 0
        matrix[:, node] = 0
        matrix[node, node] = 1

def force_vector(num_nodes, force_time_list, time):
    """
    Generate the force vector at a given time.
    """
    F = np.zeros(num_nodes)
    for t, force in force_time_list:
        if t == time:
            F[3] = force  # Assuming force is applied at node 3
    return F

def euler_integration(u, v, a, M, C, K, F, dt):
    """
    Perform one step of explicit Euler integration.
    """
    a_new = np.linalg.inv(M) @ (F - C @ v - K @ u)
    v_new = v + dt * a_new
    u_new = u + dt * v_new
    return u_new, v_new, a_new
        
def compute_and_store(force_time_list, element_list, node_list, locked_nodes):
    """
    Main function to compute the response of the system and store the results.
    """
    if not force_time_list or not element_list:
        messagebox.showerror("Incomplete data", "Please provide both force-time data and element assembly.")
        return

    num_nodes = len(node_list)
    num_steps = len(force_time_list)
    dt = 0.01  # Assuming a fixed time step, modify as needed

    u = np.zeros(num_nodes)  # Displacement
    v = np.zeros(num_nodes)  # Velocity
    a = np.zeros(num_nodes)  # Acceleration
    M = np.eye(num_nodes)  # Mass matrix, assumed to be identity for simplicity
    displacements = np.zeros((num_steps, num_nodes))
    forces = np.zeros(num_steps)

    K, C = assemble_global_matrices(num_nodes, element_list)
    apply_boundary_conditions(K, locked_nodes)
    apply_boundary_conditions(C, locked_nodes)

    times = [t for t, _ in force_time_list]

    for step, (time, force) in enumerate(force_time_list):
        F = force_vector(num_nodes, force_time_list, time)
        forces[step] = force
        u, v, a = euler_integration(u, v, a, M, C, K, F, dt)
        displacements[step] = u.copy()

    plot_results(times, forces, displacements, num_nodes)

    global stored_data
    stored_data = {node[0]: list(zip(times, displacements[:, idx])) for idx, node in enumerate(node_list)}
    messagebox.showinfo("Computation Complete", "Data computed and stored successfully.")

# Function to show node data
def show_node_data():
    try:
        node_number = int(node_data_entry.get())
        if node_number not in stored_data:
            raise ValueError("Invalid node number.")
        times, responses = zip(*stored_data[node_number])
        plt.figure()
        plt.plot(times, responses, marker='o')
        plt.xlabel('Time')
        plt.ylabel('Response')
        plt.title(f'Time Response of Node {node_number}')
        plt.show()
    except ValueError as e:
        messagebox.showerror("Invalid input", str(e))

# Frame for scrolling content
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=1)

# Add a canvas in that frame
canvas = tk.Canvas(main_frame)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

# Add a scrollbar to the canvas
scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Configure the canvas
canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

# Bind mouse wheel to scroll
canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

# Create another frame inside the canvas
second_frame = tk.Frame(canvas)

# Add that new frame to a window in the canvas
canvas.create_window((0,0), window=second_frame, anchor="nw")

# Force-time input frame
force_time_frame = ttk.LabelFrame(second_frame, text="Force-Time Input")
force_time_frame.grid(row=0, column=0, padx=10, pady=10)

ttk.Label(force_time_frame, text="Time:").grid(row=0, column=0)
time_entry = ttk.Entry(force_time_frame)
time_entry.grid(row=0, column=1)

ttk.Label(force_time_frame, text="Force:").grid(row=1, column=0)
force_entry = ttk.Entry(force_time_frame)
force_entry.grid(row=1, column=1)

add_force_time_button = ttk.Button(force_time_frame, text="Add", command=add_force_time)
add_force_time_button.grid(row=2, column=0, pady=5)
delete_force_time_button = ttk.Button(force_time_frame, text="Delete Last", command=delete_last_force_time)
delete_force_time_button.grid(row=2, column=1, pady=5)
plot_force_button = ttk.Button(force_time_frame, text="Plot", command=plot_force_time)
plot_force_button.grid(row=2, column=2, pady=5)

force_time_listbox = tk.Listbox(force_time_frame, width=50)
force_time_listbox.grid(row=3, column=0, columnspan=3, pady=5)

# Node input frame
node_frame = ttk.LabelFrame(second_frame, text="Node Input")
node_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

ttk.Label(node_frame, text="X Position:").grid(row=0, column=0)
x_position_entry = ttk.Entry(node_frame)
x_position_entry.grid(row=0, column=1)

ttk.Label(node_frame, text="Fixed:").grid(row=1, column=0)
is_fixed_var = tk.IntVar()
is_fixed_checkbutton = ttk.Checkbutton(node_frame, variable=is_fixed_var)
is_fixed_checkbutton.grid(row=1, column=1)

add_node_button = ttk.Button(node_frame, text="Add Node", command=add_node)
add_node_button.grid(row=2, column=0, pady=5)
delete_node_button = ttk.Button(node_frame, text="Delete Node", command=delete_node)
delete_node_button.grid(row=2, column=1, pady=5)
delete_node_number_entry = ttk.Entry(node_frame)
delete_node_number_entry.grid(row=2, column=2)

node_listbox = tk.Listbox(node_frame, width=50)
node_listbox.grid(row=3, column=0, columnspan=3, pady=5)

# Element (spring/dashpot) input frame
element_frame = ttk.LabelFrame(second_frame, text="Element Input")
element_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

ttk.Label(element_frame, text="Element Type:").grid(row=0, column=0)
element_type_var = tk.StringVar(value="spring")
element_type_combobox = ttk.Combobox(element_frame, textvariable=element_type_var, values=["spring", "dashpot"], state="readonly")
element_type_combobox.grid(row=0, column=1)

ttk.Label(element_frame, text="Node 1:").grid(row=1, column=0)
node1_entry = ttk.Entry(element_frame)
node1_entry.grid(row=1, column=1)

ttk.Label(element_frame, text="Node 2:").grid(row=2, column=0)
node2_entry = ttk.Entry(element_frame)
node2_entry.grid(row=2, column=1)

ttk.Label(element_frame, text="Value:").grid(row=3, column=0)
value_entry = ttk.Entry(element_frame)
value_entry.grid(row=3, column=1)

add_element_button = ttk.Button(element_frame, text="Add Element", command=add_element)
add_element_button.grid(row=4, column=0, pady=5)
delete_last_element_button = ttk.Button(element_frame, text="Delete Last Element", command=delete_last_element)
delete_last_element_button.grid(row=4, column=1, pady=5)
delete_element_button = ttk.Button(element_frame, text="Delete Element", command=delete_element)
delete_element_button.grid(row=4, column=2, pady=5)
delete_element_number_entry = ttk.Entry(element_frame)
delete_element_number_entry.grid(row=4, column=3)

element_listbox = tk.Listbox(element_frame, width=50)
element_listbox.grid(row=5, column=0, columnspan=4, pady=5)

show_assembly_button = ttk.Button(element_frame, text="Show Assembly", command=show_assembly)
show_assembly_button.grid(row=6, column=0, columnspan=2, pady=5)

# Computation and data frame
computation_frame = ttk.LabelFrame(second_frame, text="Computation and Data")
computation_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")

compute_and_store_button = ttk.Button(computation_frame, text="Compute and Store Data", command=compute_and_store)
compute_and_store_button.grid(row=0, column=0, pady=5)

ttk.Label(computation_frame, text="Node Number:").grid(row=1, column=0)
node_data_entry = ttk.Entry(computation_frame)
node_data_entry.grid(row=1, column=1)

show_data_button = ttk.Button(computation_frame, text="Show Node Data", command=show_node_data)
show_data_button.grid(row=1, column=2, pady=5)

update_buttons()
update_node_listbox()
root.mainloop()

