import time

from tkinter import Tk, Canvas, Text, BOTH, W, N, E, S, messagebox
from tkinter.ttk import Frame, Button, Entry, Label, Style
import numpy as np

from environment import GameBoard
from train import create_agent_and_opponent


default_max_mcts_step = 100
default_mcts_eps = 0.05

def step(mcts, game_board, max_mcts_steps, mcts_eps):
    if game_board.get_allowed_actions().any() == 1:
        picked_action = mcts.monte_carlo_tree_search(max_mcts_steps, mcts_eps)
        game_board.take_action(picked_action)
        mcts.change_root_with_action(picked_action)


def take_action(a, mcts, game_board):
    if game_board.get_allowed_actions()[a]:
        mcts.change_root_with_action(a)
        game_board.take_action(a)


def create_button(master, row, col, text, func=None, pady=7, padx=3):
    '''Takes in master, row number, col number, button display text, callback 
       function, and position padding and returns the created button widget'''
    button = Button(master, text=text, command=func, width=7)
    button.grid(row=row, column=col, pady=pady, padx=padx)
    return button


def create_label(master, row, col, text, pady=0, padx=2):
    '''Takes in master, row number, col number, label display text, and 
       position padding and returns the created labe widgetl'''
    label = Label(master, text=text)
    label.grid(row=row, column=col, pady=pady, padx=padx, sticky=W)
    return label


def create_entry(master, row, col, text, pady=0, padx=0):
    '''Takes in master, row number, col number, entry default value, and 
       position padding and returns the created entry widget'''
    entry = Entry(master)
    entry.insert(0, text)
    entry.grid(row=row, column=col, pady=pady, padx=padx)
    entry.config({"width":5})
    return entry


class Window(Frame):
    def __init__(self, root, board_size, win_length):
        super().__init__()
        self.root = root
        self.board_size = board_size
        self.win_length = win_length
        self.game_board = GameBoard(board_size, win_length)
        agent_mcts, _, _ = create_agent_and_opponent(board_size, win_length, replay_maxlen=1)
        self.mcts = agent_mcts
        self.draw_area = DrawArea(self, self.game_board, self.mcts)
        self.initUI()
        self._rules_callback()

    
    def initUI(self):
        '''Initializes the graphical interface through specified button,
           label, and entry specs'''
        self.master.title("Conway's Game of Life")
        self.pack(fill=BOTH, expand=True)

        btn_specs = [{"text": " Set\nRules", "func":self._rules_callback},
                     {"text": "Reset", "func":self._reset_callback},
                     {"text": "Step", "func":self._step_callback}]
        lbl_specs = [{"text": "max MCTS steps"},
                     {"text": "MCTS epsilon (for eps-greedy search exploration)"},
                     {"text": "Update game rules"},
                     {"text": "Reset game to empty board"},
                     {"text": "Let the agent take a step"}]
        entry_specs = [{"text": default_max_mcts_step},
                       {"text": default_mcts_eps}]
        self.labels = [create_label(self, row=j, col=4, text=spec["text"]) for j, spec in enumerate(lbl_specs)]
        self.entries = [create_entry(self, row=j, col=3, text=spec["text"]) for j, spec in enumerate(entry_specs)]
        self.buttons = [create_button(self, row=j+len(entry_specs), col=3, text=spec["text"], func=spec["func"]) for j, spec in enumerate(btn_specs)]


    def _step_callback(self):
        '''Callback function for button that steps through the game world'''
        step(self.mcts, self.game_board, self.max_mcts_steps, self.mcts_eps)
        self.draw_area.draw()
    

    def _reset_callback(self):
        '''Callback function for button that resets board and mcts'''
        self.game_board.reset()
        self.mcts.reset()
        self.draw_area.draw()

    
    def _rules_callback(self):
        '''Callback function for button that reads game rules through GUI and loads them'''
        entries = [e.get() for e in self.entries]
        try:
            entries = [float(entry) for entry in entries]
        except Exception:
            messagebox.showerror(title="Entry Error", message="Entry has to be a valid number")
        self.max_mcts_steps, self.mcts_eps = entries
        self.draw_area.draw()


class DrawArea(Canvas):
    '''A drawing canvas which has acces to the game world 
       so that it can draw and change it through the user gui'''
    def __init__(self, master, game_board, mcts, size=500, border_thickness=4):
        super().__init__(master, width=size, height=size)
        self.game_board = game_board
        self.mcts = mcts
        self.size=size
        self.border_thickness = border_thickness
        self.config({"background":"black"})
        self.bind("<Button-1>", self.mb1_callback)
        self.grid(row=0, column=0, rowspan=9, padx=5, pady=5, sticky=N+S+W+E)
        self.draw()
    

    def mb1_callback(self, event):
        '''Callback that converts mouse click event coordinates to a change in game world'''
        #convert to indices
        n = self.game_board.size
        size = self.size
        border = self.border_thickness
        grid = (size-(n-1)*border)//n
        start = (size - n*grid - (n-1)*border)//2
        coord = (np.array([event.y, event.x]) - start)//(grid+border)

        #change value
        a = n*coord[0] + coord[1]
        take_action(a, self.mcts, self.game_board)
        self.draw()
        

    def draw(self):
        '''Draws game world on canvas'''
        self.delete("all")
        n = self.game_board.size
        size = self.size
        border = self.border_thickness
        grid = (size-(n-1)*border)//n
        start = (size - n*grid - (n-1)*border)//2

        terminate, coords = self.game_board.check_win_position()
        color = np.array([["white"]*n]*n)
        for coord in coords:
            color[coord[0], coord[1]] = "blue"
        if terminate and not coords:
            color = np.array([["grey"]*n]*n)

        for y in range(n):
            for x in range(n):
                x0 = start+ x*(grid+border)
                y0 = start + y*(grid+border)
                x1 = x0+grid
                y1 = y0+grid
                if self.game_board.board[y, x] == 1:
                    self.create_rectangle(x0, y0, x1, y1, fill=color[y, x])
                    self.create_line(x0+10, y0+10, x1-10, y1-10, fill="black", width=10)
                    self.create_line(x0+10, y1-10, x1-10, y0+10, fill="black", width=10)
                elif self.game_board.board[y, x] == -1:
                    self.create_rectangle(x0, y0, x1, y1, fill=color[y, x])
                    self.create_oval(x0+10, y0+10, x1-10, y1-10, fill="black")
                else:
                    self.create_rectangle(x0, y0, x1, y1, fill=color[y, x])


def main():
    board_size = 8 #make sure that this is the same as the net was trained on
    win_length = 5
    root = Tk()
    root.geometry("900x513+450+100")
    app = Window(root, board_size, win_length)
    root.mainloop()


if __name__ == '__main__':
    main()