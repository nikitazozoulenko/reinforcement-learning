import time

from tkinter import Tk, Canvas, Text, BOTH, W, N, E, S, messagebox
from tkinter.ttk import Frame, Button, Entry, Label, Style
import numpy as np

from game_of_life import GameWorld, read_coords_from_file, default_birth, default_overpop, default_starve, default_stride, default_game_size


class DrawArea(Canvas):
    '''A drawing canvas which has acces to the game world 
       so that it can draw and change it through the user gui'''
    def __init__(self, master, game_world, size=500, border_thickness=1):
        super().__init__(master, width=size, height=size)
        self.game_world = game_world
        self.size=size
        self.border_thickness = border_thickness
        self.config({"background":"black"})
        self.bind("<Button-1>", self.mb1_callback)
        self.grid(row=0, column=0, rowspan=9, padx=5, pady=5, sticky=N+S+W+E)
        self.draw()
    

    def mb1_callback(self, event):
        '''Callback that converts mouse click event coordinates to a change in game world'''
        #convert to indices
        n = self.game_world.size
        size = self.size
        border = self.border_thickness
        grid = (size-(n-1)*border)//n
        start = (size - n*grid - (n-1)*border)//2
        coord = (np.array([event.y, event.x]) - start)//(grid+border)

        #change value
        value = self.game_world.world[coord[0], coord[1]]
        if value == True:
            self.game_world.world[coord[0], coord[1]]=False
        else:
            self.game_world.world[coord[0], coord[1]]=True
        self.draw()
        

    def draw(self):
        '''Draws game world on canvas'''
        self.delete("all")
        n = self.game_world.size
        size = self.size
        border = self.border_thickness
        grid = (size-(n-1)*border)//n
        start = (size - n*grid - (n-1)*border)//2
        for y in range(n):
            for x in range(n):
                x0 = start+ x*(grid+border)
                y0 = start + y*(grid+border)
                x1 = x0+grid
                y1 = y0+grid
                if self.game_world.world[y, x]:
                    self.create_rectangle(x0, y0, x1, y1, fill="black")
                else:
                    self.create_rectangle(x0, y0, x1, y1, fill="white")


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
    '''User GUI which lets the user interact with a graphical display of 
       the game world and change the game logic at any time'''
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.game_world = GameWorld(default_starve, default_overpop, default_birth, default_stride, size=default_game_size)
        self.draw_area = DrawArea(self, self.game_world)
        self.initUI()

    
    def initUI(self):
        '''Initializes the graphical interface through specified button,
           label, and entry specs'''
        self.master.title("Conway's Game of Life")
        self.pack(fill=BOTH, expand=True)

        btn_specs = [{"text": " Set\nRules", "func":self._rules_callback},
                     {"text": " Load\nCoords", "func":self._coord_callback},
                     {"text": "Step", "func":self._step_callback},
                     {"text": "Animate", "func":self._animate_callback}]
        lbl_specs = [{"text": "A cell starves to death below this many neighbours (Default = "+str(default_starve)+")"},
                     {"text": "A cell dies by overpopulation above this many neighbours (Default = "+str(default_overpop)+")"},
                     {"text": "A cell is birthed if it has this many neighbours (Default = "+str(default_birth)+")"},
                     {"text": "Generation skip stride (Default = "+str(default_stride)+")"},
                     {"text": "Game world size (Default = "+str(default_game_size)+")"},
                     {"text": "Update game rules"},
                     {"text": "Reset game and load coordinates from file"},
                     {"text": "Step one iteration with given stride"},
                     {"text": "Animate 25 steps"}]
        entry_specs = [{"text": default_starve},
                       {"text": default_overpop},
                       {"text": default_birth},
                       {"text": default_stride},
                       {"text": default_game_size}]
        self.labels = [create_label(self, row=j, col=4, text=spec["text"]) for j, spec in enumerate(lbl_specs)]
        self.entries = [create_entry(self, row=j, col=3, text=spec["text"]) for j, spec in enumerate(entry_specs)]
        self.buttons = [create_button(self, row=j+len(entry_specs), col=3, text=spec["text"], func=spec["func"]) for j, spec in enumerate(btn_specs)]


    def _step_callback(self):
        '''Callback function for button that steps through the game world'''
        self.game_world.step()
        self.draw_area.draw()
    

    def _coord_callback(self):
        '''Callback function for button that reads coordinates from file and loads them'''
        coords = read_coords_from_file()
        self.game_world.reset()
        self.game_world.set_from_coords(coords)
        self.draw_area.draw()

    
    def _rules_callback(self):
        '''Callback function for button that reads game rules through GUI and loads them'''
        entries = [e.get() for e in self.entries]
        try:
            entries = [int(entry) for entry in entries]
        except Exception:
            messagebox.showerror(title="Entry Error", message="Rule entry has to be a valid number")
        self.game_world.update_rules(*entries)
        self.draw_area.draw()

    
    def _animate_callback(self):
        '''Callback function for button that animates the game world'''
        for _ in range(25):
            self.game_world.step()
            self.draw_area.draw()
            self.root.update()
            time.sleep(0.05)


def main():
    root = Tk()
    root.geometry("1020x513+450+100")
    app = Window(root)
    root.mainloop()


if __name__ == '__main__':
    main()