import numpy as np 
import matplotlib
from tkinter import Frame, Button, Label, StringVar, Tk, filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class sorter:
    def __init__(self,window,n):
        self.n = n
        self.window = window
        self.data = np.argsort(np.random.uniform(size=n))
        self.swaps = bubble(self.data.copy())[1]
        self.initpaint()
        self.update()
    def initpaint(self):
        self.fig = Figure(figsize=(6,6))
        self.fig.patch.set_facecolor('black')
        self.fig.subplots_adjust(left=0,bottom=0,top=1,right=1)
        self.ax = self.fig.add_subplot(111)
        self.ax.scatter(np.arange(self.n),self.data,color='yellow',edgecolors='black')
        self.ax.axis('off')
        self.canvas = FigureCanvasTkAgg(self.fig,master=self.window)
        self.canvas.get_tk_widget().pack()
        self.canvas.draw()
    def update(self):
        if not self.swaps: return
        # Write code here
        self.window.after(1,self.update)

def bubble(data):
    y = data.copy()
    swaps = []
    return(y,swaps)

root = Tk()
#my_gui = myclass(root)
my_gui = sorter(root,20)
root.mainloop()
