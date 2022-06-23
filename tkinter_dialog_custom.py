import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import os


def askdirectory(title="Select folder", initialdir=""):
    root=tk.Tk()
    root.overrideredirect(True)
    root.attributes("-alpha", 0)
    if initialdir == "" or not os.path.exists(initialdir): initialdir = os.path.dirname(os.path.abspath(__file__))
    path = filedialog.askdirectory(title=title, initialdir=initialdir)
    root.destroy()
    return path

def askYesNo(title="", message=""):
    root=tk.Tk()
    root.overrideredirect(True)
    root.attributes("-alpha", 0)
    answer = messagebox.askyesno(title=title, message=message)
    root.destroy()
    return answer

def askopenfilename(title="Select file", initialdir="",  filetypes =[('All files', '*.*')]):
    root=tk.Tk()
    root.overrideredirect(True)
    root.attributes("-alpha", 0)
    if initialdir == "" or not os.path.exists(initialdir): initialdir = os.path.dirname(os.path.abspath(__file__))
    path = filedialog.askopenfilename(title=title, initialdir=initialdir, filetypes=filetypes)
    root.destroy()
    return path

def choosefromlist(list_=list(), title="Choose one option", width=100):
    root=tk.Tk()
    root.title(title)
    #root.geometry(f"{500}x{300}")
    listbox = tk.Listbox(master=root, selectmode="SINGLE", width=width)
    for i, element in enumerate(list_):
        listbox.insert(i,str(element))
    listbox.pack(fill=tk.BOTH)
    root.update()
    while True:
        chosen = listbox.curselection()
        if len(chosen) > 0: break
        root.update()
    root.destroy()
    return list_[chosen[0]]

