import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox


def askdirectory(title="Select folder"):
    root=tk.Tk()
    root.overrideredirect(True)
    root.attributes("-alpha", 0)
    path = filedialog.askdirectory(title=title)
    root.destroy()
    return path

def askYesNo(title="", message=""):
    root=tk.Tk()
    root.overrideredirect(True)
    root.attributes("-alpha", 0)
    answer = messagebox.askyesno(title=title, message=message)
    root.destroy()
    return answer