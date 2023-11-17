# Import Module
from tkinter import *
from tkinter.ttk import *
import numpy as np

# create root window
root = Tk()

# root window title and dimension
root.title("Welcome to GeekForGeeks")
# Set geometry(widthxheight)
root.geometry('350x200')

# Adding widgets to the root window 
Label(root, text = 'GeeksforGeeks', font =( 
  'Verdana', 15)).pack(side = TOP, pady = 10) 
  
# Creating a photoimage object to use image 
# photos = np.empty(5)
# print(photos.__sizeof__)

b = []
p = []

for i in range(2):
	for j in range(2):
		name = "output/River_" + str(i) + "_" + str(j) + ".png"
		photo = PhotoImage(file = name)
		p.append(photo.subsample(2,2))
		photoimage = p[i*2+j]

		# here, image option is used to 
		# set image on button 
		# button.append(Button(root, text = 'Click Me !', image = photoimage).pack(side = TOP))
		button = Button(root, text = 'Click Me !', image = photoimage)
		button.pack()
		b.append(button)


# Execute Tkinter
root.mainloop()