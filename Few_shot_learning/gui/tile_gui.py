# Import Module
from tkinter import *
from tkinter.ttk import *
from PIL import Image, ImageTk

# create root window
root = Tk()

# root window title and dimension
root.title("Tile Classification")
# Set geometry(widthxheight)
# root.geometry('350x200')

# Adding widgets to the root window 
# Label(root, text = 'GeeksforGeeks', font =( 
#   'Verdana', 15)).pack(side = TOP, pady = 10) 
  
# Creating a photoimage object to use image 
b = []
p = []

for i in range(12):
	for j in range(16):
		name = "output/River_" + str(i) + "_" + str(j) + ".png"
		image = Image.open(name)
		image = image.resize((64, 64))
		photo = ImageTk.PhotoImage(image)

		p.append(photo)
		photoimage = p[-1]

		# here, image option is used to 
		# set image on button 
		# button.append(Button(root, text = 'Click Me !', image = photoimage).pack(side = TOP))
		# button_border = Frame(root, highlightbackground = "green",highlightthickness = 2, bd=0) 
		button = Button(root, text = 'Click Me !', image = photoimage)
		button.grid(row=i,column=j)
		b.append(button)

Label(root, text="meh").grid()
Entry(root).grid(row = 1,column = 16)

# Execute Tkinter
root.mainloop()