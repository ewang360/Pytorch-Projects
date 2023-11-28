# Import Module
from tkinter import *
from PIL import Image, ImageTk

classes = ["clouds","darkness"]

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

def click(row,col):
	index = row*16+col
	b[index].config(highlightbackground='yellow')

for i in range(12):
	for j in range(16):
		name = "output/River_" + str(i) + "_" + str(j) + ".png"
		image = Image.open(name)
		image = image.resize((32,32))
		photo = ImageTk.PhotoImage(image)

		p.append(photo)
		photoimage = p[-1]

		# here, image option is used to 
		# set image on button 
		button = Button(root, command=lambda i=i,j=j: click(i,j), highlightbackground='black', image = photoimage)
		button.grid(row=i,column=j)
		b.append(button)

Label(root, text="meh").grid()
Entry(root).grid(row = 1,column = 16)

print("hi")

# Execute Tkinter
root.mainloop()