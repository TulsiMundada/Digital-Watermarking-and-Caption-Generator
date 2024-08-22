# Import module
import tkinter
from tkinter import *
from PIL import ImageTk, Image

# Create object
root = Tk()

# Adjust size
root.geometry("400x400")

frame = Frame(root, width=600, height=400)
frame.pack()
frame.place(anchor='center', relx=0.1, rely=0.1)

img = Image.open("bgimage.png")
# Create an object of tkinter ImageTk
image1 = img.resize((1000, 1000), Image.ANTIALIAS)
imgg = ImageTk.PhotoImage(image1)

# Create a Label Widget to display the text or Image
label = Label(frame,image = imgg)
label.pack()


# Add text
label2 = Label(root, text="Welcome",
               bg="#88cffa")

label2.pack(pady=50)

# Create Frame
frame1 = Frame(root, bg="#88cffa")
frame1.pack(pady=20)

# Add buttons
button1 = Button(frame1, text="Exit")
button1.pack(pady=20)

button2 = Button(frame1, text="Start")
button2.pack(pady=20)



# Execute tkinter
root.mainloop()