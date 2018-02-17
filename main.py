# -*- coding: utf-8 -*-
from Tkinter import *
from PIL import Image,ImageDraw,ImageTk
import os
from GCR import *
from numpy import array
import tkMessageBox
from random import randint

from copy import deepcopy




b1 = "up"
xold, yold = None, None



##class Dialog:
##    def __init__(self,parent,img,char):
##        top = self.top = Toplevel(parent)
##        top.title('')
##        top.geometry('200x150')
##
##        self.img = img
##        
##
##        self.photo = ImageTk.PhotoImage(self.img)
##        self.pic = Label(image=self.photo)
##        self.pic.grid(row = 0, column = 0)
##        
##        Label(top,text = 'ეს ასო ').grid(row = 1, column = 0)
##        Label(top,text = char,width = 1).grid(row = 1,column = 1)
##        Label(top,text = ' არის ხო?!').grid(row = 1, column = 2)
##        
##    
##        
##
##        yesbutton = Button(top, text = 'კი', command = self.yes, width = 3)
##        yesbutton.grid(row = 2,column =1)
##        nobutton = Button(top,text = 'არა', command = self.no, width = 3)
##        nobutton.grid(row = 2,column = 2)
##
##
##        
##    def yes(self):
##        
##        print 'yes'
##        self.top.destroy()
##        
##    def no(self):
##
##       print 'no'
##       self.top.destroy()
		


		


		
def main():
	global drawing_area,image,draw
	root = Tk()
	root.geometry('350x500')
	root.title('ასოს ამოცნობა')
	drawing_area = Canvas(root, width = 256, height = 256, bg = 'white')
	drawing_area.grid(row = 0, column = 0)
	drawing_area.bind("<Motion>", motion)
	drawing_area.bind("<ButtonPress-1>", b1down)
	drawing_area.bind("<ButtonRelease-1>", b1up)

	saveButton = Button(bg = 'white',text = 'სურათის შენახვა', width = 15, command = saveim)
	saveButton.grid(row = 4,column = 0)
	
	
	delAllButton = Button(text = 'გასუფთავება',bg = 'white', command =  deleteboard)
	delAllButton.grid(row = 0, column = 1)

	textarea = Entry(width = 17, bg  = 'white')
	textarea.grid(row = 1, column = 0)
	
	
	whichButton = Button(root, bg = 'white',text = '?', width = 15, command = whichCharIsIt)
	whichButton.grid(row = 2,column = 0)

	addButton = Button(root,bg = 'white', text = 'დამატება',  command = lambda: add(textarea) )
	addButton.grid(column = 1, row = 1)

	trainButton = Button(root,bg = 'white', text  = 'სწავლა', width = 15, command = trainOnSamples)
	trainButton.grid(row = 3, column = 0)
	
	image = Image.new("RGB",(256,256),(255,255,255))
	draw = ImageDraw.Draw(image)
	

	
	
	
	
	

	
	
	root.mainloop()



def trainOnSamples():
        train()
	

def add(entry):
        char = entry.get()

        
        
        #entry.insert(0,str(type(char)))
        

        if len(char) == 0 or len(char) > 1:
                entry['bg'] = '#F49C9C'
        else:
                vals = [i.decode('utf-8') for i in alphabet.values()]
                if char in vals:
                        
                        entry['bg'] = 'white'
                        dim = array(image)
                        box = getBox(dim)
                        crimage = image.crop(box)
                        crimage = crimage.resize((16,16),Image.ANTIALIAS)
                        rand = randint(1,100000)
                        rand = str(rand)
                        charkey = getcharkey(char)
                        crimage.save('samples/'+ str(charkey)+'-'+rand+'.png')
                        
                        dim = array(crimage)
                        
                        
                        dim = blackwhite(dim)
                        dim = makelist(dim)
                        
                        inp = makestring(dim)
                        
                        
                        
                        tar = [0]*33
                        tar[charkey] = 1
                        tar = makestring(tar)
                        sample = Sample(inp,tar)
                        try:
                                
                                addSample(sample)
                        except Exception as e:
                                tkMessageBox.showerror('ERROR',e.message)
                else:
                        entry['bg'] = '#F49C9C'
                        
                        
	
		    
		

		    
def whichCharIsIt():
        dim = array(image)
        box = getBox(dim)
        crimage = image.crop(box)
        crimage = crimage.resize((16,16),Image.ANTIALIAS)
        crimage.save('which.jpg')
        dim = array(crimage)
        dim = blackwhite(dim)
        which(dim)
	


	
def deleteboard():
	
	drawing_area.delete('all')

	draw = ImageDraw.Draw(image)
	draw.rectangle([0,0,256,256],fill = (255,255,255))

	
def saveim():
	image.resize((32,32),Image.ANTIALIAS).save('tkinterImagelittle.jpg')
	image.save('tkinterImage.jpg')

	cr = image.crop(getBox(array(image)))
	cr.save('crop.jpg')


	cr = cr.resize((32,32),Image.ANTIALIAS)
	cr.save('32x32.jpg')


	tkMessageBox.showinfo('','შენახულია')
	
	

def b1down(event):
	global b1
	b1 = "down"           

def b1up(event):
	global b1, xold, yold
	b1 = "up"
	xold = None           
	yold = None

def motion(event):
	if b1 == "down":
		global xold, yold
		if xold is not None and yold is not None:
			event.widget.create_line(xold,yold,event.x,event.y,smooth=TRUE,width=5)
			drawing_area.create_line([xold,yold,event.x,event.y],smooth=TRUE,width = 5)
			
			
			draw.line([xold,yold,event.x,event.y],fill = (0,0,0),width=10)
			
						  
		xold = event.x
		yold = event.y

if __name__ == "__main__":
	main()
