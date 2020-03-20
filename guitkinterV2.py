from tkinter import *
import DriverActionDistractionV4
import threading 



root=Tk()
root.geometry('400x400')
root.title("Driver Distraction System")
lab1=Label(root,text='Driver Distraction System Using CNN',bg='powder blue',fg='black',font=('arial 16 bold')).pack()
root.config(background='powder blue')

lab2=Label(root,text='Software for alerting the Distracted Driver',font=('arial 16'),bg='white',fg='black').pack()

def  automate():

    timer = threading.Timer(10.0, predict) 
    timer.start()

    timer1 = threading.Timer(20.0, predict) 
    timer1.start()
     
    timer2 = threading.Timer(30.0, predict) 
    timer2.start()
    
     


def predict(): 
    alert = DriverActionDistractionV4.test_driver()
    lab1=Label(root,text=alert,bg='powder blue',fg='Red',font=('arial 16 bold')).pack()

     

# ent1=Entry(root,text="HELLO",font=('arial 13')).pack()

i = 0 
but2 = Button(root,text='Start Evaluating',width=20,bg='brown',fg='white',command = automate).place(x=100,y=200)


root.mainloop()
