from tkinter import *
import driverDistractionCapturingV3


root=Tk()
root.geometry('400x400')
root.title("Driver Distraction System")
lab1=Label(root,text='Driver Distraction System Using CNN',bg='powder blue',fg='black',font=('arial 16 bold')).pack()
root.config(background='powder blue')

lab2=Label(root,text='Software for alerting the Distracted Driver',font=('arial 16'),bg='powder blue',fg='black').pack()


def predict():
    alert = driverDistractionCapturingV3.test_driver()
    lab1=Label(root,text=alert,bg='powder blue',fg='Red',font=('arial 16 bold')).pack()
    

ent1=Entry(root,text="HELLO",font=('arial 13')).pack()


but2=Button(root,text='Start Evaluating',width=20,bg='brown',fg='white',command=predict).place(x=100,y=150)


root.mainloop()
