import turtle 
import winsound

try:
    tess = turtle.Turtle()
except:
    tess = turtle.Turtle() 

wn = turtle.Screen()
wn.title('Maxos Pong')
wn.bgcolor('black')
wn.setup(width = 800, height = 600)
wn.tracer(0) #updates window, try after finish


score_a = 0
score_b = 0


#Paddle A
paddle_a = turtle.Turtle()
paddle_a.speed(0) #speed of animation, set animation speed to maximum
paddle_a.shape('square')
paddle_a.shapesize(stretch_wid= 5, stretch_len=(1))


paddle_a.color('white')
paddle_a.penup()
paddle_a.goto(-350,0)

#Paddle B
paddle_b = turtle.Turtle()
paddle_b.speed(0) #speed of animation, set animation speed to maximum
paddle_b.shape('square')
paddle_b.shapesize(stretch_wid= 5, stretch_len=(1))
paddle_b.color('white')
paddle_b.penup()
paddle_b.goto(350,0)

#Ball
ball = turtle.Turtle()
ball.speed(0)
ball.shape('square')
ball.color('white')
ball.penup()
ball.dx = 0.2 #maybe change evertime it moves it moves by 2 pixels
ball.dy = 0.2 #maybe change

# Pen
pen = turtle.Turtle()
pen.speed(0)
pen.color('white')
pen.penup()
pen.hideturtle()
pen.goto(0,260)
pen.write('Player A:0  Player B:0', align = 'center', font = ('Courier', 24,'normal'))


def paddle_a_up():
    y = paddle_a.ycor()
    y += 20
    paddle_a.sety(y)
      
def paddle_a_down():
    y = paddle_a.ycor()
    y -= 20
    paddle_a.sety(y)

def paddle_b_up():
    y = paddle_b.ycor()
    y +=20
    paddle_b.sety(y)


def paddle_b_down():
    y = paddle_b.ycor()
    y -= 20
    paddle_b.sety(y)

    

#Keyboard binding
wn.listen()
#movement of paddle_a
wn.onkeypress(paddle_a_up, 'w')
wn.onkeypress(paddle_a_down, 's')
#movement of paddle b
wn.onkeypress(paddle_b_up, 'Up')
wn.onkeypress(paddle_b_down, 'Down')


#Main game loop
while True:
    wn.update() #updates screen everytime it runs

    #Move the ball
    ball.setx(ball.xcor() + ball.dx)
    ball.sety(ball.ycor() + ball.dy)
    
    # #Border checking
    if ball.ycor() > 290:
        ball.sety(290)
        ball.dy *= -1
        winsound.PlaySound('bounce.wav', winsound.SND_ASYNC)  
        
        
    if ball.ycor() < -290:
        ball.sety(-290)
        ball.dy *= -1
        winsound.PlaySound('bounce.wav', winsound.SND_ASYNC) 
        
    if ball.xcor() > 390:
        ball.goto(0,0)
        ball.dx *= -1
        score_a += 1 
        pen.clear()
        pen.write(f'Player A: {score_a}  Player B:{score_b}', align = 'center', font = ('Courier', 24,'normal'))
        
    if ball.xcor() < -390:
        ball.goto(0,0)
        ball.dx *= -1
        score_b += 1
        pen.clear()
        pen.write(f'Player A: {score_a}  Player B:{score_b}', align = 'center', font = ('Courier', 24,'normal'))
        
    #Paddle and ball colisions
    if (ball.xcor() > 340 and ball.xcor() < 350) and (ball.ycor()<paddle_b.ycor() + 40) and (ball.ycor() > paddle_b.ycor() -40):    
        ball.setx(340)
        ball.dx *= -1
        winsound.PlaySound('bounce.wav', winsound.SND_ASYNC)  
        
    if (ball.xcor() < -340 and ball.xcor() < -350) and (ball.ycor()<paddle_a.ycor() + 40) and (ball.ycor() > paddle_a.ycor() -40):    
        ball.setx(-340)
        ball.dx *= -1
        winsound.PlaySound('bounce.wav', winsound.SND_ASYNC) 
        
from sys import platform
if platform=='win32':
    wn.exitonclick()    