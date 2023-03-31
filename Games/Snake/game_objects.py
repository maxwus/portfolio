import pygame
import random

SIZE = 40

class Apple:
    def __init__(self, parent_screen):
        self.image = pygame.image.load('resources/apple.jpg').convert()
        self.parent_screen = parent_screen
        self.x = SIZE*3
        self.y = SIZE*3
             
    def draw(self):
        self.parent_screen.blit(self.image, (self.x, self.y))#draws this image
        # pygame.display.flip() 

    def move(self):
        self.x = random.randint(0, 24)*SIZE #windows lenth / SIze of apple 
        self.y = random.randint(0,19)*SIZE
        self.draw()
        


class Snake:
    def __init__(self, parent_screen, length):
        self.length = length
        self.parent_screen = parent_screen
        self.block = pygame.image.load('resources/block.jpg').convert()
        self.x = [SIZE]*length 
        self.y = [SIZE]*length
        self.direction = 'right'
        
    def draw(self):
        for i in range(self.length):
            self.parent_screen.blit(self.block, (self.x[i], self.y[i]))#draws this image
   
    def increase_length(self):
        self.length += 1
        self.x.append(-1)
        self.y.append(-1)
        
    def walk(self):
    
        for i in range(self.length-1,0, -1):
            self.x[i] = self.x[i-1]
            self.y[i] = self.y[i-1]
    
        if self.direction == 'up':
            self.y[0] -= SIZE            
    
        if self.direction == 'down':
           self.y[0] += SIZE
        
        if self.direction == 'right':
           self.x[0] += SIZE
          
        if self.direction == 'left':
           self.x[0] -= SIZE
        
        self.draw()

    def move_left(self):
        self.direction = 'left'

    def move_right(self):
        self.direction = 'right'
        
    def move_up(self):
        self.direction = 'up'
        
    def move_down(self):
        self.direction = 'down'