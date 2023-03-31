import pygame, sys
import time
from game_objects import *

#%% game settings
level = 1

difficulty = 'medium'
if difficulty != 'easy' and difficulty != 'medium' and difficulty != 'hard':
    raise ValueError('Please choose difficulty easy/medium/hard')

#%%
SIZE = 40 # size of the block in pixels
BACKGROUND_COLOR = (110,110,5)



class Game:
    def __init__(self):
        pygame.init()  
        self.surface = pygame.display.set_mode((1000,800))
        
        pygame.mixer.init()
        # self.play_background_music()
        self.snake = Snake(self.surface, 2)
        self.snake.draw()
        self.apple = Apple(self.surface)
        self.apple.draw()
      
    
    def is_collision(self, x1, y1, x2, y2):   
        if y1 == y2 and x1==x2:
                return True        
        return False
        
    def display_score(self):
        font = pygame.font.SysFont('arial', 30)
        score = font.render(f'Score:{self.snake.length}', True, (255,255,255))
        self.surface.blit(score, (800,10))
      
        
    def show_game_over(self):
        self.render_background()
        font = pygame.font.SysFont('arial', 30)
        line1 = font.render(f"Game is over! Your score is {self.snake.length}", True, (255, 255, 255))
        self.surface.blit(line1, (200, 300))
        line2 = font.render("To play again press Enter. To exit press Escape!", True, (255, 255, 255))
        self.surface.blit(line2, (200, 350))
        pygame.display.flip()
        
        pygame.mixer.music.pause()
        
# =============================================================================
#     def play_sound(self,sound):
#         sound = pygame.mixer.Sound(f'resources/{sound}.mp3')
#         pygame.mixer.Sound.play(sound)
# =============================================================================
    
# =============================================================================
#     def play_background_music(self):
#         pygame.mixer.music.load('resources/bg_music_1.mp3')
#         pygame.mixer.music.play()
# =============================================================================
    
    def render_background(self):
        bg = pygame.image.load('resources/background.jpg')
        self.surface.blit(bg, (0,0))
    
    def play(self):
        self.render_background()
        self.snake.walk()
        self.apple.draw()
        self.display_score()
        pygame.display.flip()
        # snake colliding with apple
        if self.is_collision(self.snake.x[0], self.snake.y[0], self.apple.x, self.apple.y):
            # self.play_sound('ding')
            self.snake.increase_length()
            self.apple.move()
            
        # snake colliding with himself    
        for i in range(3, self.snake.length):
            if self.is_collision(self.snake.x[0], self.snake.y[0],self.snake.x[i], self.snake.y[i]):
                # self.play_sound('crash')
                raise ValueError('Game over')
        
        if level==1:
            for i in range(self.snake.length):
                
                if self.snake.x[i] == -SIZE and self.snake.y[i] <= 800:
                    self.snake.x[i] = 1000
                    self.snake.move_left()
                
                if self.snake.x[i] == 1000 + SIZE and self.snake.y[i] <= 800:
                    self.snake.x[i] = 0
                    self.snake.move_right()
                
                if self.snake.x[i] <= 1000 and self.snake.y[i] == -SIZE:
                    self.snake.y[i] = 800
                    self.snake.move_up()
                
                if self.snake.x[i] <= 1000 and self.snake.y[i] == 800 + SIZE:
                    self.snake.y[i] = 0
                    self.snake.move_down()
                
        if level == 2:
            for i in range(self.snake.length):
                if self.snake.x[0] == -SIZE and self.snake.y[0] <= 800:
                    running = False
                    self.show_game_over()
                    raise ValueError('Game over')
                
                if self.snake.x[0] == 1000 + SIZE and self.snake.y[0] <= 800:
                    running = False
                    self.show_game_over()
                    raise ValueError('Game over')
                
                if self.snake.x[0] <= 1000 and self.snake.y[0] == -SIZE:
                    running = False
                    self.show_game_over()
                    raise ValueError('Game over')
                    
                if self.snake.x[0] <= 1000 and self.snake.y[0] == 800 + SIZE:
                    running = False
                    self.show_game_over()
                    raise ValueError('Game over')
                    
    def reset(self):        
        self.snake = Snake(self.surface, 2)
        self.apple = Apple(self.surface)  
  
    def run(self):
        running = True    
        pause = False
        while running:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN: #came form pygame locals
                
                    if event.key == pygame.K_ESCAPE:
                        # ending program after hitting escape
                        running = False
                    
                        pygame.quit()
                        sys.exit()
                       
                    
                    if event.key == pygame.K_RETURN:
                       pause = False  
                       pygame.mixer.music.unpause()
                     
                    if not pause:   
                        if event.key == pygame.K_UP:
                            self.snake.move_up()      
                        
                        if event.key == pygame.K_DOWN:
                            self.snake.move_down()  
                    
                        if event.key == pygame.K_LEFT:
                            self.snake.move_left()  
                            
                        if event.key == pygame.K_RIGHT:
                            self.snake.move_right()  
                    
                                     
                elif event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    sys.exit()
                    
            try:
                if not pause:
                    self.play() 
            except Exception as e:
                self.show_game_over()
                pause = True
                self.reset()
                print(e)
            if difficulty == 'easy':
                time.sleep(0.2)
            elif difficulty == 'medium':
                time.sleep(0.175)
            elif difficulty == 'hard':
                time.sleep(0.15)

if __name__ == '__main__':
    game = Game()
    game.run()
      
  
             
                
                