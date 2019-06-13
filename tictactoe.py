from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import random
import json
import sys

class TTTagent:
    def __init__(self,discountFactor=0.2,learningRate=0.005,NNagent=True):
        """ 
            Input : 
                - discountFactor : the discount factor hyperparameter.
                - learningRate : the learning rate hyperparameter.
                - NNagent : specifies wether a deep Q learning approach is used or simply a Q learning approach.
        """
        self.discountFactor=discountFactor
        self.learningRate=learningRate
        self.NNagent=NNagent
        if NNagent:
            self.createModels()
        else:
            self.table=np.zeros((3,3,3,3,3,3,3,3,3,9))
        self.state=None
        self.randomMoveProb=0
        self.states=list()
        self.predictions=list()
    
    def createModels(self):
        """ Initializes the identical Neural Network models """
        self.model = tf.keras.Sequential()
        self.model.add(layers.Dense(64, activation='relu', input_dim=9,kernel_initializer=tf.keras.initializers.glorot_normal,bias_initializer=tf.keras.initializers.constant(0))) 
        self.model.add(layers.Dropout(0.05))
        self.model.add(layers.Dense(64, activation='relu',kernel_initializer=tf.keras.initializers.glorot_normal,bias_initializer=tf.keras.initializers.constant(0) ))       
        self.model.add(layers.Dropout(0.05))
        self.model.add(layers.Dense(9,activation="linear"))
        self.model.compile(loss="mse", optimizer=tf.train.AdamOptimizer(self.learningRate))
        
        model_copy = tf.keras.Sequential()
        model_copy.add(layers.Dense(64, activation='relu', input_dim=9,kernel_initializer=tf.keras.initializers.glorot_normal,bias_initializer=tf.keras.initializers.constant(0))) 
        model_copy.add(layers.Dropout(0.05))
        model_copy.add(layers.Dense(64, activation='relu',kernel_initializer=tf.keras.initializers.glorot_normal,bias_initializer=tf.keras.initializers.constant(0) ))       
        model_copy.add(layers.Dropout(0.05))
        model_copy.add(layers.Dense(9,activation="linear"))
        model_copy.compile(loss="mse", optimizer=tf.train.AdamOptimizer(self.learningRate))
        self.model_copy=model_copy
        
    
    def play(self,state,actions):
        """ Decides which action to make.
            Input : 
                - state : current state of the game.
                - actions : a list of possible actions.
            Output : 
                - choice : action to be made by the player
        """
        selectMove=random.uniform(0,1)
        self.state=state
        predictions=list()
        if selectMove<self.randomMoveProb:
            #Selects a random action
            choices= [pos for pos,item in enumerate(state) if item==0]
            choice = random.choice(choices)
            return actions[choice]
        else:
            if self.NNagent:
                #Uses its Neural Network to decide the next possible action
                choices= [pos for pos,item in enumerate(state) if item!=0]
                predictions=self.model.predict(np.array(self.state).reshape((1,9)))[0]
                predictions[choices]=np.amin(predictions)-1
                action=actions[np.argmax(predictions)]
            else:
                #Uses its Q-table to decide the next possible action
                table=self.table[state]
                action=actions[np.argmax(table)]
            return action
    
    def update(self,newState,iplayed,lastAction,reward,finished):
        """ Updates itself based on the latest state and action in the game
            Input : 
                - newState : current state of the game.
                - iplayed : whether this player played the last action or not.
                - lastAction : last action played.
                - reward : reward provided by the game.
        """
        #updates only if this played did the last action or if the game is finished
        if iplayed or finished:
            if iplayed:
                self.lastAction=lastAction
            if not self.NNagent:
                #Updating Q-table
                new_Qvalue=(1-self.learningRate)*self.table[self.state][lastAction]+self.learningRate*reward
                if not finished:
                    new_Qvalue+=self.learningRate*(self.discountFactor*np.amax(self.table[newState]))
                self.table[self.state][lastAction]=new_Qvalue
            else:    
                #Processing and storing new data in memory
                choices= [pos for pos,item in enumerate(newState) if item!=0]
                new_Qvalue=reward
                if not finished:
                    prediction=self.model_copy.predict(np.array(newState).reshape((1,9)))[0]
                    prediction[choices]=np.amin(prediction)-1
                    new_Qvalue += self.discountFactor*np.amax(prediction)
                predictions=self.model.predict(np.array(self.state).reshape((1,9)))
                predictions[0][self.lastAction]=new_Qvalue
                self.states.append(np.array(self.state))
                self.predictions.append(predictions[0])
            self.state=newState
            
    def fit_game(self):
        """ Updates the model after the game has finished."""
        self.model.fit(np.array(self.states), np.array(self.predictions), epochs=1)
        self.states=list()
        self.predictions=list()
    
    
    def update_model(self):
        """ Updates the copy of the model"""
        self.model_copy.set_weights(self.model.get_weights())
    
    def save_model(self,name,path=None):
        """ Saves the model weights.
            Input : 
                - name : name of the player or agent.
                - path : path of the save file.
        """
        if self.NNagent:
            if path is None:
                path='./models/player_'+name
            self.model.save_weights(path +".h5" , save_format='h5')
    
    def load_model(self,name,path):
        """ loads the model weights.
            Input : 
                - name : name of the player or agent.
                - path : path of the save file.
        """
        if self.NNagent:
            if path is None:
                path='./models/player_'+name
            self.model.load_weights(path+'.h5')
            self.model_copy.load_weights(path+'.h5')
    
    def isNNagent(self):
        return self.NNagent
    
    def setRandomMoveProb(self,prob):
        self.randomMoveProb=prob
    
    
    def train(self,numberGames=1000,player=None,enemy_type="random",MinrandomMoveProb=0.3):
        """ Trains the model against the specified enemy type using a decaying epsilon greedy method. 
            Input : 
                - numberGames : number of games to be trained in.
                - player : predefined player to use in training.
                - enemy_type : type of the enemy to be trained on.
                - MinrandomMoveProb : minimum probability for selecting a random action.
            Output :
                - training_results : (wins,losses,draws)
        """
        enemy=Player("Enemy",enemy_type)
        if enemy_type=="TTTagent":
            enemy.load('./models/player_AI')
        results=[0,0,0]
        if player==None:
            me=Player("Me","TTTagent",self)
        else:
            me=player
        for i in range(numberGames):
            self.randomMoveProb=max(1-i/numberGames,MinrandomMoveProb)
            game=TicTacToe(me,enemy)
            print("training game "+str(i)+" : ",end="")
            winner=game.play(False)   
            if self.NNagent:
                self.fit_game()
                #after every 100 iterations the model copy is updated
                if i%100==99:
                    self.update_model()
            if winner==None:
                results[2]+=1
            elif winner==enemy:
                results[1]+=1
            else:
                results[0]+=1
        self.randomMoveProb=0
        print("wins = "+str(results[0])+" || losses = "+str(results[1])+" || draws = "+str(results[2]))
        
    
    def test(self,numberGames=200,player=None,enemy_type="random"):
        """ 
            Input : 
                - numberGames : number of games to be tested in.
                - player : predefined player to use in training.
                - enemy_type : type of the enemy to be tested on.
            Output :
                - test_results : (wins,losses,draws)
        """
        enemy=Player("Enemy",enemy_type)
        if enemy_type=="TTTagent":
            enemy.load('./models/player_AI')
        results=[0,0,0]
        if player==None:
            me=Player("Me","TTTagent",self)
        else:
            me=player
        for i in range(numberGames):
            game=TicTacToe(me,enemy)
            print("test game "+str(i)+" : ",end="")
            winner=game.play(False)   
            if winner==None:
                results[2]+=1
            elif winner==enemy:
                results[1]+=1
            else:
                results[0]+=1
        print("wins = "+str(results[0])+" || losses = "+str(results[1])+" || draws = "+str(results[2]))
        return results
        
        
        
class Player:
    """A player in the game TicTacToe"""
    def __init__(self,name,ptype="random",agent=None):
        """ Input : 
                 - name : name of the player.
                 - ptype : can be either a human player, a bot who randomely selects a move or an agent.
                 - agent : predefined agent.
        """
        assert ptype in ["human","random","TTTagent"]
        assert not(ptype!="TTTagent" and agent!=None)
        self.ptype=ptype
        self.name=name
        if ptype=="TTTagent":
            if agent is None:
                self.agent=TTTagent()
            else:
                self.agent=agent
    
    def train(self,numberGames=200,enemy_type="random"):
        """ 
            Input : 
                - numberGames : number of games to be trained in.
                - enemy_type : type of the enemy to be trained on.
            Output :
                - training_results : (wins,losses,draws)
        """
        if not(self.ptype=="TTTagent"):
            print("I'm already as good as I can ever be !!!")
        else:
            return self.agent.train(numberGames,self,enemy_type)
            
            
    def test(self,numberGames=200,enemy_type="random"):
        """ 
            Input : 
                - numberGames : number of games to be tested in.
                - enemy_type : type of the enemy to be tested on.
            Output :
                - test_results : (wins,losses,draws)
        """
        if not(self.ptype=="TTTagent"):
            print("I don't need testing !!!")
        else:
            return self.agent.test(numberGames,self,enemy_type)
    
    def play(self,state=None,actions=None):
        """ 
            Input : 
                - state : current state of the game.
                - actions : list of possible actions.
            
            Output :
                - choice : next move of the player ( should be an element in actions variable ).
        """
        assert not(self.ptype!="human" and state==None and actions==None)
        if self.ptype=="TTTagent":
            return self.agent.play(state,actions)
        elif self.ptype=="random":
            choices= [pos for pos,item in enumerate(state) if item==0]
            choice = random.choice(choices)
            return actions[choice]
        else:
            print('Write coordinates of the cell you want to play, example 00 for the left top cell  ')
            choice = input(self.name+' : ')
            print("\n",end="")
            return choice 
        
    def updateState(self,newState,iplayed,lastAction,reward,finished):
        """ Updates the players on the new state.
            Input : 
                - newState : current state of the game.
                - iplayed : boolean showing if this player played the last move.
                - lastAction : last move played.
                - reward : reward given by the game.
                - finshed : secifies wether the game is finished or not.
        """
        if self.ptype=="TTTagent":
            self.agent.update(newState,iplayed,lastAction,reward,finished)
            
    def save(self,path=None):
        """ 
            Input : 
                - path : path for save file.
        """
        if self.ptype=="TTTagent":
            self.agent.save_model(self.name,path)
    
    def load(self,path=None):
        """ 
            Input : 
                - path : path for save file.
        """
        if self.ptype=="TTTagent":
            self.agent.load_model(self.name,path)
    
    
    def isHuman(self):
        return self.ptype=="human"
    
    def getName(self):
        return self.name
        
        
class TicTacToe:
    """ TicTacToe game main class. """
    
    def __init__(self,player1,player2):
        """ 
            Input : 
                - player1 : first player of the game of the type Player.
                - player2 : second player of the game of the type Player.
        """
        assert isinstance(player1,Player) and isinstance(player2,Player) 
        
        #randomely select first player
        select_players=random.randint(1,2)
        if select_players==1:
            self.player1=player1
            self.player2=player2
        else:
            self.player1=player2
            self.player2=player1
        self.StateArray=np.zeros(shape=(3,3),dtype=np.int64)
        self.rewards={"default":0,"not empty":0,"win":1,"loss":-1,"draw":0}
        self.actions=list()
        for i in range(3):
            for j in range(3):
                self.actions.append(str(i)+str(j))


    def getState(self,player_number):
        """ Returns the state of the game in a list format where 1 corresponds to a cell filled by the player requesting the state, -1 a cell filled by his enemy and 0 an empty cell.
            Input :
                - player_number : number of the player requesting the state.
            Output : 
                - state : current state of the game in a list format.
                - player_number : player 1 or player 2.
        """
        
        state = self.StateArray.flatten().tolist()
        m=1
        if player_number==2:
            m=-1
        for i in range(len(state)):
            if state[i]==1:
                state[i]=1*m
            elif state[i]==2:
                state[i]=-1*m
        return state
    
    def updatePlayers(self,winner,player,lastAction,reward,finished):
        """ Update players on the new state and give them their correspoding rewards.
            Input : 
                - winner : the winner if any.
                - player : which player played the last move.
                - lastAction : last move played.
                - reward : reward given by the game.
        """
        lastAction=self.actions.index(lastAction)
        if winner==self.player1:
            self.player1.updateState(self.getState(1),player==self.player1,lastAction,reward,finished)
            self.player2.updateState(self.getState(2),not player==self.player1,lastAction,-1*reward,finished)
        elif winner==self.player2:
            self.player1.updateState(self.getState(1),player==self.player1,lastAction,-1*reward,finished)
            self.player2.updateState(self.getState(2),not player==self.player1,lastAction,reward,finished)
        else:
            self.player1.updateState(self.getState(1),player==self.player1,lastAction,reward,finished)
            self.player2.updateState(self.getState(2),not player==self.player1,lastAction,reward,finished)
    
    def findWinner(self):
        """
            Output : 
                - player : the player that won the game. None if it's a draw or the game isn't done yet.
                - unfinished : False if the game is done.
        """
        if ((self.StateArray[2][0]==self.StateArray[1][1]==self.StateArray[0][2] or self.StateArray[0][0]==self.StateArray[1][1]==self.StateArray[2][2]) and self.StateArray[1][1]!=0):
            if (self.StateArray[1][1]==1):
                return(self.player1,False)
            else:
                return(self.player2,False)
        for i in range(self.StateArray.shape[0]):
            if (self.StateArray[i][0]==self.StateArray[i][1]==self.StateArray[i][2] and self.StateArray[i][0]!=0):
                if (self.StateArray[i][0]==1):
                    return(self.player1,False)
                else:
                    return(self.player2,False)
            if (self.StateArray[0][i]==self.StateArray[1][i]==self.StateArray[2][i] and self.StateArray[0][i]!=0):
                if (self.StateArray[0][i]==1):
                    return(self.player1,False)
                else:
                    return(self.player2,False)
        unfinished=False
        for x in range(3):
            if self.StateArray[x][0]==0 or self.StateArray[x][1]==0 or self.StateArray[x][2]==0:
               unfinished=True
               break
        return (None,unfinished)
    
    def play(self,show=True):
        """
            Input : 
                - show : True to allow print statements.
        """
        unfinished=True            
        if show:
            print("\n",self,"\n")
            print("=====================================================================================")
        while(unfinished):
            #First player's turn
            correct=False
            while not correct:
                choice = self.player1.play(self.getState(1),self.actions)
                if choice=="exit":
                    return
                correct=True
                for x in choice:
                    if not choice in self.actions:
                        if show:
                            print("Wrong values !!")
                        correct=False
                        break
                if correct :
                    reward=self.rewards["default"]
                    player,unfinished=None,True
                    if self.StateArray[int(choice[0])][int(choice[1])]!=0:
                        if show:
                            print("This case is not empty !!")
                        correct=False
                        reward=self.rewards["not empty"]
                    else:
                        self.StateArray[int(choice[0])][int(choice[1])]=1
                        player,unfinished=self.findWinner()
                        if player==self.player1:
                            reward=self.rewards["win"]
                            print(self.player1.getName()+" has won !!!")
                        elif player==self.player2:
                            reward=self.rewards["loss"]
                            print(self.player2.getName()+" has won !!!")
                        else:
                            if not unfinished:
                                print("It's a draw !!!")
                                reward=self.rewards["draw"]
                    self.updatePlayers(player,self.player1,choice,reward,not unfinished)
            
            if show:
                print("\n",self,"\n")
                print("=====================================================================================")
            if not unfinished:
                break
              
            #second player's turn.    
            correct=False
            while not correct:
                choice = self.player2.play(self.getState(2),self.actions)
                if choice=="exit":
                    return
                correct=True
                for x in choice:
                    if not choice in self.actions:
                        if show:
                            print("Wrong values !!")
                        correct=False
                        break
                if correct :
                    reward=self.rewards["default"]
                    player,unfinished=None,True
                    if self.StateArray[int(choice[0])][int(choice[1])]!=0:
                        if show:
                            print("This case is not empty !!")
                        correct=False
                        reward=self.rewards["not empty"]
                    else:
                        self.StateArray[int(choice[0])][int(choice[1])]=2
                        player,unfinished=self.findWinner()
                        if player==self.player2:
                            reward=self.rewards["win"]
                            print(self.player2.getName()+" has won !!!")
                        elif player==self.player1:
                            reward=self.rewards["loss"]
                            print(self.player1.getName()+" has won !!!")
                        else:
                            if not unfinished:
                                print("It's a draw !!!")
                                reward=self.rewards["draw"]
                    self.updatePlayers(player,self.player2,choice,reward,not unfinished)
                    
            if show:
                print("\n",self,"\n")
                print("=====================================================================================")
        return player
               
                        
                        
    def __str__(self):
        text="( 1 : "+self.player1.getName()+" : X || 2 : "+self.player2.getName()+" : O )\n"
        text+=" |0|1|2|\n"
        text+="--------\n"
        for x in range(self.StateArray.shape[0]):
            text+=str(x)+"|"
            for y in range(self.StateArray.shape[1]):
                if self.StateArray[x][y]==2:
                    text+="O|"
                elif self.StateArray[x][y]==1:
                    text+="X|"
                else:
                    text+=" |"
            text+="\n--------\n"
        return text


def train_agents(numberGames,agent1,agent2):
    """ Trains two agents against each other.
        Input : 
            - numberGames : number of training games.
            - agent1 : first agent to train.
            - agent2 : second agent to train.
    """
    player1=Player("agent1","TTTagent",agent1)
    player2=Player("agent2","TTTagent",agent2)
    results=[0,0,0]
    for i in range(numberGames):
        agent1.setRandomMoveProb(1-i/numberGames)
        agent2.setRandomMoveProb(1-i/numberGames)
        game=TicTacToe(player1,player2)
        print("train game "+str(i)+" : ",end="")
        winner=game.play(False)   
        if agent1.isNNagent():
            agent1.fit_game()
            if i%100==99:
                agent1.update_model()
        if agent2.isNNagent():
            agent2.fit_game()
            if i%100==99:
                agent2.update_model()
        if winner==None:
            results[2]+=1
        elif winner==player2:
            results[1]+=1
        else:
            results[0]+=1
    print("player1 = "+str(results[0])+" || player1 = "+str(results[1])+" || draws = "+str(results[2]))



def verify_arguments():
    arguments=dict()
    if len(sys.argv)==1:
        return arguments
    for i,arg in enumerate(sys.argv):
        if arg=="--myname" or arg=="-m":
            if i+1<len(sys.argv) and not "myname" in arguments:
                arguments["myname"]=sys.argv[i+1]
                del sys.argv[i]
                del sys.argv[i]
                if i<len(sys.argv):
                    arg=sys.argv[i]
                else:
                    break
            else:
                return None
        
        if arg=="--training" or arg=="-r":
            if i+1<len(sys.argv) and not "training" in arguments:
                try:
                    arguments["training"]=int(sys.argv[i+1])
                except TypeError:
                    return None
                del sys.argv[i]
                del sys.argv[i]
                if i<len(sys.argv):
                    arg=sys.argv[i]
                else:
                    break
            else:
                return None
               
        if arg=="--test" or arg=="-e":
            if i+1<len(sys.argv) and not "test" in arguments:
                try:
                    arguments["test"]=int(sys.argv[i+1])
                except TypeError:
                    return None
                del sys.argv[i]
                del sys.argv[i]
                if i<len(sys.argv):
                    arg=sys.argv[i]
                else:
                    break
            else:
                return None
    if len(arguments)!=0 and len(sys.argv)==1:
        return arguments
    else:
        return None
    

arguments = verify_arguments()
if arguments is None:
    print(f'Usage: python {sys.argv[0]} [ --myname <name> ] [ --training <training> ] [ --test <test> ] ')
else:
    if "myname" in arguments:
        name=arguments["myname"]
    else:
        name="me"
    if "training" in arguments:
        training=arguments["training"]
    else:
        training=1000
    if "test" in arguments:
        test=arguments["test"]
    else:
        test=0
            
    player1=Player(name,"human")
    player2=Player("TicTacToeAgent","TTTagent")
    training_results=player2.train(training,"TTTagent")
    if test>0:
        test_results=player2.test(test,"random")        
    play=input("play a game ? (y/n) : ")
    while play!='n':
        game=TicTacToe(player1,player2)
        game.play()  
        play=input("play again ? (y/n) : ") 
