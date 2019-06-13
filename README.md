# TicTacToe
An implmentation of TicTacToe and an AI capable of playing it using deep reinforcement learning.

## Description
This is a small project I made to learn and implement Deep Q-learning. It's a basic TicTacToe game with an AI agent capable of learning 	how to play it using Q-learning or deep Q-learning ( both choices are available ). 
This is achieved by rewarding the agent if it wins and penalizing it if it loses. It then updates itself using the Q-value formula.   

## Results
The agent was trained against a player that selects randomly an empty cell to play. Most of the time the agent would win more than 80% of its test games. However, all of the stragtegies it developed were very aggressive and would lose quickly against a real player which isn't surprising considering it's a 2-player game and Q-learning alone won't work for that.

## Install
This project requires Python 3.7 installed and Keras on top of Tensorflow.

## Usage
To try out the script, simply run the command :
    
    python tictactoe.py

Optional arguments : 
  - myname (or m) : name of the player (default "me")
  - training (or r) : number of training games (default 1000)
  - test (or e) : number of test games (default 0)
Example :

    python tictactoe.py --myname khaled --training 5000 --test 1000

You can tweak as well the hyperparameters, save your model, train against a different type of agent, etc.
