import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizer_v2.adam import Adam
from collections import deque
import random
import tkinter as tk
from datetime import datetime

# Função para criar o modelo de rede neural
def create_model():
    print('Etapa 1: Criando modelo sequencial')
    model = Sequential()
    print('Etapa 2: Adicionando primeira camada densa (1 neurônio, 9 entradas, função de ativação ReLU)')
    model.add(Dense(50, input_dim=9, activation="relu"))

    print('Etapa 3: Adicionando segunda camada densa (1 neurônio, função de ativação ReLU)')
    model.add(Dense(100, activation="relu"))
    
    print('Etapa 4: Adicionando terceira camada densa (1 neurônio, função de ativação linear)')
    model.add(Dense(50, activation="relu"))
    
    print('Etapa 5: Adicionando camada de saída (9 neurônios, função de ativação linear)')
    model.add(Dense(9, activation="linear"))

    print('Etapa 6: Compilando o modelo (usando erro quadrático médio e otimizador Adam)')
    model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=0.001))

    print('Etapa 7: Retornando o modelo criado')
    return model

# Função para converter o tabuleiro em uma matriz unidimensional
def board_to_input(board):
    input_values = []
    for row in board:
        for value in row:
            input_values.append(value)
    return np.array(input_values)

# Função para converter a matriz unidimensional de volta para o tabuleiro
def input_to_board(input_values):
    board = []
    for i in range(0, 9, 3):
        board.append(input_values[i:i+3].tolist())
    return board

# Função para verificar se o jogo terminou
def is_game_over(board):
    for row in board:
        if row[0] == row[1] == row[2] != 0:
            return True
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] != 0:
            return True
    if board[0][0] == board[1][1] == board[2][2] != 0:
        return True
    if board[0][2] == board[1][1] == board[2][0] != 0:
        return True
    for row in board:
        for value in row:
            if value == 0:
                return False
    return True

# Função para treinar a rede neural jogando partidas aleatórias
# games = 1000 por padrão - pode ser alterado para treinar mais ou menos
# Quanto menos, mais rápido o treinamento será feito (e menos eficiente)
# Quanto mais, mais lento o treinamento será feito (e mais eficiente)

def train_model(model, games=100): 
    print('Etapa 1: Inicializando memória de treinamento')
    memory = deque(maxlen=2000)

    for game in range(games):
        print(f'Etapa 2: Iniciando jogo {game + 1} - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        
        while not is_game_over(board):
            print('Etapa 3: Escolhendo uma jogada e atualizando o tabuleiro')
            old_board = board_to_input(board)
            valid_moves = [(i, j) for i in range(3) for j in range(3) if board[i][j] == 0]
            move = random.choice(valid_moves)
            board[move[0]][move[1]] = 1
            reward = -1 if is_game_over(board) else 0
            new_board = board_to_input(board)
            memory.append((old_board, move, reward, new_board))

            if len(memory) > 32:
                print('Etapa 4: Treinando o modelo com um batch de memórias')
                batch = random.sample(memory, 32)
                for old_board, move, reward, new_board in batch:
                    target = reward
                    if not is_game_over(input_to_board(new_board)):
                        target = reward + 0.95 * np.amax(model.predict(np.array([new_board]))[0])
                    target_f = model.predict(np.array([old_board]))
                    target_f[0][move[0] * 3 + move[1]] = target
                    model.fit(np.array([old_board]), target_f, epochs=1, verbose=0)

# Função para escolher o próximo movimento da IA
def choose_best_move(model, board):
    print('Etapa 1: Identificando jogadas válidas')
    valid_moves = [(i, j) for i in range(3) for j in range(3) if board[i][j] == 0]

    print('Etapa 2: Inicializando o melhor valor e a melhor jogada')
    best_value = -np.inf
    best_move = None

    print('Etapa 3: Avaliando cada jogada válida')
    for move in valid_moves:
        new_board = [row.copy() for row in board]
        new_board[move[0]][move[1]] = 1
        value = model.predict(np.array([board_to_input(new_board)]))[0][move[0] * 3 + move[1]]
        
        print(f'Etapa 4: Comparando o valor da jogada atual ({value}) com o melhor valor ({best_value})')
        if value > best_value:
            best_value = value
            best_move = move
            print(f'Etapa 5: Atualizando o melhor valor e a melhor jogada para {best_value} e {best_move}')

    print('Etapa 6: Retornando a melhor jogada')
    return best_move


def print_board(board):
    print('Etapa 1: Imprimindo tabuleiro')
    for row in board:
        print(" ".join("X" if value == 1 else "O" if value == -1 else "." for value in row))
    print()

def play_against_model(model):
    print('Etapa 1: Inicializando tabuleiro vazio')
    board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    while not is_game_over(board):
        print('Etapa 2: Exibindo tabuleiro')
        print_board(board)
        
        print('Etapa 3: Lendo a jogada do usuário')
        move = tuple(map(int, input("Enter your move (row col): ").split()))
        
        while board[move[0]][move[1]] != 0:
            print("Invalid move. Try again.")
            move = tuple(map(int, input("Enter your move (row col): ").split()))

        print('Etapa 4: Atualizando tabuleiro com a jogada do usuário')
        board[move[0]][move[1]] = -1

        if not is_game_over(board):
            print('Etapa 5: Escolhendo a melhor jogada para a IA')
            best_move = choose_best_move(model, board)
            
            print('Etapa 6: Atualizando tabuleiro com a jogada da IA')
            board[best_move[0]][best_move[1]] = 1

    print('Etapa 7: Exibindo tabuleiro final')
    print_board(board)
    print("Game over!")


# Função para atualizar o tabuleiro na interface gráfica
def update_board_ui(board, buttons):
    print('Etapa 1: Atualizando a interface gráfica do tabuleiro')
    for i in range(3):
        for j in range(3):
            if board[i][j] == 1:
                buttons[i][j].config(text="X", state=tk.DISABLED)
            elif board[i][j] == -1:
                buttons[i][j].config(text="O", state=tk.DISABLED)

def check_winner(board):
    print("Etapa 1: Verificando vitória nas linhas")
    for row in board:
        if row[0] == row[1] == row[2] != 0:
            return row[0]

    print("Etapa 2: Verificando vitória nas colunas")
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] != 0:
            return board[0][col]

    print("Etapa 3: Verificando vitória na diagonal principal")
    if board[0][0] == board[1][1] == board[2][2] != 0:
        return board[0][0]

    print("Etapa 4: Verificando vitória na diagonal secundária")
    if board[0][2] == board[1][1] == board[2][0] != 0:
        return board[0][2]

    print("Etapa 5: Sem vencedor")
    return None

# Função para reiniciar o jogo
def reset_game(board, buttons, wins_label):
    print("Etapa 1: Reiniciando o jogo")
    for i in range(3):
        for j in range(3):
            board[i][j] = 0
            buttons[i][j].config(text=" ", state=tk.NORMAL)
    wins_label["text"] = f"Jogador: {player_wins.get()}, Máquina: {machine_wins.get()}"

def on_click(i, j, board, buttons, model, wins_label):
    print("Etapa 1: Verificando se a posição clicada está vazia")
    if board[i][j] == 0:
        print("Etapa 2: Atualizando tabuleiro com a jogada do usuário")
        board[i][j] = -1
        update_board_ui(board, buttons)

        print("Etapa 3: Verificando vencedor após a jogada do usuário")
        winner = check_winner(board)
        if winner is not None:
            print("Etapa 4: Atualizando pontuação e reiniciando o jogo")
            player_wins.set(player_wins.get() + 1)
            reset_game(board, buttons, wins_label)
        elif not is_game_over(board):
            print("Etapa 5: Escolhendo a melhor jogada para a IA")
            best_move = choose_best_move(model, board)
            board[best_move[0]][best_move[1]] = 1
            update_board_ui(board, buttons)

            print("Etapa 6: Verificando vencedor após a jogada da IA")
            winner = check_winner(board)
            if winner is not None:
                print("Etapa 7: Atualizando pontuação e reiniciando o jogo")
                machine_wins.set(machine_wins.get() + 1)
                reset_game(board, buttons, wins_label)

def play_against_model_gui(model):
    print("Etapa 1: Inicializando tabuleiro vazio")
    board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    print("Etapa 2: Criando a janela da interface gráfica")
    root = tk.Tk()
    root.title("Tic-Tac-Toe")

    print("Etapa 3: Inicializando contadores de vitórias")
    global player_wins, machine_wins
    player_wins = tk.IntVar(value=0)
    machine_wins = tk.IntVar(value=0)

    print("Etapa 4: Criando botões para cada posição do tabuleiro")
    buttons = [[None, None, None] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            buttons[i][j] = tk.Button(root, text=" ", width=10, height=3,
                                      command=lambda i=i, j=j: on_click(i, j, board, buttons, model, wins_label))
            buttons[i][j].grid(row=i, column=j)

    print("Etapa 5: Criando label para exibir número de vitórias")
    wins_label = tk.Label(root, text=f"Jogador: {player_wins.get()}, Máquina: {machine_wins.get()}")
    wins_label.grid(row=3, column=0, columnspan=2)

    print("Etapa 6: Adicionando botão de reiniciar jogo")
    reset_button = tk.Button(root, text="Reiniciar jogo", command=lambda: reset_game(board, buttons, wins_label))
    reset_button.grid(row=3, column=2)

    print("Etapa 7: Iniciando o loop da interface gráfica")
    root.mainloop()


# Treinar o modelo e jogar contra a IA usando a interface gráfica
model = create_model()
train_model(model)
play_against_model_gui(model)


