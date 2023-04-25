# Script de Jogo da Velha com IA
O script a seguir implementa um jogo da velha com uma Inteligência Artificial (IA) utilizando a biblioteca TensorFlow para treinar uma rede neural que aprende a jogar o jogo. O script contém várias funções para criar e treinar o modelo, jogar contra o modelo e uma interface gráfica para jogar o jogo.

```sh
pip install -r requirements.txt
```

## Funções do script
### 1. Função create_model()
Esta função cria e compila um modelo sequencial de rede neural usando a biblioteca Keras do TensorFlow. O modelo tem várias camadas densas e usa a função de perda de erro quadrático médio e o otimizador Adam.

### 2. Função board_to_input(board) e input_to_board(input_values)
Essas funções são usadas para converter o tabuleiro em uma matriz unidimensional e vice-versa, o que é necessário para alimentar a rede neural com os dados corretos.

### 3. Função is_game_over(board)
Esta função verifica se o jogo terminou, ou seja, se há um vencedor ou se o tabuleiro está cheio.

### 4. Função train_model(model, games=100)
Esta função treina a rede neural jogando um número específico de partidas aleatórias (por padrão, 100 jogos). A IA aprende com cada jogada, atualizando os pesos do modelo.

### 5. Função choose_best_move(model, board)
Esta função usa a rede neural treinada para escolher a melhor jogada para a IA, dado um tabuleiro atual.

### 6. Função print_board(board)
Esta função imprime o tabuleiro no terminal.

### 7. Função play_against_model(model)
Esta função permite que o usuário jogue contra a IA usando o terminal.

### 8. Função update_board_ui(board, buttons) e outras funções relacionadas à interface gráfica
Essas funções são usadas para criar e atualizar a interface gráfica do jogo, permitindo ao usuário jogar contra a IA usando uma interface gráfica simples.

## Como usar o script
- Primeiro, o modelo de rede neural é criado chamando a função create_model().
- Em seguida, o modelo é treinado chamando a função train_model(model).
- Por fim, a função play_against_model_gui(model) é chamada para iniciar a interface gráfica e permitir que o usuário jogue contra a IA treinada.

# Ao executar o script, o usuário pode jogar o jogo da velha contra a IA treinada usando a interface gráfica.