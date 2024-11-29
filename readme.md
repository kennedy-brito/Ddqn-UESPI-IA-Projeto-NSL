# Projeto de IA - Aprendizagem Não-supervisionada

Esse projeto apresenta uma competição entre dois times de 3 agentes. Agentes podem realizar 10 ações diferentes:

0. Mover para a direita
1. Mover para a baixo-direita
2. Mover para baixo
3. Mover para a baixo-esquerda
4. Mover para a esquerda
5. Mover para a cima-esquerda
6. Mover para cima
7. Mover para a cima-direita
8. Atacar (causa sua força em dano nos seus 8-vizinhos)
9. Esperar (não fazer nada)

Modelos devem ser treinados para sair o índice de uma dessas ações (0-9).

**Exemplo de chamada - (treinamento)**: `python main.py -t -t0m="meu_modulo" -t0c="minha_classe" -t1m="random_ai" -t1c="DumbAI" -mw=20 -mh=20`

**Exemplo de chamada - (apresentação)**: `python main.py -p -t0m="meu_modulo" -t0c="minha_classe" -t1m="random_ai" -t1c="DumbAI" -mw=20 -mh=20`

**Chamando o modelo treinado (Ddqn versus modelo aleatório):** `python main.py -p -t0m="ddqn" -t0c="Ddqn" -t1m="random_ai" -t1c="RandomAI" -mw=20 -mh=20 -l0='model/best.pt' `
