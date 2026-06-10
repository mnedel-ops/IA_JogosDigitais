import random

class Perceptron:
    # Metodo construtor.
    def __init__(self, amostras, saidas, taxa_aprendizado=0.1, geracoes=1000, limiar=1):
        self.amostras = [amostra.copy() for amostra in amostras]
        self.saidas = saidas
        self.taxa_aprendizado = taxa_aprendizado
        self.geracoes = geracoes
        self.limiar = limiar
        self.n_amostras = len(amostras)
        self.n_atributos = len(amostras[0])
        self.pesos = []

    def treinar(self):
        # Insere o limiar na posicao 0 de cada amostra.
        for amostra in self.amostras:
            if len(amostra) == self.n_atributos:
                amostra.insert(0, self.limiar)

        # Gera pesos aleatorios entre 0 e 1.
        self.pesos = [self.limiar]
        for _ in range(self.n_atributos):
            self.pesos.append(random.random())

        geracoes = 0
        while True:
            aprendeu = True

            for i in range(self.n_amostras):
                soma = 0

                for j in range(self.n_atributos + 1):
                    soma += self.pesos[j] * self.amostras[i][j]

                saida_gerada = self.funcao_ativacao_signal(soma)

                if saida_gerada != self.saidas[i]:
                    erro = self.saidas[i] - saida_gerada

                    for j in range(self.n_atributos + 1):
                        self.pesos[j] = (
                            self.pesos[j]
                            + self.taxa_aprendizado * erro * self.amostras[i][j]
                        )

                    aprendeu = False

            geracoes += 1

            if aprendeu or geracoes > self.geracoes:
                print('Quantidade de geracoes para aprender: %d\n' % geracoes)
                break

    def teste(self, amostra):
        amostra = amostra.copy()
        amostra.insert(0, self.limiar)

        soma = 0

        for i in range(self.n_atributos + 1):
            soma += self.pesos[i] * amostra[i]

        saida_gerada = self.funcao_ativacao_signal(soma)

        if saida_gerada == 1:
            print('Dia de chuva! Leve um guarda-chuva!')
        else:
            print('Dia de sol! Não esqueça do protetor solar!')

    def funcao_ativacao_signal(self, soma):
        if soma >= 0:
            return 1
        return -1

    def plotar_grafico(self, amostras, saidas, nome_arquivo='grafico.png'):
        try:
            import matplotlib  # type: ignore
            matplotlib.use('Agg')  # Para nao abrir janela de plotagem.
            import matplotlib.pyplot as plt  # type: ignore
            import numpy as np
        except ModuleNotFoundError as erro:
            nome_svg = nome_arquivo.rsplit('.', 1)[0] + '.svg'
            print(f'Biblioteca "{erro.name}" nao instalada. Gerando grafico em SVG.')
            self.plotar_grafico_svg(amostras, saidas, nome_svg)
            return

        fig, ax = plt.subplots(figsize=(8, 6))

        sol_x = [amostras[i][0] for i in range(len(saidas)) if saidas[i] == -1]
        sol_y = [amostras[i][1] for i in range(len(saidas)) if saidas[i] == -1]
        chuva_x = [amostras[i][0] for i in range(len(saidas)) if saidas[i] == 1]
        chuva_y = [amostras[i][1] for i in range(len(saidas)) if saidas[i] == 1]

        ax.scatter(sol_x, sol_y, color='orange', marker='D', s=100, label='Sol (-1)', zorder=5)
        ax.scatter(chuva_x, chuva_y, color='royalblue', marker='s', s=100, label='Chuva (1)', zorder=5)

        w0, w1, w2 = self.pesos[0], self.pesos[1], self.pesos[2]
        x_vals = np.linspace(-0.1, 1.1, 200)

        if abs(w2) > 1e-6:
            y_vals = -(w0 + w1 * x_vals) / w2
            mask = (y_vals >= -0.1) & (y_vals <= 1.1)
            ax.plot(
                x_vals[mask],
                y_vals[mask],
                color='green',
                linewidth=2.5,
                label='Fronteira de Decisao',
                zorder=4,
            )

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Umidade Normalizada', fontsize=12)
        ax.set_ylabel('Pressao Normalizada', fontsize=12)
        ax.set_title('Perceptron - Classificacao Chuva vs Sol', fontsize=14)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(nome_arquivo, dpi=150)
        print(f'Grafico salvo em: {nome_arquivo}')
        plt.close()

    def plotar_grafico_svg(self, amostras, saidas, nome_arquivo='grafico.svg'):
        largura = 800
        altura = 600
        margem_esq = 90
        margem_dir = 40
        margem_topo = 70
        margem_baixo = 80
        x_min, x_max = -0.05, 1.05
        y_min, y_max = -0.05, 1.05
        area_largura = largura - margem_esq - margem_dir
        area_altura = altura - margem_topo - margem_baixo

        def mapa_x(x):
            return margem_esq + (x - x_min) / (x_max - x_min) * area_largura

        def mapa_y(y):
            return margem_topo + (y_max - y) / (y_max - y_min) * area_altura

        linhas = [
            '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600" viewBox="0 0 800 600">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<text x="400" y="35" text-anchor="middle" font-family="Arial" font-size="20" font-weight="bold">Perceptron - Classificacao Chuva vs Sol</text>',
            f'<line x1="{margem_esq}" y1="{margem_topo + area_altura}" x2="{margem_esq + area_largura}" y2="{margem_topo + area_altura}" stroke="black" stroke-width="2"/>',
            f'<line x1="{margem_esq}" y1="{margem_topo}" x2="{margem_esq}" y2="{margem_topo + area_altura}" stroke="black" stroke-width="2"/>',
        ]

        for i in range(6):
            valor = i / 5
            x = mapa_x(valor)
            y = mapa_y(valor)
            linhas.append(f'<line x1="{x:.1f}" y1="{margem_topo}" x2="{x:.1f}" y2="{margem_topo + area_altura}" stroke="#dddddd"/>')
            linhas.append(f'<line x1="{margem_esq}" y1="{y:.1f}" x2="{margem_esq + area_largura}" y2="{y:.1f}" stroke="#dddddd"/>')
            linhas.append(f'<text x="{x:.1f}" y="{margem_topo + area_altura + 25}" text-anchor="middle" font-family="Arial" font-size="12">{valor:.1f}</text>')
            linhas.append(f'<text x="{margem_esq - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{valor:.1f}</text>')

        if len(self.pesos) >= 3:
            w0, w1, w2 = self.pesos[0], self.pesos[1], self.pesos[2]
            pontos_linha = []

            if abs(w2) > 1e-6:
                for x_real in (x_min, x_max):
                    y_real = -(w0 + w1 * x_real) / w2
                    if y_min <= y_real <= y_max:
                        pontos_linha.append((x_real, y_real))

            if abs(w1) > 1e-6:
                for y_real in (y_min, y_max):
                    x_real = -(w0 + w2 * y_real) / w1
                    if x_min <= x_real <= x_max:
                        pontos_linha.append((x_real, y_real))

            if len(pontos_linha) >= 2:
                p1, p2 = pontos_linha[0], pontos_linha[1]
                linhas.append(
                    f'<line x1="{mapa_x(p1[0]):.1f}" y1="{mapa_y(p1[1]):.1f}" '
                    f'x2="{mapa_x(p2[0]):.1f}" y2="{mapa_y(p2[1]):.1f}" '
                    'stroke="green" stroke-width="4"/>'
                )

        for amostra, saida in zip(amostras, saidas):
            x = mapa_x(amostra[0])
            y = mapa_y(amostra[1])

            if saida == 1:
                linhas.append(f'<rect x="{x - 6:.1f}" y="{y - 6:.1f}" width="12" height="12" fill="royalblue"/>')
            else:
                pontos = f'{x:.1f},{y - 8:.1f} {x + 8:.1f},{y:.1f} {x:.1f},{y + 8:.1f} {x - 8:.1f},{y:.1f}'
                linhas.append(f'<polygon points="{pontos}" fill="orange"/>')

        linhas.extend([
            '<text x="400" y="570" text-anchor="middle" font-family="Arial" font-size="15">Umidade Normalizada</text>',
            '<text x="25" y="300" text-anchor="middle" font-family="Arial" font-size="15" transform="rotate(-90 25 300)">Pressao Normalizada</text>',
            '<rect x="575" y="75" width="14" height="14" fill="royalblue"/>',
            '<text x="598" y="87" font-family="Arial" font-size="13">Chuva (1)</text>',
            '<polygon points="582,110 590,118 582,126 574,118" fill="orange"/>',
            '<text x="598" y="122" font-family="Arial" font-size="13">Sol (-1)</text>',
            '<line x1="574" y1="145" x2="590" y2="145" stroke="green" stroke-width="4"/>',
            '<text x="598" y="149" font-family="Arial" font-size="13">Fronteira de Decisao</text>',
            '</svg>',
        ])

        with open(nome_arquivo, 'w', encoding='utf-8') as arquivo:
            arquivo.write('\n'.join(linhas))

        print(f'Grafico salvo em: {nome_arquivo}')


def main():
    # Amostras para treinamento: [umidade, pressao].
    amostras = [
            # --- Sol ---
    [0.20, 0.90],  # Ar seco, pressão alta
    [0.35, 0.85],  # Ar seco, pressão alta
    [0.15, 0.75],  # Muito seco, pressão média
    [0.40, 0.80],  # Umidade média-baixa, pressão alta
    [0.10, 0.95],  # Muito seco, pressão muito alta
    [0.25, 0.70],  # Seco, pressão média-alta
    [0.30, 0.88],  # Seco, pressão alta
    [0.45, 0.72],  # Umidade média-baixa, pressão média-alta
    [0.18, 0.82],  # Muito seco, pressão alta
    [0.38, 0.91],  # Seco, pressão muito alta
    [0.12, 0.68],  # Muito seco, pressão média
    [0.42, 0.77],  # Umidade média-baixa, pressão alta
    # --- Chuva ---
    [0.55, 0.45],  # Umidade média, pressão baixa
    [0.85, 0.30],  # Muito úmido, pressão baixa
    [0.90, 0.20],  # Muito úmido, pressão muito baixa
    [0.75, 0.35],  # Úmido, pressão baixa
    [0.95, 0.10],  # Saturado, pressão muito baixa
    [0.70, 0.25],  # Úmido, pressão muito baixa
    [0.60, 0.40],  # Úmido, pressão baixa
    [0.80, 0.15],  # Muito úmido, pressão muito baixa
    [0.65, 0.32],  # Úmido, pressão baixa
    [0.78, 0.42],  # Muito úmido, pressão baixa
    [0.88, 0.22],  # Muito úmido, pressão muito baixa
    [0.58, 0.38],  # Úmido, pressão baixa
    ]

    saidas = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
           1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]


    rede = Perceptron(amostras, saidas)
    rede.treinar()
    rede.plotar_grafico(amostras, saidas, 'grafico.png')

    while True:
        x = float(input('Valor da umidade: '))
        y = float(input('Valor da pressao de ar: '))
        print('Valores: ', x, ' , ', y)
        rede.teste([x, y])


if __name__ == '__main__':
    main()
