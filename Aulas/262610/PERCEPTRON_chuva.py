import random
from pathlib import Path

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
            nome_svg = Path(nome_arquivo).with_suffix('.svg')
            print(
                f'Biblioteca "{erro.name}" nao instalada. '
                f'Gerando grafico em SVG: {nome_svg}'
            )
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
        margem_esquerda = 80
        margem_direita = 40
        margem_topo = 60
        margem_baixo = 80
        grafico_largura = largura - margem_esquerda - margem_direita
        grafico_altura = altura - margem_topo - margem_baixo

        def escala_x(valor):
            return margem_esquerda + valor * grafico_largura

        def escala_y(valor):
            return margem_topo + (1 - valor) * grafico_altura

        elementos = [
            '<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{largura}" height="{altura}" viewBox="0 0 {largura} {altura}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<text x="400" y="32" text-anchor="middle" '
            'font-family="Arial" font-size="22" font-weight="bold">'
            'Perceptron - Classificacao Chuva vs Sol</text>',
        ]

        for i in range(11):
            valor = i / 10
            x = escala_x(valor)
            y = escala_y(valor)

            elementos.append(
                f'<line x1="{x:.1f}" y1="{margem_topo}" '
                f'x2="{x:.1f}" y2="{altura - margem_baixo}" '
                'stroke="#dddddd" stroke-width="1"/>'
            )
            elementos.append(
                f'<line x1="{margem_esquerda}" y1="{y:.1f}" '
                f'x2="{largura - margem_direita}" y2="{y:.1f}" '
                'stroke="#dddddd" stroke-width="1"/>'
            )
            elementos.append(
                f'<text x="{x:.1f}" y="{altura - margem_baixo + 24}" '
                'text-anchor="middle" font-family="Arial" font-size="12">'
                f'{valor:.1f}</text>'
            )
            elementos.append(
                f'<text x="{margem_esquerda - 12}" y="{y + 4:.1f}" '
                'text-anchor="end" font-family="Arial" font-size="12">'
                f'{valor:.1f}</text>'
            )

        elementos.extend([
            f'<line x1="{margem_esquerda}" y1="{altura - margem_baixo}" '
            f'x2="{largura - margem_direita}" y2="{altura - margem_baixo}" '
            'stroke="black" stroke-width="2"/>',
            f'<line x1="{margem_esquerda}" y1="{margem_topo}" '
            f'x2="{margem_esquerda}" y2="{altura - margem_baixo}" '
            'stroke="black" stroke-width="2"/>',
            f'<text x="{margem_esquerda + grafico_largura / 2}" '
            f'y="{altura - 25}" text-anchor="middle" '
            'font-family="Arial" font-size="16">Umidade Normalizada</text>',
            f'<text x="22" y="{margem_topo + grafico_altura / 2}" '
            'text-anchor="middle" font-family="Arial" font-size="16" '
            'transform="rotate(-90 22 300)">Pressao Normalizada</text>',
        ])

        if len(self.pesos) >= 3:
            w0, w1, w2 = self.pesos[0], self.pesos[1], self.pesos[2]

            if abs(w2) > 1e-6:
                pontos_linha = []

                for i in range(101):
                    x = i / 100
                    y = -(w0 + w1 * x) / w2

                    if 0 <= y <= 1:
                        pontos_linha.append(f'{escala_x(x):.1f},{escala_y(y):.1f}')

                if pontos_linha:
                    elementos.append(
                        '<polyline points="{}" fill="none" stroke="green" '
                        'stroke-width="4"/>'.format(' '.join(pontos_linha))
                    )

        for amostra, saida in zip(amostras, saidas):
            x = escala_x(amostra[0])
            y = escala_y(amostra[1])

            if saida == 1:
                elementos.append(
                    f'<rect x="{x - 7:.1f}" y="{y - 7:.1f}" width="14" height="14" '
                    'fill="royalblue" stroke="black" stroke-width="1"/>'
                )
            else:
                elementos.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" '
                    'fill="orange" stroke="black" stroke-width="1"/>'
                )

        elementos.extend([
            '<circle cx="610" cy="92" r="8" fill="orange" stroke="black"/>',
            '<text x="628" y="97" font-family="Arial" font-size="14">Sol (-1)</text>',
            '<rect x="602" y="115" width="16" height="16" fill="royalblue" stroke="black"/>',
            '<text x="628" y="128" font-family="Arial" font-size="14">Chuva (1)</text>',
            '<line x1="600" y1="150" x2="620" y2="150" stroke="green" stroke-width="4"/>',
            '<text x="628" y="155" font-family="Arial" font-size="14">Fronteira</text>',
            '</svg>',
        ])

        with open(nome_arquivo, 'w', encoding='utf-8') as arquivo:
            arquivo.write('\n'.join(elementos))

        print(f'Grafico salvo em: {nome_arquivo}')


def main():
    # Amostras para treinamento: [umidade, pressao].
    amostras = [
        [0.20, 0.90],  # Ar seco, pressao alta -> Sol
        [0.35, 0.85],  # Ar seco, pressao alta -> Sol
        [0.15, 0.75],  # Muito seco, pressao media -> Sol
        [0.40, 0.80],  # Umidade media, pressao alta -> Sol
        [0.55, 0.45],  # Umidade media, pressao baixa -> Chuva
        [0.85, 0.30],  # Muito umido, pressao baixa -> Chuva
        [0.90, 0.20],  # Muito umido, pressao muito baixa -> Chuva
        [0.75, 0.35],  # Umido, pressao baixa -> Chuva
    ]

    saidas = [-1, -1, -1, -1, 1, 1, 1, 1]

    rede = Perceptron(amostras, saidas)
    rede.treinar()
    rede.plotar_grafico(amostras, saidas, 'grafico.svg')

    while True:
        x = float(input('Valor da umidade: '))
        y = float(input('Valor da pressao de ar: '))
        print('Valores: ', x, ' , ', y)
        rede.teste([x, y])


if __name__ == '__main__':
    main()
