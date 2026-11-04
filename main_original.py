import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path
import urllib.request

# =====================================================
# 1. PARAMETROS GENERALES
# =====================================================

CATEGORIA = "cat"
LATENT_DIM = 100
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.0002
MAX_DIBUJOS = 5000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CARPETA = Path(__file__).resolve().parent
DATA = CARPETA / "data"
RESULTADOS = CARPETA / "resultados"

DATA.mkdir(exist_ok=True)
RESULTADOS.mkdir(exist_ok=True)

URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/cat.npy"
ARCHIVO = DATA / "cat.npy"


# =====================================================
# 2. CARGAR LOS DATOS DE GOOGLE QUICK DRAW
# =====================================================

def cargar_datos():

    if not ARCHIVO.exists():
        print("Descargando dibujos de gatos...")
        urllib.request.urlretrieve(URL, ARCHIVO)

    dibujos = np.load(ARCHIVO)
    dibujos = dibujos[:MAX_DIBUJOS]

    # Normalizamos los pixeles entre -1 y 1.
    dibujos = dibujos.astype(np.float32) / 127.5 - 1

    # Convertimos cada imagen en 784 valores.
    dibujos = dibujos.reshape(-1, IMAGE_PIXELS)

    tensor = torch.tensor(dibujos, dtype=torch.float32)

    conjunto = TensorDataset(tensor)

    return DataLoader(
        conjunto,
        batch_size=BATCH_SIZE,
        shuffle=True
    )


# =====================================================
# 3. RED NEURONAL GENERADORA (GENERATOR)
# =====================================================

class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        # AQUI SE CREA LA RED GENERADORA.
        self.red = nn.Sequential(

            # Entrada: 100 valores aleatorios.
            nn.Linear(LATENT_DIM, 256),
            nn.ReLU(),

            # Primera capa oculta: 256 neuronas.
            nn.Linear(256, 512),
            nn.ReLU(),

            # Segunda capa oculta: 512 neuronas.
            nn.Linear(512, 1024),
            nn.ReLU(),

            # Salida: 784 pixeles.
            nn.Linear(1024, IMAGE_PIXELS),
            nn.Tanh()
        )

    def forward(self, ruido):
        return self.red(ruido)


# =====================================================
# 4. RED NEURONAL DISCRIMINADORA (DISCRIMINATOR)
# =====================================================

class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        # AQUI SE CREA LA RED DISCRIMINADORA.
        self.red = nn.Sequential(

            # Entrada: 784 valores de la imagen.
            nn.Linear(IMAGE_PIXELS, 512),
            nn.LeakyReLU(0.2),

            # Primera capa oculta: 512 neuronas.
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),

            # Segunda capa oculta: 256 neuronas.
            nn.Linear(256, 1)
        )

    def forward(self, imagen):
        return self.red(imagen)


# =====================================================
# 5. FUNCION PARA MOSTRAR DIBUJOS GENERADOS
# =====================================================

def guardar_dibujos(generador, epoca):

    generador.eval()

    with torch.no_grad():
        ruido = torch.randn(16, LATENT_DIM).to(DEVICE)
        imagenes = generador(ruido).cpu().numpy()

    figura, ejes = plt.subplots(4, 4, figsize=(6, 6))

    for i, eje in enumerate(ejes.flat):
        dibujo = imagenes[i].reshape(IMAGE_SIZE, IMAGE_SIZE)
        eje.imshow(dibujo, cmap="gray", vmin=-1, vmax=1)
        eje.axis("off")

    figura.suptitle(f"Dibujos generados - Epoca {epoca}")
    figura.tight_layout()
    figura.savefig(RESULTADOS / f"generados_{epoca}.png")
    plt.close(figura)

    generador.train()


# =====================================================
# 6. ENTRENAMIENTO DE LA GAN
# =====================================================

def entrenar():

    datos = cargar_datos()

    # AQUI SE CREA EL GENERADOR.
    generador = Generator().to(DEVICE)

    # AQUI SE CREA EL DISCRIMINADOR.
    discriminador = Discriminator().to(DEVICE)

    print("\nRED GENERADORA:")
    print(generador)

    print("\nRED DISCRIMINADORA:")
    print(discriminador)

    # Funcion de perdida de ambas redes.
    funcion_perdida = nn.BCEWithLogitsLoss()

    # Optimizadores que actualizan los pesos.
    optimizador_G = torch.optim.Adam(
        generador.parameters(),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    optimizador_D = torch.optim.Adam(
        discriminador.parameters(),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    historial_G = []
    historial_D = []

    print("\nINICIANDO ENTRENAMIENTO")
    print("Categoria:", CATEGORIA)
    print("Dispositivo:", DEVICE)

    for epoca in range(1, EPOCHS + 1):

        perdida_G_total = 0
        perdida_D_total = 0
        lotes = 0

        for (reales,) in datos:

            reales = reales.to(DEVICE)
            cantidad = reales.size(0)

            etiquetas_reales = torch.ones(cantidad, 1).to(DEVICE)
            etiquetas_falsas = torch.zeros(cantidad, 1).to(DEVICE)

            # -----------------------------------------
            # ENTRENAMIENTO DEL DISCRIMINADOR
            # -----------------------------------------

            ruido = torch.randn(cantidad, LATENT_DIM).to(DEVICE)

            falsas = generador(ruido).detach()

            salida_real = discriminador(reales)
            salida_falsa = discriminador(falsas)

            perdida_real = funcion_perdida(
                salida_real, etiquetas_reales
            )

            perdida_falsa = funcion_perdida(
                salida_falsa, etiquetas_falsas
            )

            # PERDIDA DEL DISCRIMINADOR.
            loss_discriminator = (
                perdida_real + perdida_falsa
            ) / 2

            optimizador_D.zero_grad()
            loss_discriminator.backward()
            optimizador_D.step()

            # -----------------------------------------
            # ENTRENAMIENTO DEL GENERADOR
            # -----------------------------------------

            for parametro in discriminador.parameters():
                parametro.requires_grad_(False)

            ruido = torch.randn(cantidad, LATENT_DIM).to(DEVICE)
            falsas = generador(ruido)

            salida_generada = discriminador(falsas)

            # El generador intenta engañar al discriminador.
            # Por eso utiliza etiquetas reales (1).
            loss_generator = funcion_perdida(
                salida_generada, etiquetas_reales
            )

            optimizador_G.zero_grad()
            loss_generator.backward()
            optimizador_G.step()

            for parametro in discriminador.parameters():
                parametro.requires_grad_(True)

            perdida_G_total += loss_generator.item()
            perdida_D_total += loss_discriminator.item()
            lotes += 1

        promedio_G = perdida_G_total / lotes
        promedio_D = perdida_D_total / lotes

        historial_G.append(promedio_G)
        historial_D.append(promedio_D)

        print(
            f"Epoca {epoca}/{EPOCHS} | "
            f"Generador: {promedio_G:.4f} | "
            f"Discriminador: {promedio_D:.4f}"
        )

        if epoca == 1 or epoca % 5 == 0:
            guardar_dibujos(generador, epoca)

    # -----------------------------------------
    # MOSTRAR VALORES APRENDIDOS
    # -----------------------------------------

    print("\nVALORES DEL GENERADOR:")
    peso_G = generador.red[0].weight.detach()

    print("Forma:", tuple(peso_G.shape))
    print("Minimo:", peso_G.min().item())
    print("Maximo:", peso_G.max().item())
    print("Promedio:", peso_G.mean().item())

    print("\nVALORES DEL DISCRIMINADOR:")
    peso_D = discriminador.red[0].weight.detach()

    print("Forma:", tuple(peso_D.shape))
    print("Minimo:", peso_D.min().item())
    print("Maximo:", peso_D.max().item())
    print("Promedio:", peso_D.mean().item())

    # -----------------------------------------
    # GRAFICA DE LAS FUNCIONES DE PERDIDA
    # -----------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(historial_G, label="Generador")
    plt.plot(historial_D, label="Discriminador")

    plt.xlabel("Epocas")
    plt.ylabel("Perdida")
    plt.title("Funciones de perdida de la GAN")
    plt.legend()
    plt.grid()

    plt.savefig(RESULTADOS / "grafica_perdidas.png")
    plt.close()

    # Guardar los modelos entrenados.
    torch.save(
        generador.state_dict(),
        RESULTADOS / "generador.pth"
    )

    torch.save(
        discriminador.state_dict(),
        RESULTADOS / "discriminador.pth"
    )

    print("\nENTRENAMIENTO FINALIZADO")
    print("Revisa la carpeta resultados.")


# =====================================================
# 7. EJECUTAR EL PROYECTO
# =====================================================

if __name__ == "__main__":
    entrenar()
