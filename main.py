
# ============================================================
# PROYECTO GAN - CUARTA PRUEBA
# DCGAN MEJORADA PARA GENERAR DIBUJOS DE GATOS
# ============================================================

import csv
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 1. CONFIGURACION GENERAL
# ============================================================

CARPETA = Path(__file__).resolve().parent

CARPETA_DATOS = CARPETA / "data"
CARPETA_RESULTADOS = CARPETA / "resultados_dcgan_v4"

CARPETA_DATOS.mkdir(exist_ok=True)
CARPETA_RESULTADOS.mkdir(exist_ok=True)

CATEGORIA = "cat"

LATENT_DIM = 100
IMAGE_SIZE = 28

BATCH_SIZE = 32
EPOCHS = 30

MAX_DIBUJOS = 30000

LEARNING_RATE_G = 0.0002
LEARNING_RATE_D = 0.0001

ETIQUETA_REAL = 0.9

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(42)
np.random.seed(42)


# ============================================================
# 2. CARGAR EL CONJUNTO DE DATOS
# ============================================================

def cargar_datos():

    archivo = CARPETA_DATOS / f"{CATEGORIA}.npy"

    url = (
        "https://storage.googleapis.com/"
        "quickdraw_dataset/full/numpy_bitmap/"
        f"{CATEGORIA}.npy"
    )

    if not archivo.exists():

        print("Descargando dibujos de gatos...")

        urllib.request.urlretrieve(url, archivo)

    print("\nCargando conjunto de dibujos...")

    dibujos = np.load(archivo)

    print(f"Total de dibujos disponibles: {len(dibujos)}")

    # Seleccionar 30.000 dibujos aleatorios
    cantidad = min(MAX_DIBUJOS, len(dibujos))

    indices = np.random.choice(
        len(dibujos),
        cantidad,
        replace=False
    )

    dibujos = dibujos[indices]

    print(f"Dibujos seleccionados: {len(dibujos)}")

    # Convertir los dibujos a formato de imagen 28x28
    dibujos = dibujos.reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE)

    # Normalizar los valores entre -1 y 1
    dibujos = dibujos.astype(np.float32) / 127.5 - 1.0

    datos = torch.tensor(dibujos, dtype=torch.float32)

    conjunto = TensorDataset(datos)

    cargador = DataLoader(
        conjunto,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )

    print(f"Lotes por epoca: {len(cargador)}")

    return cargador, len(dibujos)


# ============================================================
# 3. RED GENERADORA MEJORADA
# ============================================================

class Generator(nn.Module):

    def __init__(self):

        super().__init__()

        self.red = nn.Sequential(

            # Entrada: ruido aleatorio de 100 dimensiones
            # Genera una primera representacion de 7x7
            nn.ConvTranspose2d(
                LATENT_DIM,
                256,
                kernel_size=7,
                stride=1,
                padding=0,
                bias=False
            ),

            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # Aumenta la imagen de 7x7 a 14x14
            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # Aumenta la imagen de 14x14 a 28x28
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # Produce la imagen final de un canal
            nn.Conv2d(
                64,
                1,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            # Normaliza los valores de salida entre -1 y 1
            nn.Tanh()
        )

    def forward(self, ruido):

        return self.red(ruido)


# ============================================================
# 4. RED DISCRIMINADORA MEJORADA
# ============================================================

class Discriminator(nn.Module):

    def __init__(self):

        super().__init__()

        self.red = nn.Sequential(

            # Reduce la imagen de 28x28 a 14x14
            nn.Conv2d(
                1,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.LeakyReLU(0.2, inplace=True),

            # Reduce la imagen de 14x14 a 7x7
            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Aprende caracteristicas adicionales
            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Genera una puntuacion final
            nn.Conv2d(
                256,
                1,
                kernel_size=7,
                stride=1,
                padding=0
            )
        )

    def forward(self, imagen):

        return self.red(imagen).view(-1, 1)


# ============================================================
# 5. INICIALIZACION DE LOS PESOS
# ============================================================

def inicializar_pesos(modelo):

    for capa in modelo.modules():

        if isinstance(capa, (nn.Conv2d, nn.ConvTranspose2d)):

            nn.init.normal_(
                capa.weight.data,
                0.0,
                0.02
            )

        elif isinstance(capa, nn.BatchNorm2d):

            nn.init.normal_(
                capa.weight.data,
                1.0,
                0.02
            )

            nn.init.constant_(
                capa.bias.data,
                0
            )


# ============================================================
# 6. GENERAR Y GUARDAR IMAGENES
# ============================================================

def guardar_imagenes(generador, ruido_fijo, epoca):

    generador.eval()

    with torch.no_grad():

        imagenes = generador(ruido_fijo).cpu()

    generador.train()

    # Convertir los valores de -1 a 1 en valores de 0 a 1
    imagenes = (imagenes + 1) / 2

    fig, ejes = plt.subplots(4, 4, figsize=(8, 8))

    for i, eje in enumerate(ejes.flat):

        imagen = imagenes[i].squeeze(0).numpy()

        eje.imshow(
            imagen,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        eje.axis("off")

    plt.suptitle(
        f"DCGAN mejorada - Categoria: {CATEGORIA} - Epoca: {epoca}"
    )

    plt.tight_layout()

    archivo = CARPETA_RESULTADOS / f"generados_{epoca}.png"

    plt.savefig(
        archivo,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Imagen guardada: {archivo.name}")


# ============================================================
# 7. GUARDAR LA GRAFICA DE PERDIDAS
# ============================================================

def guardar_grafica(perdidas_g, perdidas_d):

    plt.figure(figsize=(10, 6))

    plt.plot(
        range(1, len(perdidas_g) + 1),
        perdidas_g,
        label="Perdida del generador"
    )

    plt.plot(
        range(1, len(perdidas_d) + 1),
        perdidas_d,
        label="Perdida del discriminador"
    )

    plt.xlabel("Epocas")
    plt.ylabel("Perdida")

    plt.title("Funciones de perdida de la DCGAN mejorada")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        CARPETA_RESULTADOS / "grafica_perdidas.png",
        dpi=150
    )

    plt.close()


# ============================================================
# 8. GUARDAR EL HISTORIAL DEL ENTRENAMIENTO
# ============================================================

def guardar_historial(perdidas_g, perdidas_d):

    archivo = CARPETA_RESULTADOS / "historial_entrenamiento.csv"

    with open(
        archivo,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        escritor = csv.writer(f)

        escritor.writerow([
            "Epoca",
            "Perdida_Generador",
            "Perdida_Discriminador"
        ])

        for i in range(len(perdidas_g)):

            escritor.writerow([
                i + 1,
                perdidas_g[i],
                perdidas_d[i]
            ])


# ============================================================
# 9. ANALISIS DE LOS PESOS APRENDIDOS
# ============================================================

def analizar_pesos(generador, discriminador):

    archivo = CARPETA_RESULTADOS / "valores_pesos.csv"

    with open(
        archivo,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        escritor = csv.writer(f)

        escritor.writerow([
            "Red",
            "Capa",
            "Forma",
            "Minimo",
            "Maximo",
            "Promedio"
        ])

        for nombre_red, modelo in [
            ("Generador", generador),
            ("Discriminador", discriminador)
        ]:

            print(f"\nVALORES APRENDIDOS: {nombre_red.upper()}")

            for nombre, parametro in modelo.named_parameters():

                if "weight" not in nombre:
                    continue

                pesos = parametro.detach().cpu()

                minimo = pesos.min().item()
                maximo = pesos.max().item()
                promedio = pesos.mean().item()

                print(f"\nCapa: {nombre}")
                print(f"Forma: {tuple(pesos.shape)}")
                print(f"Minimo: {minimo:.6f}")
                print(f"Maximo: {maximo:.6f}")
                print(f"Promedio: {promedio:.6f}")

                escritor.writerow([
                    nombre_red,
                    nombre,
                    str(tuple(pesos.shape)),
                    minimo,
                    maximo,
                    promedio
                ])


# ============================================================
# 10. EVALUACION DEL DISCRIMINADOR
# ============================================================

def evaluar_redes(generador, discriminador, cargador):

    generador.eval()
    discriminador.eval()

    datos_reales = next(iter(cargador))[0].to(DEVICE)

    cantidad = min(32, datos_reales.size(0))

    datos_reales = datos_reales[:cantidad]

    ruido = torch.randn(
        cantidad,
        LATENT_DIM,
        1,
        1,
        device=DEVICE
    )

    with torch.no_grad():

        imagenes_falsas = generador(ruido)

        puntuacion_real = torch.sigmoid(
            discriminador(datos_reales)
        ).mean().item()

        puntuacion_falsa = torch.sigmoid(
            discriminador(imagenes_falsas)
        ).mean().item()

    print("\nEVALUACION DEL DISCRIMINADOR")

    print(
        f"Promedio para dibujos reales: {puntuacion_real:.4f}"
    )

    print(
        f"Promedio para dibujos generados: {puntuacion_falsa:.4f}"
    )

    generador.train()
    discriminador.train()


# ============================================================
# 11. ENTRENAMIENTO PRINCIPAL
# ============================================================

def entrenar():

    print("\n========================================")
    print(" DCGAN - CUARTA PRUEBA MEJORADA")
    print("========================================")

    # Cargar dibujos
    cargador, cantidad_dibujos = cargar_datos()

    # Crear las redes
    generador = Generator().to(DEVICE)

    discriminador = Discriminator().to(DEVICE)

    # Inicializar pesos
    generador.apply(inicializar_pesos)

    discriminador.apply(inicializar_pesos)

    print("\nRED GENERADORA:")
    print(generador)

    print("\nRED DISCRIMINADORA:")
    print(discriminador)

    # Funcion de perdida
    funcion_perdida = nn.BCEWithLogitsLoss()

    # Optimizadores independientes
    optimizador_g = optim.Adam(
        generador.parameters(),
        lr=LEARNING_RATE_G,
        betas=(0.5, 0.999)
    )

    optimizador_d = optim.Adam(
        discriminador.parameters(),
        lr=LEARNING_RATE_D,
        betas=(0.5, 0.999)
    )

    # Ruido fijo para comparar las mismas entradas
    ruido_fijo = torch.randn(
        16,
        LATENT_DIM,
        1,
        1,
        device=DEVICE
    )

    perdidas_g = []
    perdidas_d = []

    print("\n========================================")
    print(" CONFIGURACION DEL ENTRENAMIENTO")
    print("========================================")

    print(f"Categoria: {CATEGORIA}")
    print(f"Cantidad de dibujos: {cantidad_dibujos}")
    print(f"Epocas: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Aprendizaje generador: {LEARNING_RATE_G}")
    print(f"Aprendizaje discriminador: {LEARNING_RATE_D}")
    print(f"Etiqueta real: {ETIQUETA_REAL}")
    print(f"Entrada aleatoria: {LATENT_DIM}")
    print(f"Dispositivo: {DEVICE}")

    # --------------------------------------------------------
    # CICLO DE ENTRENAMIENTO
    # --------------------------------------------------------

    for epoca in range(1, EPOCHS + 1):

        perdida_g_total = 0.0
        perdida_d_total = 0.0

        for lote in cargador:

            imagenes_reales = lote[0].to(DEVICE)

            cantidad = imagenes_reales.size(0)

            # Crear etiquetas
            etiquetas_reales = torch.full(
                (cantidad, 1),
                ETIQUETA_REAL,
                device=DEVICE
            )

            etiquetas_falsas = torch.zeros(
                cantidad,
                1,
                device=DEVICE
            )

            # ------------------------------------------------
            # ENTRENAMIENTO DEL DISCRIMINADOR
            # ------------------------------------------------

            optimizador_d.zero_grad(set_to_none=True)

            # Evaluar dibujos reales
            salida_real = discriminador(imagenes_reales)

            perdida_real = funcion_perdida(
                salida_real,
                etiquetas_reales
            )

            # Generar dibujos falsos
            ruido = torch.randn(
                cantidad,
                LATENT_DIM,
                1,
                1,
                device=DEVICE
            )

            imagenes_falsas = generador(ruido)

            # Evaluar dibujos generados
            salida_falsa = discriminador(
                imagenes_falsas.detach()
            )

            perdida_falsa = funcion_perdida(
                salida_falsa,
                etiquetas_falsas
            )

            # Calcular perdida total del discriminador
            perdida_d = (perdida_real + perdida_falsa) / 2

            perdida_d.backward()

            optimizador_d.step()

            # ------------------------------------------------
            # ENTRENAMIENTO DEL GENERADOR
            # ------------------------------------------------

            optimizador_g.zero_grad(set_to_none=True)

            # Congelar los parametros del discriminador
            for parametro in discriminador.parameters():

                parametro.requires_grad_(False)

            # El generador intenta engañar al discriminador
            salida_generada = discriminador(imagenes_falsas)

            perdida_g = funcion_perdida(
                salida_generada,
                etiquetas_reales
            )

            perdida_g.backward()

            optimizador_g.step()

            # Reactivar los parametros del discriminador
            for parametro in discriminador.parameters():

                parametro.requires_grad_(True)

            perdida_g_total += perdida_g.item()
            perdida_d_total += perdida_d.item()

        # Calcular promedios de cada epoca
        promedio_g = perdida_g_total / len(cargador)
        promedio_d = perdida_d_total / len(cargador)

        perdidas_g.append(promedio_g)
        perdidas_d.append(promedio_d)

        print(
            f"Epoca {epoca}/{EPOCHS} | "
            f"Generador: {promedio_g:.4f} | "
            f"Discriminador: {promedio_d:.4f}"
        )

        # Guardar imagenes solamente en las epocas 1, 15 y 30
        if epoca in [1, 15, 30]:

            guardar_imagenes(
                generador,
                ruido_fijo,
                epoca
            )

    # ========================================================
    # 12. RESULTADOS FINALES
    # ========================================================

    print("\n========================================")
    print(" ANALISIS DE LOS PESOS APRENDIDOS")
    print("========================================")

    analizar_pesos(generador, discriminador)

    evaluar_redes(
        generador,
        discriminador,
        cargador
    )

    # Guardar grafica e historial
    guardar_grafica(perdidas_g, perdidas_d)

    guardar_historial(perdidas_g, perdidas_d)

    # Guardar modelos entrenados
    torch.save(
        generador.state_dict(),
        CARPETA_RESULTADOS / "generador_dcgan_v4.pth"
    )

    torch.save(
        discriminador.state_dict(),
        CARPETA_RESULTADOS / "discriminador_dcgan_v4.pth"
    )

    # Guardar configuracion
    configuracion = {
        "categoria": CATEGORIA,
        "cantidad_dibujos": cantidad_dibujos,
        "epocas": EPOCHS,
        "batch_size": BATCH_SIZE,
        "latent_dim": LATENT_DIM,
        "image_size": IMAGE_SIZE,
        "learning_rate_g": LEARNING_RATE_G,
        "learning_rate_d": LEARNING_RATE_D,
        "etiqueta_real": ETIQUETA_REAL
    }

    torch.save(
        configuracion,
        CARPETA_RESULTADOS / "configuracion_v4.pth"
    )

    print("\n========================================")
    print(" ENTRENAMIENTO DCGAN FINALIZADO")
    print("========================================")

    print(f"Categoria: {CATEGORIA}")
    print(f"Dibujos utilizados: {cantidad_dibujos}")
    print(f"Epocas realizadas: {EPOCHS}")

    print(
        f"Resultados guardados en: {CARPETA_RESULTADOS}"
    )


# ============================================================
# 13. EJECUTAR EL PROGRAMA
# ============================================================

if __name__ == "__main__":

    entrenar()
