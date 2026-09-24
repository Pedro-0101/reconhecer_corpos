### RECONHECER CORPOS
# O objetivo desse codigo é reconhecer e ressaltar corpos em geral em videos, nao exatamente corpos humanos, mas qualquer tipo de corpo ou objeto que se mova.
# Para isso, utilizarei a biblioteca OpenCV para processar o vídeo e detectar de forma visual os corpos em movimento.

# 1. Importar as bibliotecas necessarias.
import argparse
import os
import tkinter

import cv2
import numpy as np
import psutil

# 1.1 Caminhos e limites de arquivo.
# Onde os videos de entrada ficam e onde os resultados sao salvos.
BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
VIDEO_FOLDER = os.path.join(BASE_FOLDER, "videos")        # pasta dos videos de entrada
OUTPUT_FOLDER = os.path.join(BASE_FOLDER, "videos_result")  # pasta dos videos processados
VIDEO_FORMAT = ".mp4"          # extensao aceita para os videos
MAX_SIZE_VIDEO = 1000000000    # tamanho maximo do video em bytes (1GB)
MAX_SIZE_FRAME_WIDTH = 3840    # largura maxima do frame em pixels (4K)
MAX_SIZE_FRAME_HEIGHT = 3840   # altura maxima do frame em pixels (cobre retrato 4K)
MIN_FPS = 1                    # fps minimo aceito para processar o video

# 1.2 Configuracoes de deteccao (ajustadas para funcionar bem na maioria dos casos).
# Edite estes valores aqui em vez de passar muitos parametros na linha de comando.
ESCALA_DETECCAO = 0.5  # fator de reducao do frame usado so na deteccao (menor = mais rapido)
HISTORY = 100          # MOG2: nº de frames do modelo de fundo (menor = adapta mais rapido)
VAR_THRESHOLD = 12.0   # MOG2: sensibilidade (menor = detecta variacoes menores de brilho)
AREA_MIN = 100         # area minima, em pixels do frame original, para ser um objeto
TAMANHO_MIN = 3        # menor lado (largura/altura) minimo em px p/ desenhar a caixa (0 = desligado)
TAMANHO_MAX = 500      # menor lado (largura/altura) maximo em px p/ desenhar a caixa (0 = desligado)
EROSAO = 0             # erosao no mask p/ separar objetos colados (0 = desligado)
ALTURA_SAIDA = 0       # altura do video de saida em px (0 = manter resolucao original)

# 1.3 Configuracoes de exibicao (como os objetos sao desenhados no video).
# Lista de cores no formato BGR. A cor de cada caixa e escolhida pela posicao da
# caixa na faixa de tamanho: a 1a cor e para as menores e a ultima para as maiores.
CORES_CAIXA = [
    # caixas menores
    (255, 255, 255),    # Branco
    (0, 255, 255),      # Amarelo
    (0, 255, 0),        # Verde
    (0, 165, 255),      # Laranja
    (0, 0, 255),        # Vermelho
    # Caixas maiores
]
COR_TAMANHO_REFERENCIA = 20000       # area (px originais) que corresponde a cor final
FONTE = cv2.FONT_HERSHEY_SIMPLEX      # fonte usada no texto da posicao
# Tamanho da fonte do texto, proporcional ao tamanho do retangulo (menor = fonte maior).
FONTE_DIVISOR_RETANGULO = 90          # divide o menor lado do retangulo p/ achar a escala
FONTE_ESCALA_MIN = 0.4                # escala minima da fonte (evita texto ilegivel)
FONTE_ESCALA_MAX = 1.5                # escala maxima da fonte (evita texto gigante)
FONTE_ESPESSURA = 1                   # espessura do traco do texto
MARGEM_TEXTO = 4                      # distancia do texto ate a base do retangulo, em px
# Espessura da linha do retangulo, proporcional ao tamanho da caixa (tende ao menor).
ESPESSURA_MIN = 1                     # espessura minima da linha, em pixels
ESPESSURA_MAX = 5                     # espessura maxima da linha, em pixels
ESPESSURA_DIVISOR_RETANGULO = 60      # divide o menor lado do retangulo p/ achar a espessura


# 2. Carregar os videos.
def listar_videos(
    video_folder=VIDEO_FOLDER,
    video_format=VIDEO_FORMAT,
    max_size_video=MAX_SIZE_VIDEO,
    max_size_frame_width=MAX_SIZE_FRAME_WIDTH,
    max_size_frame_height=MAX_SIZE_FRAME_HEIGHT,
    min_fps=MIN_FPS,
    videos=None,
):
    if videos is None:
        videos = []

    if not os.path.isdir(video_folder):
        os.makedirs(video_folder, exist_ok=True)
        print(f"Pasta de videos criada: {video_folder}. Coloque arquivos {video_format} nela.")
        return videos

    for file in os.listdir(video_folder):
        if not file.lower().endswith(video_format):
            continue

        video_path = os.path.join(video_folder, file)

        # Verificar se o video tem mais de 1GB de tamanho.
        if os.path.getsize(video_path) > max_size_video:
            print(f"Video {file} excede o tamanho maximo de {max_size_video} bytes. Ignorando.")
            continue

        # Abrir o video uma unica vez e ler as propriedades.
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Nao foi possivel abrir o video {file}. Ignorando.")
            cap.release()
            continue

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Verificar se o video tem mais de 1080p de largura.
        if width > max_size_frame_width:
            print(f"Video {file} excede a largura maxima de {max_size_frame_width} pixels. Ignorando.")
            cap.release()
            continue

        # Verificar se o video tem mais de 1080p de altura.
        if height > max_size_frame_height:
            print(f"Video {file} excede a altura maxima de {max_size_frame_height} pixels. Ignorando.")
            cap.release()
            continue

        # Verificar se o video tem menos de 1 frame por segundo.
        if fps < min_fps:
            print(f"Video {file} tem menos de {min_fps} frame por segundo. Ignorando.")
            cap.release()
            continue

        videos.append((video_path, cap))

    return videos


# 2.1 Desenhar uma barra de progresso simples no terminal.
# prefixo (ex.: nome do video) fica na mesma linha da barra.
def exibir_progresso(atual, total, prefixo="", largura=40, uso=None):
    if total > 0:
        progresso = min(atual / total, 1.0)
    else:
        progresso = 0.0
    preenchido = int(largura * progresso)
    barra = "#" * preenchido + "-" * (largura - preenchido)
    if total > 0:
        linha = f"{prefixo}[{barra}] {progresso * 100:5.1f}% ({atual}/{total} frames)"
    else:
        linha = f"{prefixo}[{barra}] {atual} frames processados"
    if uso:
        linha += f" | {uso}"
    print(f"\r{linha}", end="", flush=True)


# 2.2 Coletar o uso de CPU e memoria (do processo e do sistema).
def coletar_uso(processo):
    cpu_proc = processo.cpu_percent(interval=None)
    mem_proc = processo.memory_info().rss / (1024 * 1024)  # MB
    cpu_sys = psutil.cpu_percent(interval=None)
    mem_sys = psutil.virtual_memory()
    return (
        f"CPU proc: {cpu_proc:5.1f}% | RAM proc: {mem_proc:7.1f} MB | "
        f"CPU sist: {cpu_sys:5.1f}% | RAM sist: {mem_sys.percent:5.1f}%"
    )


# 2.3 Detectar objetos no mask e retornar (x, y, w, h, area) em coordenadas da deteccao.
# area_min, tamanho_min e tamanho_max filtram objetos fora do tamanho desejado
# (0 = sem limite).
def detectar_objetos(fgmask, area_min, tamanho_min=0, tamanho_max=0):
    num, _, stats, _ = cv2.connectedComponentsWithStats(fgmask, connectivity=8)
    objetos = []
    for i in range(1, num):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < area_min:
            continue
        lado = min(w, h)
        if tamanho_min > 0 and lado < tamanho_min:
            continue
        if tamanho_max > 0 and lado > tamanho_max:
            continue
        objetos.append((x, y, w, h, area))
    return objetos


# 3. Processar o video frame a frame.
def processar_video(
    video_path,
    cap,
    exibir=True,
    output_folder=OUTPUT_FOLDER,
    escala_deteccao=0.5,
    altura_saida=0,
    history=500,
    var_threshold=16.0,
    area_min=100,
    tamanho_min=0,
    tamanho_max=0,
    erosao=0,
):
    # Parametros do MOG2:
    #  - history menor: o modelo de fundo se adapta mais rapido (bom p/ eventos curtos
    #    como fogos de artificio, que aparecem e somem rapidamente).
    #  - var_threshold menor: mais sensivel, detecta variacoes menores de brilho.
    #  - detectShadows=False: mais rapido e com menos ruido.
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=False,
    )
    window_name = f"Video Processado - {os.path.basename(video_path)}"

    if exibir:
        # Descobrir o tamanho da tela para exibir o video inteiro sem cortes.
        try:
            root = tkinter.Tk()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
        except Exception:
            screen_width, screen_height = 1920, 1080

        # Reservar uma margem para a barra de titulo e bordas da janela.
        max_width = int(screen_width * 0.9)
        max_height = int(screen_height * 0.9)

        # WINDOW_AUTOSIZE exibe o frame 1:1 (sem reescalonamento extra que causa desfoque),
        # pois o redimensionamento ja e feito manualmente abaixo.
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    else:
        # Preparar a escrita do video processado em videos_result.
        os.makedirs(output_folder, exist_ok=True)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Definir a resolucao de saida (mantendo a proporcao).
        if altura_saida and altura_saida < height:
            out_height = altura_saida
            out_width = int(round(width * out_height / height))
        else:
            out_width, out_height = width, height

        nome_base = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_folder, f"{nome_base}_processado{VIDEO_FORMAT}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_atual = 0

        # Preparar a leitura do uso de CPU e memoria.
        processo = psutil.Process(os.getpid())
        processo.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        uso_texto = coletar_uso(processo)

        # Nome do video e barra de progresso na mesma linha.
        prefixo = f"{os.path.basename(video_path):<45} "
        exibir_progresso(0, total_frames, prefixo=prefixo, uso=uso_texto)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not exibir:
            frame_atual += 1
            # Atualizar as estatisticas de uso periodicamente (a cada 10 frames).
            if frame_atual % 10 == 0:
                uso_texto = coletar_uso(processo)

        # Detectar o movimento numa versao reduzida do frame (muito mais rapido em 4K).
        if escala_deteccao != 1.0:
            det_frame = cv2.resize(
                frame, None, fx=escala_deteccao, fy=escala_deteccao,
                interpolation=cv2.INTER_AREA,
            )
        else:
            det_frame = frame

        # Aplicar a subtração de fundo.
        fgmask = fgbg.apply(det_frame)

        # Erosao para separar objetos que estao colados (ex.: varios fogos proximos).
        # Quanto maior o valor, mais os objetos sao separados (ao custo de encolher).
        if erosao > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (erosao * 2 + 1, erosao * 2 + 1)
            )
            fgmask = cv2.erode(fgmask, kernel, iterations=1)

        # Fator para converter as coordenadas da deteccao de volta para o frame original.
        fator = 1.0 / escala_deteccao
        area_min_det = max(1, int(area_min * escala_deteccao * escala_deteccao))
        tamanho_min_det = int(tamanho_min * escala_deteccao)
        tamanho_max_det = int(tamanho_max * escala_deteccao) if tamanho_max > 0 else 0

        # Detectar os corpos em movimento.
        objetos = detectar_objetos(fgmask, area_min_det, tamanho_min_det, tamanho_max_det)

        # Faixa de tamanho que corresponde a cada cor da lista.
        passo_cor = COR_TAMANHO_REFERENCIA / len(CORES_CAIXA)

        # Ressaltar os corpos detectados.
        for x, y, w, h, _ in objetos:
            # Converter as coordenadas para a resolucao original do frame.
            x = int(x * fator)
            y = int(y * fator)
            w = int(w * fator)
            h = int(h * fator)

            # Cor escolhida pela posicao da caixa na faixa de tamanho (area da caixa).
            indice_cor = min(int((w * h) / passo_cor), len(CORES_CAIXA) - 1)
            cor = CORES_CAIXA[indice_cor]

            # Espessura proporcional ao tamanho da caixa, tendendo ao menor valor.
            espessura = min(w, h) / ESPESSURA_DIVISOR_RETANGULO
            espessura = int(max(ESPESSURA_MIN, min(espessura, ESPESSURA_MAX)))

            # Desenhar o retangulo no frame original.
            cv2.rectangle(frame, (x, y), (x + w, y + h), cor, espessura)

            # Escala da fonte proporcional ao tamanho do retangulo, com limites.
            escala_fonte = min(w, h) / FONTE_DIVISOR_RETANGULO
            escala_fonte = max(FONTE_ESCALA_MIN, min(escala_fonte, FONTE_ESCALA_MAX))

            # Centro do retangulo e sua posicao no canto inferior esquerdo.
            centro_x = x + w // 2
            centro_y = y + h // 2
            cv2.putText(
                frame,
                f"({centro_x}, {centro_y})",
                (x + 2, y + h - MARGEM_TEXTO),
                FONTE,
                escala_fonte,
                cor,
                FONTE_ESPESSURA,
            )

        if exibir:
            # Redimensionar o frame para caber na tela mantendo a proporcao,
            # evitando que videos 4K fiquem deslocados e cortados.
            frame_height, frame_width = frame.shape[:2]
            scale = min(max_width / frame_width, max_height / frame_height, 1.0)
            if scale < 1.0:
                display_frame = cv2.resize(
                    frame,
                    (int(frame_width * scale), int(frame_height * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                display_frame = frame

            # Exibir o video processado em tempo real.
            cv2.imshow(window_name, display_frame)

            # Permitir que o usuario pressione 'q' para sair do video.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Redimensionar o frame para a resolucao de saida, se necessario.
            if (out_width, out_height) != (frame.shape[1], frame.shape[0]):
                out_frame = cv2.resize(
                    frame, (out_width, out_height), interpolation=cv2.INTER_AREA
                )
            else:
                out_frame = frame

            # Escrever o frame processado no video de saida.
            writer.write(out_frame)
            exibir_progresso(frame_atual, total_frames, prefixo=prefixo, uso=uso_texto)

    cap.release()
    if exibir:
        cv2.destroyWindow(window_name)
    else:
        writer.release()
        uso_texto = coletar_uso(processo)
        exibir_progresso(
            frame_atual,
            total_frames if total_frames > 0 else frame_atual,
            prefixo=prefixo,
            uso=uso_texto,
        )
        print()


# 4. Detectar os corpos em movimento e exibir/salvar o resultado.
def detectar_corpos_em_movimento(
    videos,
    exibir=True,
    escala_deteccao=0.5,
    altura_saida=0,
    history=500,
    var_threshold=16.0,
    area_min=100,
    tamanho_min=0,
    tamanho_max=0,
    erosao=0,
):
    if not exibir:
        # Cabecalho mostrado uma unica vez, no topo.
        resolucao = f"{altura_saida}px de altura" if altura_saida else "original"
        print(f"Pasta de saida: {OUTPUT_FOLDER}")
        print(f"Resolucao de saida: {resolucao}")

    for video_path, cap in videos:
        processar_video(
            video_path,
            cap,
            exibir=exibir,
            escala_deteccao=escala_deteccao,
            altura_saida=altura_saida,
            history=history,
            var_threshold=var_threshold,
            area_min=area_min,
            tamanho_min=tamanho_min,
            tamanho_max=tamanho_max,
            erosao=erosao,
        )
    if exibir:
        cv2.destroyAllWindows()


# 5. Ponto de entrada.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reconhecer e ressaltar corpos em movimento em videos."
    )
    parser.add_argument(
        "--salvar",
        action="store_true",
        help="Processa sem exibir na tela e salva o resultado em videos_result.",
    )
    parser.add_argument(
        "--altura-saida",
        type=int,
        default=ALTURA_SAIDA,
        help="Altura do video de saida (ex.: 1080). 0 = resolucao original.",
    )
    parser.add_argument(
        "--tamanho-min",
        type=int,
        default=TAMANHO_MIN,
        help="Menor lado (px) minimo para desenhar a caixa. 0 = sem limite.",
    )
    parser.add_argument(
        "--tamanho-max",
        type=int,
        default=TAMANHO_MAX,
        help="Menor lado (px) maximo para desenhar a caixa. 0 = sem limite.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Numero de threads do OpenCV (0 = automatico, usa todos os nucleos).",
    )
    args = parser.parse_args()

    if args.threads > 0:
        cv2.setNumThreads(args.threads)

    videos = listar_videos()
    if not videos:
        print("Nenhum video valido encontrado.")
    else:
        detectar_corpos_em_movimento(
            videos,
            exibir=not args.salvar,
            escala_deteccao=ESCALA_DETECCAO,
            altura_saida=args.altura_saida,
            history=HISTORY,
            var_threshold=VAR_THRESHOLD,
            area_min=AREA_MIN,
            tamanho_min=args.tamanho_min,
            tamanho_max=args.tamanho_max,
            erosao=EROSAO,
        )
