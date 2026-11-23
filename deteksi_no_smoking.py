"""
SISTEM DETEKSI AKTIVITAS DI AREA NO SMOKING
Penerapan Logika Boolean (Matematika Diskrit)

Proposisi:
    P = tangan terdeteksi
    Q = wajah terdeteksi
    R = ujung jari dekat dengan mulut (rasio jarak < threshold)
    S = kondisi P, Q, R bertahan >= N frame berturut-turut

    ALARM = P AND Q AND R AND S

Kontrol keyboard:
    q  -> keluar
    +  -> naikkan threshold (lebih sensitif)
    -  -> turunkan threshold (kurang sensitif)

Instalasi (disarankan Python 3.9 - 3.11):
    pip install opencv-python mediapipe==0.10.14
"""

import math
import itertools

import cv2
import mediapipe as mp

# ====================== PENGATURAN ======================
RASIO_THRESHOLD = 0.25    # jarak jari-mulut / lebar wajah. Atur sesuai uji coba
FRAME_KONFIRMASI = 10     # jumlah frame berturut-turut sebelum alarm menyala
INDEKS_KAMERA = 0         # ganti ke 1 jika webcam tidak terbuka
UJUNG_JARI = [8, 12]      # 8 = ujung telunjuk, 12 = ujung jari tengah
TITIK_MULUT = [13, 14]    # bibir atas dan bibir bawah (bagian dalam) pada Face Mesh
TITIK_PIPI = (234, 454)   # sisi kiri dan kanan wajah, dipakai sebagai skala
# ========================================================


def cetak_tabel_kebenaran():
    """Mencetak tabel kebenaran ALARM = P AND Q AND R AND S ke terminal."""
    print("\nTABEL KEBENARAN: ALARM = P ^ Q ^ R ^ S")
    print("-" * 36)
    print(" P  Q  R  S | ALARM")
    print("-" * 36)
    for p, q, r, s in itertools.product([True, False], repeat=4):
        alarm = p and q and r and s
        print(f" {int(p)}  {int(q)}  {int(r)}  {int(s)} |   {int(alarm)}")
    print("-" * 36 + "\n")


def jarak(titik_a, titik_b):
    """Jarak Euclidean antara dua titik (x, y)."""
    return math.hypot(titik_a[0] - titik_b[0], titik_a[1] - titik_b[1])


def ke_piksel(landmark, lebar, tinggi):
    """Ubah koordinat landmark (0-1) menjadi koordinat piksel."""
    return (landmark.x * lebar, landmark.y * tinggi)


def main():
    cetak_tabel_kebenaran()

    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_mesh
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    face = mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    kamera = cv2.VideoCapture(INDEKS_KAMERA)
    if not kamera.isOpened():
        print("Kamera tidak dapat dibuka. Coba ubah INDEKS_KAMERA.")
        return

    threshold = RASIO_THRESHOLD
    hitung_frame = 0

    while True:
        berhasil, frame = kamera.read()
        if not berhasil:
            break

        frame = cv2.flip(frame, 1)  # cermin agar natural
        tinggi, lebar = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hasil_tangan = hands.process(rgb)
        hasil_wajah = face.process(rgb)

        # --- Proposisi P dan Q ---
        P = hasil_tangan.multi_hand_landmarks is not None
        Q = hasil_wajah.multi_face_landmarks is not None

        # --- Proposisi R ---
        R = False
        rasio = None

        if P and Q:
            lm_wajah = hasil_wajah.multi_face_landmarks[0].landmark

            # titik tengah mulut
            mulut_atas = ke_piksel(lm_wajah[TITIK_MULUT[0]], lebar, tinggi)
            mulut_bawah = ke_piksel(lm_wajah[TITIK_MULUT[1]], lebar, tinggi)
            mulut = ((mulut_atas[0] + mulut_bawah[0]) / 2,
                     (mulut_atas[1] + mulut_bawah[1]) / 2)

            # lebar wajah sebagai skala, supaya tidak bergantung jarak ke kamera
            pipi_kiri = ke_piksel(lm_wajah[TITIK_PIPI[0]], lebar, tinggi)
            pipi_kanan = ke_piksel(lm_wajah[TITIK_PIPI[1]], lebar, tinggi)
            lebar_wajah = jarak(pipi_kiri, pipi_kanan)

            # cari jarak terdekat antara ujung jari manapun ke mulut
            jarak_min = float("inf")
            for tangan in hasil_tangan.multi_hand_landmarks:
                for idx in UJUNG_JARI:
                    ujung = ke_piksel(tangan.landmark[idx], lebar, tinggi)
                    jarak_min = min(jarak_min, jarak(ujung, mulut))

            if lebar_wajah > 0:
                rasio = jarak_min / lebar_wajah
                R = rasio < threshold

            cv2.circle(frame, (int(mulut[0]), int(mulut[1])), 5, (0, 255, 0), -1)

        # --- Proposisi S (konfirmasi beberapa frame) ---
        if P and Q and R:
            hitung_frame += 1
        else:
            hitung_frame = 0
        S = hitung_frame >= FRAME_KONFIRMASI

        # --- Keputusan akhir: Logika Boolean ---
        ALARM = P and Q and R and S

        # --- Gambar landmark tangan ---
        if P:
            for tangan in hasil_tangan.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, tangan, mp_hands.HAND_CONNECTIONS)

        # --- Panel informasi ---
        cv2.rectangle(frame, (0, 0), (330, 175), (30, 30, 30), -1)
        teks = [
            f"P Tangan terdeteksi : {int(P)}",
            f"Q Wajah terdeteksi  : {int(Q)}",
            f"R Jari dekat mulut  : {int(R)}",
            f"S Konfirmasi        : {int(S)} ({min(hitung_frame, FRAME_KONFIRMASI)}/{FRAME_KONFIRMASI})",
            f"Rasio jarak : {rasio:.2f}" if rasio is not None else "Rasio jarak : -",
            f"Threshold   : {threshold:.2f}  (+/-)",
        ]
        for i, baris in enumerate(teks):
            cv2.putText(frame, baris, (10, 25 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # --- Status ---
        if ALARM:
            cv2.rectangle(frame, (0, tinggi - 90), (lebar, tinggi), (0, 0, 255), -1)
            cv2.putText(frame, "!!! DILARANG MEROKOK DI AREA INI !!!",
                        (20, tinggi - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 2)
            cv2.putText(frame, "Matikan rokok Anda",
                        (20, tinggi - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
        else:
            cv2.putText(frame, "STATUS: AMAN", (10, tinggi - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

        cv2.imshow("Deteksi Area No Smoking", frame)

        tombol = cv2.waitKey(1) & 0xFF
        if tombol == ord("q"):
            break
        elif tombol in (ord("+"), ord("=")):
            threshold = min(1.0, threshold + 0.01)
        elif tombol in (ord("-"), ord("_")):
            threshold = max(0.05, threshold - 0.01)

    kamera.release()
    cv2.destroyAllWindows()
    hands.close()
    face.close()


if __name__ == "__main__":
    main()
